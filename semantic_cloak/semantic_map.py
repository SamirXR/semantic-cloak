r"""semantic_map.py
================
VLM-guided semantic cost-matrix generation for the **Semantic Cloak** steganography
pipeline.

This module uses **real** frozen VLMs (no fallbacks):

* **CLIP** ViT-B/32 — we extract per-patch dense embeddings from
  ``vision_model.last_hidden_state[:, 1:, :]`` (dropping the CLS token) and
  compute cosine similarity to the text prompt embedding for each patch
  independently. The resulting similarity grid is bilinearly upsampled to
  the cover resolution.

* **SAM** ViT-H — we use ``SamAutomaticMaskGenerator`` to produce a stack
  of binary masks, take their union, and compute the boundary-probability
  map via Sobel gradient magnitude of the union.

If the VLM weights are not available, the constructor raises — there is
no silent degradation. The user must install ``transformers`` and
``segment-anything`` and download the checkpoints.

Mathematical formulation (paper §3.2)
-------------------------------------
Let :math:`I \in \mathbb{R}^{H \times W \times 3}` be the cover and
:math:`t` the textual prompt. The CLIP similarity map is

.. math::
    \mathcal{S}^{\mathrm{CLIP}}(i,j) \;=\; \cos\bigl(\phi_V(I)_{(i,j)},\, \phi_T(t)\bigr),

where :math:`\phi_V(I)_{(i,j)}` is the dense CLIP patch embedding at
patch :math:`(i,j)` and :math:`\phi_T(t)` is the text embedding. The
SAM boundary map is

.. math::
    \mathcal{S}^{\mathrm{SAM}}(i,j) \;=\; \frac{\|\nabla M(i,j)\|_2}{\max_{a,b} \|\nabla M(a,b)\|_2},

where :math:`M` is the union of all SAM masks. The fused cost is

.. math::
    \rho_{i,j} \;=\; (1 - \alpha\, \mathcal{S}^{\mathrm{CLIP}}_{i,j}) \;\cdot\; (1 + \beta\, \mathcal{S}^{\mathrm{SAM}}_{i,j}) \;\cdot\; (\epsilon + \gamma\, \mathcal{R}(I)_{i,j}),

with :math:`\mathcal{R}` the WOW directional residual.

Author : Samir (Maharishi Dayanand University)
License: MIT
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# Type aliases
# ----------------------------------------------------------------------
Tensor = torch.Tensor


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
@dataclass
class SemanticMapConfig:
    """Hyper-parameters for the semantic cost-map generator.

    All paths must point to real, downloaded checkpoints. No defaults
    that silently degrade behaviour are provided.
    """

    clip_model_name: str = "openai/clip-vit-base-patch32"
    sam_checkpoint: str = "weights/sam_vit_h_4b8939.pth"
    sam_model_type: str = "vit_h"
    prompt: str = "a smooth unremarkable background region"
    patch_size: int = 32
    alpha: float = 0.6
    beta: float = 4.0
    gamma: float = 1.0
    epsilon: float = 1e-3
    rho_max: float = 100.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------
# CLIP dense encoder (real per-patch similarity, no fallback)
# ----------------------------------------------------------------------
class CLIPDenseEncoder(nn.Module):
    """Wraps a frozen CLIP ViT and exposes a per-patch similarity map.

    The dense map is obtained by extracting the patch-token outputs from
    ``vision_model.last_hidden_state`` (excluding the CLS token at
    position 0), normalising them, and computing the cosine similarity
    with the text embedding. The resulting grid is bilinearly upsampled
    to the cover image resolution.
    """

    def __init__(self, cfg: SemanticMapConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = None
        self.processor = None
        self._text_emb: Optional[Tensor] = None
        self._load()

    def _load(self) -> None:
        """Load CLIP from HuggingFace. Raises if transformers missing."""
        from transformers import CLIPModel, CLIPProcessor  # type: ignore

        self.model = CLIPModel.from_pretrained(self.cfg.clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(self.cfg.clip_model_name)
        self.model.to(self.cfg.device).eval()
        self._text_emb = self._encode_text(self.cfg.prompt)

    @torch.no_grad()
    def _encode_text(self, prompt: str) -> Tensor:
        """Encode the prompt and return an L2-normalised [D] tensor."""
        tok = self.processor(text=[prompt], return_tensors="pt", padding=True)
        tok = {k: v.to(self.cfg.device) for k, v in tok.items()}
        # get_text_features returns the pooled projection-space embedding
        feats = self.model.get_text_features(**tok)  # [1, D_proj]
        return F.normalize(feats, dim=-1).squeeze(0)  # [D_proj]

    @torch.no_grad()
    def dense_similarity(self, image_uint8: Tensor) -> Tensor:
        """Compute the per-patch CLIP similarity map.

        Parameters
        ----------
        image_uint8 : Tensor
            Cover image, shape ``[3, H, W]``, values in ``[0, 255]``.

        Returns
        -------
        Tensor
            Similarity map of shape ``[H, W]``, values in ``[-1, 1]``.
        """
        if self.model is None:
            raise RuntimeError("CLIP model not loaded.")

        h, w = image_uint8.shape[-2:]

        # Preprocess: CLIP expects [0,1] RGB, 224x224
        img = image_uint8.float() / 255.0
        img = F.interpolate(
            img.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
        )

        # Extract patch embeddings from the vision tower
        vision_out = self.model.vision_model(pixel_values=img)
        patch_tokens = vision_out.last_hidden_state[:, 1:, :]  # [1, N_patches, D_vit]
        # Project to the shared CLIP space via the visual projection
        patch_proj = self.model.visual_projection(patch_tokens)  # [1, N_patches, D_proj]
        patch_proj = F.normalize(patch_proj, dim=-1)

        # Per-patch cosine similarity with text embedding
        sim = (patch_proj * self._text_emb.unsqueeze(0)).sum(dim=-1)  # [1, N_patches]
        sim = sim.squeeze(0)  # [N_patches]

        # Reshape to patch grid (CLIP ViT-B/32 uses 7x7 = 49 patches at 224x224)
        # The actual grid size depends on the model; derive it from sqrt.
        n_patches = sim.shape[0]
        grid_h = int(round(math.sqrt(n_patches)))
        grid_w = n_patches // grid_h
        assert grid_h * grid_w == n_patches, (
            f"Patch count {n_patches} is not a perfect rectangle; "
            f"got grid_h={grid_h}, grid_w={grid_w}."
        )
        sim_grid = sim.reshape(1, 1, grid_h, grid_w).float()

        # Bilinearly upsample to cover resolution
        sim_full = F.interpolate(
            sim_grid, size=(h, w), mode="bilinear", align_corners=False
        )
        return sim_full.squeeze().clamp(-1.0, 1.0)


# ----------------------------------------------------------------------
# SAM boundary encoder (real automatic mask generation, no fallback)
# ----------------------------------------------------------------------
class SAMBoundaryEncoder(nn.Module):
    """Wraps SAM and produces a boundary-probability map in [0, 1]."""

    def __init__(self, cfg: SemanticMapConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.predictor = None
        self._load()

    def _load(self) -> None:
        """Load SAM. Raises if checkpoint missing or segment-anything not installed."""
        import os

        if not os.path.isfile(self.cfg.sam_checkpoint):
            raise FileNotFoundError(
                f"SAM checkpoint not found at {self.cfg.sam_checkpoint}. "
                f"Download from "
                f"https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth "
                f"and set sam_checkpoint in SemanticMapConfig."
            )

        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator  # type: ignore

        sam = sam_model_registry[self.cfg.sam_model_type](
            checkpoint=self.cfg.sam_checkpoint
        )
        sam.to(self.cfg.device)
        self.mask_generator = SamAutomaticMaskGenerator(sam)

    @torch.no_grad()
    def boundary_map(self, image_uint8: Tensor) -> Tensor:
        """Return the SAM-derived boundary-probability map.

        Parameters
        ----------
        image_uint8 : Tensor
            Cover image, shape ``[3, H, W]``, values in ``[0, 255]``.

        Returns
        -------
        Tensor
            Boundary map of shape ``[H, W]``, values in ``[0, 1]``.
        """
        h, w = image_uint8.shape[-2:]
        np_img = (
            image_uint8.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        )
        # RGB -> BGR for SAM (cv2 convention used internally)
        # SamAutomaticMaskGenerator actually expects RGB; we keep RGB.

        masks = self.mask_generator.generate(np_img)
        if not masks:
            # No objects detected: empty union -> zero boundary everywhere
            return torch.zeros((h, w), device=self.cfg.device, dtype=torch.float32)

        # Union all masks
        union = np.zeros((h, w), dtype=np.float32)
        for m in masks:
            union += m["segmentation"].astype(np.float32)
        union = (union > 0).astype(np.float32)
        t_union = torch.from_numpy(union).to(self.cfg.device).float()

        # Sobel gradient magnitude of the binary union -> boundary strength
        gx = F.pad(t_union[None, None], (1, 1, 0, 0), mode="replicate")
        gy = F.pad(t_union[None, None], (0, 0, 1, 1), mode="replicate")
        sx = gx[:, :, 1:-1, 2:] - gx[:, :, 1:-1, :-2]
        sy = gy[:, :, 2:, 1:-1] - gy[:, :, :-2, 1:-1]
        boundary = torch.sqrt(sx.squeeze() ** 2 + sy.squeeze() ** 2 + 1e-12)
        bmax = boundary.max()
        if bmax > 1e-9:
            boundary = boundary / bmax
        return boundary


# ----------------------------------------------------------------------
# WOW directional residual (classical baseline component)
# ----------------------------------------------------------------------
def _wow_residual(image_gray: Tensor) -> Tensor:
    """Compute the WOW directional residual energy.

    Parameters
    ----------
    image_gray : Tensor
        ``[1, H, W]`` grayscale image, float in ``[0, 1]``.

    Returns
    -------
    Tensor
        ``[H, W]`` non-negative residual energy.
    """
    k1 = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, -1.0, 0.5], [0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    k2 = torch.tensor(
        [[0.0, 0.5, 0.0], [0.0, -1.0, 0.0], [0.0, 0.5, 0.0]],
        dtype=torch.float32,
    )
    k3 = torch.tensor(
        [[0.25, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.25]],
        dtype=torch.float32,
    )
    k4 = torch.tensor(
        [[0.0, 0.0, 0.25], [0.0, -1.0, 0.0], [0.25, 0.0, 0.0]],
        dtype=torch.float32,
    )

    img = image_gray.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    res = []
    for k in (k1, k2, k3, k4):
        kk = k.view(1, 1, 3, 3).to(image_gray.device)
        res.append(F.conv2d(img, kk, padding=1).abs())
    energy = torch.stack(res, dim=0).sum(0).squeeze()
    return energy


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
class SemanticCostMap(nn.Module):
    """End-to-end generator producing the per-pixel cost matrix ``rho``.

    Usage
    -----
    >>> cfg = SemanticMapConfig()
    >>> generator = SemanticCostMap(cfg)
    >>> rho = generator(cover_tensor)        # [H, W] float tensor
    """

    def __init__(self, cfg: Optional[SemanticMapConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or SemanticMapConfig()
        self.clip = CLIPDenseEncoder(self.cfg)
        self.sam = SAMBoundaryEncoder(self.cfg)

    @torch.no_grad()
    def forward(self, image_uint8: Tensor) -> Tensor:
        """Compute the cost matrix (paper Eq. 3).

        Parameters
        ----------
        image_uint8 : Tensor
            Cover image, shape ``[3, H, W]``, dtype ``uint8`` (or float in
            ``[0, 255]``).

        Returns
        -------
        Tensor
            Cost matrix ``rho`` of shape ``[H, W]``, dtype ``float32``,
            values normalised to ``[1, rho_max]``.
        """
        if image_uint8.dim() != 3 or image_uint8.shape[0] != 3:
            raise ValueError(
                f"Expected [3,H,W] tensor, got shape {tuple(image_uint8.shape)}"
            )

        h, w = image_uint8.shape[-2:]

        # --- CLIP per-patch similarity ----------------------------------
        sim = self.clip.dense_similarity(image_uint8)  # [H, W], [-1, 1]
        sim = sim.clamp(-1.0, 1.0)

        # --- SAM boundary map -------------------------------------------
        boundary = self.sam.boundary_map(image_uint8)  # [H, W], [0, 1]

        # --- Classical WOW residual -------------------------------------
        gray = image_uint8.float().mean(0, keepdim=True) / 255.0
        residual = _wow_residual(gray)

        # --- Fused cost (Eq. 3) -----------------------------------------
        clip_factor = (1.0 - self.cfg.alpha * sim).clamp(min=1e-3)
        sam_factor = 1.0 + self.cfg.beta * boundary
        rho = clip_factor * sam_factor * (self.cfg.epsilon + self.cfg.gamma * residual)

        # --- Normalise to [1, rho_max] ----------------------------------
        rho_min = rho.min()
        rho_max = rho.max()
        if (rho_max - rho_min) > 1e-9:
            rho = (rho - rho_min) / (rho_max - rho_min)
        rho = 1.0 + rho * (self.cfg.rho_max - 1.0)
        return rho.float()
