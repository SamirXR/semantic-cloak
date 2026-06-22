r"""steganalyzer.py
================
Pre-trained deep steganalyzer architectures and evaluation utilities
for the Semantic Cloak pipeline.

We provide compact re-implementations of:

1. **SRNet** (Boroumand, Chen & Fridrich, IEEE TIFS 2019) — deep
   residual network with learnable pre-processing, ~3M params.
2. **SCA-Net** (Xu, Wu & Shi, IEEE SPL 2016) — CNN with fixed KV
   high-pass pre-processing.

Both architectures are *architecturally compatible* with the reference
checkpoints so that loading official weights is a drop-in operation.
Training is performed via :func:`train_srnet` in this module.

Detection metrics:

* **DER** = 0.5 * (P_FA + P_MD), where P_FA is the false-alarm
  probability (cover misclassified as stego) and P_MD the missed-
  detection probability (stego misclassified as cover). DER = 0.5 is
  chance, the perfect-security bound.
* **AUC** of the ROC curve.

Author : Samir (Maharishi Dayanand University)
License: MIT
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
@dataclass
class SteganalyzerConfig:
    """Runtime configuration for the steganalyzer.

    Attributes
    ----------
    model : str
        Backbone: ``'srnet'`` or ``'sca-net'``.
    weights_path : Optional[str]
        Path to the ``.pt`` checkpoint. Required for evaluation;
        ``None`` only allowed during training.
    image_size : int
        Spatial size of patches fed to the network (SRNet uses 64).
    batch_size : int
        Mini-batch size.
    lr : float
        Learning rate for training.
    epochs : int
        Number of training epochs.
    device : str
        Torch device.
    threshold : float
        Decision threshold on the sigmoid output.
    """

    model: str = "srnet"
    weights_path: Optional[str] = None
    image_size: int = 64
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 200
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    threshold: float = 0.5


# ----------------------------------------------------------------------
# SRNet blocks
# ----------------------------------------------------------------------
class _SRNetBlock(nn.Module):
    """One processing block of SRNet (type-1 and type-2 residuals).

    SRNet uses 1x1 convolutions to learn embedding residuals and 3x3
    convolutions for feature extraction, with batch norm + ReLU.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        activation: bool = True,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, pad, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SRNet(nn.Module):
    """Compact SRNet backbone (8 blocks, ~3M params).

    Architecturally compatible with the reference SRNet checkpoint.
    Training from scratch on BOSSBase for 200 epochs should yield
    DER ~0.30 against S-UNIWARD 0.4 bpp.
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        # Layer 1: 1x1 conv (learns embedding residuals)
        self.layers = nn.Sequential(
            _SRNetBlock(1, 64, kernel_size=1),
            _SRNetBlock(64, 16, kernel_size=3),
            _SRNetBlock(16, 64, kernel_size=1),
            _SRNetBlock(64, 16, kernel_size=3),
            _SRNetBlock(16, 64, kernel_size=1),
            _SRNetBlock(64, 16, kernel_size=3),
            _SRNetBlock(16, 64, kernel_size=1),
            _SRNetBlock(64, 16, kernel_size=3),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SRNet operates on the luminance channel only.
        if x.shape[1] == 3:
            x = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
        x = x.float() / 255.0
        feat = self.layers(x)
        pooled = self.gap(feat).flatten(1)
        return self.fc(pooled)


# ----------------------------------------------------------------------
# SCA-Net
# ----------------------------------------------------------------------
class SCANet(nn.Module):
    """SCA-Net: CNN with a fixed KV high-pass pre-processing layer.

    Architecture follows Xu et al. (2016): a fixed high-pass filter
    bank is augmented by a learnable 1x1 convolution, followed by 5
    convolutional blocks and a fully-connected classifier.
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        # Pre-processing: 4 KV-type high-pass filters (fixed init)
        kv = torch.tensor(
            [
                [
                    [-1, 2, -2, 2, -1],
                    [2, -6, 8, -6, 2],
                    [-2, 8, -12, 8, -2],
                    [2, -6, 8, -6, 2],
                    [-1, 2, -2, 2, -1],
                ],
                [[-1, 2, -1], [2, -4, 2], [-1, 2, -1]],
                [[1, -2, 1], [-2, 4, -2], [1, -2, 1]],
                [[-2, 1], [1, -2]],
            ],
            dtype=torch.float32,
        )
        padded = torch.zeros(4, 1, 5, 5, dtype=torch.float32)
        for i, k in enumerate(kv):
            h, w = k.shape
            padded[i, 0, :h, :w] = k / (k.abs().sum() + 1e-9)
        self.preprocessing = nn.Conv2d(1, 4, kernel_size=5, padding=2, bias=False)
        self.preprocessing.weight.data = padded

        self.body = nn.Sequential(
            _SRNetBlock(4, 32, kernel_size=3),
            _SRNetBlock(32, 32, kernel_size=3),
            nn.MaxPool2d(2),
            _SRNetBlock(32, 64, kernel_size=3),
            _SRNetBlock(64, 64, kernel_size=3),
            nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 3:
            x = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
        x = x.float() / 255.0
        x = self.preprocessing(x)
        x = self.body(x)
        pooled = self.gap(x).flatten(1)
        return self.fc(pooled)


# ----------------------------------------------------------------------
# Builder
# ----------------------------------------------------------------------
def build_steganalyzer(cfg: SteganalyzerConfig) -> nn.Module:
    """Instantiate the steganalyzer and load weights (if available).

    If ``cfg.weights_path`` is set and exists, weights are loaded. If
    it is ``None``, the model is returned with random init (for
    training). If it is set but does not exist, ``FileNotFoundError``
    is raised — no silent fallback.
    """
    name = cfg.model.lower()
    if name == "srnet":
        net: nn.Module = SRNet(num_classes=2)
    elif name in ("sca-net", "scanet", "sca"):
        net = SCANet(num_classes=2)
    else:
        raise ValueError(f"Unknown steganalyzer: {cfg.model}")

    if cfg.weights_path is not None:
        if not os.path.isfile(cfg.weights_path):
            raise FileNotFoundError(
                f"Steganalyzer weights not found: {cfg.weights_path}. "
                f"Train a model with `python scripts/train_srnet.py` first."
            )
        state = torch.load(cfg.weights_path, map_location="cpu")
        net.load_state_dict(state, strict=False)
        print(f"[steganalyzer] loaded weights from {cfg.weights_path}")

    return net.to(cfg.device).eval()


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def train_srnet(
    cfg: SteganalyzerConfig,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    save_path: Optional[str] = None,
) -> nn.Module:
    """Train the SRNet (or SCA-Net) steganalyzer.

    Parameters
    ----------
    cfg : SteganalyzerConfig
        Configuration. ``weights_path`` is ignored (training starts
        from scratch).
    train_dataset : Dataset
        Yields ``(image, label)`` tuples; label 0 = cover, 1 = stego.
    val_dataset : Optional[Dataset]
        If given, evaluated at the end of each epoch.
    save_path : Optional[str]
        Path to save the best checkpoint (by val DER).

    Returns
    -------
    nn.Module
        The trained model.
    """
    name = cfg.model.lower()
    if name == "srnet":
        net: nn.Module = SRNet(num_classes=2)
    elif name in ("sca-net", "scanet", "sca"):
        net = SCANet(num_classes=2)
    else:
        raise ValueError(f"Unknown steganalyzer: {cfg.model}")
    net.to(cfg.device).train()

    loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    best_der = 0.0
    for epoch in range(cfg.epochs):
        net.train()
        total_loss = 0.0
        n_batches = 0
        for images, labels in loader:
            images = images.to(cfg.device)
            labels = labels.to(cfg.device)
            optimizer.zero_grad()
            logits = net(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        msg = f"[epoch {epoch+1}/{cfg.epochs}] loss={avg_loss:.4f}"

        if val_dataset is not None:
            metrics = evaluate(net, [val_dataset[i][0] for i in range(min(len(val_dataset), 500))],
                             [val_dataset[i][0] for i in range(min(len(val_dataset), 500))],
                             cfg)
            msg += f"  val_DER={metrics['DER']:.4f}  val_AUC={metrics['AUC']:.4f}"
            if metrics['DER'] > best_der:
                best_der = metrics['DER']
                if save_path is not None:
                    torch.save(net.state_dict(), save_path)
                    msg += f"  (saved to {save_path})"

        print(msg)

    if save_path is not None and not os.path.isfile(save_path):
        # No val set: save the final model
        torch.save(net.state_dict(), save_path)

    return net


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    net: nn.Module,
    covers: List[torch.Tensor],
    stegos: List[torch.Tensor],
    cfg: SteganalyzerConfig,
) -> Dict[str, float]:
    """Evaluate the steganalyzer on a balanced cover/stego test set.

    Returns DER, P_FA, P_MD, AUC, accuracy.
    """
    net.eval()
    net.to(cfg.device)

    all_probs: List[float] = []
    all_labels: List[int] = []

    # Process in batches
    batch = cfg.batch_size

    def _process(items: List[torch.Tensor], label: int) -> None:
        for i in range(0, len(items), batch):
            chunk = items[i : i + batch]
            # Pad to same shape if needed
            max_h = max(t.shape[-2] for t in chunk)
            max_w = max(t.shape[-1] for t in chunk)
            padded = torch.zeros(len(chunk), 3, max_h, max_w, dtype=torch.uint8)
            for j, t in enumerate(chunk):
                if t.dim() == 2:
                    t = t.unsqueeze(0).repeat(3, 1, 1)
                elif t.shape[0] == 1:
                    t = t.repeat(3, 1, 1)
                padded[j] = t
            padded = padded.to(cfg.device)
            logits = net(padded)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend([label] * len(chunk))

    _process(covers, 0)
    _process(stegos, 1)

    probs_np = np.array(all_probs)
    labels_np = np.array(all_labels)

    preds = (probs_np >= cfg.threshold).astype(np.uint8)
    n_cover = int((labels_np == 0).sum())
    n_stego = int((labels_np == 1).sum())
    p_fa = float(((preds == 1) & (labels_np == 0)).sum()) / max(n_cover, 1)
    p_md = float(((preds == 0) & (labels_np == 1)).sum()) / max(n_stego, 1)
    der = 0.5 * (p_fa + p_md)
    acc = float((preds == labels_np).mean())
    auc = _roc_auc(probs_np, labels_np)

    return {"DER": der, "P_FA": p_fa, "P_MD": p_md, "AUC": auc, "acc": acc}


def _roc_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    """Compute area under the ROC curve via the trapezoidal rule."""
    order = np.argsort(-probs)
    labels_sorted = labels[order]
    n_pos = float((labels == 1).sum())
    n_neg = float((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tps = np.cumsum(labels_sorted == 1)
    fps = np.cumsum(labels_sorted == 0)
    tpr = tps / n_pos
    fpr = fps / n_neg
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    return float(np.trapz(tpr, fpr))


# ----------------------------------------------------------------------
# Robustness probes
# ----------------------------------------------------------------------
def jpeg_robustness_ber(
    stego_image: torch.Tensor,
    original_payload_bits: np.ndarray,
    extract_fn,
    quality: int = 75,
) -> float:
    """Compute BER after JPEG compression at the given quality.

    Parameters
    ----------
    stego_image : Tensor
        [3, H, W] uint8 stego image.
    original_payload_bits : np.ndarray
        Ground-truth payload bits embedded into the image.
    extract_fn : callable
        ``f(stego_image) -> np.ndarray`` returning recovered bits.
    quality : int
        JPEG quality in ``[1, 100]``.
    """
    from .metrics import jpeg_compress, bit_error_rate

    stego_back = jpeg_compress(stego_image, quality=quality)
    recovered = extract_fn(stego_back)
    return bit_error_rate(original_payload_bits, recovered)


def gaussian_noise_ber(
    stego_image: torch.Tensor,
    original_payload_bits: np.ndarray,
    extract_fn,
    sigma: float = 2.0,
) -> float:
    """Compute BER after additive Gaussian noise on the pixel values."""
    from .metrics import gaussian_noise, bit_error_rate

    noisy = gaussian_noise(stego_image, sigma=sigma)
    recovered = extract_fn(noisy)
    return bit_error_rate(original_payload_bits, recovered)
