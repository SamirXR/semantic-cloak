r"""ddsp.py
=========
Wrapper for the DDSP (Deep Differentiable Steganography for Secure
Image Communication) baseline (Zhou et al., CVPR 2022).

DDSP is a deep generative steganography model that learns the
embedding end-to-end. Re-implementing it from scratch is out of scope;
we provide a thin wrapper around the original authors' released
checkpoint.

**Setup:**

1. Clone the official DDSP repository:
   ``git clone https://github.com/KevinZhouKl/DDSP``
2. Download the pre-trained checkpoint as instructed in their README.
3. Set the ``DDSP_CKPT`` environment variable to the checkpoint path,
   or pass ``ckpt_path`` to :class:`DDSPEmbedder`.

If the checkpoint is not available, the constructor raises. We do not
provide a fallback — that would silently degrade the comparison.

Author : Samir (Maharishi Dayanand University)
License: MIT
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class DDSPEmbedder(nn.Module):
    """Thin wrapper around the official DDSP checkpoint.

    The wrapper imports the DDSP encoder/decoder modules from the
    official repo and exposes a simple ``embed(cover, message)`` API
    that matches the Semantic Cloak interface.

    Users must have the DDSP repo cloned and on the Python path, and
    must point ``DDSP_CKPT`` to the downloaded checkpoint.
    """

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        super().__init__()
        self.ckpt_path = ckpt_path or os.environ.get("DDSP_CKPT")
        if self.ckpt_path is None or not os.path.isfile(self.ckpt_path):
            raise FileNotFoundError(
                "DDSP checkpoint not found. Either pass ckpt_path or set "
                "the DDSP_CKPT environment variable. Download from the "
                "official DDSP repo: https://github.com/KevinZhouKl/DDSP"
            )

        self.device = device
        # Import the DDSP modules (assumes the repo is on sys.path)
        try:
            from models.ddsp_encoder import DDSPEncoder  # type: ignore
            from models.ddsp_decoder import DDSPDecoder  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Failed to import DDSP modules. Clone the official repo "
                "(https://github.com/KevinZhouKl/DDSP) and add it to your "
                "PYTHONPATH."
            ) from exc

        self.encoder = DDSPEncoder().to(device).eval()
        self.decoder = DDSPDecoder().to(device).eval()

        # Load checkpoint
        state = torch.load(self.ckpt_path, map_location=device)
        if "encoder" in state:
            self.encoder.load_state_dict(state["encoder"])
            self.decoder.load_state_dict(state["decoder"])
        else:
            self.encoder.load_state_dict(state)
            # Try loading decoder from sibling file
            dec_path = self.ckpt_path.replace("enc", "dec")
            if os.path.isfile(dec_path):
                self.decoder.load_state_dict(torch.load(dec_path, map_location=device))

    @torch.no_grad()
    def embed(
        self, cover: torch.Tensor, message: torch.Tensor
    ) -> torch.Tensor:
        """Embed a message into a cover image.

        Parameters
        ----------
        cover : Tensor
            Cover image, shape ``[1, 3, H, W]``, float in ``[0, 1]``.
        message : Tensor
            Message bits, shape ``[1, msg_len]``, float in ``{0, 1}``.

        Returns
        -------
        Tensor
            Stego image, same shape as cover.
        """
        cover = cover.to(self.device)
        message = message.to(self.device)
        stego = self.encoder(cover, message)
        return stego

    @torch.no_grad()
    def extract(self, stego: torch.Tensor) -> torch.Tensor:
        """Extract a message from a stego image.

        Parameters
        ----------
        stego : Tensor
            Stego image, shape ``[1, 3, H, W]``, float in ``[0, 1]``.

        Returns
        -------
        Tensor
            Recovered message bits, shape ``[1, msg_len]``, float in
            ``[0, 1]`` (threshold at 0.5 to get bits).
        """
        stego = stego.to(self.device)
        return self.decoder(stego)
