r"""embed.py
=========
STC embedding for the **Semantic Cloak** pipeline, using AES-256-GCM
authenticated encryption of the payload.

Pipeline
--------
1.  Compress (LZ4) the message, then encrypt with AES-256-GCM under a
    PBKDF2-derived key.
2.  Pack (length, nonce, tag, ciphertext) into a single bit-stream and
    pad to the target payload length with cryptographic randomness
    (hides payload length).
3.  Run the STC encoder on the per-pixel cost matrix and the payload
    bit-stream to obtain the minimum-distortion LSB modification
    pattern.
4.  Apply the modifications (XOR LSB) to the blue channel of the cover.

This module relies on :mod:`semantic_cloak.stc` for the STC Viterbi
encoder and :mod:`semantic_cloak.crypto` for the AES-GCM encryption.

Author : Samir (Maharishi Dayanand University)
License: MIT
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .crypto import (
    CryptoConfig,
    encrypt_payload,
    derive_key,
    pack_header,
)
from .stc import STCConfig, STCEncoder


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
@dataclass
class EmbedConfig:
    """Hyper-parameters for the STC embedder.

    Attributes
    ----------
    target_bpp : float
        Target payload capacity in bits per pixel.
    stc : STCConfig
        STC submatrix configuration.
    crypto : CryptoConfig
        AES-GCM + PBKDF2 configuration.
    """

    target_bpp: float = 0.4
    stc: STCConfig = None  # type: ignore
    crypto: CryptoConfig = None  # type: ignore

    def __post_init__(self) -> None:
        if self.stc is None:
            self.stc = STCConfig()
        if self.crypto is None:
            self.crypto = CryptoConfig()


# ----------------------------------------------------------------------
# Public container
# ----------------------------------------------------------------------
@dataclass
class EmbeddingResult:
    """Container for the output of :func:`embed`."""

    stego_image: torch.Tensor        # [3, H, W] uint8
    payload_bits: np.ndarray         # [N] uint8 actually embedded
    cost_matrix: torch.Tensor        # [H, W] float
    modification_mask: torch.Tensor  # [H, W] uint8 (1 = pixel was flipped)
    nonce: bytes                     # 12 bytes
    tag: bytes                       # 16 bytes
    ciphertext_len: int              # bytes


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def embed(
    cover_image: torch.Tensor,
    message: bytes,
    passphrase: str,
    cost_matrix: torch.Tensor,
    cfg: Optional[EmbedConfig] = None,
) -> EmbeddingResult:
    """Top-level embedding entry point.

    Parameters
    ----------
    cover_image : Tensor
        Cover image, shape ``[3, H, W]``, dtype ``uint8``.
    message : bytes
        Plaintext message to embed.
    passphrase : str
        Passphrase used to derive the AES-256 key.
    cost_matrix : Tensor
        Per-pixel cost matrix ``rho`` of shape ``[H, W]``, output of
        :class:`semantic_cloak.semantic_map.SemanticCostMap`.
    cfg : Optional[EmbedConfig]
        Embedding hyper-parameters. ``None`` => use defaults.

    Returns
    -------
    EmbeddingResult
    """
    cfg = cfg or EmbedConfig()

    if cover_image.dim() != 3 or cover_image.shape[0] != 3:
        raise ValueError("cover_image must be [3,H,W].")
    h, w = cover_image.shape[-2:]
    n_pixels = h * w

    # ----- 1. Compress + encrypt ------------------------------------
    key = derive_key(passphrase, cfg.crypto)
    ciphertext, nonce, tag = encrypt_payload(message, key)

    # ----- 2. Pack header + ciphertext -> bit-stream ---------------
    packed = pack_header(ciphertext, nonce, tag)
    bitstream = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))

    # ----- 3. Pad to target length with cryptographic randomness ---
    target_payload_bits = int(round(cfg.target_bpp * n_pixels))
    if bitstream.shape[0] > target_payload_bits:
        raise ValueError(
            f"Payload ({bitstream.shape[0]} bits) exceeds capacity "
            f"({target_payload_bits} bits at {cfg.target_bpp} bpp)."
        )
    pad_len = target_payload_bits - bitstream.shape[0]
    if pad_len > 0:
        # Cryptographic randomness: os.urandom is CSPRNG.
        pad_bytes = os.urandom(pad_len // 8 + 1)
        pad_bits = np.unpackbits(np.frombuffer(pad_bytes, dtype=np.uint8))
        bitstream = np.concatenate([bitstream, pad_bits[:pad_len]])

    # ----- 4. Reshape cover & cost into pixel streams --------------
    # Embed in the blue channel (least perceptually salient in natural
    # images, per Wenger et al., IEEE WIFS 2015).
    blue = cover_image[2].clone().to(torch.uint8)
    flat = blue.view(-1).to(torch.uint8).cpu().numpy().astype(np.uint8)
    original_lsb = (flat & 0x01).astype(np.uint8)
    cost_flat = cost_matrix.view(-1).to(torch.float64).cpu().numpy()
    # Rescale cost so the lowest-cost pixel is 1.0 (STC expects
    # strictly positive costs).
    cost_flat = cost_flat / (cost_flat.min() + 1e-12)

    # ----- 5. STC embed --------------------------------------------
    # The STC decoder computes syndrome = H * stego_lsb (mod 2), where
    # stego_lsb = original_lsb XOR y (y = modification vector).
    # We want syndrome = message, so we need:
    #   H * (original_lsb XOR y) = message
    #   H*original_lsb XOR H*y = message
    #   H*y = message XOR H*original_lsb
    # So the target syndrome for the encoder is message XOR H*original_lsb.
    stc = STCEncoder(cfg.stc)
    # Compute the syndrome of the original LSBs using the decoder's
    # schedule (the decoder is the authoritative source of the syndrome
    # computation).
    from .stc import STCDecoder
    decoder = STCDecoder(cfg.stc)
    original_syndrome = decoder.decode(original_lsb, n_message_bits=len(bitstream))
    # Target syndrome = message XOR original_syndrome
    target_syndrome = np.bitwise_xor(bitstream, original_syndrome)
    modifications = stc.encode(target_syndrome, cost_flat)

    # ----- 6. Apply modifications to LSB of blue channel -----------
    modified = flat.copy()
    modified[modifications == 1] ^= 0x01
    modified_flat = torch.from_numpy(modified).to(cover_image.device)
    stego = cover_image.clone()
    stego[2] = modified_flat.view(h, w).to(cover_image.dtype)

    mod_mask = torch.from_numpy(modifications.astype(np.uint8)).to(
        cover_image.device
    ).view(h, w)

    return EmbeddingResult(
        stego_image=stego,
        payload_bits=bitstream,
        cost_matrix=cost_matrix,
        modification_mask=mod_mask,
        nonce=nonce,
        tag=tag,
        ciphertext_len=len(ciphertext),
    )
