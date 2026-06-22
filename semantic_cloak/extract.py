r"""extract.py
===========
Reverse pipeline for the **Semantic Cloak** steganography scheme.

Given a stego image and the same passphrase used at embed time, this
module:

1.  Reads back the LSB stream from the blue channel.
2.  Runs the STC syndrome computation to recover the embedded bit-stream.
3.  Splits the bit-stream into (length, nonce, tag, ciphertext).
4.  Decrypts (AES-GCM) and decompresses to recover the original message.

Because the STC parity-check submatrix ``H_hat`` is shared between
embedder and extractor by construction (same deterministic seed), the
extractor simply computes the syndrome of the LSB stream.

Author : Samir (Maharishi Dayanand University)
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from .crypto import (
    CryptoConfig,
    decrypt_payload,
    derive_key,
    parse_header,
)
from .stc import STCConfig, STCDecoder


# ----------------------------------------------------------------------
# Public container
# ----------------------------------------------------------------------
@dataclass
class ExtractionResult:
    """Container for the output of :func:`extract`."""

    message: bytes                  # decrypted plaintext
    payload_len: int                # bytes (header field)
    bitstream: np.ndarray           # raw recovered bits, before parsing
    verified: bool                  # whether AES-GCM tag matched
    n_bits_read: int                # how many bits were read from the image


# ----------------------------------------------------------------------
# Bitstream <-> bytes helpers
# ----------------------------------------------------------------------
def _bits_to_bytes(bits: np.ndarray) -> bytes:
    """Pack a 1-D bit array (length multiple of 8) into ``bytes``."""
    if bits.shape[0] % 8 != 0:
        bits = np.concatenate([bits, np.zeros(8 - bits.shape[0] % 8, dtype=np.uint8)])
    return np.packbits(bits).tobytes()


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def extract(
    stego_image: torch.Tensor,
    passphrase: str,
    image_size_hw: Tuple[int, int],
    cfg: Optional["EmbedConfig"] = None,
) -> ExtractionResult:
    """Top-level extraction entry point.

    Parameters
    ----------
    stego_image : Tensor
        Stego image, shape ``[3, H, W]``, dtype ``uint8``.
    passphrase : str
        Same passphrase used at embed time.
    image_size_hw : Tuple[int, int]
        Original ``(H, W)`` of the cover. Used to recover the payload
        length (``target_bpp * H * W``).
    cfg : Optional[EmbedConfig]
        Embedding hyper-parameters (must match the embedder). Pass
        ``None`` for defaults; this works only if the embedder also
        used defaults.

    Returns
    -------
    ExtractionResult
    """
    # Late import to avoid circular dependency
    from .embed import EmbedConfig

    cfg = cfg or EmbedConfig()

    if stego_image.dim() != 3 or stego_image.shape[0] != 3:
        raise ValueError("stego_image must be [3,H,W].")
    h, w = stego_image.shape[-2:]
    if (h, w) != image_size_hw:
        raise ValueError(
            f"Image size mismatch: stego is {(h,w)}, expected {image_size_hw}."
        )

    n_pixels = h * w
    n_bits = int(round(cfg.target_bpp * n_pixels))

    # ----- 1. Read LSBs from the blue channel ----------------------
    blue = stego_image[2].to(torch.uint8).cpu().numpy().astype(np.uint8)
    flat = blue.reshape(-1)
    lsb_stream = (flat & 0x01).astype(np.uint8)

    # ----- 2. STC decode -------------------------------------------
    decoder = STCDecoder(cfg.stc)
    bitstream = decoder.decode(lsb_stream, n_bits)

    # ----- 3. Parse header -----------------------------------------
    bytestream = _bits_to_bytes(bitstream[: (len(bitstream) // 8) * 8])
    try:
        payload_len, nonce, tag, ciphertext = parse_header(bytestream)
    except ValueError:
        return ExtractionResult(
            message=b"",
            payload_len=0,
            bitstream=bitstream,
            verified=False,
            n_bits_read=len(lsb_stream),
        )

    # ----- 4. AES-GCM decrypt + decompress -------------------------
    key = derive_key(passphrase, cfg.crypto)
    try:
        message = decrypt_payload(ciphertext, nonce, tag, key)
        verified = True
    except Exception:
        message = b""
        verified = False

    return ExtractionResult(
        message=message,
        payload_len=payload_len,
        bitstream=bitstream,
        verified=verified,
        n_bits_read=len(lsb_stream),
    )
