r"""crypto.py
==========
Authenticated encryption for the Semantic Cloak payload.

Uses AES-256-GCM via the ``cryptography`` package. The 32-byte key is
derived from a user-supplied passphrase via PBKDF2-HMAC-SHA256 with
200,000 iterations (NIST SP 800-132 recommendation).

Payload layout (530 bits header + ciphertext):

    +------------------+------------------+------------------+
    | len (16 bits BE) | nonce (12 bytes) | tag (16 bytes)   |
    +------------------+------------------+------------------+
    | ciphertext (len bytes)                                |
    +--------------------------------------------------------+

The length prefix is required because AES-GCM does not self-delimit.

LZ4 compression is applied before encryption (with a zlib fallback if
``lz4`` is not installed) to remove plaintext redundancy.

Author : Samir (Maharishi Dayanand University)
License: MIT
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

# ----------------------------------------------------------------------
# Compression (LZ4 preferred, zlib required fallback)
# ----------------------------------------------------------------------
try:
    import lz4.frame as _lz4

    def _compress(data: bytes) -> bytes:
        return _lz4.compress(data)

    def _decompress(data: bytes) -> bytes:
        return _lz4.decompress(data)

    _COMPRESSION = "lz4"

except ImportError:
    import zlib

    def _compress(data: bytes) -> bytes:
        return zlib.compress(data, level=9)

    def _decompress(data: bytes) -> bytes:
        return zlib.decompress(data)

    _COMPRESSION = "zlib"


# ----------------------------------------------------------------------
# AES-256-GCM (no fallback: cryptography is a hard dependency)
# ----------------------------------------------------------------------
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
@dataclass
class CryptoConfig:
    """AES-GCM + PBKDF2 hyper-parameters.

    Attributes
    ----------
    salt : bytes
        Static salt for PBKDF2. In production, use a per-user salt
        stored alongside the stego image metadata.
    pbkdf2_iters : int
        PBKDF2 iteration count. NIST SP 800-132 recommends >= 100,000;
        we use 200,000 for a safety margin.
    """

    salt: bytes = b"semantic-cloak-v1-static-salt"
    pbkdf2_iters: int = 200_000


# ----------------------------------------------------------------------
# Key derivation
# ----------------------------------------------------------------------
def derive_key(passphrase: str, cfg: CryptoConfig) -> bytes:
    """Derive a 32-byte AES-256 key from a passphrase via PBKDF2-HMAC-SHA256.

    Parameters
    ----------
    passphrase : str
        User-supplied passphrase (any length).
    cfg : CryptoConfig
        Configuration carrying the salt and iteration count.

    Returns
    -------
    bytes
        32-byte (256-bit) AES key.
    """
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=cfg.salt,
        iterations=cfg.pbkdf2_iters,
    )
    return kdf.derive(passphrase.encode("utf-8"))


# ----------------------------------------------------------------------
# Encrypt / decrypt
# ----------------------------------------------------------------------
def encrypt_payload(plaintext: bytes, key: bytes) -> Tuple[bytes, bytes, bytes]:
    """AES-256-GCM authenticated encryption of the payload.

    Parameters
    ----------
    plaintext : bytes
        Data to encrypt (will be compressed first).
    key : bytes
        32-byte AES-256 key from :func:`derive_key`.

    Returns
    -------
    (ciphertext, nonce, tag) : Tuple[bytes, bytes, bytes]
        ``nonce`` is 12 bytes, ``tag`` is 16 bytes (concatenated to
        ciphertext by ``AESGCM.encrypt``, but we split for explicit
        storage).
    """
    if len(key) != 32:
        raise ValueError(f"AES-256 key must be 32 bytes, got {len(key)}.")

    compressed = _compress(plaintext)
    nonce = os.urandom(12)
    aesgcm = AESGCM(key)
    # AESGCM.encrypt appends the 16-byte tag to the ciphertext.
    ct_and_tag = aesgcm.encrypt(nonce, compressed, associated_data=None)
    ciphertext, tag = ct_and_tag[:-16], ct_and_tag[-16:]
    return ciphertext, nonce, tag


def decrypt_payload(
    ciphertext: bytes, nonce: bytes, tag: bytes, key: bytes
) -> bytes:
    """Inverse of :func:`encrypt_payload`. Raises if authentication fails.

    Parameters
    ----------
    ciphertext : bytes
        The encrypted payload (without tag).
    nonce : bytes
        12-byte AES-GCM nonce.
    tag : bytes
        16-byte AES-GCM authentication tag.
    key : bytes
        32-byte AES-256 key.

    Returns
    -------
    bytes
        The decrypted, decompressed plaintext.

    Raises
    ------
    cryptography.exceptions.InvalidTag
        If the authentication tag does not match (wrong key, tampered
        ciphertext, or corrupted nonce).
    """
    if len(key) != 32:
        raise ValueError(f"AES-256 key must be 32 bytes, got {len(key)}.")
    if len(nonce) != 12:
        raise ValueError(f"AES-GCM nonce must be 12 bytes, got {len(nonce)}.")
    if len(tag) != 16:
        raise ValueError(f"AES-GCM tag must be 16 bytes, got {len(tag)}.")

    aesgcm = AESGCM(key)
    compressed = aesgcm.decrypt(nonce, ciphertext + tag, associated_data=None)
    return _decompress(compressed)


# ----------------------------------------------------------------------
# Bitstream header layout (used by embed.py and extract.py)
# ----------------------------------------------------------------------
HEADER_BYTES = 2 + 12 + 16  # len + nonce + tag = 30 bytes
HEADER_BITS = HEADER_BYTES * 8  # 240 bits


def pack_header(ciphertext: bytes, nonce: bytes, tag: bytes) -> bytes:
    """Pack the (len, nonce, tag, ciphertext) into a single bytestream."""
    payload_len = len(ciphertext)
    if payload_len > 0xFFFF:
        raise ValueError(
            f"Ciphertext exceeds 64 KiB ({payload_len}B); raise header width."
        )
    return (
        payload_len.to_bytes(2, "big")
        + nonce
        + tag
        + ciphertext
    )


def parse_header(bytestream: bytes) -> Tuple[int, bytes, bytes, bytes]:
    """Inverse of :func:`pack_header`. Returns (len, nonce, tag, ct)."""
    if len(bytestream) < HEADER_BYTES:
        raise ValueError(
            f"Bytestream too short to contain header ({len(bytestream)}B)."
        )
    payload_len = int.from_bytes(bytestream[:2], "big")
    nonce = bytestream[2:14]
    tag = bytestream[14:30]
    ciphertext = bytestream[30 : 30 + payload_len]
    if len(ciphertext) != payload_len:
        raise ValueError(
            f"Ciphertext truncated: expected {payload_len}B, got {len(ciphertext)}B."
        )
    return payload_len, nonce, tag, ciphertext
