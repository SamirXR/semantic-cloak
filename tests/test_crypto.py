"""Tests for the crypto module (AES-256-GCM, PBKDF2, header packing).

These tests require the `cryptography` package.
"""

import os

import pytest
import numpy as np

cryptography = pytest.importorskip("cryptography")

from semantic_cloak.crypto import (
    CryptoConfig, derive_key, encrypt_payload, decrypt_payload,
    pack_header, parse_header, HEADER_BYTES,
)


# ----------------------------------------------------------------------
# Key derivation
# ----------------------------------------------------------------------
class TestKeyDerivation:
    def test_key_length(self):
        key = derive_key("password", CryptoConfig())
        assert len(key) == 32

    def test_same_passphrase_same_key(self):
        cfg = CryptoConfig()
        k1 = derive_key("mypassword", cfg)
        k2 = derive_key("mypassword", cfg)
        assert k1 == k2

    def test_different_passphrase_different_key(self):
        cfg = CryptoConfig()
        k1 = derive_key("password1", cfg)
        k2 = derive_key("password2", cfg)
        assert k1 != k2

    def test_different_salt_different_key(self):
        k1 = derive_key("password", CryptoConfig(salt=b"salt1"))
        k2 = derive_key("password", CryptoConfig(salt=b"salt2"))
        assert k1 != k2


# ----------------------------------------------------------------------
# Encrypt / decrypt round-trip
# ----------------------------------------------------------------------
class TestRoundTrip:
    def test_short_message(self):
        key = derive_key("pw", CryptoConfig())
        plaintext = b"Hello, Semantic Cloak!"
        ct, nonce, tag = encrypt_payload(plaintext, key)
        recovered = decrypt_payload(ct, nonce, tag, key)
        assert recovered == plaintext

    def test_empty_message(self):
        key = derive_key("pw", CryptoConfig())
        plaintext = b""
        ct, nonce, tag = encrypt_payload(plaintext, key)
        recovered = decrypt_payload(ct, nonce, tag, key)
        assert recovered == plaintext

    def test_large_message(self):
        key = derive_key("pw", CryptoConfig())
        plaintext = os.urandom(10000)
        ct, nonce, tag = encrypt_payload(plaintext, key)
        recovered = decrypt_payload(ct, nonce, tag, key)
        assert recovered == plaintext

    def test_wrong_passphrase_fails(self):
        """Decryption with the wrong key should raise InvalidTag."""
        from cryptography.exceptions import InvalidTag
        ct, nonce, tag = encrypt_payload(b"secret", derive_key("right", CryptoConfig()))
        wrong_key = derive_key("wrong", CryptoConfig())
        with pytest.raises(InvalidTag):
            decrypt_payload(ct, nonce, tag, wrong_key)

    def test_tampered_ciphertext_fails(self):
        """Tampering with the ciphertext should fail authentication."""
        from cryptography.exceptions import InvalidTag
        key = derive_key("pw", CryptoConfig())
        ct, nonce, tag = encrypt_payload(b"secret", key)
        # Flip a bit in the ciphertext
        tampered = bytes([ct[0] ^ 1]) + ct[1:]
        with pytest.raises(InvalidTag):
            decrypt_payload(tampered, nonce, tag, key)


# ----------------------------------------------------------------------
# Header packing
# ----------------------------------------------------------------------
class TestHeader:
    def test_round_trip(self):
        ct = os.urandom(100)
        nonce = os.urandom(12)
        tag = os.urandom(16)
        packed = pack_header(ct, nonce, tag)
        assert len(packed) == HEADER_BYTES + len(ct)
        payload_len, n, t, c = parse_header(packed)
        assert payload_len == len(ct)
        assert n == nonce
        assert t == tag
        assert c == ct

    def test_oversized_ciphertext_raises(self):
        """Ciphertext > 64 KiB should raise."""
        ct = os.urandom(0x10001)  # 64 KiB + 1
        nonce = os.urandom(12)
        tag = os.urandom(16)
        with pytest.raises(ValueError, match="exceeds 64 KiB"):
            pack_header(ct, nonce, tag)

    def test_short_bytestream_raises(self):
        """Parsing a too-short bytestream should raise."""
        with pytest.raises(ValueError, match="too short"):
            parse_header(b"short")
