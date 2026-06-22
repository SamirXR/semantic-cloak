"""Integration test for the embed -> extract round-trip.

This test uses a pure-Python STC config (small h, small image) so it
runs in seconds without any VLM dependencies. It validates that the
end-to-end pipeline (compress -> encrypt -> STC embed -> STC extract ->
decrypt -> decompress) recovers the original message bit-exactly.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from semantic_cloak.embed import embed, EmbedConfig
from semantic_cloak.extract import extract
from semantic_cloak.stc import STCConfig


@pytest.fixture
def small_embed_cfg():
    """Small config for fast tests: 128x128 image, 0.05 bpp, h=5.

    A 128x128 image at 0.05 bpp gives 819 bits of capacity, which
    comfortably fits the 240-bit AES-GCM header plus a test message.
    We use h=5 (instead of the default h=9) for faster Viterbi.
    """
    return EmbedConfig(
        target_bpp=0.05,
        stc=STCConfig(submatrix_height=5, submatrix_width=11),
    )


@pytest.fixture
def dummy_cover():
    """A 128x128 RGB cover with some texture."""
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, (3, 128, 128), dtype=np.uint8)
    return torch.from_numpy(arr)


@pytest.fixture
def dummy_cost():
    """A 128x128 cost matrix with a mix of cheap and expensive pixels."""
    return torch.rand(128, 128) * 99 + 1


class TestEmbedExtractRoundTrip:
    def test_short_message_roundtrip(self, dummy_cover, dummy_cost, small_embed_cfg):
        """A short message should round-trip exactly."""
        message = b"Hello, Semantic Cloak!"
        result = embed(
            cover_image=dummy_cover,
            message=message,
            passphrase="test",
            cost_matrix=dummy_cost,
            cfg=small_embed_cfg,
        )
        assert result.stego_image.shape == dummy_cover.shape
        assert result.stego_image.dtype == dummy_cover.dtype

        # The stego should differ from the cover only in the blue channel
        # and only at LSB level
        assert torch.equal(result.stego_image[0], dummy_cover[0])
        assert torch.equal(result.stego_image[1], dummy_cover[1])
        # Blue channel may differ

        # Extract
        out = extract(
            stego_image=result.stego_image,
            passphrase="test",
            image_size_hw=(128, 128),
            cfg=small_embed_cfg,
        )
        assert out.verified, "AES-GCM tag verification failed"
        assert out.message == message, (
            f"Round-trip failed: expected {message!r}, got {out.message!r}"
        )

    def test_wrong_passphrase_fails(self, dummy_cover, dummy_cost, small_embed_cfg):
        """Extraction with the wrong passphrase should fail verification."""
        message = b"secret"
        result = embed(
            cover_image=dummy_cover,
            message=message,
            passphrase="right",
            cost_matrix=dummy_cost,
            cfg=small_embed_cfg,
        )
        out = extract(
            stego_image=result.stego_image,
            passphrase="wrong",
            image_size_hw=(128, 128),
            cfg=small_embed_cfg,
        )
        assert not out.verified
        assert out.message == b""

    def test_modification_mask(self, dummy_cover, dummy_cost, small_embed_cfg):
        """The modification mask should be a subset of the blue channel pixels."""
        result = embed(
            cover_image=dummy_cover,
            message=b"test",
            passphrase="pw",
            cost_matrix=dummy_cost,
            cfg=small_embed_cfg,
        )
        # Mask is binary
        assert result.modification_mask.dtype == torch.uint8
        assert result.modification_mask.shape == (128, 128)
        # All values are 0 or 1
        assert ((result.modification_mask == 0) | (result.modification_mask == 1)).all()

    def test_stego_close_to_cover(self, dummy_cover, dummy_cost, small_embed_cfg):
        """The stego should be very close to the cover (LSB modifications only)."""
        result = embed(
            cover_image=dummy_cover,
            message=b"x" * 20,
            passphrase="pw",
            cost_matrix=dummy_cost,
            cfg=small_embed_cfg,
        )
        diff = (result.stego_image.int() - dummy_cover.int()).abs()
        # Only the blue channel may differ, and only by at most 1
        assert diff[0].max() == 0
        assert diff[1].max() == 0
        assert diff[2].max() <= 1
