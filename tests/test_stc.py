"""Tests for the STC encoder/decoder round-trip.

These tests do NOT require torch, VLMs, or any heavyweight dependencies.
They validate the STC Viterbi algorithm in isolation using NumPy only.
"""

import numpy as np
import pytest

from semantic_cloak.stc import STCEncoder, STCDecoder, STCConfig, build_submatrix


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def small_cfg():
    """Small STC config for fast tests (h=5, w_hat=11)."""
    return STCConfig(submatrix_height=5, submatrix_width=11)


@pytest.fixture
def default_cfg():
    """Default STC config (h=9, w_hat=19)."""
    return STCConfig()


# ----------------------------------------------------------------------
# Submatrix tests
# ----------------------------------------------------------------------
class TestSubmatrix:
    def test_shape(self, small_cfg):
        H = build_submatrix(small_cfg)
        assert H.shape == (5, 11)

    def test_no_zero_columns(self, small_cfg):
        H = build_submatrix(small_cfg)
        for j in range(H.shape[1]):
            assert H[:, j].sum() > 0, f"Column {j} is all zeros"

    def test_no_duplicate_columns(self, small_cfg):
        H = build_submatrix(small_cfg)
        cols = set()
        for j in range(H.shape[1]):
            col = tuple(H[:, j].tolist())
            assert col not in cols, f"Column {j} duplicates an earlier column"
            cols.add(col)

    def test_deterministic(self, small_cfg):
        """Same config => same submatrix."""
        H1 = build_submatrix(small_cfg)
        H2 = build_submatrix(small_cfg)
        np.testing.assert_array_equal(H1, H2)


# ----------------------------------------------------------------------
# Round-trip tests
# ----------------------------------------------------------------------
class TestRoundTrip:
    def test_short_message_roundtrip(self, small_cfg):
        """A short message should round-trip exactly."""
        n_pixels = 500
        message = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=np.uint8)
        cost = np.random.default_rng(42).uniform(1.0, 100.0, n_pixels)

        encoder = STCEncoder(small_cfg)
        decoder = STCDecoder(small_cfg)

        y = encoder.encode(message, cost)
        assert y.dtype == np.uint8
        assert y.shape == (n_pixels,)

        # Decode
        recovered = decoder.decode(y, n_message_bits=len(message))
        np.testing.assert_array_equal(recovered, message)

    def test_random_message_roundtrip(self, default_cfg):
        """Random messages of various lengths should round-trip."""
        rng = np.random.default_rng(123)
        n_pixels = 5000

        for msg_len in [10, 50, 100, 200]:
            message = rng.integers(0, 2, msg_len).astype(np.uint8)
            cost = rng.uniform(1.0, 100.0, n_pixels)

            encoder = STCEncoder(default_cfg)
            decoder = STCDecoder(default_cfg)

            y = encoder.encode(message, cost)
            recovered = decoder.decode(y, n_message_bits=msg_len)
            np.testing.assert_array_equal(
                recovered, message,
                err_msg=f"Round-trip failed for msg_len={msg_len}"
            )

    def test_low_cost_pixels_modified_more(self, small_cfg):
        """Pixels with lower cost should be more likely to be modified."""
        n_pixels = 2000
        message = np.random.default_rng(7).integers(0, 2, 50).astype(np.uint8)

        # Half pixels cheap, half expensive
        cost = np.concatenate([
            np.full(n_pixels // 2, 1.0),
            np.full(n_pixels // 2, 100.0),
        ])

        encoder = STCEncoder(small_cfg)
        y = encoder.encode(message, cost)

        cheap_mods = y[:n_pixels // 2].sum()
        expensive_mods = y[n_pixels // 2:].sum()
        # The encoder should prefer cheap pixels
        assert cheap_mods >= expensive_mods, (
            f"Expected cheap pixels to be modified more, "
            f"got cheap={cheap_mods}, expensive={expensive_mods}"
        )

    def test_too_long_message_raises(self, small_cfg):
        """Messages that exceed capacity should raise ValueError."""
        n_pixels = 100
        # h=5, w_hat=11 => max payload ≈ n_pixels / w_hat ≈ 9 bits
        message = np.ones(50, dtype=np.uint8)
        cost = np.ones(n_pixels)
        encoder = STCEncoder(small_cfg)
        with pytest.raises(ValueError, match="Message too long"):
            encoder.encode(message, cost)
