"""Tests for the baseline cost functions (S-UNIWARD, WOW, HUGO).

These tests require scipy and pywavelets (for S-UNIWARD). They validate
that the cost functions produce reasonable outputs on synthetic images.
"""

import numpy as np
import pytest
import torch

# Skip the entire module if scipy/pywt not available
scipy = pytest.importorskip("scipy")
pywt = pytest.importorskip("pywt")

from semantic_cloak.baselines import s_uniward_cost, wow_cost, hugo_cost


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def synthetic_cover():
    """A 64x64 RGB cover with a textured region and a flat region."""
    rng = np.random.default_rng(42)
    img = np.zeros((3, 64, 64), dtype=np.uint8)
    # Left half: flat gray
    img[:, :, :32] = 128
    # Right half: random texture
    img[:, :, 32:] = rng.integers(0, 256, (3, 64, 32)).astype(np.uint8)
    return torch.from_numpy(img)


@pytest.fixture
def flat_cover():
    """A 32x32 flat cover (no texture)."""
    img = np.full((3, 32, 32), 100, dtype=np.uint8)
    return torch.from_numpy(img)


# ----------------------------------------------------------------------
# Shape and type
# ----------------------------------------------------------------------
class TestCostShape:
    @pytest.mark.parametrize("cost_fn", [s_uniward_cost, wow_cost, hugo_cost])
    def test_shape_matches_input(self, synthetic_cover, cost_fn):
        cost = cost_fn(synthetic_cover)
        h, w = synthetic_cover.shape[-2:]
        assert cost.shape == (h, w)

    @pytest.mark.parametrize("cost_fn", [s_uniward_cost, wow_cost, hugo_cost])
    def test_dtype_is_float32(self, synthetic_cover, cost_fn):
        cost = cost_fn(synthetic_cover)
        assert cost.dtype == torch.float32

    @pytest.mark.parametrize("cost_fn", [s_uniward_cost, wow_cost, hugo_cost])
    def test_cost_in_range(self, synthetic_cover, cost_fn):
        """Cost should be normalised to [1, 100]."""
        cost = cost_fn(synthetic_cover)
        assert cost.min() >= 1.0 - 1e-5
        assert cost.max() <= 100.0 + 1e-5


# ----------------------------------------------------------------------
# Behavioural tests
# ----------------------------------------------------------------------
class TestCostBehavior:
    @pytest.mark.parametrize("cost_fn", [s_uniward_cost, wow_cost, hugo_cost])
    def test_flat_cover_uniform_cost(self, flat_cover, cost_fn):
        """A completely flat cover should produce approximately uniform cost.

        Note: S-UNIWARD may show higher variance on flat covers due to
        wavelet boundary effects; we use a relaxed threshold.
        """
        cost = cost_fn(flat_cover)
        std = cost.std().item()
        mean = cost.mean().item()
        # All three baselines should produce relatively uniform cost on
        # a flat cover. We allow up to 80% coefficient of variation to
        # account for boundary effects in the wavelet decomposition.
        assert std / mean < 0.8, (
            f"Flat cover cost has high variance: std={std}, mean={mean}"
        )

    def test_textured_region_cheaper_than_flat(self, synthetic_cover):
        """Textured regions should have lower cost (cheaper to modify)."""
        # The synthetic cover has flat left, textured right.
        # S-UNIWARD, WOW, HUGO all assign LOW cost to textured regions.
        for cost_fn in [s_uniward_cost, wow_cost, hugo_cost]:
            cost = cost_fn(synthetic_cover)
            left_mean = cost[:, :32].mean().item()
            right_mean = cost[:, 32:].mean().item()
            # Textured (right) should be cheaper than flat (left).
            # Note: after normalisation to [1, 100], "cheap" means
            # closer to 1 and "expensive" means closer to 100. But
            # the convention in our implementation is the opposite:
            # high cost = expensive to modify, low cost = cheap.
            # So we expect right_mean < left_mean.
            assert right_mean < left_mean, (
                f"{cost_fn.__name__}: expected textured (right) to be "
                f"cheaper than flat (left), got left={left_mean}, "
                f"right={right_mean}"
            )


# ----------------------------------------------------------------------
# Grayscale support
# ----------------------------------------------------------------------
class TestGrayscale:
    @pytest.mark.parametrize("cost_fn", [s_uniward_cost, wow_cost, hugo_cost])
    def test_grayscale_input(self, cost_fn):
        """Cost functions should accept [1, H, W] grayscale input."""
        img = torch.randint(0, 256, (1, 32, 32), dtype=torch.uint8)
        cost = cost_fn(img)
        assert cost.shape == (32, 32)
