"""Tests for the metrics module (PSNR, SSIM, BER).

These tests use only torch and numpy (no LPIPS, no PIL).
"""

import numpy as np
import pytest
import torch

from semantic_cloak.metrics import (
    psnr, ssim, bit_error_rate, gaussian_noise, _gaussian_window,
)


# ----------------------------------------------------------------------
# PSNR
# ----------------------------------------------------------------------
class TestPSNR:
    def test_identical_images_inf(self):
        img = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
        assert psnr(img, img) == float("inf")

    def test_different_images_finite(self):
        a = torch.zeros((3, 32, 32), dtype=torch.uint8)
        b = torch.ones((3, 32, 32), dtype=torch.uint8) * 10
        result = psnr(a, b)
        assert 20 < result < 100  # 10-level difference should give moderate PSNR

    def test_psnr_decreases_with_noise(self):
        """More noise => lower PSNR."""
        img = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
        psnr_low = psnr(img, gaussian_noise(img, sigma=1.0))
        psnr_high = psnr(img, gaussian_noise(img, sigma=20.0))
        assert psnr_low > psnr_high


# ----------------------------------------------------------------------
# SSIM
# ----------------------------------------------------------------------
class TestSSIM:
    def test_identical_images_one(self):
        img = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
        result = ssim(img, img)
        assert abs(result - 1.0) < 1e-5

    def test_different_images_less_than_one(self):
        a = torch.zeros((3, 64, 64), dtype=torch.uint8)
        b = torch.full((3, 64, 64), 50, dtype=torch.uint8)
        result = ssim(a, b)
        assert 0 <= result < 1.0

    def test_ssim_2d_input(self):
        """SSIM should work with 2D (grayscale) input."""
        img = torch.randint(0, 256, (64, 64), dtype=torch.uint8)
        result = ssim(img, img)
        assert abs(result - 1.0) < 1e-5


# ----------------------------------------------------------------------
# Bit Error Rate
# ----------------------------------------------------------------------
class TestBER:
    def test_identical_bits_zero(self):
        bits = np.array([0, 1, 1, 0, 1], dtype=np.uint8)
        assert bit_error_rate(bits, bits) == 0.0

    def test_all_flipped(self):
        a = np.array([0, 0, 0, 0], dtype=np.uint8)
        b = np.array([1, 1, 1, 1], dtype=np.uint8)
        assert bit_error_rate(a, b) == 1.0

    def test_half_flipped(self):
        a = np.array([0, 0, 1, 1], dtype=np.uint8)
        b = np.array([0, 1, 1, 0], dtype=np.uint8)
        assert bit_error_rate(a, b) == 0.5

    def test_different_lengths(self):
        """BER compares only the first min(len, len) bits."""
        a = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
        b = np.array([0, 1, 0], dtype=np.uint8)
        assert bit_error_rate(a, b) == 0.0


# ----------------------------------------------------------------------
# Gaussian noise
# ----------------------------------------------------------------------
class TestGaussianNoise:
    def test_shape_preserved(self):
        img = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
        noisy = gaussian_noise(img, sigma=2.0)
        assert noisy.shape == img.shape
        assert noisy.dtype == img.dtype

    def test_values_in_range(self):
        img = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
        noisy = gaussian_noise(img, sigma=100.0)
        assert noisy.min() >= 0
        assert noisy.max() <= 255

    def test_zero_sigma_identity(self):
        """With sigma=0, noise should be approximately identity."""
        img = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
        noisy = gaussian_noise(img, sigma=0.0)
        torch.testing.assert_close(noisy, img)


# ----------------------------------------------------------------------
# Gaussian window
# ----------------------------------------------------------------------
class TestGaussianWindow:
    def test_shape(self):
        w = _gaussian_window(window_size=11, sigma=1.5)
        assert w.shape == (11, 11)

    def test_sums_to_one(self):
        """A normalised Gaussian window should sum to 1."""
        w = _gaussian_window(window_size=11, sigma=1.5)
        assert abs(w.sum().item() - 1.0) < 1e-5
