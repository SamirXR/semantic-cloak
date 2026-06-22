r"""metrics.py
============
Imperceptibility and robustness metrics for steganography evaluation.

* **PSNR** — Peak Signal-to-Noise Ratio (in dB).
* **SSIM** — Structural Similarity Index (Wang et al., 2004).
* **LPIPS** — Learned Perceptual Image Patch Similarity (Zhang et al.,
  2018), using the official VGG-based weights.
* **BER** — Bit Error Rate for robustness evaluation.

All metrics operate on ``torch.Tensor`` images in ``[0, 255]`` uint8 or
float format. LPIPS requires the ``lpips`` Python package.

Author : Samir (Maharishi Dayanand University)
License: MIT
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------
# PSNR
# ----------------------------------------------------------------------
def psnr(
    cover: torch.Tensor,
    stego: torch.Tensor,
    max_val: float = 255.0,
) -> float:
    """Peak Signal-to-Noise Ratio.

    Parameters
    ----------
    cover, stego : Tensor
        Images of the same shape ``[...]``, dtype ``uint8`` or float
        in ``[0, max_val]``.
    max_val : float
        Maximum possible pixel value (255 for 8-bit).

    Returns
    -------
    float
        PSNR in dB. Returns ``inf`` if images are identical.
    """
    cover = cover.float()
    stego = stego.float()
    mse = F.mse_loss(stego, cover)
    if mse.item() == 0:
        return float("inf")
    return float(
        20.0 * np.log10(max_val) - 10.0 * np.log10(mse.item())
    )


# ----------------------------------------------------------------------
# SSIM (Wang et al. 2004, 11x11 Gaussian window, sigma=1.5)
# ----------------------------------------------------------------------
def _gaussian_window(
    window_size: int = 11, sigma: float = 1.5
) -> torch.Tensor:
    """Create a 2D Gaussian window for SSIM convolution."""
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(0) * g.unsqueeze(1)
    return window


def ssim(
    cover: torch.Tensor,
    stego: torch.Tensor,
    max_val: float = 255.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """Structural Similarity Index (Wang et al., 2004).

    Parameters
    ----------
    cover, stego : Tensor
        Images of shape ``[C, H, W]`` or ``[H, W]``, dtype ``uint8`` or
        float in ``[0, max_val]``.
    max_val : float
        Maximum pixel value.
    window_size : int
        Size of the Gaussian window (default 11, per the paper).
    sigma : float
        Standard deviation of the Gaussian window (default 1.5).

    Returns
    -------
    float
        Mean SSIM over all pixels (1.0 = perfect match).
    """
    if cover.dim() == 2:
        cover = cover.unsqueeze(0)
        stego = stego.unsqueeze(0)
    if cover.dim() == 3:
        cover = cover.unsqueeze(0)
        stego = stego.unsqueeze(0)

    c = cover.shape[1]
    window = _gaussian_window(window_size, sigma).to(cover.device)
    window = window.unsqueeze(0).unsqueeze(0).expand(c, 1, -1, -1).contiguous()

    cover = cover.float()
    stego = stego.float()

    pad = window_size // 2
    mu1 = F.conv2d(cover, window, padding=pad, groups=c)
    mu2 = F.conv2d(stego, window, padding=pad, groups=c)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(cover * cover, window, padding=pad, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(stego * stego, window, padding=pad, groups=c) - mu2_sq
    sigma12 = F.conv2d(cover * stego, window, padding=pad, groups=c) - mu1_mu2

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    ssim_map = (
        (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    ) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(ssim_map.mean().item())


# ----------------------------------------------------------------------
# LPIPS (requires the `lpips` package)
# ----------------------------------------------------------------------
_LPIPS_MODEL = None  # cached singleton


def _get_lpips_model(net: str = "vgg", device: str = "cpu"):
    """Lazy-load and cache the LPIPS model."""
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None:
        import lpips  # type: ignore

        _LPIPS_MODEL = lpips.LPIPS(net=net).to(device).eval()
    return _LPIPS_MODEL


def lpips(
    cover: torch.Tensor,
    stego: torch.Tensor,
    net: str = "vgg",
    device: Optional[str] = None,
) -> float:
    """Learned Perceptual Image Patch Similarity (Zhang et al., 2018).

    Lower = more similar. 0 = identical.

    Parameters
    ----------
    cover, stego : Tensor
        Images of shape ``[C, H, W]`` or ``[1, C, H, W]``, dtype
        ``uint8`` or float in ``[0, 255]``.
    net : str
        Backbone network: ``'vgg'``, ``'alex'``, or ``'squeeze'``.
    device : Optional[str]
        Torch device. Defaults to cover's device.

    Returns
    -------
    float
        LPIPS distance (0 = identical, higher = more different).
    """
    if device is None:
        device = str(cover.device)

    if cover.dim() == 3:
        cover = cover.unsqueeze(0)
        stego = stego.unsqueeze(0)

    # LPIPS expects [-1, 1] float
    cover_n = cover.float() * 2.0 / 255.0 - 1.0
    stego_n = stego.float() * 2.0 / 255.0 - 1.0
    cover_n = cover_n.to(device)
    stego_n = stego_n.to(device)

    model = _get_lpips_model(net=net, device=device)
    with torch.no_grad():
        dist = model(cover_n, stego_n)
    return float(dist.squeeze().item())


# ----------------------------------------------------------------------
# Bit Error Rate (for robustness evaluation)
# ----------------------------------------------------------------------
def bit_error_rate(
    original_bits: np.ndarray,
    recovered_bits: np.ndarray,
) -> float:
    """Compute the Bit Error Rate between two bit arrays.

    Parameters
    ----------
    original_bits, recovered_bits : np.ndarray
        1-D ``uint8`` arrays of message bits in ``{0, 1}``. Need not be
        the same length; only the first ``min(len, len)`` bits are
        compared.

    Returns
    -------
    float
        BER in ``[0, 1]``. 0 = perfect recovery.
    """
    n = min(len(original_bits), len(recovered_bits))
    if n == 0:
        return 0.0
    errors = int(np.bitwise_xor(
        original_bits[:n], recovered_bits[:n]
    ).sum())
    return errors / n


# ----------------------------------------------------------------------
# JPEG robustness probe
# ----------------------------------------------------------------------
def jpeg_compress(
    image: torch.Tensor,
    quality: int = 75,
) -> torch.Tensor:
    """JPEG-compress a torch image and return the result as a torch tensor.

    Parameters
    ----------
    image : Tensor
        Image of shape ``[3, H, W]`` or ``[1, H, W]``, dtype ``uint8``.
    quality : int
        JPEG quality in ``[1, 100]``.

    Returns
    -------
    Tensor
        JPEG-compressed image, same shape and dtype as input.
    """
    from PIL import Image
    import io

    if image.dim() == 2:
        image = image.unsqueeze(0)
    if image.shape[0] == 1:
        # Grayscale
        arr = image.squeeze(0).cpu().numpy().astype(np.uint8)
        pil = Image.fromarray(arr, mode="L")
    else:
        arr = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        pil = Image.fromarray(arr, mode="RGB")

    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    pil_back = Image.open(buf)

    if image.shape[0] == 1:
        arr_back = np.array(pil_back)
        return torch.from_numpy(arr_back).unsqueeze(0).to(image.device)
    else:
        arr_back = np.array(pil_back)
        if arr_back.ndim == 2:
            arr_back = np.stack([arr_back] * 3, axis=-1)
        return (
            torch.from_numpy(arr_back).permute(2, 0, 1).contiguous().to(image.device)
        )


def gaussian_noise(
    image: torch.Tensor,
    sigma: float = 2.0,
) -> torch.Tensor:
    """Add Gaussian noise N(0, sigma^2) to an image.

    Parameters
    ----------
    image : Tensor
        Image of shape ``[C, H, W]``, dtype ``uint8``.
    sigma : float
        Standard deviation of the noise (in pixel intensity units).

    Returns
    -------
    Tensor
        Noised image, same shape and dtype as input.
    """
    noise = torch.randn_like(image.float()) * sigma
    noisy = (image.float() + noise).round().clamp(0, 255).to(image.dtype)
    return noisy
