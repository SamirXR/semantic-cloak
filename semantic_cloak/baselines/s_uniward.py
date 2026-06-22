r"""s_uniward.py
==============
S-UNIWARD distortion cost (Holub, Fridrich & Denemark, EURASIP JIS 2014).

The cost is computed in the wavelet domain using Daubechies 8-tap
wavelets. For each pixel, the cost is the sum of absolute wavelet
coefficients in a :math:`3 \times 3` neighborhood (in the wavelet
domain) across the three high-pass directions (LH, HL, HH):

.. math::
    \rho_i = \frac{1}{\sigma + \sum_{k \in \{LH, HL, HH\}}
              \sum_{j \in \mathcal{N}(i)} |W_k(j)|},

where :math:`\mathcal{N}(i)` is the 3x3 neighborhood of pixel ``i`` in
the wavelet subband, :math:`W_k(j)` is the wavelet coefficient at
position ``j`` in direction ``k``, and :math:`\sigma` is a stabilising
constant.

The final per-pixel distortion for STC is :math:`\rho_i^{-1}`, since
pixels with high wavelet energy (textured) are cheaper to modify.

Requires ``PyWavelets`` (``pip install pywavelets``) and ``scipy``.

Author : Samir (Maharishi Dayanand University)
License: MIT
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import correlate

try:
    import pywt
except ImportError as exc:
    raise ImportError(
        "S-UNIWARD baseline requires PyWavelets. "
        "Install with `pip install pywavelets`."
    ) from exc


def _wavelet_residuals(cover_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the LH, HL, HH wavelet subbands of a grayscale cover.

    Uses Daubechies 8-tap wavelets (``db8`` in PyWavelets), as in the
    original S-UNIWARD implementation. We use ``periodization`` boundary
    mode so the subbands are exactly half the cover size, which
    simplifies spatial alignment.
    """
    _, (LH, HL, HH) = pywt.dwt2(
        cover_gray.astype(np.float64), "db8", mode="periodization"
    )
    return LH, HL, HH


def _upsample_to_full(
    subband: np.ndarray, target_h: int, target_w: int
) -> np.ndarray:
    """Zero-stuff a DWT subband back to the cover resolution.

    The DWT halves the spatial resolution. We use zero-insertion
    upsampling: each wavelet coefficient at position ``(i, j)`` in the
    subband is placed at position ``(2*i, 2*j)`` in the full-resolution
    array, with zeros in between. This preserves the spatial alignment
    between cover pixels and wavelet coefficients.
    """
    out = np.zeros((target_h, target_w), dtype=np.float64)
    h, w = subband.shape
    # Zero-insertion: coefficient (i,j) -> pixel (2i, 2j)
    out[: 2 * h : 2, : 2 * w : 2] = subband
    return out


def s_uniward_cost(cover: torch.Tensor, sigma: float = 1e-6) -> torch.Tensor:
    """Compute the S-UNIWARD distortion cost.

    Parameters
    ----------
    cover : Tensor
        Cover image of shape ``[3, H, W]`` (RGB) or ``[1, H, W]``
        (grayscale), dtype ``uint8`` or float in ``[0, 255]``.
    sigma : float
        Stabilising constant to prevent division by zero.

    Returns
    -------
    Tensor
        Per-pixel cost ``rho`` of shape ``[H, W]``, dtype ``float32``,
        values normalised to ``[1, 100]`` (matching Semantic Cloak's
        convention so the same STC encoder can be used).
    """
    if cover.dim() != 3:
        raise ValueError(f"Expected [C,H,W] tensor, got {cover.shape}")
    if cover.shape[0] == 3:
        # S-UNIWARD operates on luminance (Rec. 601)
        gray = (
            0.299 * cover[0] + 0.587 * cover[1] + 0.114 * cover[2]
        ).cpu().numpy()
    else:
        gray = cover[0].cpu().numpy()
    H, W = gray.shape

    # Compute wavelet subbands and upsample to full resolution
    LH, HL, HH = _wavelet_residuals(gray)
    LH_f = _upsample_to_full(np.abs(LH), H, W)
    HL_f = _upsample_to_full(np.abs(HL), H, W)
    HH_f = _upsample_to_full(np.abs(HH), H, W)

    # 3x3 neighborhood sum (reflect boundary)
    kernel = np.ones((3, 3), dtype=np.float64)
    LH_nb = correlate(LH_f, kernel, mode="reflect")
    HL_nb = correlate(HL_f, kernel, mode="reflect")
    HH_nb = correlate(HH_f, kernel, mode="reflect")

    # S-UNIWARD cost: 1 / (sigma + sum of neighborhood residuals)
    # High textured energy => low cost (easy to modify)
    # Low textured energy => high cost (hard to modify)
    rho = 1.0 / (sigma + LH_nb + HL_nb + HH_nb)

    # Convert to distortion (high cost = hard to modify)
    # S-UNIWARD distortion = 1/rho (invert so textured = expensive)
    # Wait — the convention is: rho = cost, low rho = cheap to modify.
    # In S-UNIWARD, textured regions have HIGH wavelet energy, so
    # 1/(sigma + energy) is LOW => cheap to modify (correct).
    # We use rho directly as the cost.

    # Normalise to [1, 100] for STC compatibility
    rho_min = rho.min()
    rho_max = rho.max()
    if (rho_max - rho_min) > 1e-9:
        rho = (rho - rho_min) / (rho_max - rho_min)
    rho = 1.0 + rho * 99.0
    return torch.from_numpy(rho.astype(np.float32))
