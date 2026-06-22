r"""wow.py
========
WOW (Wavelet-Obtained Weights) distortion cost
(Holub & Fridrich, IEEE WIFS 2012).

WOW uses a bank of four directional KV filters (3x3 and 5x5) to
compute directional residuals. The cost is a weighted sum of the
absolute residuals, where the weights are inversely proportional to
the maximum residual magnitude:

.. math::
    \rho_i = \sum_k w_k(i) \cdot |R_k(i)|, \quad
    w_k(i) = \frac{|R_k(i)|}{\max_j |R_k(j)|}.

Pixels with high residual energy (textured) are assigned low cost
(cheap to modify).

Author : Samir (Maharishi Dayanand University)
License: MIT
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import correlate


# ----------------------------------------------------------------------
# KV directional filter bank (3x3, as in the original WOW paper)
# ----------------------------------------------------------------------
_KV_FILTERS = [
    # Horizontal edge
    np.array(
        [[0.0, 0.0, 0.0],
         [0.5, -1.0, 0.5],
         [0.0, 0.0, 0.0]],
        dtype=np.float64,
    ),
    # Vertical edge
    np.array(
        [[0.0, 0.5, 0.0],
         [0.0, -1.0, 0.0],
         [0.0, 0.5, 0.0]],
        dtype=np.float64,
    ),
    # Diagonal (45°)
    np.array(
        [[0.25, 0.0, 0.0],
         [0.0, -1.0, 0.0],
         [0.0, 0.0, 0.25]],
        dtype=np.float64,
    ),
    # Diagonal (135°)
    np.array(
        [[0.0, 0.0, 0.25],
         [0.0, -1.0, 0.0],
         [0.25, 0.0, 0.0]],
        dtype=np.float64,
    ),
]


def wow_cost(cover: torch.Tensor) -> torch.Tensor:
    """Compute the WOW distortion cost.

    Parameters
    ----------
    cover : Tensor
        Cover image of shape ``[3, H, W]`` or ``[1, H, W]``, dtype
        ``uint8`` or float in ``[0, 255]``.

    Returns
    -------
    Tensor
        Per-pixel cost ``rho`` of shape ``[H, W]``, dtype ``float32``,
        normalised to ``[1, 100]``.
    """
    if cover.dim() != 3:
        raise ValueError(f"Expected [C,H,W] tensor, got {cover.shape}")
    if cover.shape[0] == 3:
        gray = (
            0.299 * cover[0] + 0.587 * cover[1] + 0.114 * cover[2]
        ).cpu().numpy().astype(np.float64)
    else:
        gray = cover[0].cpu().numpy().astype(np.float64)

    # Compute directional residuals
    residuals = []
    for kernel in _KV_FILTERS:
        r = np.abs(correlate(gray, kernel, mode="reflect"))
        residuals.append(r)

    # WOW weighting: w_k(i) = |R_k(i)| / max_j |R_k(j)|
    # Cost = sum_k w_k(i) * |R_k(i)|
    rho = np.zeros_like(gray)
    for r in residuals:
        r_max = r.max()
        if r_max > 1e-9:
            w = r / r_max
        else:
            w = np.zeros_like(r)
        rho += w * r

    # Invert: high residual energy should be LOW cost (cheap to modify)
    # rho currently = sum of weighted residuals (high for textured)
    # For STC: low cost = cheap to modify
    # So distortion = 1 / (1 + rho)
    rho = 1.0 / (1.0 + rho)

    # Normalise to [1, 100]
    rho_min = rho.min()
    rho_max = rho.max()
    if (rho_max - rho_min) > 1e-9:
        rho = (rho - rho_min) / (rho_max - rho_min)
    rho = 1.0 + rho * 99.0
    return torch.from_numpy(rho.astype(np.float32))
