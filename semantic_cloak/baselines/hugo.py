r"""hugo.py
=========
HUGO (Highly Undetectable steGO) distortion cost
(Pevny, Filler & Bas, IH 2010).

HUGO computes the cost from the SPAM (Subspace-based Active
Measurement) feature residual. For each pixel, the cost is based on
the directional differences to its 8 neighbors:

.. math::
    \rho_i = \sum_{d \in \mathcal{D}} \sum_{j \in \mathcal{N}_d(i)}
             |I_i - I_j|^{2},

where :math:`\mathcal{D}` is the set of 4 directions (right, down,
diagonal-45, diagonal-135) and :math:`\mathcal{N}_d(i)` is the
immediate neighbor of ``i`` in direction ``d``.

The cost is then transformed via a soft-thresholding function so that
small differences (smooth regions) get high cost and large differences
(textured regions) get low cost.

Author : Samir (Maharishi Dayanand University)
License: MIT
"""

from __future__ import annotations

import numpy as np
import torch


def hugo_cost(
    cover: torch.Tensor,
    sigma: float = 2.0,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Compute the HUGO distortion cost.

    Parameters
    ----------
    cover : Tensor
        Cover image of shape ``[3, H, W]`` or ``[1, H, W]``, dtype
        ``uint8`` or float in ``[0, 255]``.
    sigma : float
        Soft-threshold parameter. Controls how quickly the cost rises
        for small residuals.
    gamma : float
        Exponent for the residual aggregation.

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

    # Compute directional differences (reflect boundary)
    d_right = np.zeros_like(gray)
    d_right[:, :-1] = np.abs(gray[:, :-1] - gray[:, 1:])

    d_down = np.zeros_like(gray)
    d_down[:-1, :] = np.abs(gray[:-1, :] - gray[1:, :])

    d_diag45 = np.zeros_like(gray)
    d_diag45[:-1, :-1] = np.abs(gray[:-1, :-1] - gray[1:, 1:])

    d_diag135 = np.zeros_like(gray)
    d_diag135[:-1, 1:] = np.abs(gray[:-1, 1:] - gray[1:, :-1])

    # Sum of squared differences across directions
    residual = (
        d_right ** gamma
        + d_down ** gamma
        + d_diag45 ** gamma
        + d_diag135 ** gamma
    )

    # HUGO soft-thresholding: high residual => low cost (cheap to modify)
    # rho = 1 / (1 + residual / sigma)
    rho = 1.0 / (1.0 + residual / sigma)

    # Normalise to [1, 100]
    rho_min = rho.min()
    rho_max = rho.max()
    if (rho_max - rho_min) > 1e-9:
        rho = (rho - rho_min) / (rho_max - rho_min)
    rho = 1.0 + rho * 99.0
    return torch.from_numpy(rho.astype(np.float32))
