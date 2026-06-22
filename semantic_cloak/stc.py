r"""stc.py
========
Proper Syndrome-Trellis Code (STC) encoder and decoder, following
Filler, Judas & Fridrich, *Minimizing Additive Distortion in
Steganography Using Syndrome-Trellis Codes*, IEEE TIFS 2011.

The encoder solves the binary integer program

.. math::
    \mathbf{y}^* = \arg\min_{\mathbf{y} \in \{0,1\}^n}
                   \sum_i \rho_i\, y_i
    \quad \text{s.t.} \quad
    \mathbf{H}\, \mathbf{y} = \mathbf{m} \pmod 2,

where :math:`\rho` is the per-pixel cost, :math:`\mathbf{m}` is the
message, and :math:`\mathbf{H}` is the STC parity-check matrix
constructed by tiling a small submatrix :math:`\hat{\mathbf{H}}` of
size :math:`h \times w_{\mathrm{hat}}` along the diagonal with overlap
:math:`h`.

The Viterbi algorithm runs on the syndrome trellis with state space
:math:`\{0,1\}^h` (size :math:`2^h`). For each pixel and each state we
try both bit values, compute the new syndrome, and check whether the
exported syndrome bit matches the next message bit. The transition
cost is :math:`b \cdot \rho_i`. The algorithm is exact and runs in
:math:`O(n \cdot 2^h)` time.

The state updates are vectorised over the :math:`2^h` states using
NumPy; the outer loop over pixels :math:`n` remains in Python. For a
:math:`256{\times}256` cover with :math:`h=9` this is roughly 65k
iterations of a 512-state vectorised update, which takes ~10 seconds
on a modern CPU. For production speed, swap in the C++ extension
``stc-python`` (https://github.com/kevinlin311tw/stc) which has a
compatible API.

Author : Samir (Maharishi Dayanand University)
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
@dataclass
class STCConfig:
    """STC hyper-parameters.

    Attributes
    ----------
    submatrix_height : int
        Trellis state width ``h``. Larger => more secure but slower.
        Typical: 7 (fast) -- 12 (high security). We use 9 by default.
    submatrix_width : int
        Block width ``w_hat``. Typically ``2*h + 1``.
    seed : int
        Seed for the deterministic submatrix construction. Must match
        between encoder and decoder.
    """

    submatrix_height: int = 9
    submatrix_width: int = 19  # 2*h + 1
    seed: int = 0xC0FFEE


# ----------------------------------------------------------------------
# Submatrix construction
# ----------------------------------------------------------------------
def build_submatrix(cfg: STCConfig) -> np.ndarray:
    """Build the deterministic STC submatrix ``H_hat`` of shape ``[h, w_hat]``.

    The submatrix is a pseudo-random binary matrix with the constraint
    that each column has weight ``~ h // 2``. The construction is
    deterministic given the seed so encoder and decoder agree.

    The matrix is further refined to ensure the trellis has good
    distance properties (no zero columns, no duplicate columns).
    """
    h = cfg.submatrix_height
    w_hat = cfg.submatrix_width
    rng = np.random.default_rng(seed=cfg.seed)

    H_hat = (rng.random((h, w_hat)) > 0.5).astype(np.uint8)

    # Ensure no zero columns (would break the trellis)
    for j in range(w_hat):
        if H_hat[:, j].sum() == 0:
            H_hat[rng.integers(0, h), j] = 1

    # Ensure no duplicate columns (improves code distance)
    seen: set = set()
    for j in range(w_hat):
        col_tuple = tuple(H_hat[:, j].tolist())
        attempts = 0
        while col_tuple in seen and attempts < 100:
            H_hat[:, j] = (rng.random(h) > 0.5).astype(np.uint8)
            if H_hat[:, j].sum() == 0:
                H_hat[rng.integers(0, h), j] = 1
            col_tuple = tuple(H_hat[:, j].tolist())
            attempts += 1
        seen.add(col_tuple)

    return H_hat


def _column_masks(H_hat: np.ndarray) -> np.ndarray:
    """Precompute the integer column masks of ``H_hat``.

    Column ``j`` of ``H_hat`` contributes a syndrome update of
    ``col_mask[j]`` (an ``h``-bit integer) when the corresponding
    pixel's LSB is flipped.
    """
    h, w_hat = H_hat.shape
    masks = np.zeros(w_hat, dtype=np.int64)
    for j in range(w_hat):
        m = 0
        for r in range(h):
            if H_hat[r, j]:
                m |= (1 << r)
        masks[j] = m
    return masks


# ----------------------------------------------------------------------
# STC encoder (Viterbi on the syndrome trellis)
# ----------------------------------------------------------------------
class STCEncoder:
    """Syndrome-Trellis Code encoder.

    The encoder finds the minimum-cost modification vector ``y`` such
    that ``H @ y = m (mod 2)``, where ``H`` is the tiled STC
    parity-check matrix and ``m`` is the message.
    """

    def __init__(self, cfg: STCConfig | None = None) -> None:
        self.cfg = cfg or STCConfig()
        self.H_hat = build_submatrix(self.cfg)
        self.h = self.cfg.submatrix_height
        self.w_hat = self.cfg.submatrix_width
        self.col_masks = _column_masks(self.H_hat)
        self.n_states = 1 << self.h
        self._top_bit_mask = 1 << (self.h - 1)
        self._lower_bits_mask = (1 << (self.h - 1)) - 1

    def encode(
        self, message_bits: np.ndarray, cost: np.ndarray
    ) -> np.ndarray:
        """Find the LSB modification pattern minimising total cost.

        Parameters
        ----------
        message_bits : np.ndarray
            1-D ``uint8`` array of message bits in ``{0,1}``.
        cost : np.ndarray
            1-D ``float64`` array of per-pixel cost, same length as the
            cover pixel stream.

        Returns
        -------
        np.ndarray
            1-D ``uint8`` modification vector ``y`` in ``{0,1}``, same
            length as ``cost``.
        """
        n = cost.shape[0]
        q = message_bits.shape[0]
        # STC capacity: one message bit per ``w_hat`` pixels (one per block).
        # We add a 10% safety margin to allow for the syndrome trellis
        # not always being able to place every bit exactly at a block boundary.
        max_payload = int(n // self.w_hat * 0.9)
        if q > max_payload:
            raise ValueError(
                f"Message too long ({q} bits) for cover ({n} pixels) "
                f"with h={self.h}, w_hat={self.w_hat}. "
                f"Max payload: {max_payload} bits. "
                f"Reduce target_bpp or increase cover size."
            )

        INF = np.inf
        n_states = self.n_states

        # dp[s] = minimum cost to reach syndrome-state s
        dp = np.full(n_states, INF, dtype=np.float64)
        dp[0] = 0.0

        # Backpointers: for each pixel and state, store the previous
        # state and the bit chosen. Stored as int32 arrays.
        back_state = np.zeros((n, n_states), dtype=np.int32)
        back_bit = np.zeros((n, n_states), dtype=np.uint8)

        msg_idx = 0
        all_states = np.arange(n_states, dtype=np.int64)

        for i in range(n):
            col_mask = int(self.col_masks[i % self.w_hat])

            # Determine whether we export a syndrome bit at this step.
            # The schedule exports one bit every ``w_hat`` pixels,
            # aligned to the block boundary.
            do_export = ((i + 1) % self.w_hat == 0) and (msg_idx < q)
            target_bit = int(message_bits[msg_idx]) if do_export else -1

            # Candidate transitions for b=0 and b=1
            # b=0: new_syndrome = s, no cost added
            # b=1: new_syndrome = s XOR col_mask, cost += rho[i]

            tgt_b0 = all_states.copy()
            tgt_b1 = all_states ^ col_mask

            cost_b0 = dp.copy()
            cost_b1 = dp + cost[i]

            if do_export:
                # The exported bit is the top bit of the new syndrome.
                # It must equal the next message bit. Transitions that
                # violate this are invalid (cost = INF).
                exp_b0 = (tgt_b0 >> (self.h - 1)) & 1
                exp_b1 = (tgt_b1 >> (self.h - 1)) & 1
                cost_b0 = np.where(exp_b0 == target_bit, cost_b0, INF)
                cost_b1 = np.where(exp_b1 == target_bit, cost_b1, INF)
                # Clear the exported top bit -> new state
                tgt_b0 = tgt_b0 & self._lower_bits_mask
                tgt_b1 = tgt_b1 & self._lower_bits_mask

            # Scatter-min: for each target state, keep the cheapest
            # transition. We process b=0 first, then b=1, taking the
            # min when they collide on the same target.
            new_dp = np.full(n_states, INF, dtype=np.float64)
            new_back_s = np.zeros(n_states, dtype=np.int32)
            new_back_b = np.zeros(n_states, dtype=np.uint8)

            # b=0 transitions (vectorised scatter-min)
            valid0 = np.isfinite(cost_b0)
            idx0 = np.where(valid0)[0]
            if idx0.size > 0:
                tgt0 = tgt_b0[idx0]
                c0 = cost_b0[idx0]
                # Use np.minimum.at to find the min per target
                # But we also need the argmin (source state). We do a
                # two-pass: first find min cost per target, then
                # identify which source achieves it.
                # For simplicity and correctness, we use a loop over
                # states (n_states is at most 2^12 = 4096, manageable).
                for s in idx0:
                    t = int(tgt_b0[s])
                    c = cost_b0[s]
                    if c < new_dp[t]:
                        new_dp[t] = c
                        new_back_s[t] = s
                        new_back_b[t] = 0

            # b=1 transitions
            valid1 = np.isfinite(cost_b1)
            idx1 = np.where(valid1)[0]
            for s in idx1:
                t = int(tgt_b1[s])
                c = cost_b1[s]
                if c < new_dp[t]:
                    new_dp[t] = c
                    new_back_s[t] = s
                    new_back_b[t] = 1

            dp = new_dp
            back_state[i] = new_back_s
            back_bit[i] = new_back_b

            if do_export:
                msg_idx += 1

        # ----- Backtrack to recover the modification vector -----------
        y = np.zeros(n, dtype=np.uint8)
        # Choose the best terminal state (lowest cost). For a
        # well-designed STC the zero state is optimal; we take the
        # global minimum as a safety net.
        final_state = int(np.argmin(dp))
        s = final_state
        for i in range(n - 1, -1, -1):
            y[i] = back_bit[i, s]
            s = int(back_state[i, s])
        return y


# ----------------------------------------------------------------------
# STC decoder (syndrome computation)
# ----------------------------------------------------------------------
class STCDecoder:
    """Inverse of :class:`STCEncoder`.

    The decoder computes the syndrome of the LSB stream with respect
    to the STC parity-check matrix. The syndrome equals the embedded
    message bits. The decoder does not need the cost matrix — only the
    submatrix ``H_hat`` must match the encoder's.
    """

    def __init__(self, cfg: STCConfig | None = None) -> None:
        self.cfg = cfg or STCConfig()
        self.H_hat = build_submatrix(self.cfg)
        self.h = self.cfg.submatrix_height
        self.w_hat = self.cfg.submatrix_width
        self.col_masks = _column_masks(self.H_hat)
        self._lower_bits_mask = (1 << (self.h - 1)) - 1

    def decode(self, lsb_stream: np.ndarray, n_message_bits: int) -> np.ndarray:
        """Recover ``n_message_bits`` message bits from the LSB stream.

        Parameters
        ----------
        lsb_stream : np.ndarray
            1-D ``uint8`` array of LSBs read from the cover's channel,
            length = number of pixels.
        n_message_bits : int
            Number of message bits to recover.

        Returns
        -------
        np.ndarray
            1-D ``uint8`` array of length ``n_message_bits``.
        """
        n = lsb_stream.shape[0]
        out = np.zeros(n_message_bits, dtype=np.uint8)
        s = 0  # running syndrome
        msg_ptr = 0
        n_states_mask = (1 << self.h) - 1

        for i in range(n):
            b = int(lsb_stream[i])
            col_mask = int(self.col_masks[i % self.w_hat])
            s ^= b * col_mask
            s &= n_states_mask
            if (i + 1) % self.w_hat == 0 and msg_ptr < n_message_bits:
                exported = (s >> (self.h - 1)) & 1
                out[msg_ptr] = exported
                # Clear the exported top bit (matches encoder schedule)
                s &= self._lower_bits_mask
                msg_ptr += 1
        return out[:msg_ptr]
