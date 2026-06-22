r"""Baselines for steganography comparison.

This package provides from-scratch implementations of the three
classical content-adaptive steganography baselines used in the paper:

* S-UNIWARD (Holub, Fridrich & Denemark, 2014)
* WOW (Holub & Fridrich, 2012)
* HUGO (Pevny, Filler & Bas, 2010)

Each baseline computes a per-pixel distortion cost using the original
paper's formulation. The cost is then fed to the same STC encoder as
Semantic Cloak, ensuring a fair comparison (only the cost function
differs, not the coder).

For the DDSP baseline, we provide a thin wrapper around the original
authors' released checkpoint, since re-implementing a deep generative
model from scratch is out of scope.
"""

from .s_uniward import s_uniward_cost
from .wow import wow_cost
from .hugo import hugo_cost

__all__ = ["s_uniward_cost", "wow_cost", "hugo_cost"]
