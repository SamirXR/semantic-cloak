"""Semantic Cloak: VLM-guided adaptive steganography.

Author: Samir (Maharishi Dayanand University, Rohtak, India)
Email:  samiryzy@gmail.com
License: MIT

Modules
-------
- ``semantic_map`` : CLIP/SAM integration producing the per-pixel cost.
- ``stc``          : Syndrome-Trellis Code encoder/decoder.
- ``crypto``       : AES-256-GCM authenticated encryption.
- ``embed``        : Top-level embedding pipeline.
- ``extract``      : Top-level extraction pipeline.
- ``steganalyzer`` : SRNet / SCA-Net architectures + training + eval.
- ``metrics``      : PSNR, SSIM, LPIPS, BER, JPEG/noise probes.
- ``baselines``    : S-UNIWARD, WOW, HUGO distortion costs.
- ``data``         : BOSSBase / ALASKA loaders.
"""

from .semantic_map import SemanticCostMap, SemanticMapConfig
from .stc import STCEncoder, STCDecoder, STCConfig
from .crypto import CryptoConfig, derive_key, encrypt_payload, decrypt_payload
from .embed import embed, EmbedConfig, EmbeddingResult
from .extract import extract, ExtractionResult
from .steganalyzer import (
    SteganalyzerConfig,
    SRNet,
    SCANet,
    build_steganalyzer,
    train_srnet,
    evaluate,
)
from .metrics import psnr, ssim, lpips, bit_error_rate, jpeg_compress, gaussian_noise
from .baselines import s_uniward_cost, wow_cost, hugo_cost
from .data import BOSSBase, ALASKA2, CoverStegoPair

__all__ = [
    # Semantic map
    "SemanticCostMap", "SemanticMapConfig",
    # STC
    "STCEncoder", "STCDecoder", "STCConfig",
    # Crypto
    "CryptoConfig", "derive_key", "encrypt_payload", "decrypt_payload",
    # Embed / extract
    "embed", "EmbedConfig", "EmbeddingResult",
    "extract", "ExtractionResult",
    # Steganalyzer
    "SteganalyzerConfig", "SRNet", "SCANet",
    "build_steganalyzer", "train_srnet", "evaluate",
    # Metrics
    "psnr", "ssim", "lpips", "bit_error_rate", "jpeg_compress", "gaussian_noise",
    # Baselines
    "s_uniward_cost", "wow_cost", "hugo_cost",
    # Data
    "BOSSBase", "ALASKA2", "CoverStegoPair",
]

__version__ = "0.2.0"
__author__ = "Samir <samiryzy@gmail.com>"
