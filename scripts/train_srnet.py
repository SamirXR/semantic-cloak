#!/usr/bin/env python3
"""train_srnet.py
================
Train an SRNet (or SCA-Net) steganalyzer.

Two modes are supported:

1. **Unadapted** (default): train on S-UNIWARD stego at 0.5 bpp,
   evaluated on all schemes' stego. Measures transferability defense.

2. **Adaptive** (``--adaptive``): train on the specified scheme's own
   stego at the target bpp. Measures security against an informed
   adversary.

Usage
-----
    # Unadapted SRNet (trained on S-UNIWARD)
    python scripts/train_srnet.py \\
        --bossbase data/BOSSBase_1.01 \\
        --output weights/srnet_unadapted.pt \\
        --scheme s_uniward --bpp 0.5 --epochs 200

    # Adaptive SRNet (trained on Semantic Cloak)
    python scripts/train_srnet.py \\
        --bossbase data/BOSSBase_1.01 \\
        --output weights/srnet_adaptive_semantic_cloak.pt \\
        --scheme semantic_cloak --bpp 0.4 --epochs 200 \\
        --adaptive \\
        --sam-checkpoint weights/sam_vit_h_4b8939.pth

Author : Samir (Maharishi Dayanand University)
License: MIT
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from semantic_cloak import (
    SemanticCostMap, SemanticMapConfig,
    embed, EmbedConfig,
    SteganalyzerConfig, train_srnet,
    s_uniward_cost, wow_cost, hugo_cost,
    BOSSBase, CoverStegoPair,
)


# ----------------------------------------------------------------------
# Dataset that generates stego on-the-fly
# ----------------------------------------------------------------------
class StegoDataset(Dataset):
    """Generates cover/stego pairs on-the-fly for steganalyzer training.

    For each cover, computes the cost via the specified scheme and
    embeds a random payload, producing a stego. Returns ``(image,
    label)`` where label is 0 for cover and 1 for stego.
    """

    def __init__(
        self,
        covers: BOSSBase,
        scheme: str,
        bpp: float,
        semantic_map: "SemanticCostMap | None" = None,
        passphrase: str = "train",
        seed: int = 0,
    ) -> None:
        self.covers = covers
        self.scheme = scheme
        self.bpp = bpp
        self.semantic_map = semantic_map
        self.passphrase = passphrase
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.covers)

    def __getitem__(self, idx: int):
        cover = self.covers[idx]
        # Random label: 0 = cover, 1 = stego
        label = int(self.rng.integers(0, 2))
        if label == 0:
            return cover, 0

        # Generate stego
        cost = self._compute_cost(cover)
        n_pixels = cover.shape[-1] * cover.shape[-2]
        target_bits = int(round(self.bpp * n_pixels))
        msg_len = max(8, (target_bits - 240) // 8 - 8)
        message = bytes(self.rng.integers(0, 256, msg_len).tolist())
        emb_cfg = EmbedConfig(target_bpp=self.bpp)
        result = embed(
            cover_image=cover,
            message=message,
            passphrase=self.passphrase,
            cost_matrix=cost,
            cfg=emb_cfg,
        )
        return result.stego_image, 1

    def _compute_cost(self, cover: torch.Tensor) -> torch.Tensor:
        if self.scheme == "semantic_cloak":
            if self.semantic_map is None:
                raise ValueError("Semantic Cloak requires a SemanticCostMap.")
            return self.semantic_map(cover)
        elif self.scheme == "s_uniward":
            return s_uniward_cost(cover)
        elif self.scheme == "wow":
            return wow_cost(cover)
        elif self.scheme == "hugo":
            return hugo_cost(cover)
        else:
            raise ValueError(f"Unknown scheme: {self.scheme}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train an SRNet steganalyzer.")
    parser.add_argument("--bossbase", type=str, required=True,
                        help="Path to BOSSBase 1.01 directory.")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save the trained checkpoint.")
    parser.add_argument("--scheme", type=str, default="s_uniward",
                        choices=["s_uniward", "wow", "hugo", "semantic_cloak"],
                        help="Steganographic scheme to train against.")
    parser.add_argument("--bpp", type=float, default=0.5,
                        help="Embedding rate (bits per pixel).")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model", type=str, default="srnet",
                        choices=["srnet", "sca-net"],
                        help="Steganalyzer architecture.")
    parser.add_argument("--adaptive", action="store_true",
                        help="Adaptive mode: train on the scheme's own stego "
                             "(as opposed to S-UNIWARD for unadapted training).")
    parser.add_argument("--sam-checkpoint", type=str, default=None,
                        help="Path to SAM ViT-H checkpoint (required for "
                             "semantic_cloak scheme).")
    parser.add_argument("--clip-model", type=str,
                        default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device (default: cuda if available).")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Build the Semantic Cloak cost generator if needed
    semantic_map = None
    if args.scheme == "semantic_cloak":
        if args.sam_checkpoint is None:
            parser.error("--sam-checkpoint is required when --scheme semantic_cloak")
        sm_cfg = SemanticMapConfig(
            clip_model_name=args.clip_model,
            sam_checkpoint=args.sam_checkpoint,
            device=device,
        )
        semantic_map = SemanticCostMap(sm_cfg)
        print(f"[setup] Semantic Cloak cost generator ready on {device}")

    # Load BOSSBase
    print(f"[setup] Loading BOSSBase from {args.bossbase}")
    train_covers = BOSSBase(root=args.bossbase, split="train", resize=(256, 256))
    val_covers = BOSSBase(root=args.bossbase, split="val", resize=(256, 256))
    print(f"[setup] Train: {len(train_covers)} images, Val: {len(val_covers)} images")

    # Build stego datasets
    train_scheme = args.scheme if args.adaptive else "s_uniward"
    train_ds = StegoDataset(
        covers=train_covers,
        scheme=train_scheme,
        bpp=args.bpp,
        semantic_map=semantic_map,
        seed=0,
    )
    val_ds = StegoDataset(
        covers=val_covers,
        scheme=train_scheme,
        bpp=args.bpp,
        semantic_map=semantic_map,
        seed=1,
    )

    # Build the steganalyzer config
    sa_cfg = SteganalyzerConfig(
        model=args.model,
        weights_path=None,  # training from scratch
        image_size=256,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        device=device,
    )

    # Train
    print(f"[train] Training {args.model} for {args.epochs} epochs "
          f"on {train_scheme} stego at {args.bpp} bpp...")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    train_srnet(
        cfg=sa_cfg,
        train_dataset=train_ds,
        val_dataset=val_ds,
        save_path=args.output,
    )
    print(f"[done] Checkpoint saved to {args.output}")


if __name__ == "__main__":
    main()
