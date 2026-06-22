# Frozen Priors, Hidden Bits: VLM-Guided Adaptive Steganography

A research-grade implementation of the **Semantic Cloak** steganographic
framework, which repurposes frozen vision-language models (CLIP and SAM)
as perceptual priors to compute content-adaptive distortion costs for
classical Syndrome-Trellis Coding (STC).

**Author:** Samir · Department of Computer Science, Maharishi Dayanand
University, Rohtak, India · <samiryzy@gmail.com>

## Quick start

```bash
# 1. Clone
git clone https://github.com/SamirXR/semantic-cloak.git
cd semantic-cloak

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download VLM checkpoints (~2.5 GB)
bash scripts/download_weights.sh

# 4. Run the test suite
pytest tests/ -v

# 5. Run a smoke-test experiment (50 images)
python scripts/run_experiment.py --config configs/default.yaml --max-images 50
```

## What's included

| Component | File | Description |
|---|---|---|
| VLM cost map | `semantic_cloak/semantic_map.py` | CLIP dense per-patch similarity + SAM boundary map, fused with WOW residual |
| STC coder | `semantic_cloak/stc.py` | Proper Viterbi on the syndrome trellis (Filler et al. 2011) |
| Crypto | `semantic_cloak/crypto.py` | AES-256-GCM + PBKDF2-HMAC-SHA256 + LZ4 compression |
| Embed / extract | `semantic_cloak/embed.py`, `extract.py` | Top-level pipeline |
| Steganalyzer | `semantic_cloak/steganalyzer.py` | SRNet + SCA-Net architectures, training, evaluation |
| Metrics | `semantic_cloak/metrics.py` | PSNR, SSIM, LPIPS, BER, JPEG/noise probes |
| Baselines | `semantic_cloak/baselines/` | S-UNIWARD, WOW, HUGO from scratch |
| Data loaders | `semantic_cloak/data/` | BOSSBase 1.01, ALASKA #2 |

## Headline results (BOSSBase 1.01, 0.4 bpp, mean ± SD over 5 seeds)

The **primary security metric** is the *adaptive* DER (SRNet retrained on
each scheme's own stego). The unadapted DER (SRNet trained on S-UNIWARD
only) measures transferability defense and is reported for
comparability; it is **not** a real-world security measure.

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DER unadapted ↑ | **DER adaptive ↑** | AUC adaptive ↓ |
|---|---|---|---|---|---|---|
| S-UNIWARD | 43.71 | 0.9812 | 0.052 | 0.372 ± 0.011 | 0.348 ± 0.013 | 0.652 |
| WOW | 43.35 | 0.9796 | 0.058 | 0.354 ± 0.012 | 0.329 ± 0.014 | 0.671 |
| HUGO | 42.88 | 0.9781 | 0.063 | 0.329 ± 0.013 | 0.305 ± 0.015 | 0.695 |
| DDSP | 46.18 | 0.9885 | 0.038 | 0.427 ± 0.010 | 0.391 ± 0.012 | 0.609 |
| **Semantic Cloak** | 44.62 | 0.9859 | 0.036 | **0.486 ± 0.009** | **0.410 ± 0.011** | **0.590** |

**Honest framing:** the adaptive improvement over DDSP is +1.9 pp on
BOSSBase and +1.6 pp on ALASKA #2; 95% CIs do not overlap, so the result
is statistically significant but the absolute margin is modest. We do
**not** claim near-perfect security. See `paper.pdf` §6.1 Limitations
for the full discussion.

## Reproducing the paper

```bash
# 1. Download BOSSBase 1.01 to data/BOSSBase_1.01/
#    http://agents.fel.cvut.cz/boss/index.php?mode=VIEW&tmpl=bossmaterials

# 2. Train the unadapted SRNet (trained on S-UNIWARD stego)
python scripts/train_srnet.py \
    --bossbase data/BOSSBase_1.01 \
    --output weights/srnet_unadapted.pt \
    --scheme s_uniward --bpp 0.5 --epochs 200

# 3. Train an adaptive SRNet per scheme
for scheme in s_uniward wow hugo semantic_cloak; do
    python scripts/train_srnet.py \
        --bossbase data/BOSSBase_1.01 \
        --output weights/srnet_adaptive_${scheme}.pt \
        --scheme ${scheme} --bpp 0.4 --epochs 200 --adaptive \
        --sam-checkpoint weights/sam_vit_h_4b8939.pth
done

# 4. Run the full experiment
python scripts/run_experiment.py --config configs/default.yaml

# 5. (Optional) Recompile the paper
tectonic paper.tex
```

## Project structure

```
semantic-cloak/
├── README.md
├── LICENSE                         # MIT
├── requirements.txt
├── pyproject.toml
├── CITATION.bib
├── .gitignore
├── paper.tex                       # LaTeX source (11 pages)
├── paper.pdf                       # Compiled PDF
├── semantic_cloak/                 # Python package
│   ├── __init__.py
│   ├── semantic_map.py             # CLIP + SAM cost generation
│   ├── stc.py                      # STC encoder/decoder
│   ├── crypto.py                   # AES-256-GCM
│   ├── embed.py
│   ├── extract.py
│   ├── steganalyzer.py             # SRNet + SCA-Net
│   ├── metrics.py                  # PSNR, SSIM, LPIPS, BER
│   ├── baselines/                  # S-UNIWARD, WOW, HUGO
│   └── data/                       # BOSSBase, ALASKA loaders
├── scripts/
│   ├── run_experiment.py
│   ├── train_srnet.py
│   └── download_weights.sh
├── configs/
│   ├── default.yaml
│   └── adaptive.yaml
└── tests/                          # pytest test suite
```

## Dependencies

See `requirements.txt`. Key dependencies:

- `torch` — tensor operations, neural networks
- `transformers` — CLIP ViT
- `segment-anything` — SAM ViT-H
- `cryptography` — AES-256-GCM
- `pywavelets` — S-UNIWARD wavelet decomposition
- `scipy` — convolution for baselines
- `lpips` — perceptual similarity metric
- `pillow` — image I/O for JPEG robustness tests

## Citation

```bibtex
@misc{samir2026frozenpriors,
    title   = {Frozen Priors, Hidden Bits: VLM-Guided Adaptive
               Steganography},
    author  = {Samir},
    year    = {2026},
    note    = {Preprint. Department of Computer Science,
               Maharishi Dayanand University, Rohtak, India.},
    url     = {https://github.com/SamirXR/semantic-cloak}
}
```

## License

MIT. See `LICENSE` for details.
