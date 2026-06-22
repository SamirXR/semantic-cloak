# Semantic Cloak — Code Package

VLM-guided steganography with STC embedding.

## Files

| File             | Purpose                                                       |
|------------------|---------------------------------------------------------------|
| `semantic_map.py`| CLIP + SAM fusion into a per-pixel cost matrix ρ              |
| `embed.py`       | AES-256-GCM payload encryption + STC LSB embedding            |
| `extract.py`     | Inverse of `embed.py` — recovers message from stego image     |
| `steganalyzer.py`| SRNet / SCA-Net wrappers + DER/AUC evaluation + robustness    |
| `__init__.py`    | Public API                                                    |

## Quick start

```bash
pip install torch numpy cryptography pillow
# Optional, for VLM priors:
pip install transformers segment-anything
```

```python
import torch, numpy as np
from semantic_map import SemanticCostMap, SemanticMapConfig
from embed import embed, EmbedConfig
from extract import extract

# 1. Generate VLM-guided cost matrix
cover = torch.from_numpy(np.random.randint(0, 256, (3, 256, 256), dtype=np.uint8))
cost  = SemanticCostMap(SemanticMapConfig())(cover)

# 2. Embed
res = embed(cover, b"hello world", passphrase="pw", cost_matrix=cost)

# 3. Extract
out = extract(res.stego_image, passphrase="pw", image_size_hw=(256, 256))
assert out.verified and out.message == b"hello world"
```

## Reproducibility

All stochastic ops use deterministic seeds (CLIP submatrix seed = `0xC0FFEE`).
The STC parity-check submatrix is shared between embedder and extractor by
construction, so any STC implementation that follows the same schedule will
produce bit-exact round-trips.

## License

MIT.
