"""UCE (Universal Cell Embeddings) adapter — STUB.

UCE (Rosen et al., 2024) is a 33-layer transformer that uses learned protein-
sequence-derived gene embeddings as inputs (instead of token IDs from a
discrete vocabulary). Pretrained on 36M cells across species.

Checkpoint:
  - Code: github.com/snap-stanford/UCE
  - Weights: their repo or huggingface mirror (model files several GB)

What to fill in:
  1. load(): UCE has a custom model class with frozen ESM protein embeddings
     as gene encoder inputs. The "vocab" is essentially ESM gene embeddings.
     Their repo provides eval_single_anndata.py — adapt that loader here.
  2. preprocess(): UCE tokenizes via gene -> ESM embedding lookup, then ranks
     by expression. Use their utility scripts for consistency.
  3. iter_layer_activations(): UCE's transformer is standard; hook the 33
     blocks under model.transformer_encoder.

Notes:
  - UCE works across species (mouse + human); to keep comparisons clean here,
    use human samples only.
  - 33 layers × 1280 d_model is bigger than scGPT — token-level cache for SAE
    will be ~8x larger per layer. Plan disk space accordingly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np
import torch

from ..core.adapter import BioFMAdapter


class UCEAdapter(BioFMAdapter):
    name = "uce"
    cls_position = 0   # UCE prepends a CLS-like token; verify

    def load(self, model_dir: str, device: str = "cuda") -> None:
        raise NotImplementedError(
            "UCE adapter is a stub. To complete:\n"
            "  1. Clone github.com/snap-stanford/UCE\n"
            "  2. Use their eval_single_anndata.py as reference\n"
            "  3. Their model = TransformerModel(... 33 layers ...)\n"
        )

    def preprocess(self, adata):
        raise NotImplementedError(
            "UCE preprocessing requires ESM-derived gene embeddings; see "
            "their data_proc utils."
        )

    def iter_layer_activations(self, adata, batch_size, device):
        raise NotImplementedError("see load() docstring")
