"""scBERT adapter — STUB.

scBERT (Yang et al., Nature Machine Intelligence 2022) is one of the earliest
single-cell transformer foundation models. Performer-based attention, smaller
than scGPT (~5M params), pretrained on PanglaoDB.

Checkpoint:
  - Code: github.com/TencentAILabHealthcare/scBERT
  - Weights: GitHub release (panglao_pretrain.pth)

What to fill in:
  1. load(): import the model class from scBERT's repo (typically performer-
     pytorch + a custom embedding head). Their preprint shipped a single
     ~250MB checkpoint.
  2. preprocess(): scBERT bins each gene's count value into 5 classes
     (0..4). Sequence length is the full gene panel (~16k), so attention
     uses Performer kernels rather than full attention.
  3. iter_layer_activations(): standard, hook each Performer block.

Notes:
  - scBERT has no <cls> token in early versions; pooling is mean over genes.
    If that's the case here, set cls_position = None.
  - Performer's randomly-initialized projection matrix has nondeterministic
    behavior across runs unless seeded; set torch.manual_seed before forward
    if reproducibility matters.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np
import torch

from ..core.adapter import BioFMAdapter


class ScBERTAdapter(BioFMAdapter):
    name = "scbert"
    cls_position = None   # scBERT (early version) has no CLS; verify

    def load(self, model_dir: str, device: str = "cuda") -> None:
        raise NotImplementedError(
            "scBERT adapter is a stub. To complete:\n"
            "  1. Clone github.com/TencentAILabHealthcare/scBERT\n"
            "  2. Import their PerformerLM_AE / PerformerLM class\n"
            "  3. Load panglao_pretrain.pth\n"
            "  4. Set n_layers, d_model from the loaded checkpoint\n"
        )

    def preprocess(self, adata):
        raise NotImplementedError(
            "scBERT bins counts into 5 classes. See their preprocess utils."
        )

    def iter_layer_activations(self, adata, batch_size, device):
        raise NotImplementedError("see load() docstring")
