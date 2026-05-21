"""scFoundation adapter — STUB.

scFoundation (Hao et al., Nature Methods 2024) is a 100M-param model pretrained
on ~50M cells. Uses xTrimoGene tokenization (binned expression + Performer
attention). Architecture is similar to scGPT in spirit but bigger.

Checkpoint:
  - Code: github.com/biomap-research/scFoundation
  - Weights: shared via biomap (need to register / download from their links)
  - Some HuggingFace mirrors exist but vary in completeness; check the repo's
    README for the canonical link at the time of running.

What to fill in (~80 lines if model loads cleanly):
  1. load(): import the model class from biomap's scFoundation repo (need to
     `pip install -e` their code), load weights, set n_layers / d_model.
  2. preprocess(): binned expression encoding (likely 11 bins via their
     tokenizer); cache input_ids / values for forward.
  3. iter_layer_activations(): hook each Performer/Mamba block + pre-encoder.
     scFoundation's blocks are under model.encoder.layers (verify in their
     code); CLS-equivalent token position is 0 if they prepend.

Open questions to verify before wiring:
  - Does scFoundation use a discrete CLS token at position 0, or pool the
    output differently? If no CLS, set cls_position = None and CLS metrics
    will be skipped automatically.
  - Performer attention may need export of the layer outputs explicitly (not
    all custom layers play nicely with register_forward_hook).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np
import torch

from ..core.adapter import BioFMAdapter


class ScFoundationAdapter(BioFMAdapter):
    name = "scfoundation"
    cls_position = 0   # TODO verify in scFoundation code

    def __init__(self):
        self.model = None
        self.device = None
        # TODO add caches for tokenized input

    def load(self, model_dir: str, device: str = "cuda") -> None:
        raise NotImplementedError(
            "scFoundation adapter is a stub. To complete:\n"
            "  1. pip install -e git+https://github.com/biomap-research/scFoundation\n"
            "  2. Import their model class (likely from scFoundation.model import ...)\n"
            "  3. Load checkpoint via their utility (custom format)\n"
            "  4. Set self.n_layers and self.d_model from the model config\n"
            "  5. Disable any fast-path that breaks hooks (FlashAttention etc.)\n"
            "See bio_fm_probe/adapters/_template.py for the contract."
        )

    def preprocess(self, adata):
        raise NotImplementedError(
            "scFoundation tokenization: binned expression with their own bin "
            "boundaries; check scFoundation/utils.py for the canonical fn."
        )

    def iter_layer_activations(self, adata, batch_size, device):
        raise NotImplementedError("see load() docstring")
