"""SCARF adapter — PLACEHOLDER.

SCARF (RNA+ATAC multimodal foundation model with Mamba backbone, ~270M cells
according to the paper). The advisor-suggested target for cross-modal
alignment work; the prime question is whether a Mamba-based bio FM shows the
same low-rank collapse + PCA-50 inferiority pattern that scGPT does.

Status: checkpoint NOT publicly released as of writing. Authors must be
contacted for academic access.

Action items:
  1. Email the SCARF authors with research-use justification; ask for the
     pretrained RNA encoder weights (the multimodal RNA+ATAC model may have
     separate per-modality encoders, and only the RNA encoder is comparable
     to scGPT here).
  2. Once you have weights, wire up load(), preprocess(), and
     iter_layer_activations() following _template.py.
  3. The SCARF audit is the punchline of the cross-model trend table: if
     SCARF behaves like scGPT (low-rank collapse, doesn't beat PCA-50), the
     advisor's "switch to Mamba" hypothesis loses its main argument and the
     direction needs rethinking.

Until checkpoint is in hand, do NOT register this adapter in
run_audit.ADAPTER_REGISTRY.
"""
from __future__ import annotations

from ..core.adapter import BioFMAdapter


class SCARFAdapter(BioFMAdapter):
    name = "scarf"

    def load(self, model_dir, device="cuda"):
        raise NotImplementedError(
            "SCARF checkpoint not publicly released. Contact authors first."
        )

    def preprocess(self, adata):
        raise NotImplementedError("see load() docstring")

    def iter_layer_activations(self, adata, batch_size, device):
        raise NotImplementedError("see load() docstring")
