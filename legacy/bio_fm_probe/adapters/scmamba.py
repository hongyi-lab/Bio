"""scMamba adapter — PLACEHOLDER.

scMamba (Tang et al., 2024-25, preprint) uses Mamba state-space blocks to
handle long single-cell sequences. RNA-focused but designed for multi-modal
extension.

Status: checkpoint public-availability **unverified** as of writing. Most
Mamba-based single-cell papers from 2024-25 have not released pretrained
weights, only code.

Action items before this adapter can be filled in:
  1. Check the scMamba GitHub for a checkpoint release.
  2. If only code is released, you'll need to either pretrain yourself
     (expensive) or contact the authors.
  3. If a checkpoint exists, follow _template.py to wire it through. The
     Mamba block API differs from Transformer — use the SSM block's forward
     hook (state_space.MambaBlock typically exposes its output cleanly).

Compatibility considerations:
  - Mamba's selective scan kernel may not expose intermediate hidden states
    via standard hooks. May need to monkey-patch the block's forward or
    use an intermediate-output return path.
  - Mamba's "valid mask" semantics differ: it processes the whole sequence
    sequentially; pad positions should be masked at the output, not at the
    input (they still get processed but their output is irrelevant).
"""
from __future__ import annotations

from ..core.adapter import BioFMAdapter


class ScMambaAdapter(BioFMAdapter):
    name = "scmamba"

    def load(self, model_dir, device="cuda"):
        raise NotImplementedError(
            "scMamba checkpoint availability unverified. "
            "Verify https://github.com/... has a public weight release "
            "before wiring up this adapter."
        )

    def preprocess(self, adata):
        raise NotImplementedError("see load() docstring")

    def iter_layer_activations(self, adata, batch_size, device):
        raise NotImplementedError("see load() docstring")
