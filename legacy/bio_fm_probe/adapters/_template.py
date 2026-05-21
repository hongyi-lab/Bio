"""Template adapter for a new bio foundation model.

How to add a new model (~50 lines if the model is well-behaved):

  1. Copy this file to bio_fm_probe/adapters/<your_model>.py
  2. Rename TemplateAdapter -> YourModelAdapter, set `name`.
  3. Fill in the three abstractmethods. Common pitfalls noted inline.
  4. Register in bio_fm_probe/run_audit.py:
         ADAPTER_REGISTRY["<your_model>"] = "bio_fm_probe.adapters.<your_model>:YourModelAdapter"
  5. Run:
         python -m bio_fm_probe.run_audit --adapter <your_model> --model_dir ... --data ...

You should NOT need to touch anything in core/. If you do, that probably means
the adapter interface needs to grow — flag it and we will extend the ABC.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np
import torch
from tqdm import tqdm

from ..core.adapter import BioFMAdapter


class TemplateAdapter(BioFMAdapter):
    name = "template"
    cls_position = 0   # set to None if your model has no CLS

    def __init__(self):
        self.model = None
        # store anything else preprocess() needs to keep around for iter_layer_activations()
        # (vocabularies, gene-id lookup tables, tokenizers, etc.)

    # -------------------------------------------------------------------------
    def load(self, model_dir: str, device: str = "cuda") -> None:
        """Build model, load weights, populate self.n_layers and self.d_model.

        Tips:
          - Set model to eval() and move to device.
          - If the model has a "fast path" (nested tensors, flash-attn fused
            kernels) that breaks forward hooks, disable it here.
        """
        # model = YourModel(...)
        # model.load_state_dict(torch.load(Path(model_dir) / "weights.pt",
        #                                   map_location=device))
        # model.to(device).eval()
        # self.model = model
        # self.n_layers = len(model.encoder.layers)
        # self.d_model  = model.d_model
        raise NotImplementedError("fill in load()")

    # -------------------------------------------------------------------------
    def preprocess(self, adata):
        """Return adata with any model-specific inputs prepared.

        Examples of what to do here:
          - Filter genes to the model's vocab (drop unknown symbols).
          - Normalize / log1p / bin / rank-encode expression to whatever the
            model was pretrained on.
          - Cache gene-id arrays on self for use during forward.

        Do NOT subset cells silently — caller controls that.
        """
        # adata = adata[:, adata.var.gene_name.isin(self.vocab)].copy()
        # adata.layers["X_input"] = some_normalization(adata.X)
        # self._gene_ids = ...
        # return adata
        raise NotImplementedError("fill in preprocess()")

    # -------------------------------------------------------------------------
    def iter_layer_activations(
        self, adata, batch_size: int, device: str = "cuda",
    ) -> Iterator[Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
        """Forward all cells in batches, yielding for each batch:
          captured: {layer_name: (B, seq, d) tensor} including "layer_00_input"
          valid:    (B, seq) bool, True where token is a real gene (not pad, not CLS)

        Pattern:

          captured = {}
          handles = []

          def pre_hook(_m, inputs):
              captured["layer_00_input"] = inputs[0].detach()
          handles.append(self.model.encoder.register_forward_pre_hook(pre_hook))

          for i, block in enumerate(self.model.encoder.layers):
              def make_hook(i=i):
                  name = f"layer_{i+1:02d}"
                  def h(_m, _i, out):
                      captured[name] = out.detach()
                  return h
              handles.append(block.register_forward_hook(make_hook()))

          try:
              for start in tqdm(range(0, adata.n_obs, batch_size), ...):
                  end = min(start + batch_size, adata.n_obs)
                  ... build batch_inputs ...
                  with torch.no_grad():
                      self.model.forward(batch_inputs)
                  valid = build_valid_mask(batch_inputs)  # bool (B, seq)
                  yield dict(captured), valid
                  captured.clear()
          finally:
              for h in handles:
                  h.remove()

        Caveats:
          - DO NOT yield the captured dict directly without making a copy if you
            also clear it after yield. `dict(captured)` is enough — values are
            tensor references, not deep copies, which is what we want.
          - If your encoder has a fast path that wraps intermediate activations
            in NestedTensor or similar non-indexable structures, disable it in
            load().
          - valid mask MUST exclude both pad positions and any CLS-like sentinel
            tokens. Mean-pool correctness depends on this.
        """
        raise NotImplementedError("fill in iter_layer_activations()")
