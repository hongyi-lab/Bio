"""Abstract base for a bio-foundation-model adapter.

A new adapter implements three things and gets all probes for free:

  1. load(model_dir, device)         -> set self.model, self.n_layers, self.d_model
  2. preprocess(adata)               -> adata (with any model-specific inputs)
  3. iter_layer_activations(adata, batch_size, device)
        yields (captured: {layer_name: (B, seq, d) tensor}, valid_mask: (B, seq) bool)
     for each forward batch. layer_name conventions are fixed:
       "layer_00_input" = pre-encoder activations (post gene+value embedding)
       "layer_{NN:02d}" = output of transformer/mamba/... block NN (1-indexed)
     valid_mask is True where the position is a real input token (excludes pad
     and CLS — that's what we want for mean-pool over genes).

Default poolings (pool_cls at position 0, pool_mean over valid) cover most
CLS-bearing transformer-encoder models. Override for models that put CLS at a
different position or don't have one.
"""
from __future__ import annotations

import abc
from typing import Dict, Iterator, List, Optional, Tuple

import torch


class BioFMAdapter(abc.ABC):
    name: str
    n_layers: int = 0   # populated by load()
    d_model: int = 0    # populated by load()
    cls_position: Optional[int] = 0  # override to None for models without CLS

    @abc.abstractmethod
    def load(self, model_dir: str, device: str = "cuda") -> None:
        """Build model, load weights, set n_layers and d_model."""

    @abc.abstractmethod
    def preprocess(self, adata):
        """Return adata with any model-specific inputs added in layers/var/obs.

        Should also stash anything the iter_layer_activations loop needs
        (gene-id lookup tables etc.) on self.
        """

    @abc.abstractmethod
    def iter_layer_activations(
        self, adata, batch_size: int, device: str = "cuda",
    ) -> Iterator[Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
        """Forward all cells in `adata` in batches of `batch_size`.

        For each batch yield (captured, valid_mask) where:
          captured: dict mapping layer name -> (B, seq, d) detached tensor,
                    one entry per layer including "layer_00_input".
          valid_mask: (B, seq) bool, True where the token is a real input gene
                      (excludes pad AND CLS).
        After each yield the adapter is free to overwrite/clear `captured`.
        """

    # ----- helpers, override only if your model is unusual --------------------
    @property
    def layer_names(self) -> List[str]:
        return ["layer_00_input"] + [f"layer_{i + 1:02d}" for i in range(self.n_layers)]

    def pool_cls(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """(B, seq, d) -> (B, d). Default: pick position cls_position."""
        if self.cls_position is None:
            raise NotImplementedError(
                f"{self.name} has no CLS; override pool_cls if you want CLS-pooling"
            )
        return x[:, self.cls_position, :]

    def pool_mean(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """(B, seq, d) -> (B, d). Mean over valid (non-pad, non-CLS) positions."""
        w = valid.float().unsqueeze(-1)              # (B, seq, 1)
        counts = w.sum(dim=1).clamp(min=1)           # (B, 1)
        return (x * w).sum(dim=1) / counts
