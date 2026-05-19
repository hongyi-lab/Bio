"""Model-agnostic activation extraction. Reads from BioFMAdapter only."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from .adapter import BioFMAdapter


def extract_cls_and_mean_per_layer(
    adapter: BioFMAdapter, adata, batch_size: int, device: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Forward all cells. At every layer return both CLS-pool and mean-pool.

    Returns:
        cls:  dict[layer_name] -> (n_cells, d_model)
        mean: dict[layer_name] -> (n_cells, d_model)
    """
    cls_buf: Dict[str, List[np.ndarray]] = {}
    mean_buf: Dict[str, List[np.ndarray]] = {}

    for captured, valid in adapter.iter_layer_activations(adata, batch_size, device):
        for name, tensor in captured.items():
            cls_v = adapter.pool_cls(tensor, valid).cpu().numpy()
            mean_v = adapter.pool_mean(tensor, valid).cpu().numpy()
            cls_buf.setdefault(name, []).append(cls_v)
            mean_buf.setdefault(name, []).append(mean_v)

    return (
        {k: np.concatenate(v, axis=0) for k, v in sorted(cls_buf.items())},
        {k: np.concatenate(v, axis=0) for k, v in sorted(mean_buf.items())},
    )


def extract_tokens_one_layer(
    adapter: BioFMAdapter, adata, target_layer: str,
    batch_size: int, device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Forward all cells. At ONE specified layer collect every valid token's
    activation as a flat array.

    Returns:
        token_acts:     (N_tokens, d_model) float32
        token_cell_idx: (N_tokens,) int64 — which cell each token came from
    """
    if target_layer not in adapter.layer_names:
        raise ValueError(
            f"unknown layer {target_layer!r}; available: {adapter.layer_names}"
        )

    acts_chunks: List[np.ndarray] = []
    cell_chunks: List[np.ndarray] = []
    cells_processed = 0

    for captured, valid in adapter.iter_layer_activations(adata, batch_size, device):
        x = captured[target_layer]  # (B, seq, d)
        B = x.shape[0]
        cell_idx_full = (
            torch.arange(cells_processed, cells_processed + B, device=x.device)
            .unsqueeze(1).expand_as(valid)
        )
        acts_chunks.append(x[valid].cpu().numpy().astype(np.float32))
        cell_chunks.append(cell_idx_full[valid].cpu().numpy().astype(np.int64))
        cells_processed += B

    return (
        np.concatenate(acts_chunks, axis=0),
        np.concatenate(cell_chunks, axis=0),
    )
