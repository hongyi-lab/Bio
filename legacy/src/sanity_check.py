"""End-to-end sanity check: load data → load model → embed 50 cells → UMAP.

Saves:
    results/sanity_embeddings.npz   (resumable checkpoint of cell embeddings)
    results/sanity_check_umap.png   (UMAP of cell embeddings, colored by cell type)
    results/sanity_check_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import torch

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(THIS.parent))

from load_scgpt import (  # noqa: E402
    embed_cells,
    load_scgpt_model,
    preprocess_adata_for_scgpt,
)

MODEL_DIR = ROOT / "checkpoints" / "scGPT_human"
DATA_PATH = ROOT / "data" / "pbmc3k.h5ad"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

N_CELLS = 50


def pick_celltype_col(adata) -> str:
    for c in ("louvain", "leiden", "cell_type", "celltype", "CellType"):
        if c in adata.obs.columns:
            return c
    return adata.obs.columns[0]


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[sanity] device={device}")
    print(f"[sanity] torch.cuda.is_available()={torch.cuda.is_available()}")

    adata = sc.read_h5ad(DATA_PATH)
    print(f"[sanity] full adata: {adata.shape}")

    # pbmc3k_processed stores the scaled matrix in .X (has negatives), but the
    # log1p-normalized matrix in .raw — that's what scGPT's binning step needs.
    if adata.raw is not None:
        print(f"[sanity] swapping .X for .raw (shape {adata.raw.X.shape}, non-negative)")
        import anndata as ad
        adata = ad.AnnData(
            X=adata.raw.X,
            obs=adata.obs.copy(),
            var=adata.raw.var.copy(),
            obsm=dict(adata.obsm),
        )

    rng = np.random.default_rng(0)
    idx = rng.choice(adata.n_obs, size=min(N_CELLS, adata.n_obs), replace=False)
    adata = adata[np.sort(idx)].copy()
    color_col = pick_celltype_col(adata)
    print(f"[sanity] sampled {adata.n_obs} cells, color column: {color_col!r}")

    model, vocab, model_args = load_scgpt_model(MODEL_DIR, device=device)
    adata = preprocess_adata_for_scgpt(adata, vocab)

    save_path = RESULTS / "sanity_embeddings.npz"
    out = embed_cells(
        adata, model, vocab,
        device=device, batch_size=16, save_path=save_path,
    )
    cell_emb = out["cell_embeddings"]
    per_gene = out["per_gene_embeddings_last_batch"]
    print(f"[sanity] cell_embeddings:           {cell_emb.shape}")
    print(f"[sanity] per_gene (last batch):     {per_gene.shape}")

    adata.obsm["X_scgpt"] = cell_emb
    sc.pp.neighbors(adata, use_rep="X_scgpt", n_neighbors=min(15, adata.n_obs - 1))
    sc.tl.umap(adata)

    fig, ax = plt.subplots(figsize=(6, 5))
    sc.pl.umap(adata, color=color_col, ax=ax, show=False, frameon=False, legend_loc="right margin")
    fig.suptitle(f"scGPT cell embeddings — {adata.n_obs} cells", y=1.02)
    out_png = RESULTS / "sanity_check_umap.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[sanity] saved {out_png}")

    summary = {
        "n_cells": int(adata.n_obs),
        "n_genes_in_vocab": int(adata.n_vars),
        "cell_emb_dim": int(cell_emb.shape[1]),
        "per_gene_last_batch_shape": list(per_gene.shape) if per_gene is not None else None,
        "color_col": color_col,
        "device": device,
        "model_embsize": model_args.get("embsize"),
        "model_nlayers": model_args.get("nlayers"),
    }
    (RESULTS / "sanity_check_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
