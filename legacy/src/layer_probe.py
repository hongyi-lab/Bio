"""Layer-wise linear probe: cell-type accuracy at each layer of scGPT.

For each layer L in [embedding, block_1, ..., block_N], we:
  1. Extract the CLS-token representation of every cell at layer L.
  2. Train a multinomial logistic regression to predict cell type.
  3. Record test accuracy + macro-F1 on a held-out split.

Output:
  results/layer_embeddings.npz       per-layer CLS embeddings (resumable cache)
  results/layer_probe_results.json   per-layer accuracy + macro-F1
  results/layer_probe_curve.png      accuracy vs. layer index plot

Usage:
  python src/layer_probe.py                                  # default: pbmc3k
  python src/layer_probe.py --data data/your.h5ad --label_col cell_type
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(THIS.parent))

from load_scgpt import (  # noqa: E402
    MAX_LEN,
    N_BINS,
    PAD_TOKEN,
    PAD_VALUE,
    load_scgpt_model,
    preprocess_adata_for_scgpt,
)


def _attach_layer_hooks(model) -> Tuple[Dict[str, torch.Tensor], list, int]:
    """Hook every transformer block + the pre-encoder input. Returns (captured, handles, n_layers)."""
    captured: Dict[str, torch.Tensor] = {}
    handles = []
    encoder = model.transformer_encoder

    # Nested-tensor mode (PyTorch fast-path) turns intermediate activations into
    # NestedTensors that don't support .__getitem__ slicing. Disable both the public
    # flag *and* the internal gate used in forward().
    for attr in ("enable_nested_tensor", "use_nested_tensor"):
        if hasattr(encoder, attr):
            setattr(encoder, attr, False)

    n_layers = len(encoder.layers)

    def pre_hook(_module, inputs):
        captured["layer_00_input"] = inputs[0].detach()

    handles.append(encoder.register_forward_pre_hook(pre_hook))

    def make_hook(idx: int):
        name = f"layer_{idx + 1:02d}"

        def hook(_module, _inputs, output):
            captured[name] = output.detach()

        return hook

    for i, layer in enumerate(encoder.layers):
        handles.append(layer.register_forward_hook(make_hook(i)))

    return captured, handles, n_layers


def extract_per_layer_cls(
    adata,
    model,
    vocab,
    device: str,
    batch_size: int = 16,
    gene_col: str = "gene_name",
) -> Dict[str, np.ndarray]:
    """Forward all cells; collect the CLS-token activation at every layer.

    Returns dict[layer_name] -> array shape (n_cells, d_model).
    """
    from scgpt.tokenizer import tokenize_and_pad_batch

    if gene_col not in adata.var.columns:
        adata.var["gene_name"] = adata.var.index.astype(str)
        gene_col = "gene_name"

    captured, handles, n_layers = _attach_layer_hooks(model)
    print(f"[probe] hooked {n_layers} transformer blocks + input embedding")

    genes = adata.var[gene_col].tolist()
    gene_ids_all = np.array(vocab(genes), dtype=int)

    binned = adata.layers["X_binned"]
    if hasattr(binned, "toarray"):
        binned = binned.toarray()
    binned = np.asarray(binned)
    pad_idx = vocab[PAD_TOKEN]

    per_layer: Dict[str, List[np.ndarray]] = {}

    try:
        for start in tqdm(range(0, adata.n_obs, batch_size), desc="forward", unit="batch"):
            end = min(start + batch_size, adata.n_obs)
            batch_X = binned[start:end].astype(np.int64)

            tokenized = tokenize_and_pad_batch(
                batch_X,
                gene_ids_all,
                max_len=MAX_LEN,
                vocab=vocab,
                pad_token=PAD_TOKEN,
                pad_value=PAD_VALUE,
                append_cls=True,
                include_zero_gene=False,
            )
            input_gene_ids = tokenized["genes"].to(device)
            input_values = tokenized["values"].to(device).float()
            pad_mask = input_gene_ids.eq(pad_idx)

            with torch.no_grad():
                model._encode(input_gene_ids, input_values, src_key_padding_mask=pad_mask)

            for name, tensor in captured.items():
                per_layer.setdefault(name, []).append(tensor[:, 0, :].cpu().numpy())
            captured.clear()
    finally:
        for h in handles:
            h.remove()

    return {k: np.concatenate(v, axis=0) for k, v in sorted(per_layer.items())}


def _save_layer_embeddings(path: Path, embeddings: Dict[str, np.ndarray], y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_base = str(path.with_suffix(""))
    np.savez(tmp_base + ".tmp", _labels=y, **embeddings)
    Path(tmp_base + ".tmp.npz").replace(path)


def _load_layer_embeddings(path: Path):
    data = np.load(path, allow_pickle=False)
    y = data["_labels"]
    embeddings = {k: data[k] for k in data.files if k != "_labels"}
    return embeddings, y


def linear_probe(X: np.ndarray, y: np.ndarray, seed: int = 0) -> Dict[str, float]:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    clf = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        n_jobs=-1,
    )
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    return {
        "accuracy": float(accuracy_score(yte, pred)),
        "macro_f1": float(f1_score(yte, pred, average="macro")),
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
    }


def _load_adata_for_probe(data_path: Path, label_col: str):
    adata = sc.read_h5ad(data_path)
    print(f"[probe] loaded {data_path}: shape={adata.shape}")

    # pbmc3k_processed convention: scaled matrix in .X (has negatives), log-norm in .raw
    if adata.raw is not None and (adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()).min() < 0:
        print(f"[probe] .X has negatives; swapping in .raw ({adata.raw.X.shape})")
        import anndata as ad
        adata = ad.AnnData(
            X=adata.raw.X,
            obs=adata.obs.copy(),
            var=adata.raw.var.copy(),
            obsm=dict(adata.obsm),
        )

    if label_col not in adata.obs.columns:
        candidates = [c for c in ("louvain", "leiden", "cell_type", "celltype", "CellType")
                      if c in adata.obs.columns]
        if not candidates:
            raise SystemExit(f"no label column in obs; have {list(adata.obs.columns)}")
        label_col = candidates[0]
    print(f"[probe] using label column {label_col!r} with {adata.obs[label_col].nunique()} classes")
    return adata, label_col


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/pbmc3k.h5ad")
    p.add_argument("--label_col", default="louvain")
    p.add_argument("--model_dir", default="checkpoints/scGPT_human")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_cells", type=int, default=20000,
                   help="cap on cells (subsample if larger)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force_extract", action="store_true",
                   help="ignore cached embeddings and re-extract")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    data_path = ROOT / args.data if not Path(args.data).is_absolute() else Path(args.data)
    model_dir = ROOT / args.model_dir if not Path(args.model_dir).is_absolute() else Path(args.model_dir)
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    emb_path = results_dir / "layer_embeddings.npz"
    json_path = results_dir / "layer_probe_results.json"
    png_path = results_dir / "layer_probe_curve.png"

    adata, label_col = _load_adata_for_probe(data_path, args.label_col)

    if adata.n_obs > args.max_cells:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(adata.n_obs, size=args.max_cells, replace=False)
        adata = adata[np.sort(idx)].copy()
        print(f"[probe] subsampled to {adata.n_obs} cells")

    y_str = adata.obs[label_col].astype(str).values
    classes, y = np.unique(y_str, return_inverse=True)
    print(f"[probe] classes: {list(classes)}")
    print(f"[probe] class counts: {dict(zip(*np.unique(y_str, return_counts=True)))}")

    # 1) Per-layer embeddings (with caching)
    if emb_path.exists() and not args.force_extract:
        print(f"[probe] loading cached embeddings from {emb_path}")
        embeddings, y_cached = _load_layer_embeddings(emb_path)
        if y_cached.shape != y.shape or not np.array_equal(y_cached, y):
            print("[probe] cached labels differ from current selection — re-extracting")
            embeddings = None
        else:
            print(f"[probe] cache OK ({len(embeddings)} layers, shape {next(iter(embeddings.values())).shape})")
    else:
        embeddings = None

    if embeddings is None:
        print(f"[probe] loading scGPT from {model_dir}")
        model, vocab, model_args = load_scgpt_model(model_dir, device=args.device)
        adata = preprocess_adata_for_scgpt(adata, vocab)
        embeddings = extract_per_layer_cls(
            adata, model, vocab,
            device=args.device, batch_size=args.batch_size,
        )
        _save_layer_embeddings(emb_path, embeddings, y)
        print(f"[probe] cached embeddings to {emb_path}")
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 2) Per-layer linear probe
    layer_names = sorted(embeddings.keys())
    if json_path.exists() and not args.force_extract:
        existing = json.loads(json_path.read_text())
        results = {r["layer"]: r for r in existing.get("per_layer", [])}
    else:
        results = {}

    per_layer = []
    for name in tqdm(layer_names, desc="probe", unit="layer"):
        if name in results:
            r = results[name]
        else:
            X = embeddings[name]
            r = linear_probe(X, y, seed=args.seed)
            r["layer"] = name
            results[name] = r
            (json_path).write_text(json.dumps(
                {"label_col": label_col, "classes": classes.tolist(),
                 "n_cells": int(adata.n_obs),
                 "per_layer": [results[k] for k in layer_names if k in results]},
                indent=2))
        per_layer.append(r)
        print(f"  {name}: acc={r['accuracy']:.4f}  macro_f1={r['macro_f1']:.4f}")

    # 3) Plot
    xs = list(range(len(layer_names)))
    accs = [results[n]["accuracy"] for n in layer_names]
    f1s = [results[n]["macro_f1"] for n in layer_names]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, accs, marker="o", label="accuracy")
    ax.plot(xs, f1s, marker="s", linestyle="--", label="macro F1")
    ax.set_xticks(xs)
    ax.set_xticklabels(
        ["embed"] + [str(i + 1) for i in range(len(layer_names) - 1)],
        rotation=0,
    )
    ax.set_xlabel("layer (0 = input embedding, 1..N = transformer block output)")
    ax.set_ylabel("linear probe score (cell type)")
    ax.set_title(
        f"scGPT layer-wise cell-type probe — pbmc3k\n"
        f"{adata.n_obs} cells × {len(classes)} classes"
    )
    ax.grid(alpha=0.3)
    ax.legend()
    best_idx = int(np.argmax(accs))
    ax.axvline(best_idx, color="red", linestyle=":", alpha=0.5,
               label=f"peak: layer {best_idx} ({accs[best_idx]:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"[probe] wrote {png_path}")
    print(f"[probe] wrote {json_path}")

    print("\n=== SUMMARY ===")
    print(f"layers: {len(layer_names)}, cells: {adata.n_obs}, classes: {len(classes)}")
    print(f"best layer: {layer_names[best_idx]}  acc={accs[best_idx]:.4f}")
    print(f"embed (layer 0): acc={accs[0]:.4f}")
    print(f"final layer:     acc={accs[-1]:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
