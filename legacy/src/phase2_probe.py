"""phase2: rigorous controls for the per-layer scGPT probe.

What phase1 showed (single seed, CLS-only, no baseline):
  layer 0  acc=0.43  F1=0.08   <- LR collapsing to majority class
  layer 1  acc=0.89  F1=0.85   <- one block does most of the work
  layer 5  acc=0.92  F1=0.89   <- "peak" — only 1.3pp above layer 1
  layer 12 acc=0.88  F1=0.78   <- macro F1 drop at final layer

Phase1 left four hypotheses open:
  H1 (artifact): layer_00_input CLS is constant across cells (CLS slot before
                 attention has no cell-specific information). If so, the 0->1
                 jump is "empty slot getting filled", not "info entering model",
                 and layer 0 is not a valid baseline.
  H2 (noise):    layers 2-9 fall within 1.3pp on a single 528-cell test set.
                 The "layer 5 peak" may be seed-dependent.
  H3 (baseline): pbmc3k cell type may be 90%+ recoverable from any reasonable
                 representation (PCA, raw genes). The absolute 92% scGPT score
                 may say little about scGPT itself.
  H4 (pooling):  CLS-only probing may under-represent late layers if cell-type
                 information diffuses back onto gene tokens.

This script tests each in one pass:
  1. layer_00_input variance across cells, vs. a reference middle layer (H1).
  2. PCA-50, PCA-512, raw-log1p LR baselines × N seeds (H3).
  3. Per-layer CLS probe × N seeds -> mean +/- std bands (H2).
  4. Per-layer mean-pool (non-pad, non-CLS gene tokens) × N seeds (H4).

Outputs (under results/phase2/):
  layer_activations.npz       per-layer CLS+mean cache (resumable)
  X_log1p_for_baselines.npy   cached log1p matrix used by baselines
  layer0_sanity.json          variance stats: layer 0 vs reference middle layer
  baselines.json              PCA-50/512/raw LR × seeds
  per_layer_probe.json        per-layer × {cls,mean} × seeds
  layer_probe_curve_v2.png    accuracy and macro-F1 with mean+/-std bands +
                              horizontal baseline lines
  SUMMARY.md                  short markdown digest with key deltas

Usage:
  python src/phase2_probe.py                                  # pbmc3k, seeds=[0..4]
  python src/phase2_probe.py --force_extract                  # ignore cache, re-extract
  python src/phase2_probe.py --seeds 0 1 2 3 4 5 6 7 8 9      # custom seeds
  python src/phase2_probe.py --skip_baselines                 # H1+H2+H4 only
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
from sklearn.decomposition import PCA
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


# ---------------------------------------------------------------------------
# Activation extraction with full (B, seq, d) capture (so caller can pool both
# CLS and gene-token-mean on the same forward pass).
# ---------------------------------------------------------------------------
def _attach_layer_hooks_full(model):
    captured: Dict[str, torch.Tensor] = {}
    handles = []
    encoder = model.transformer_encoder
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


def extract_per_layer_cls_and_mean(
    adata,
    model,
    vocab,
    device: str,
    batch_size: int = 16,
    gene_col: str = "gene_name",
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Forward all cells; for each layer return CLS-position and
    mean-over-real-gene-tokens representations.

    Returns:
        cls_per_layer:  dict[layer] -> (n_cells, d)
        mean_per_layer: dict[layer] -> (n_cells, d)
    """
    from scgpt.tokenizer import tokenize_and_pad_batch

    if gene_col not in adata.var.columns:
        adata.var["gene_name"] = adata.var.index.astype(str)
        gene_col = "gene_name"

    captured, handles, n_layers = _attach_layer_hooks_full(model)
    print(f"[phase2] hooked {n_layers} blocks + pre-encoder input")

    genes = adata.var[gene_col].tolist()
    gene_ids_all = np.array(vocab(genes), dtype=int)
    binned = adata.layers["X_binned"]
    if hasattr(binned, "toarray"):
        binned = binned.toarray()
    binned = np.asarray(binned)
    pad_idx = vocab[PAD_TOKEN]

    cls_buf: Dict[str, List[np.ndarray]] = {}
    mean_buf: Dict[str, List[np.ndarray]] = {}

    try:
        for start in tqdm(range(0, adata.n_obs, batch_size), desc="forward", unit="batch"):
            end = min(start + batch_size, adata.n_obs)
            batch_X = binned[start:end].astype(np.int64)
            tok = tokenize_and_pad_batch(
                batch_X, gene_ids_all, max_len=MAX_LEN,
                vocab=vocab, pad_token=PAD_TOKEN, pad_value=PAD_VALUE,
                append_cls=True, include_zero_gene=False,
            )
            input_gene_ids = tok["genes"].to(device)
            input_values = tok["values"].to(device).float()
            pad_mask = input_gene_ids.eq(pad_idx)  # (B, seq) True where pad

            with torch.no_grad():
                model._encode(input_gene_ids, input_values, src_key_padding_mask=pad_mask)

            # mean-pool over real gene tokens only (exclude pad AND CLS at pos 0)
            valid = (~pad_mask).clone()
            valid[:, 0] = False
            valid_f = valid.float().unsqueeze(-1)                       # (B, seq, 1)
            counts = valid.sum(dim=1, keepdim=True).clamp(min=1).float()  # (B, 1)

            for name, tensor in captured.items():
                cls_v = tensor[:, 0, :].cpu().numpy()
                mean_v = ((tensor * valid_f).sum(dim=1) / counts).cpu().numpy()
                cls_buf.setdefault(name, []).append(cls_v)
                mean_buf.setdefault(name, []).append(mean_v)
            captured.clear()
    finally:
        for h in handles:
            h.remove()

    cls_full = {k: np.concatenate(v, axis=0) for k, v in sorted(cls_buf.items())}
    mean_full = {k: np.concatenate(v, axis=0) for k, v in sorted(mean_buf.items())}
    return cls_full, mean_full


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _save_activations(path: Path, cls: Dict[str, np.ndarray],
                      mean: Dict[str, np.ndarray], y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_base = str(path.with_suffix(""))
    payload = {"_labels": y}
    for name, arr in cls.items():
        payload[f"cls__{name}"] = arr
    for name, arr in mean.items():
        payload[f"mean__{name}"] = arr
    np.savez(tmp_base + ".tmp", **payload)
    Path(tmp_base + ".tmp.npz").replace(path)


def _load_activations(path: Path):
    data = np.load(path, allow_pickle=False)
    y = data["_labels"]
    cls = {k[len("cls__"):]: data[k] for k in data.files if k.startswith("cls__")}
    mean = {k[len("mean__"):]: data[k] for k in data.files if k.startswith("mean__")}
    return cls, mean, y


# ---------------------------------------------------------------------------
# Adata loader (same convention as phase1 layer_probe.py)
# ---------------------------------------------------------------------------
def _load_adata_for_probe(data_path: Path, label_col: str):
    adata = sc.read_h5ad(data_path)
    print(f"[phase2] loaded {data_path}: shape={adata.shape}")

    if adata.raw is not None and (adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()).min() < 0:
        print(f"[phase2] .X has negatives; swapping in .raw ({adata.raw.X.shape})")
        import anndata as ad
        adata = ad.AnnData(
            X=adata.raw.X, obs=adata.obs.copy(), var=adata.raw.var.copy(),
            obsm=dict(adata.obsm),
        )

    if label_col not in adata.obs.columns:
        candidates = [c for c in ("louvain", "leiden", "cell_type", "celltype", "CellType")
                      if c in adata.obs.columns]
        if not candidates:
            raise SystemExit(f"no label column in obs; have {list(adata.obs.columns)}")
        label_col = candidates[0]
    print(f"[phase2] label column {label_col!r} with {adata.obs[label_col].nunique()} classes")
    return adata, label_col


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------
def _lr(high_dim: bool = False) -> LogisticRegression:
    # lbfgs works fine on multinomial even with ~12k features when there are
    # ~2k samples — just bump max_iter so we do not hit the warning cap.
    return LogisticRegression(
        max_iter=5000 if high_dim else 2000,
        solver="lbfgs",
        n_jobs=-1,
        C=1.0,
    )


def probe(X: np.ndarray, y: np.ndarray, splits, high_dim: bool = False):
    out = []
    for s, (tr, te) in enumerate(splits):
        clf = _lr(high_dim=high_dim)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        out.append({
            "seed_idx": s,
            "accuracy": float(accuracy_score(y[te], pred)),
            "macro_f1": float(f1_score(y[te], pred, average="macro")),
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
        })
    return out


def probe_with_pca(X: np.ndarray, y: np.ndarray, splits, n_components: int):
    out = []
    for s, (tr, te) in enumerate(splits):
        n_comp = min(n_components, X.shape[1], len(tr) - 1)
        pca = PCA(n_components=n_comp, random_state=0)
        Xtr = pca.fit_transform(X[tr])
        Xte = pca.transform(X[te])
        clf = _lr(high_dim=False)
        clf.fit(Xtr, y[tr])
        pred = clf.predict(Xte)
        out.append({
            "seed_idx": s,
            "accuracy": float(accuracy_score(y[te], pred)),
            "macro_f1": float(f1_score(y[te], pred, average="macro")),
            "n_components_used": int(pca.n_components_),
            "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        })
    return out


def _agg(runs):
    accs = np.array([r["accuracy"] for r in runs])
    f1s = np.array([r["macro_f1"] for r in runs])
    return {
        "accuracy_mean": float(accs.mean()),
        "accuracy_std": float(accs.std(ddof=1) if len(accs) > 1 else 0.0),
        "macro_f1_mean": float(f1s.mean()),
        "macro_f1_std": float(f1s.std(ddof=1) if len(f1s) > 1 else 0.0),
        "n_seeds": int(len(runs)),
        "per_seed": runs,
    }


# ---------------------------------------------------------------------------
# Layer 0 sanity check (H1)
# ---------------------------------------------------------------------------
def layer0_sanity(cls_per_layer: Dict[str, np.ndarray], ref_layer: str) -> dict:
    L0 = cls_per_layer["layer_00_input"]
    L_ref = cls_per_layer[ref_layer]

    def _stats(M: np.ndarray) -> dict:
        std_per_dim = M.std(axis=0)
        return {
            "shape": list(M.shape),
            "mean_abs_value": float(np.abs(M).mean()),
            "std_per_dim_mean": float(std_per_dim.mean()),
            "std_per_dim_median": float(np.median(std_per_dim)),
            "std_per_dim_max": float(std_per_dim.max()),
            "max_abs_deviation_from_mean": float(
                np.abs(M - M.mean(axis=0, keepdims=True)).max()
            ),
        }

    ratio = float(L0.std(axis=0).mean() / max(1e-12, L_ref.std(axis=0).mean()))
    return {
        "layer_00_input_cls": _stats(L0),
        f"{ref_layer}_cls_reference": _stats(L_ref),
        "ratio_l0_to_ref_std": ratio,
        "interpretation_note": (
            "ratio_l0_to_ref_std reports how much per-dim std the CLS slot has "
            "at layer 0 relative to a mid-layer CLS. If << 1, layer 0 CLS is "
            "near-constant across cells and the phase1 layer-0 baseline is "
            "degenerate. Numbers, not verdicts — judge in context."
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/pbmc3k.h5ad")
    p.add_argument("--label_col", default="louvain")
    p.add_argument("--model_dir", default="checkpoints/scGPT_human")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_cells", type=int, default=20000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--force_extract", action="store_true")
    p.add_argument("--skip_baselines", action="store_true",
                   help="skip PCA / raw-log1p baselines (H1+H2+H4 only)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    data_path = ROOT / args.data if not Path(args.data).is_absolute() else Path(args.data)
    model_dir = ROOT / args.model_dir if not Path(args.model_dir).is_absolute() else Path(args.model_dir)
    out_dir = ROOT / "results" / "phase2"
    out_dir.mkdir(parents=True, exist_ok=True)

    acts_path = out_dir / "layer_activations.npz"
    feat_path = out_dir / "X_log1p_for_baselines.npy"
    sanity_path = out_dir / "layer0_sanity.json"
    baselines_path = out_dir / "baselines.json"
    probe_path = out_dir / "per_layer_probe.json"
    plot_path = out_dir / "layer_probe_curve_v2.png"
    summary_path = out_dir / "SUMMARY.md"

    adata, label_col = _load_adata_for_probe(data_path, args.label_col)

    if adata.n_obs > args.max_cells:
        rng = np.random.default_rng(args.seeds[0])
        idx = rng.choice(adata.n_obs, size=args.max_cells, replace=False)
        adata = adata[np.sort(idx)].copy()
        print(f"[phase2] subsampled to {adata.n_obs} cells")

    y_str = adata.obs[label_col].astype(str).values
    classes, y = np.unique(y_str, return_inverse=True)
    counts = dict(zip(*np.unique(y_str, return_counts=True)))
    print(f"[phase2] {len(classes)} classes: {list(classes)}")
    print(f"[phase2] class counts: {counts}")

    # 1) Cached activations (CLS + mean-pool per layer)
    cls_acts = mean_acts = None
    if acts_path.exists() and not args.force_extract:
        print(f"[phase2] loading cached activations from {acts_path}")
        cls_acts, mean_acts, y_cached = _load_activations(acts_path)
        if y_cached.shape != y.shape or not np.array_equal(y_cached, y):
            print("[phase2] cached labels differ from current selection — re-extracting")
            cls_acts = mean_acts = None
        else:
            d = next(iter(cls_acts.values())).shape[1]
            print(f"[phase2] cache OK: {len(cls_acts)} layers, d={d}")

    if cls_acts is None:
        print(f"[phase2] loading scGPT from {model_dir}")
        model, vocab, _ = load_scgpt_model(model_dir, device=args.device)
        adata_proc = preprocess_adata_for_scgpt(adata, vocab)
        cls_acts, mean_acts = extract_per_layer_cls_and_mean(
            adata_proc, model, vocab, device=args.device, batch_size=args.batch_size,
        )
        _save_activations(acts_path, cls_acts, mean_acts, y)
        print(f"[phase2] cached activations to {acts_path}")

        # Cache log1p matrix for baselines (Preprocessor sets X_log1p layer when
        # the input was raw counts; if not, fall back to .X which is already log-normed).
        if "X_log1p" in adata_proc.layers:
            X_logn = adata_proc.layers["X_log1p"]
        else:
            X_logn = adata_proc.X
        if hasattr(X_logn, "toarray"):
            X_logn = X_logn.toarray()
        X_logn = np.asarray(X_logn).astype(np.float32)
        np.save(feat_path, X_logn)
        print(f"[phase2] cached baseline features to {feat_path}: shape={X_logn.shape}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        if feat_path.exists():
            X_logn = np.load(feat_path)
            print(f"[phase2] loaded baseline features from {feat_path}: shape={X_logn.shape}")
        elif not args.skip_baselines:
            print("[phase2] baseline feature cache missing — running preprocess once for X_log1p")
            from scgpt.tokenizer.gene_tokenizer import GeneVocab
            vocab = GeneVocab.from_file(model_dir / "vocab.json")
            adata_proc = preprocess_adata_for_scgpt(adata, vocab)
            X_logn = adata_proc.layers.get("X_log1p", adata_proc.X)
            if hasattr(X_logn, "toarray"):
                X_logn = X_logn.toarray()
            X_logn = np.asarray(X_logn).astype(np.float32)
            np.save(feat_path, X_logn)
        else:
            X_logn = None

    layer_names = sorted(cls_acts.keys())
    d_model = cls_acts[layer_names[0]].shape[1]
    print(f"[phase2] {len(layer_names)} layers, d_model={d_model}")

    # 2) Layer 0 CLS-variance sanity (H1) ------------------------------------
    middle_layers = [k for k in layer_names if k != "layer_00_input"]
    ref_layer = "layer_05" if "layer_05" in cls_acts else middle_layers[len(middle_layers) // 2]
    sanity = layer0_sanity(cls_acts, ref_layer=ref_layer)
    sanity_path.write_text(json.dumps(sanity, indent=2))
    print(f"[phase2] wrote {sanity_path}")
    print(f"[phase2] layer 0 std / {ref_layer} std = {sanity['ratio_l0_to_ref_std']:.4g}")

    # 3) Shared train/test splits — used by baselines AND per-layer probes
    splits = [
        train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=s)
        for s in args.seeds
    ]

    # 4) Baselines (H3) ------------------------------------------------------
    baselines = None
    if not args.skip_baselines and X_logn is not None:
        print(f"[phase2] running baselines × {len(args.seeds)} seeds")
        print(f"[phase2]   raw log1p LR ({X_logn.shape[1]} features) ...")
        raw_runs = probe(X_logn, y, splits, high_dim=True)
        print(f"[phase2]   PCA-50 + LR ...")
        pca50_runs = probe_with_pca(X_logn, y, splits, n_components=50)
        print(f"[phase2]   PCA-512 + LR ...")
        pca512_runs = probe_with_pca(X_logn, y, splits, n_components=512)
        baselines = {
            "X_log1p_shape": list(X_logn.shape),
            "seeds": args.seeds,
            "raw_log1p_LR": _agg(raw_runs),
            "PCA50_LR": _agg(pca50_runs),
            "PCA512_LR": _agg(pca512_runs),
        }
        baselines_path.write_text(json.dumps(baselines, indent=2))
        print(f"[phase2] wrote {baselines_path}")
        for key in ["raw_log1p_LR", "PCA50_LR", "PCA512_LR"]:
            b = baselines[key]
            print(f"[phase2]   {key}: acc={b['accuracy_mean']:.4f} +/- {b['accuracy_std']:.4f}, "
                  f"F1={b['macro_f1_mean']:.4f} +/- {b['macro_f1_std']:.4f}")
    else:
        print("[phase2] skipping baselines")

    # 5) Per-layer multi-seed probe (H2 + H4) -------------------------------
    per_layer = {"layers": layer_names, "seeds": args.seeds, "results": {}}
    for name in tqdm(layer_names, desc="per-layer probe", unit="layer"):
        per_layer["results"][name] = {
            "cls": _agg(probe(cls_acts[name], y, splits, high_dim=False)),
            "mean": _agg(probe(mean_acts[name], y, splits, high_dim=False)),
        }
    probe_path.write_text(json.dumps(per_layer, indent=2))
    print(f"[phase2] wrote {probe_path}")

    # 6) Plot ---------------------------------------------------------------
    xs = np.arange(len(layer_names))
    def get(field, pool):
        return np.array([per_layer["results"][n][pool][field] for n in layer_names])
    cls_acc_m, cls_acc_s = get("accuracy_mean", "cls"), get("accuracy_std", "cls")
    mean_acc_m, mean_acc_s = get("accuracy_mean", "mean"), get("accuracy_std", "mean")
    cls_f1_m, cls_f1_s = get("macro_f1_mean", "cls"), get("macro_f1_std", "cls")
    mean_f1_m, mean_f1_s = get("macro_f1_mean", "mean"), get("macro_f1_std", "mean")

    def add_baseline(ax, b, label, color, key):
        if b is None:
            return
        m, s = b[f"{key}_mean"], b[f"{key}_std"]
        ax.axhline(m, color=color, linestyle=":", alpha=0.8,
                   label=f"{label} ({m:.3f}±{s:.3f})")
        ax.fill_between(xs, m - s, m + s, color=color, alpha=0.08)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    for ax, c_m, c_s, m_m, m_s, ylabel, key in [
        (ax1, cls_acc_m, cls_acc_s, mean_acc_m, mean_acc_s, "accuracy", "accuracy"),
        (ax2, cls_f1_m, cls_f1_s, mean_f1_m, mean_f1_s, "macro F1", "macro_f1"),
    ]:
        ax.plot(xs, c_m, marker="o", color="C0", label="CLS")
        ax.fill_between(xs, c_m - c_s, c_m + c_s, color="C0", alpha=0.2)
        ax.plot(xs, m_m, marker="s", linestyle="--", color="C1", label="mean-pool")
        ax.fill_between(xs, m_m - m_s, m_m + m_s, color="C1", alpha=0.2)
        if baselines is not None:
            add_baseline(ax, baselines.get("PCA50_LR"), "PCA-50 LR", "tab:gray", key=key)
            add_baseline(ax, baselines.get("PCA512_LR"), "PCA-512 LR", "tab:purple", key=key)
            add_baseline(ax, baselines.get("raw_log1p_LR"), "raw log1p LR", "tab:green", key=key)
        ax.set_xticks(xs)
        ax.set_xticklabels(["embed"] + [str(i) for i in range(1, len(layer_names))], rotation=0)
        ax.set_xlabel("layer (0 = input embedding, 1..N = transformer block output)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(
        f"scGPT layer probe — phase 2  (pbmc3k, {len(args.seeds)} seeds × "
        f"{adata.n_obs} cells × {len(classes)} classes)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[phase2] wrote {plot_path}")

    # 7) SUMMARY.md ---------------------------------------------------------
    best_cls_idx = int(np.argmax(cls_acc_m))
    best_mean_idx = int(np.argmax(mean_acc_m))
    md: List[str] = []
    md.append("# Phase 2 — scGPT layer probe with controls\n")
    md.append(f"- Data: `{data_path.name}` — {adata.n_obs} cells, {len(classes)} classes, label=`{label_col}`")
    md.append(f"- Seeds: {args.seeds} (all numbers reported as mean ± std across seeds)")
    md.append(f"- d_model: {d_model}\n")

    md.append("## H1 — layer 0 CLS variance")
    r = sanity["ratio_l0_to_ref_std"]
    l0_max_dev = sanity["layer_00_input_cls"]["max_abs_deviation_from_mean"]
    md.append(f"- `layer_00_input` CLS std / `{ref_layer}` CLS std = **{r:.4g}**")
    md.append(f"- `layer_00_input` max |dev from mean| across cells = **{l0_max_dev:.4g}**")
    md.append("- (ratio ≪ 1 ⇒ CLS at layer 0 is near-constant across cells, so phase1 layer-0 "
              "baseline is degenerate and the 0→1 'jump' is artifact)\n")

    if baselines is not None:
        md.append("## H3 — baselines")
        for key in ["raw_log1p_LR", "PCA50_LR", "PCA512_LR"]:
            b = baselines[key]
            md.append(f"- **{key}**: acc = {b['accuracy_mean']:.4f} ± {b['accuracy_std']:.4f}, "
                      f"macro-F1 = {b['macro_f1_mean']:.4f} ± {b['macro_f1_std']:.4f}")
        md.append("")
        pca50_acc = baselines["PCA50_LR"]["accuracy_mean"]
        delta_best = cls_acc_m[best_cls_idx] - pca50_acc
        md.append(f"- Δ(best CLS layer − PCA-50 LR) = **{delta_best:+.4f}**")
        md.append(f"- Δ(layer 12 CLS − peak CLS)    = **{cls_acc_m[-1] - cls_acc_m[best_cls_idx]:+.4f}**\n")

    md.append("## H2 + H4 — per-layer, multi-seed, CLS vs mean-pool")
    md.append(f"- Best CLS layer:       **{layer_names[best_cls_idx]}**  "
              f"acc = {cls_acc_m[best_cls_idx]:.4f} ± {cls_acc_s[best_cls_idx]:.4f}")
    md.append(f"- Best mean-pool layer: **{layer_names[best_mean_idx]}**  "
              f"acc = {mean_acc_m[best_mean_idx]:.4f} ± {mean_acc_s[best_mean_idx]:.4f}")
    md.append(f"- Range across layers 1..N (CLS, acc): "
              f"min = {cls_acc_m[1:].min():.4f}, max = {cls_acc_m[1:].max():.4f}, "
              f"spread = {cls_acc_m[1:].max() - cls_acc_m[1:].min():.4f}")
    md.append(f"- Median seed-std across layers 1..N (CLS): {np.median(cls_acc_s[1:]):.4f}\n")
    md.append("(If layer-1..N spread is within ~2× the median seed-std, the 'inverted-U' "
              "is not statistically distinguishable from a flat plateau with noise.)\n")

    md.append("## Full per-layer table (mean ± std across seeds)\n")
    md.append("| layer | CLS acc | CLS F1 | mean-pool acc | mean-pool F1 |")
    md.append("|---|---|---|---|---|")
    for i, n in enumerate(layer_names):
        md.append(
            f"| {n} "
            f"| {cls_acc_m[i]:.4f}±{cls_acc_s[i]:.4f} "
            f"| {cls_f1_m[i]:.4f}±{cls_f1_s[i]:.4f} "
            f"| {mean_acc_m[i]:.4f}±{mean_acc_s[i]:.4f} "
            f"| {mean_f1_m[i]:.4f}±{mean_f1_s[i]:.4f} |"
        )
    summary_path.write_text("\n".join(md))
    print(f"[phase2] wrote {summary_path}")

    # Console digest
    print("\n=== PHASE 2 SUMMARY ===")
    print(f"layer 0 std / {ref_layer} std = {r:.4g}")
    if baselines is not None:
        for key in ["raw_log1p_LR", "PCA50_LR", "PCA512_LR"]:
            b = baselines[key]
            print(f"{key:18s} acc = {b['accuracy_mean']:.4f} +/- {b['accuracy_std']:.4f}")
    print(f"best CLS layer:       {layer_names[best_cls_idx]}  "
          f"acc = {cls_acc_m[best_cls_idx]:.4f} +/- {cls_acc_s[best_cls_idx]:.4f}")
    print(f"best mean-pool layer: {layer_names[best_mean_idx]}  "
          f"acc = {mean_acc_m[best_mean_idx]:.4f} +/- {mean_acc_s[best_mean_idx]:.4f}")
    print(f"layer 1..N CLS acc spread: {cls_acc_m[1:].max() - cls_acc_m[1:].min():.4f}  "
          f"(median seed-std: {np.median(cls_acc_s[1:]):.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
