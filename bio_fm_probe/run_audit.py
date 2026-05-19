"""Run the full audit pipeline on a bio foundation model.

Pipeline (all results go to results/<adapter>/audit/):
  1. Layer probe (CLS + mean-pool, multi-seed)            -> per_layer_probe.json
  2. Layer 0 CLS sanity (H1)                              -> layer0_sanity.json
  3. Baselines (raw log1p LR / PCA-50 / PCA-512)          -> baselines.json
  4. SVD spectrum diagnostic on selected layers           -> svd_diag.json
  5. TopK SAE on selected layers (with cell-type probe)   -> sae/<layer>/...
  6. Cross-section digest                                  -> AUDIT.md

Adding a new model:
  - Drop bio_fm_probe/adapters/<name>.py implementing BioFMAdapter
    (use _template.py as a starting point).
  - Register it in ADAPTER_REGISTRY below.

Usage:
    python -m bio_fm_probe.run_audit \\
        --adapter scgpt \\
        --model_dir checkpoints/scGPT_human \\
        --data data/pbmc3k.h5ad \\
        --label_col louvain

Compare across models (after running on several):
    python -m bio_fm_probe.compare_models scgpt scmamba geneformer
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import scanpy as sc
import torch

from bio_fm_probe.core.adapter import BioFMAdapter
from bio_fm_probe.core.extract import (
    extract_cls_and_mean_per_layer,
    extract_tokens_one_layer,
)
from bio_fm_probe.core.probes import (
    agg, aggregate_per_cell, encode_batched,
    layer0_cls_sanity, make_splits, probe, probe_with_pca,
    svd_spectrum_diagnostic, train_topk_sae, TopKSAE,
)


# -----------------------------------------------------------------------------
# Adapter registry. Add new models here.
# -----------------------------------------------------------------------------
ADAPTER_REGISTRY: Dict[str, str] = {
    "scgpt": "bio_fm_probe.adapters.scgpt:ScGPTAdapter",
    # "scmamba":    "bio_fm_probe.adapters.scmamba:ScMambaAdapter",
    # "geneformer": "bio_fm_probe.adapters.geneformer:GeneformerAdapter",
    # "scfoundation": "bio_fm_probe.adapters.scfoundation:ScFoundationAdapter",
    # "uce": "bio_fm_probe.adapters.uce:UCEAdapter",
}


def load_adapter(name: str) -> BioFMAdapter:
    if name not in ADAPTER_REGISTRY:
        raise SystemExit(
            f"unknown adapter {name!r}; available: {list(ADAPTER_REGISTRY)}"
        )
    mod_path, cls_name = ADAPTER_REGISTRY[name].split(":")
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)()


def _load_adata(data_path: Path, label_col: str):
    adata = sc.read_h5ad(data_path)
    print(f"[audit] loaded {data_path}: shape={adata.shape}")

    if adata.raw is not None and (
        adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
    ).min() < 0:
        print(f"[audit] .X has negatives; swapping in .raw ({adata.raw.X.shape})")
        import anndata as ad
        adata = ad.AnnData(
            X=adata.raw.X, obs=adata.obs.copy(),
            var=adata.raw.var.copy(), obsm=dict(adata.obsm),
        )

    if label_col not in adata.obs.columns:
        for c in ("louvain", "leiden", "cell_type", "celltype", "CellType"):
            if c in adata.obs.columns:
                label_col = c
                break
        else:
            raise SystemExit(f"no label column; have {list(adata.obs.columns)}")
    return adata, label_col


def _pick_default_sae_layers(n_layers: int) -> list:
    """Default = input + mid + final block."""
    mid = max(1, (n_layers + 1) // 2)
    return ["layer_00_input", f"layer_{mid:02d}", f"layer_{n_layers:02d}"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True, choices=list(ADAPTER_REGISTRY))
    p.add_argument("--model_dir", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--label_col", default="louvain")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--max_cells", type=int, default=20000)
    p.add_argument("--extract_batch_size", type=int, default=16,
                   help="cells per forward batch")
    # SAE knobs
    p.add_argument("--sae_layers", nargs="+", default=None,
                   help="layer names for SAE; default: input, mid, final")
    p.add_argument("--sae_dict", type=int, default=2048)
    p.add_argument("--sae_k", type=int, default=32)
    p.add_argument("--sae_epochs", type=int, default=20)
    p.add_argument("--sae_batch", type=int, default=4096)
    p.add_argument("--sae_lr", type=float, default=1e-3)
    p.add_argument("--skip_sae", action="store_true",
                   help="run only layer probe + baselines + SVD (no SAE)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None,
                   help="output dir; default results/<adapter>/audit/")
    args = p.parse_args()

    # ---------- adapter + data ----------
    adapter = load_adapter(args.adapter)
    print(f"[audit] loading {adapter.name} from {args.model_dir}")
    adapter.load(args.model_dir, device=args.device)

    adata, label_col = _load_adata(Path(args.data), args.label_col)
    if adata.n_obs > args.max_cells:
        rng = np.random.default_rng(args.seeds[0])
        idx = rng.choice(adata.n_obs, args.max_cells, replace=False)
        adata = adata[np.sort(idx)].copy()
        print(f"[audit] subsampled to {adata.n_obs} cells")

    y_str = adata.obs[label_col].astype(str).values
    classes, y = np.unique(y_str, return_inverse=True)
    print(f"[audit] {len(classes)} classes: {list(classes)}")

    adata = adapter.preprocess(adata)
    print(f"[audit] post-preprocess: {adata.shape}, "
          f"d_model={adapter.d_model}, n_layers={adapter.n_layers}")

    out_dir = (Path(args.out) if args.out
               else Path("results") / adapter.name / "audit")
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = make_splits(y, args.seeds)

    # ---------- 1. CLS + mean per layer ----------
    print("[audit] === step 1/6: per-layer CLS+mean extraction ===")
    cls_dict, mean_dict = extract_cls_and_mean_per_layer(
        adapter, adata, args.extract_batch_size, args.device,
    )
    np.savez(
        out_dir / "cls_mean_per_layer.npz",
        **{f"cls__{k}": v for k, v in cls_dict.items()},
        **{f"mean__{k}": v for k, v in mean_dict.items()},
        _labels=y,
    )
    layer_names = sorted(cls_dict.keys())
    print(f"[audit] extracted {len(layer_names)} layers")

    # ---------- 2. Layer 0 CLS sanity ----------
    print("[audit] === step 2/6: layer 0 CLS sanity ===")
    middle = [k for k in layer_names if k != "layer_00_input"]
    ref = "layer_05" if "layer_05" in cls_dict else middle[len(middle) // 2]
    sanity = layer0_cls_sanity(cls_dict, ref_layer=ref)
    (out_dir / "layer0_sanity.json").write_text(json.dumps(sanity, indent=2))
    print(f"[audit] layer 0 std / {ref} std = {sanity.get('ratio_l0_to_ref_std', float('nan')):.4g}")

    # ---------- 3. Per-layer probe ----------
    print("[audit] === step 3/6: per-layer LR probe (CLS + mean) ===")
    per_layer = {}
    for name in layer_names:
        per_layer[name] = {
            "cls": agg(probe(cls_dict[name], y, splits)),
            "mean": agg(probe(mean_dict[name], y, splits)),
        }
    (out_dir / "per_layer_probe.json").write_text(json.dumps({
        "layers": layer_names, "seeds": args.seeds, "results": per_layer,
    }, indent=2))

    # ---------- 4. Baselines ----------
    print("[audit] === step 4/6: baselines (raw log1p / PCA-50 / PCA-512) ===")
    if "X_log1p" in adata.layers:
        X_log = adata.layers["X_log1p"]
    elif "X_normed" in adata.layers:
        X_log = adata.layers["X_normed"]
    else:
        X_log = adata.X
    if hasattr(X_log, "toarray"):
        X_log = X_log.toarray()
    X_log = np.asarray(X_log).astype(np.float32)

    baselines = {
        "X_features_shape": list(X_log.shape),
        "seeds": args.seeds,
        "raw_log1p_LR": agg(probe(X_log, y, splits, high_dim=True)),
        "PCA50_LR": agg(probe_with_pca(X_log, y, splits, 50)),
        "PCA512_LR": agg(probe_with_pca(X_log, y, splits, 512)),
    }
    (out_dir / "baselines.json").write_text(json.dumps(baselines, indent=2))
    pca50_acc = baselines["PCA50_LR"]["accuracy_mean"]
    print(f"[audit]   raw log1p LR: {baselines['raw_log1p_LR']['accuracy_mean']:.4f}")
    print(f"[audit]   PCA-50  LR:  {pca50_acc:.4f}")
    print(f"[audit]   PCA-512 LR:  {baselines['PCA512_LR']['accuracy_mean']:.4f}")

    # ---------- 5. SVD + 6. SAE on selected layers ----------
    sae_layers = args.sae_layers or _pick_default_sae_layers(adapter.n_layers)
    print(f"[audit] === step 5-6/6: SVD diag + SAE on {sae_layers} ===")
    svd_results = {}
    sae_summary = {}

    if not args.skip_sae:
        sae_root = out_dir / "sae"
        sae_root.mkdir(exist_ok=True)
        for layer in sae_layers:
            print(f"[audit] --- layer {layer} ---")
            ld = sae_root / layer
            ld.mkdir(exist_ok=True)
            # token-level activations (fresh forward)
            tok_acts, tok_cell = extract_tokens_one_layer(
                adapter, adata, layer, args.extract_batch_size, args.device,
            )
            np.savez(ld / "token_activations.npz",
                     acts=tok_acts, cell_idx=tok_cell)
            # SVD on the same activations
            svd_results[layer] = svd_spectrum_diagnostic(tok_acts)
            print(f"[audit]   SVD: PR={svd_results[layer]['participation_ratio']:.1f}  "
                  f"k95={svd_results[layer]['k95']}  k99={svd_results[layer]['k99']}")
            # train SAE
            sae, train_log = train_topk_sae(
                tok_acts, d_in=adapter.d_model,
                n_features=args.sae_dict, k=args.sae_k,
                batch_size=args.sae_batch, epochs=args.sae_epochs,
                lr=args.sae_lr, device=args.device,
            )
            torch.save({
                "state_dict": sae.state_dict(),
                "config": {"d_in": adapter.d_model,
                           "n_features": args.sae_dict, "k": args.sae_k},
                "layer": layer,
            }, ld / "sae.pt")
            (ld / "training_log.json").write_text(json.dumps(train_log, indent=2))
            # probe SAE features
            tok_codes = encode_batched(sae, tok_acts, args.sae_batch, args.device)
            cell_feats = aggregate_per_cell(tok_codes, tok_cell, n_cells=adata.n_obs)
            np.savez(ld / "per_cell_features.npz", features=cell_feats, labels=y)
            sae_probe_res = agg(probe(cell_feats, y, splits))
            (ld / "cell_type_probe.json").write_text(json.dumps({
                "layer": layer,
                "n_features": args.sae_dict,
                "k_active": args.sae_k,
                "results": sae_probe_res,
            }, indent=2))
            sae_summary[layer] = {
                "var_explained_final": train_log["epoch_var_explained"][-1],
                "dead_features_ever": train_log["dead_features_ever"],
                "cell_type_probe": sae_probe_res,
            }
            print(f"[audit]   SAE probe acc = {sae_probe_res['accuracy_mean']:.4f} "
                  f"± {sae_probe_res['accuracy_std']:.4f}  "
                  f"(Δ vs PCA-50: {sae_probe_res['accuracy_mean'] - pca50_acc:+.4f})")
    else:
        # SVD only, on the cached cell-level mean pools (cheap)
        for layer in sae_layers:
            if layer in mean_dict:
                svd_results[layer] = svd_spectrum_diagnostic(mean_dict[layer])

    (out_dir / "svd_diag.json").write_text(json.dumps(svd_results, indent=2))

    # ---------- AUDIT.md ----------
    md = [f"# Audit — {adapter.name}\n"]
    md.append(f"- Data: `{Path(args.data).name}` — {adata.n_obs} cells, "
              f"{len(classes)} classes, label=`{label_col}`")
    md.append(f"- Model: d_model={adapter.d_model}, n_layers={adapter.n_layers}")
    md.append(f"- Seeds: {args.seeds}\n")

    md.append("## Baselines (5-seed mean ± std)")
    for k in ["raw_log1p_LR", "PCA50_LR", "PCA512_LR"]:
        b = baselines[k]
        md.append(f"- **{k}**: acc = {b['accuracy_mean']:.4f} ± {b['accuracy_std']:.4f}, "
                  f"F1 = {b['macro_f1_mean']:.4f} ± {b['macro_f1_std']:.4f}")
    md.append("")

    md.append("## Layer 0 CLS sanity (H1)")
    r = sanity.get("ratio_l0_to_ref_std", float("nan"))
    md.append(f"- layer_00_input CLS std / {ref} CLS std = **{r:.4g}**")
    md.append("- (≪ 1 ⇒ CLS slot is constant at input ⇒ layer-0 baseline is degenerate)\n")

    md.append("## Per-layer probe (5-seed mean ± std)")
    md.append("| layer | CLS acc | CLS F1 | mean-pool acc | mean-pool F1 |")
    md.append("|---|---|---|---|---|")
    cls_accs, mean_accs = [], []
    for n in layer_names:
        c = per_layer[n]["cls"]
        m = per_layer[n]["mean"]
        cls_accs.append(c["accuracy_mean"])
        mean_accs.append(m["accuracy_mean"])
        md.append(
            f"| {n} | {c['accuracy_mean']:.4f}±{c['accuracy_std']:.4f} "
            f"| {c['macro_f1_mean']:.4f}±{c['macro_f1_std']:.4f} "
            f"| {m['accuracy_mean']:.4f}±{m['accuracy_std']:.4f} "
            f"| {m['macro_f1_mean']:.4f}±{m['macro_f1_std']:.4f} |"
        )
    best = max(max(cls_accs), max(mean_accs))
    md.append(f"\n- best across layers × pooling: **{best:.4f}**  "
              f"(Δ vs PCA-50: **{best - pca50_acc:+.4f}**)\n")

    md.append("## SVD spectrum")
    md.append("| layer | PR (eff. rank) | k50 | k95 | k99 | var_per_elem |")
    md.append("|---|---|---|---|---|---|")
    for layer, s in svd_results.items():
        md.append(
            f"| {layer} | {s['participation_ratio']:.1f} | "
            f"{s['k50']} | {s['k95']} | {s['k99']} | {s['var_per_elem']:.3f} |"
        )
    md.append("")

    if sae_summary:
        md.append("## TopK SAE per selected layer")
        md.append(f"- config: dict={args.sae_dict}, k={args.sae_k}, "
                  f"epochs={args.sae_epochs}, bs={args.sae_batch}, lr={args.sae_lr}")
        md.append("\n| layer | var_exp | dead_ever | SAE probe acc | Δ vs PCA-50 |")
        md.append("|---|---|---|---|---|")
        for layer, s in sae_summary.items():
            pr_acc = s["cell_type_probe"]["accuracy_mean"]
            pr_std = s["cell_type_probe"]["accuracy_std"]
            md.append(
                f"| {layer} | {s['var_explained_final']:.3f} "
                f"| {s['dead_features_ever']}/{args.sae_dict} "
                f"| {pr_acc:.4f}±{pr_std:.4f} "
                f"| {pr_acc - pca50_acc:+.4f} |"
            )
        md.append("")

    (out_dir / "AUDIT.md").write_text("\n".join(md))
    print(f"\n[audit] === DONE ===\n[audit] wrote {out_dir / 'AUDIT.md'}")
    print(f"[audit] all outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
