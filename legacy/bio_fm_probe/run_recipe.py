"""Run a single (Model adapter × Dataset adapter) recipe end-to-end.

Pipeline (all results to results/<model>__<dataset>/audit/):
  1. Load model adapter; load dataset adapter; validate modality match.
  2. Preprocess sample inputs through the model adapter.
  3. Extract CLS + mean-pool per layer (modality-agnostic).
  4. Per-layer linear probe (5 seeds) on CLS and mean-pool.
  5. Build modality-specific baseline (log1p+PCA / kmer+PCA / onehot_aa+PCA).
  6. Probe baseline + a model PCA (PCA of layer activations) for context.
  7. SVD spectrum diagnostic per SAE layer.
  8. TopK SAE on selected layers' token activations.
  9. **SAE − PCA ablation gap** per SAE layer — the headline metric.
 10. Cross-section digest -> AUDIT.md.

Usage:
    python -m bio_fm_probe.run_recipe \\
        --model scgpt --model_dir checkpoints/scGPT_human \\
        --dataset pbmc3k
    python -m bio_fm_probe.run_recipe \\
        --model hyenadna --model_dir checkpoints/hyenadna-small-32k-seqlen-hf \\
        --dataset genomic_benchmarks
    python -m bio_fm_probe.run_recipe \\
        --model esm2 --model_dir checkpoints/esm2_t12_35M_UR50D \\
        --dataset deeploc

The model/dataset registries are populated by importing bio_fm_probe.run_audit
(ADAPTER_REGISTRY) and bio_fm_probe.datasets (DATASET_REGISTRY).
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.decomposition import PCA

# Trigger registry population
from bio_fm_probe import datasets as _datasets  # noqa: F401
from bio_fm_probe.run_audit import ADAPTER_REGISTRY, load_adapter
from bio_fm_probe.core.dataset import DATASET_REGISTRY, load_dataset_adapter
from bio_fm_probe.core.recipe import AuditRecipe, validate_modality_match
from bio_fm_probe.core.extract import (
    extract_cls_and_mean_per_layer,
    extract_tokens_one_layer,
)
from bio_fm_probe.core.probes import (
    agg, aggregate_per_cell, encode_batched,
    layer0_cls_sanity, make_splits, probe,
    svd_spectrum_diagnostic, train_topk_sae,
)
from bio_fm_probe.core.baselines import build_baseline
from bio_fm_probe.core.ablation import sae_pca_ablation_gap


def _pick_default_sae_layers(n_layers: int) -> List[str]:
    mid = max(1, (n_layers + 1) // 2)
    return ["layer_00_input", f"layer_{mid:02d}", f"layer_{n_layers:02d}"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(ADAPTER_REGISTRY))
    p.add_argument("--model_dir", required=True)
    p.add_argument("--dataset", required=True, choices=list(DATASET_REGISTRY))
    p.add_argument("--data_dir", default=None,
                   help="dataset data dir; default = adapter's DEFAULT_*")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--max_samples", type=int, default=20000)
    p.add_argument("--extract_batch_size", type=int, default=16)
    # SAE
    p.add_argument("--sae_layers", nargs="+", default=None)
    p.add_argument("--sae_expansion", type=float, default=4.0,
                   help="SAE dict_size = expansion × d_model. Uniform across "
                        "models so cross-FM comparison is fair. Default 4×; "
                        "Lucas's phase 6 spec is 16×.")
    p.add_argument("--sae_dict", type=int, default=None,
                   help="override absolute dict_size; bypasses --sae_expansion")
    p.add_argument("--sae_k", type=int, default=32)
    p.add_argument("--sae_epochs", type=int, default=20)
    p.add_argument("--sae_batch", type=int, default=4096)
    p.add_argument("--sae_lr", type=float, default=1e-3)
    p.add_argument("--skip_sae", action="store_true")
    # ablation
    p.add_argument("--ablation_K_grid", type=int, nargs="+",
                   default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    p.add_argument("--ablation_pca_dim", type=int, default=512,
                   help="PCA dim of layer activations for ablation-gap comparison")
    # baseline
    p.add_argument("--baseline_pca_dim", type=int, default=50)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    # ---------- recipe ----------
    recipe = AuditRecipe(args.model, args.dataset)
    out_dir = Path(args.out) if args.out else recipe.output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[recipe] {recipe.slug()} -> {out_dir}")

    # ---------- load dataset, then model, validate modality ----------
    ds = load_dataset_adapter(args.dataset)
    sample = ds.load(args.data_dir)
    print(f"[recipe] dataset modality = {sample.modality}, "
          f"n_samples = {sample.n_samples}, baseline_kind = {sample.baseline_kind}")

    adapter = load_adapter(args.model)
    validate_modality_match(adapter.modality, sample.modality)

    # Subsample if too large
    if sample.n_samples > args.max_samples:
        rng = np.random.default_rng(args.seeds[0])
        idx = rng.choice(sample.n_samples, args.max_samples, replace=False)
        idx = np.sort(idx)
        if sample.modality == "scrna":
            sample.inputs = sample.inputs[idx].copy()
        else:
            sample.inputs = [sample.inputs[i] for i in idx]
        sample.labels = sample.labels[idx]
        print(f"[recipe] subsampled to {sample.n_samples} samples")

    print(f"[recipe] loading {adapter.name} from {args.model_dir}")
    adapter.load(args.model_dir, device=args.device)

    # Resolve SAE dict size: explicit --sae_dict wins, else expansion × d_model.
    if args.sae_dict is None:
        sae_dict = max(64, int(round(args.sae_expansion * adapter.d_model)))
    else:
        sae_dict = args.sae_dict
    print(f"[recipe] SAE dict_size = {sae_dict}  "
          f"(expansion = {sae_dict / adapter.d_model:.2f}× over d_model={adapter.d_model})")

    y = sample.labels
    classes, y = np.unique(y, return_inverse=True)
    print(f"[recipe] {len(classes)} classes")

    # Preprocess through adapter. Some adapters mutate sample.inputs (scrna
    # returns a processed adata; sequence models stash tokenized arrays internally).
    preprocessed = adapter.preprocess(sample.inputs)
    # The adata path: adapter.preprocess returns a new adata. Update sample.inputs.
    if sample.modality == "scrna" and preprocessed is not None and preprocessed is not sample.inputs:
        sample.inputs = preprocessed

    splits = make_splits(y, args.seeds)

    # ---------- 1. CLS + mean per layer ----------
    print("[recipe] === step 1/8: per-layer CLS+mean extraction ===")
    cls_dict, mean_dict = extract_cls_and_mean_per_layer(
        adapter, sample.inputs, args.extract_batch_size, args.device,
    )
    # mean_dict is always populated; cls_dict is empty for models with cls_position=None
    layer_names = sorted(mean_dict.keys())

    # ---------- 2. Layer 0 CLS sanity (skipped for models without CLS) ----------
    if adapter.cls_position is not None:
        middle = [k for k in layer_names if k != "layer_00_input"]
        ref = ("layer_05" if "layer_05" in cls_dict
               else middle[len(middle) // 2])
        sanity = layer0_cls_sanity(cls_dict, ref_layer=ref)
        (out_dir / "layer0_sanity.json").write_text(json.dumps(sanity, indent=2))

    # ---------- 3. Per-layer probe ----------
    print("[recipe] === step 2/8: per-layer LR probe ===")
    per_layer = {}
    for name in layer_names:
        entry: Dict = {"mean": agg(probe(mean_dict[name], y, splits))}
        if adapter.cls_position is not None:
            entry["cls"] = agg(probe(cls_dict[name], y, splits))
        per_layer[name] = entry
    (out_dir / "per_layer_probe.json").write_text(json.dumps({
        "layers": layer_names, "seeds": args.seeds, "results": per_layer,
    }, indent=2))

    # ---------- 4. Modality-specific baseline ----------
    print(f"[recipe] === step 3/8: baseline ({sample.baseline_kind}) ===")
    X_base, base_info = build_baseline(sample, n_components=args.baseline_pca_dim)
    base_runs = agg(probe(X_base, y, splits))
    baselines = {
        "baseline_kind": sample.baseline_kind,
        "info": base_info,
        "probe": base_runs,
    }
    (out_dir / "baselines.json").write_text(json.dumps(baselines, indent=2))
    base_acc = base_runs["accuracy_mean"]
    print(f"[recipe]   baseline acc = {base_acc:.4f} ± {base_runs['accuracy_std']:.4f}")

    # ---------- 5-8. SVD + SAE + ablation gap on selected layers ----------
    sae_layers = args.sae_layers or _pick_default_sae_layers(adapter.n_layers)
    print(f"[recipe] === step 4-8/8: SVD+SAE+ablation on {sae_layers} ===")
    svd_results = {}
    sae_summary = {}
    ablation_results = {}

    if not args.skip_sae:
        sae_root = out_dir / "sae"
        sae_root.mkdir(exist_ok=True)
        for layer in sae_layers:
            print(f"[recipe] --- layer {layer} ---")
            ld = sae_root / layer
            ld.mkdir(exist_ok=True)

            # 5. token-level activations at this layer
            tok_acts, tok_cell = extract_tokens_one_layer(
                adapter, sample.inputs, layer,
                args.extract_batch_size, args.device,
            )
            np.savez(ld / "token_activations.npz",
                     acts=tok_acts, cell_idx=tok_cell)

            # 6. SVD spectrum
            svd_results[layer] = svd_spectrum_diagnostic(tok_acts)
            print(f"[recipe]   SVD: PR={svd_results[layer]['participation_ratio']:.1f}, "
                  f"k95={svd_results[layer]['k95']}, k99={svd_results[layer]['k99']}")

            # 7. Train SAE
            sae, train_log = train_topk_sae(
                tok_acts, d_in=adapter.d_model,
                n_features=sae_dict, k=args.sae_k,
                batch_size=args.sae_batch, epochs=args.sae_epochs,
                lr=args.sae_lr, device=args.device,
            )
            torch.save({
                "state_dict": sae.state_dict(),
                "config": {"d_in": adapter.d_model,
                           "n_features": sae_dict, "k": args.sae_k},
                "layer": layer,
            }, ld / "sae.pt")
            (ld / "training_log.json").write_text(json.dumps(train_log, indent=2))

            # Encode SAE, aggregate per cell
            tok_codes = encode_batched(sae, tok_acts, args.sae_batch, args.device)
            n_cells = sample.n_samples
            X_sae_cell = aggregate_per_cell(tok_codes, tok_cell, n_cells=n_cells)
            np.savez(ld / "per_cell_features.npz", features=X_sae_cell, labels=y)

            # Build per-cell PCA features (PCA of layer activations, mean-pooled per cell)
            print(f"[recipe]   fitting layer-PCA (n_components={args.ablation_pca_dim}) ...")
            n_pca = min(args.ablation_pca_dim, tok_acts.shape[1], tok_acts.shape[0] - 1)
            pca = PCA(n_components=n_pca, random_state=0)
            tok_pca = pca.fit_transform(tok_acts).astype(np.float32)
            X_pca_cell = aggregate_per_cell(tok_pca, tok_cell, n_cells=n_cells)
            np.savez(ld / "per_cell_pca_features.npz",
                     features=X_pca_cell, labels=y,
                     evr_sum=pca.explained_variance_ratio_.sum())

            # 8. SAE - PCA ablation gap (THE headline)
            print(f"[recipe]   running SAE-PCA ablation gap ...")
            gap = sae_pca_ablation_gap(
                X_sae_cell, X_pca_cell, y, splits,
                K_grid=args.ablation_K_grid,
                include_random_null=True,
            )
            (ld / "ablation_gap.json").write_text(json.dumps(gap, indent=2))
            ablation_results[layer] = gap
            print(f"[recipe]   gap@K={args.ablation_K_grid[-1]}: "
                  f"{gap['gap']['gap_mean'][-1]:+.4f} "
                  f"± {gap['gap']['gap_std'][-1]:.4f}")

            # Quick per-layer probe summary
            sae_probe_res = agg(probe(X_sae_cell, y, splits))
            sae_summary[layer] = {
                "var_explained_final": train_log["epoch_var_explained"][-1],
                "dead_features_ever": train_log["dead_features_ever"],
                "cell_type_probe_sae": sae_probe_res,
                "cell_type_probe_pca": agg(probe(X_pca_cell, y, splits)),
                "gap_curve_mean": gap["gap"]["gap_mean"],
                "gap_curve_std": gap["gap"]["gap_std"],
                "K_grid": gap["gap"]["K_grid"],
            }

    (out_dir / "svd_diag.json").write_text(json.dumps(svd_results, indent=2))

    # ---------- AUDIT.md ----------
    md = [f"# Audit — {recipe.slug()}\n"]
    md.append(f"- Model: `{adapter.name}` (d_model={adapter.d_model}, "
              f"n_layers={adapter.n_layers}, modality={adapter.modality})")
    md.append(f"- Dataset: `{ds.name}` "
              f"(n_samples={sample.n_samples}, classes={len(classes)}, "
              f"baseline={sample.baseline_kind})")
    md.append(f"- Seeds: {args.seeds}\n")

    md.append("## Baseline")
    md.append(f"- **{sample.baseline_kind}**: acc = {base_acc:.4f} ± "
              f"{base_runs['accuracy_std']:.4f}, F1 = "
              f"{base_runs['macro_f1_mean']:.4f} ± "
              f"{base_runs['macro_f1_std']:.4f}")
    md.append(f"- baseline PCA dim: {args.baseline_pca_dim}\n")

    md.append("## Per-layer probe (5-seed mean ± std)")
    if adapter.cls_position is not None:
        md.append("| layer | CLS acc | mean-pool acc |")
        md.append("|---|---|---|")
        for n in layer_names:
            c = per_layer[n].get("cls", {})
            m = per_layer[n]["mean"]
            cls_str = (f"{c['accuracy_mean']:.4f}±{c['accuracy_std']:.4f}"
                       if c else "—")
            md.append(f"| {n} | {cls_str} | "
                      f"{m['accuracy_mean']:.4f}±{m['accuracy_std']:.4f} |")
    else:
        md.append("| layer | mean-pool acc |")
        md.append("|---|---|")
        for n in layer_names:
            m = per_layer[n]["mean"]
            md.append(f"| {n} | {m['accuracy_mean']:.4f}±{m['accuracy_std']:.4f} |")
    md.append("")

    md.append("## SVD spectrum")
    md.append("| layer | PR | k50 | k95 | k99 | var_per_elem |")
    md.append("|---|---|---|---|---|---|")
    for layer, s in svd_results.items():
        md.append(f"| {layer} | {s['participation_ratio']:.1f} | "
                  f"{s['k50']} | {s['k95']} | {s['k99']} | "
                  f"{s['var_per_elem']:.3f} |")
    md.append("")

    if sae_summary:
        md.append("## SAE per selected layer")
        md.append("| layer | var_exp | dead_ever | SAE probe acc | PCA probe acc |")
        md.append("|---|---|---|---|---|")
        for layer, s in sae_summary.items():
            sae_pr = s["cell_type_probe_sae"]
            pca_pr = s["cell_type_probe_pca"]
            md.append(
                f"| {layer} | {s['var_explained_final']:.3f} "
                f"| {s['dead_features_ever']}/{sae_dict} "
                f"| {sae_pr['accuracy_mean']:.4f}±{sae_pr['accuracy_std']:.4f} "
                f"| {pca_pr['accuracy_mean']:.4f}±{pca_pr['accuracy_std']:.4f} |"
            )
        md.append("")

        md.append("## SAE − PCA ablation gap (the headline)")
        md.append("Positive gap ⇒ knowledge more **distributed** than PCA captures ⇒ "
                  "SAE recovers a hidden sparse dictionary PCA misses.")
        md.append(
            "Gap ≈ 0 ⇒ knowledge **concentrated** in PCA-aligned dims ⇒ "
            "no extra inversion space.\n"
        )
        K_grid = args.ablation_K_grid
        md.append("| layer | " + " | ".join(f"gap@K={k}" for k in K_grid) + " |")
        md.append("|---|" + "---|" * len(K_grid))
        for layer, s in sae_summary.items():
            row = [layer]
            for i, k in enumerate(K_grid):
                row.append(f"{s['gap_curve_mean'][i]:+.3f}±{s['gap_curve_std'][i]:.3f}")
            md.append("| " + " | ".join(row) + " |")
        md.append("")

    (out_dir / "AUDIT.md").write_text("\n".join(md))
    print(f"\n[recipe] === DONE === wrote {out_dir / 'AUDIT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
