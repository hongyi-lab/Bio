"""evo_resume.py — resume a partial evo_probe.py run from cached state.

Picks up wherever the previous run died. For each layer it inspects what's on
disk and only runs the missing steps. Uses the memory-safe fused
encode-and-aggregate helper so the 306 GB allocation that killed evo_probe.py
last night does not happen.

Files it looks for per layer dir (results/evo_1_8k__genomic_benchmarks/<layer>/):
    token_activations.npz       forward-pass cache (~39 GB per layer)
    per_cell_pca_features.npz   PCA per-cell features
    sae.pt + sae_training_log.json   trained SAE
    per_cell_features.npz       SAE per-cell features (this is what we crashed before)
    cell_type_probe.json        PCA + SAE LR probe results
    ablation_gap.json           SAE-PCA ablation gap

If any of these is missing it runs the corresponding step. If all are present
the layer is skipped.

Run after the patched common_sae.py is in place:
    python src/phase7/evo_resume.py
or with a custom layer set / fewer SAE epochs:
    python src/phase7/evo_resume.py --sae_layers layer_00_input
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch
from tqdm import tqdm

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent.parent
sys.path.insert(0, str(THIS.parent))


def _resolve(p: str) -> str:
    path = Path(p)
    return str(path if path.is_absolute() else ROOT / path)


from common_sae import (  # noqa: E402
    agg, encode_and_aggregate_per_cell, extract_summary_fields,
    extract_tokens_from_iter, fit_pca_and_aggregate_per_cell,
    make_splits, plot_phase7_summary, probe, sae_pca_ablation_gap,
    svd_spectrum_diagnostic, train_topk_sae, write_phase7_report, TopKSAE,
)
from evo_probe import load_evo2, iter_evo2_activations, load_genomic_benchmarks  # noqa: E402


def _have(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _nan_filter(token_acts, token_cell, layer_dir: Path, chunk: int = 200_000):
    """Drop tokens whose activation row contains NaN or Inf (StripedHyena fp16
    instability at deep layers). Filtered arrays are persisted as memmapped
    .npy files in `layer_dir` so re-runs skip the work.

    Memory: chunked scan + memmap-output write. Peak ~ chunk * d * 8 bytes ≈ 5 GB
    for chunk=200k, d=4096. Never allocates the full (n, d) fp32 array in RAM.
    """
    filt_acts_path = layer_dir / "filtered_acts.npy"
    filt_cell_path = layer_dir / "filtered_cell_idx.npy"
    mask_path = layer_dir / "nan_filter_mask.npy"

    # Reuse cached filter if both files exist and are non-empty
    if _have(filt_acts_path) and _have(filt_cell_path):
        new_acts = np.load(filt_acts_path, mmap_mode="r")
        new_cell = np.load(filt_cell_path)
        n = int(token_acts.shape[0])
        kept = int(new_acts.shape[0])
        print(f"[nan-filter] reusing cached filtered arrays "
              f"({kept}/{n} = {100*kept/n:.2f}% kept)")
        return new_acts, new_cell

    n, d = token_acts.shape
    print(f"[nan-filter] scanning {n} tokens for NaN/Inf ...")
    valid_mask = np.zeros(n, dtype=bool)
    n_bad = 0
    for i in tqdm(range(0, n, chunk),
                  total=(n + chunk - 1) // chunk,
                  desc="[nan-filter] scan", unit="chunk"):
        j = min(i + chunk, n)
        block = np.asarray(token_acts[i:j]).astype(np.float32)
        good = np.isfinite(block).all(axis=1)
        valid_mask[i:j] = good
        n_bad += int((~good).sum())
    if n_bad == 0:
        print(f"[nan-filter] no NaN/Inf — proceeding with original cache")
        return token_acts, token_cell

    n_good = n - n_bad
    print(f"[nan-filter] dropping {n_bad}/{n} rows ({100*n_bad/n:.2f}%); "
          f"keeping {n_good} tokens")

    np.save(mask_path, valid_mask)

    # Write filtered acts via memmap (never allocate full array in RAM)
    out_acts = np.lib.format.open_memmap(
        filt_acts_path, mode="w+", dtype=token_acts.dtype, shape=(n_good, d),
    )
    out_cell = np.empty(n_good, dtype=token_cell.dtype)
    pos = 0
    for i in tqdm(range(0, n, chunk),
                  total=(n + chunk - 1) // chunk,
                  desc="[nan-filter] write", unit="chunk"):
        j = min(i + chunk, n)
        m = valid_mask[i:j]
        if not m.any():
            continue
        good_a = np.asarray(token_acts[i:j])[m]
        good_c = np.asarray(token_cell[i:j])[m]
        out_acts[pos:pos + good_a.shape[0]] = good_a
        out_cell[pos:pos + good_a.shape[0]] = good_c
        pos += good_a.shape[0]
    out_acts.flush()
    del out_acts
    np.save(filt_cell_path, out_cell)
    print(f"[nan-filter] wrote {filt_acts_path.stat().st_size/1e9:.2f} GB filtered cache")

    # Reload via mmap for downstream
    return np.load(filt_acts_path, mmap_mode="r"), np.load(filt_cell_path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="checkpoints/evo-1-8k-base")
    p.add_argument("--data_dir",
                   default="data/genomic_benchmarks/human_nontata_promoters")
    p.add_argument("--sae_layers", nargs="+", default=None,
                   help="default: input, mid, final (mid/final inferred from n_layers)")
    p.add_argument("--sae_expansion", type=float, default=4.0)
    p.add_argument("--sae_k", type=int, default=32)
    p.add_argument("--sae_epochs", type=int, default=20)
    p.add_argument("--sae_batch", type=int, default=4096)
    p.add_argument("--sae_lr", type=float, default=1e-3)
    p.add_argument("--pca_dim", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=20000)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--extract_batch_size", type=int, default=2)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--ablation_K_grid", type=int, nargs="+",
                   default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    p.add_argument("--no_fp16", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out_dir = (Path(_resolve(args.out)) if args.out
               else ROOT / "results" / "evo_1_8k__genomic_benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[resume] output -> {out_dir}")

    # Dataset (needed for labels + cell count even if forward is cached)
    data_dir = Path(_resolve(args.data_dir))
    seqs, labels = load_genomic_benchmarks(data_dir)
    if len(seqs) > args.max_samples:
        rng = np.random.default_rng(args.seeds[0])
        idx = np.sort(rng.choice(len(seqs), args.max_samples, replace=False))
        seqs = [seqs[i] for i in idx]
        labels = labels[idx]
    y_classes, y = np.unique(labels, return_inverse=True)
    n_cells = len(seqs)
    print(f"[resume] {n_cells} samples, {len(y_classes)} classes")

    # Defer model load until we actually need it (avoids the 13 GB GPU hit if
    # all forward caches are present).
    model = tok = blocks_attr = None
    d_model = 4096   # known from Evo-1 config; verified at load time
    n_layers = 32

    sae_layers = args.sae_layers or [
        "layer_00_input", f"layer_{n_layers // 2:02d}", f"layer_{n_layers:02d}",
    ]
    print(f"[resume] target layers: {sae_layers}")

    sae_dict = max(64, int(round(args.sae_expansion * d_model)))
    pca_dim = args.pca_dim if args.pca_dim is not None else sae_dict
    print(f"[resume] SAE dict_size={sae_dict}, PCA dim={pca_dim}")
    splits = make_splits(y, args.seeds)

    svd_results = {}
    sae_summary = {}
    pca_summary = {}
    ablation_summary = {}

    for layer in sae_layers:
        print(f"\n{'=' * 60}\n[resume] === layer {layer} ===\n{'=' * 60}")
        ld = out_dir / layer
        ld.mkdir(parents=True, exist_ok=True)
        acts_cache = ld / "token_activations.npz"
        pca_cache = ld / "per_cell_pca_features.npz"
        sae_ckpt = ld / "sae.pt"
        sae_cell_cache = ld / "per_cell_features.npz"
        probe_json = ld / "cell_type_probe.json"
        ablation_json = ld / "ablation_gap.json"

        # ---- 1. Token activations (forward) ----
        if _have(acts_cache):
            sz = acts_cache.stat().st_size / 1e9
            print(f"[resume] mmap-loading cached {acts_cache} ({sz:.2f} GB)")
            d = np.load(acts_cache, allow_pickle=False, mmap_mode="r")
            token_acts, token_cell = d["acts"], d["cell_idx"]
            print(f"[resume]   acts={token_acts.shape} {token_acts.dtype}")
        else:
            # Need the model
            if model is None:
                print(f"[resume] loading {args.model_dir}")
                model, tok, d_model_real, n_layers_real, blocks_attr = load_evo2(
                    _resolve(args.model_dir), device=args.device, fp16=not args.no_fp16,
                )
                # If discovered values differ, refresh derived knobs
                if d_model_real != d_model:
                    d_model = d_model_real
                    sae_dict = max(64, int(round(args.sae_expansion * d_model)))
                    pca_dim = args.pca_dim if args.pca_dim is not None else sae_dict
                n_layers = n_layers_real
                # Re-tokenize
                print(f"[resume] tokenizing (max_len={args.max_len}) ...")
                enc = tok(seqs, padding="max_length", truncation=True,
                          max_length=args.max_len, return_tensors="np")
                input_ids = enc["input_ids"].astype(np.int64)
                if "attention_mask" in enc:
                    attn_mask = enc["attention_mask"].astype(np.int64)
                else:
                    pad_id = getattr(tok, "pad_token_id", 0) or 0
                    attn_mask = (input_ids != pad_id).astype(np.int64)

            print(f"[resume] extracting tokens for {layer} ...")
            t0 = time.time()
            token_acts, token_cell = extract_tokens_from_iter(
                iter_evo2_activations(
                    model, tok, blocks_attr, input_ids, attn_mask,
                    target_layers=[layer], batch_size=args.extract_batch_size,
                    device=args.device,
                ),
                target_layer=layer, n_samples=n_cells,
            )
            np.savez(acts_cache, acts=token_acts, cell_idx=token_cell)
            print(f"[resume]   extracted {token_acts.shape} in {time.time() - t0:.0f}s")

        # ---- 1.5 NaN/Inf filter (StripedHyena fp16 instability at deep layers) ----
        token_acts, token_cell = _nan_filter(token_acts, token_cell, ld)

        # ---- 2. SVD ----
        svd_results[layer] = svd_spectrum_diagnostic(token_acts)
        print(f"[resume] SVD: PR={svd_results[layer]['participation_ratio']:.1f}  "
              f"k95={svd_results[layer]['k95']}  k99={svd_results[layer]['k99']}")

        # ---- 3. PCA per-cell ----
        if _have(pca_cache):
            print(f"[resume] loading cached PCA per-cell features {pca_cache}")
            dd = np.load(pca_cache, allow_pickle=False)
            X_pca_cell = dd["features"]
            evr_sum = float(dd.get("evr_sum", np.array(np.nan)).item()) if "evr_sum" in dd.files else float("nan")
        else:
            print(f"[resume] fitting PCA(n={pca_dim}) on {token_acts.shape[0]} tokens (streaming) ...")
            n_pca = min(pca_dim, token_acts.shape[1], token_acts.shape[0] - 1)
            X_pca_cell, pca = fit_pca_and_aggregate_per_cell(
                token_acts, token_cell, n_cells=n_cells,
                n_components=n_pca, random_state=0,
            )
            evr_sum = float(pca.explained_variance_ratio_.sum())
            np.savez(pca_cache, features=X_pca_cell, labels=y, evr_sum=evr_sum)
        print(f"[resume]   X_pca_cell={X_pca_cell.shape}  evr_sum={evr_sum:.4f}")

        # ---- 4. SAE ----
        if _have(sae_ckpt):
            print(f"[resume] loading cached SAE {sae_ckpt}")
            ckpt = torch.load(sae_ckpt, map_location=args.device)
            sae = TopKSAE(ckpt["config"]["d_in"], ckpt["config"]["n_features"],
                          ckpt["config"]["k"]).to(args.device)
            sae.load_state_dict(ckpt["state_dict"])
            train_log = ckpt.get("training_log") or json.loads(
                (ld / "sae_training_log.json").read_text()
            )
        else:
            print(f"[resume] training SAE (dict={sae_dict}, k={args.sae_k}, "
                  f"epochs={args.sae_epochs}) ...")
            sae, train_log = train_topk_sae(
                token_acts, d_in=token_acts.shape[1], n_features=sae_dict,
                k=args.sae_k, batch_size=args.sae_batch,
                epochs=args.sae_epochs, lr=args.sae_lr, device=args.device,
            )
            torch.save({
                "state_dict": sae.state_dict(),
                "config": {"d_in": int(token_acts.shape[1]), "n_features": sae_dict,
                           "k": args.sae_k},
                "training_log": train_log,
            }, sae_ckpt)
            (ld / "sae_training_log.json").write_text(json.dumps(train_log, indent=2))

        # ---- 5. SAE per-cell features (the fix) ----
        if _have(sae_cell_cache):
            print(f"[resume] loading cached SAE per-cell features {sae_cell_cache}")
            ee = np.load(sae_cell_cache, allow_pickle=False)
            X_sae_cell = ee["features"]
        else:
            print(f"[resume] encoding SAE + aggregating per cell (streaming) ...")
            X_sae_cell = encode_and_aggregate_per_cell(
                sae, token_acts, token_cell, n_cells=n_cells,
                batch_size=args.sae_batch, device=args.device,
            )
            np.savez(sae_cell_cache, features=X_sae_cell, labels=y)
        print(f"[resume]   X_sae_cell={X_sae_cell.shape}")

        # ---- 6. Probes (PCA + SAE) ----
        if _have(probe_json):
            print(f"[resume] reusing {probe_json}")
            pj = json.loads(probe_json.read_text())
            pca_probe = pj["pca"]
            sae_probe = pj["sae"]
        else:
            print("[resume] running PCA + SAE LR probes (5 seeds each) ...")
            pca_probe = agg(probe(X_pca_cell, y, splits))
            sae_probe = agg(probe(X_sae_cell, y, splits))
            probe_json.write_text(json.dumps({
                "layer": layer,
                "n_pca_features": int(X_pca_cell.shape[1]),
                "n_sae_features": int(X_sae_cell.shape[1]),
                "pca": pca_probe, "sae": sae_probe,
            }, indent=2))
        print(f"[resume]   PCA probe acc = {pca_probe['accuracy_mean']:.4f} "
              f"± {pca_probe['accuracy_std']:.4f}")
        print(f"[resume]   SAE probe acc = {sae_probe['accuracy_mean']:.4f} "
              f"± {sae_probe['accuracy_std']:.4f}")

        # ---- 7. Ablation gap ----
        if _have(ablation_json):
            print(f"[resume] reusing {ablation_json}")
            ablation = json.loads(ablation_json.read_text())
        else:
            print("[resume] computing SAE-PCA ablation gap ...")
            ablation = sae_pca_ablation_gap(
                X_sae_cell, X_pca_cell, y, splits,
                K_grid=args.ablation_K_grid,
            )
            ablation_json.write_text(json.dumps(ablation, indent=2))

        pca_summary[layer] = pca_probe
        sae_summary[layer] = {
            "training_log": train_log,
            "cell_type_probe": sae_probe,
        }
        ablation_summary[layer] = ablation

        # Free memory between layers
        del token_acts, X_pca_cell, X_sae_cell, sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- 8. Report + plot ----
    (out_dir / "svd_diag.json").write_text(json.dumps(svd_results, indent=2))
    summary = extract_summary_fields(
        recipe="evo_1_8k__genomic_benchmarks", n_cells=n_cells,
        d_model=d_model, n_layers=n_layers,
        layer_names=sae_layers, svd=svd_results,
        pca_probe=pca_summary, sae_summary=sae_summary,
        ablation=ablation_summary,
    )
    (out_dir / "phase7_summary.json").write_text(json.dumps(summary, indent=2))
    write_phase7_report(out_dir / "REPORT.md", summary)
    try:
        plot_phase7_summary(out_dir / "phase7_summary.png", summary)
    except Exception as e:
        print(f"[resume] WARN: plot failed ({e}); JSON+REPORT.md still written")
    print(f"\n[resume] DONE — see {out_dir}/REPORT.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
