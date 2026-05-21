"""evo2_probe.py — Evo-2 7B end-to-end PCA + SAE probing on a DNA benchmark.

Flat script per the phase 7 design (no bio_fm_probe toolkit). The whole flow
in one file:

  1. Load Evo-2 7B (arcinstitute/evo2_7b) via HF AutoModel(trust_remote_code).
  2. Load dataset (default: genomic_benchmarks human_nontata_promoters CSVs).
  3. Tokenize + per-batch forward with hooks on target layers; collect
     token-level activations (N_tokens, d_model) per layer.
  4. Per layer:
       (a) SVD spectrum diagnostic
       (b) PCA fit on tokens → mean-pool per sample → PCA-probe (5 seeds)
       (c) TopK SAE training (expansion=4× by default) → encode →
           mean-pool per sample → SAE-probe (5 seeds)
       (d) SAE - PCA ablation gap + random-feature ablation null
  5. Write per-layer JSON outputs + per-recipe REPORT.md + summary plot.

Output dir: results/evo2_7b__genomic_benchmarks/ (default; --out overrides).

Usage:
    python src/evo2_probe.py
    python src/evo2_probe.py --sae_expansion 32 --sae_layers layer_00_input layer_16 layer_32
    python src/evo2_probe.py --max_samples 5000 --force   # cheap dry run

Compute on A6000 fp16: ~25-30h for default 3-layer 32x-expansion full-data run.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent.parent  # script lives at src/phase7/ → 3 hops to project root
sys.path.insert(0, str(THIS.parent))


def _resolve(p: str) -> str:
    """Anchor a relative path to ROOT so the script works from any cwd."""
    path = Path(p)
    return str(path if path.is_absolute() else ROOT / path)

from common_sae import (  # noqa: E402
    aggregate_per_cell, agg, encode_batched, extract_summary_fields,
    extract_tokens_from_iter,
    make_splits, plot_phase7_summary, probe, sae_pca_ablation_gap,
    svd_spectrum_diagnostic, train_topk_sae, write_phase7_report,
)


# ============================================================================
# Evo-2 model loader and forward hook (fresh, no adapter abstraction)
# ============================================================================
MAX_LEN_DEFAULT = 8192


def load_evo2(model_dir: str, device: str = "cuda", fp16: bool = True):
    """Load Evo-2 from HF or local dir. Returns (model, tokenizer, d_model, n_layers,
    blocks_attr_path).

    Evo-2 7B uses StripedHyena-2 with custom modeling code; needs
    trust_remote_code=True.
    """
    from transformers import AutoModel, AutoTokenizer

    print(f"[evo2] loading {model_dir} (fp16={fp16})")
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_dir, trust_remote_code=True,
        torch_dtype=torch.float16 if fp16 else torch.float32,
        output_hidden_states=False,
    )
    model.to(device).eval()

    # Locate block list. StripedHyena-2 HF wrappers vary; try common paths.
    blocks_attr = None
    for attr_path in (("backbone", "blocks"), ("backbone", "layers"),
                      ("model", "blocks"), ("model", "layers"),
                      ("blocks",), ("layers",)):
        obj = model
        ok = True
        for a in attr_path:
            obj = getattr(obj, a, None)
            if obj is None:
                ok = False
                break
        if ok:
            blocks_attr = attr_path
            n_layers = len(obj)
            break
    if blocks_attr is None:
        raise SystemExit(
            "[evo2] couldn't locate the block list on the model. Check "
            "the HF repo's modeling code and edit load_evo2 to point at the "
            "correct attribute path.")

    d_model = getattr(model.config, "d_model",
                      getattr(model.config, "hidden_size", None))
    if d_model is None:
        for p in model.parameters():
            if p.dim() == 2:
                d_model = p.shape[-1]
                break
    print(f"[evo2] loaded: n_layers={n_layers}, d_model={d_model}, "
          f"blocks_attr={blocks_attr}")
    return model, tok, int(d_model), int(n_layers), blocks_attr


def iter_evo2_activations(
    model, tokenizer, blocks_attr, input_ids: np.ndarray, attn_mask: np.ndarray,
    target_layers: List[str], batch_size: int, device: str,
) -> Iterator[Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
    """Forward all samples, hook target_layers, yield per-batch
    (captured, valid_mask).

    captured contains only target_layers (saves memory vs hooking everything).
    valid_mask is (B, seq) bool where True = real nucleotide token (no CLS for
    Evo, just exclude pad).
    """
    # Resolve block list
    blocks = model
    for a in blocks_attr:
        blocks = getattr(blocks, a)
    n_layers = len(blocks)

    captured: Dict[str, torch.Tensor] = {}
    handles = []

    if "layer_00_input" in target_layers:
        def pre_hook(_m, args):
            if not args:
                return
            captured["layer_00_input"] = args[0].detach().float()
        handles.append(blocks[0].register_forward_pre_hook(pre_hook))

    target_block_idxs = []
    for name in target_layers:
        if name == "layer_00_input":
            continue
        # parse "layer_NN"
        idx = int(name.split("_")[1]) - 1
        if not (0 <= idx < n_layers):
            raise ValueError(f"layer {name!r} out of range (n_layers={n_layers})")
        target_block_idxs.append((idx, name))

    def make_hook(name):
        def h(_m, _args, output):
            acts = output[0] if isinstance(output, tuple) else output
            captured[name] = acts.detach().float()
        return h

    for idx, name in target_block_idxs:
        handles.append(blocks[idx].register_forward_hook(make_hook(name)))

    n_samples = input_ids.shape[0]
    try:
        for start in tqdm(range(0, n_samples, batch_size),
                          desc="evo2 forward", unit="batch"):
            end = min(start + batch_size, n_samples)
            ids = torch.from_numpy(input_ids[start:end]).to(device)
            mask = torch.from_numpy(attn_mask[start:end]).to(device)
            with torch.no_grad():
                # StripedHyena doesn't accept attention_mask; only pass input_ids
                model(input_ids=ids)
            valid = (mask == 1)
            yield dict(captured), valid
            captured.clear()
    finally:
        for h in handles:
            h.remove()


# ============================================================================
# Dataset loader (CSV-flat, no bio_fm_probe imports)
# ============================================================================
def load_genomic_benchmarks(data_dir: Path) -> Tuple[List[str], np.ndarray]:
    seqs, labels = [], []
    for split in ("train", "test"):
        csv_path = data_dir / f"{split}.csv"
        if not csv_path.exists():
            raise SystemExit(
                f"missing {csv_path}; run src/download_genomic_benchmarks.py first")
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                seqs.append(row["sequence"])
                labels.append(int(row["label"]))
    return seqs, np.asarray(labels, dtype=np.int64)


# ============================================================================
# Main
# ============================================================================
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="checkpoints/evo2_7b")
    p.add_argument("--data_dir",
                   default="data/genomic_benchmarks/human_nontata_promoters")
    p.add_argument("--sae_layers", nargs="+", default=None,
                   help="default: input, mid, final")
    p.add_argument("--sae_expansion", type=float, default=4.0,
                   help="SAE dict_size = expansion × d_model. Default 4× for "
                        "cross-FM parity with phase 5 (scgpt/hyenadna).")
    p.add_argument("--sae_k", type=int, default=32)
    p.add_argument("--sae_epochs", type=int, default=20)
    p.add_argument("--sae_batch", type=int, default=4096)
    p.add_argument("--sae_lr", type=float, default=1e-3)
    p.add_argument("--pca_dim", type=int, default=None,
                   help="default: matched to SAE dict for fair comparison")
    p.add_argument("--max_samples", type=int, default=20000)
    p.add_argument("--max_len", type=int, default=MAX_LEN_DEFAULT)
    p.add_argument("--extract_batch_size", type=int, default=4)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--ablation_K_grid", type=int, nargs="+",
                   default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    p.add_argument("--no_fp16", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="ignore caches and re-extract / re-train SAE")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None,
                   help="default: results/evo2_7b__genomic_benchmarks/")
    args = p.parse_args()

    # Output dir
    out_dir = (Path(_resolve(args.out)) if args.out
               else ROOT / "results" / "evo2_7b__genomic_benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[evo2] output -> {out_dir}")

    # Data
    data_dir = Path(_resolve(args.data_dir))
    seqs, labels = load_genomic_benchmarks(data_dir)
    print(f"[evo2] loaded {len(seqs)} sequences from {data_dir}, "
          f"n_classes={len(np.unique(labels))}, "
          f"mean_len={int(np.mean([len(s) for s in seqs]))}")

    if len(seqs) > args.max_samples:
        rng = np.random.default_rng(args.seeds[0])
        idx = np.sort(rng.choice(len(seqs), args.max_samples, replace=False))
        seqs = [seqs[i] for i in idx]
        labels = labels[idx]
        print(f"[evo2] subsampled to {len(seqs)}")

    y_classes, y = np.unique(labels, return_inverse=True)

    # Load model
    model, tok, d_model, n_layers, blocks_attr = load_evo2(
        _resolve(args.model_dir), device=args.device, fp16=not args.no_fp16,
    )

    # Tokenize once for whole dataset
    print(f"[evo2] tokenizing (max_len={args.max_len}) ...")
    encoded = tok(
        seqs, padding="max_length", truncation=True,
        max_length=args.max_len, return_tensors="np",
    )
    input_ids = encoded["input_ids"].astype(np.int64)
    if "attention_mask" in encoded:
        attn_mask = encoded["attention_mask"].astype(np.int64)
    else:
        pad_id = getattr(tok, "pad_token_id", 0) or 0
        attn_mask = (input_ids != pad_id).astype(np.int64)
    print(f"[evo2] tokenized: input_ids={input_ids.shape}")

    # Default SAE layers: input + mid + final
    if args.sae_layers is None:
        mid = max(1, (n_layers + 1) // 2)
        sae_layers = ["layer_00_input", f"layer_{mid:02d}", f"layer_{n_layers:02d}"]
    else:
        sae_layers = args.sae_layers
    print(f"[evo2] SAE layers: {sae_layers}")

    sae_dict = max(64, int(round(args.sae_expansion * d_model)))
    pca_dim = args.pca_dim or min(sae_dict, d_model)
    print(f"[evo2] SAE dict_size = {sae_dict} "
          f"(expansion = {sae_dict / d_model:.2f}x over d={d_model}); "
          f"PCA dim = {pca_dim}")

    splits = make_splits(y, args.seeds)

    # Per-layer processing
    svd_results = {}
    sae_summary = {}

    for layer in sae_layers:
        print(f"\n{'=' * 60}\n[evo2] === layer {layer} ===\n{'=' * 60}")
        layer_dir = out_dir / layer
        layer_dir.mkdir(exist_ok=True)
        acts_cache = layer_dir / "token_activations.npz"
        sae_ckpt = layer_dir / "sae.pt"

        # ---- Extract token activations ----
        if acts_cache.exists() and not args.force:
            print(f"[evo2] loading cached activations from {acts_cache}")
            d = np.load(acts_cache, allow_pickle=False)
            token_acts, token_cell = d["acts"], d["cell_idx"]
        else:
            t0 = time.time()
            token_acts, token_cell = extract_tokens_from_iter(
                iter_evo2_activations(
                    model, tok, blocks_attr, input_ids, attn_mask,
                    target_layers=[layer],
                    batch_size=args.extract_batch_size, device=args.device,
                ),
                target_layer=layer, n_samples=len(seqs),
            )
            np.savez(acts_cache, acts=token_acts, cell_idx=token_cell)
            print(f"[evo2] extracted {token_acts.shape} in {time.time() - t0:.0f}s")

        n_cells = len(seqs)

        # ---- SVD diagnostic ----
        svd_results[layer] = svd_spectrum_diagnostic(token_acts)
        print(f"[evo2] SVD: PR={svd_results[layer]['participation_ratio']:.1f}, "
              f"k95={svd_results[layer]['k95']}, k99={svd_results[layer]['k99']}")

        # ---- PCA fit + per-sample features ----
        print(f"[evo2] fitting PCA(n={pca_dim}) on {token_acts.shape[0]} tokens ...")
        n_pca = min(pca_dim, token_acts.shape[1], token_acts.shape[0] - 1)
        pca = PCA(n_components=n_pca, random_state=0)
        tok_pca = pca.fit_transform(token_acts).astype(np.float32)
        X_pca_cell = aggregate_per_cell(tok_pca, token_cell, n_cells=n_cells)
        np.savez(layer_dir / "per_cell_pca_features.npz",
                 features=X_pca_cell, labels=y,
                 evr_sum=pca.explained_variance_ratio_.sum())

        # ---- SAE train ----
        if sae_ckpt.exists() and not args.force:
            print(f"[evo2] loading cached SAE from {sae_ckpt}")
            ckpt = torch.load(sae_ckpt, map_location=args.device)
            from common_sae import TopKSAE
            sae = TopKSAE(ckpt["config"]["d_in"],
                          ckpt["config"]["n_features"],
                          ckpt["config"]["k"]).to(args.device)
            sae.load_state_dict(ckpt["state_dict"])
            train_log = json.loads(
                (layer_dir / "sae_training_log.json").read_text())
        else:
            sae, train_log = train_topk_sae(
                token_acts, d_in=d_model, n_features=sae_dict, k=args.sae_k,
                batch_size=args.sae_batch, epochs=args.sae_epochs,
                lr=args.sae_lr, device=args.device,
            )
            torch.save({
                "state_dict": sae.state_dict(),
                "config": {"d_in": d_model, "n_features": sae_dict, "k": args.sae_k},
                "layer": layer,
            }, sae_ckpt)
            (layer_dir / "sae_training_log.json").write_text(
                json.dumps(train_log, indent=2))

        # ---- SAE encode + per-sample features ----
        print(f"[evo2] encoding SAE features ...")
        tok_codes = encode_batched(sae, token_acts,
                                   batch_size=args.sae_batch,
                                   device=args.device)
        X_sae_cell = aggregate_per_cell(tok_codes, token_cell, n_cells=n_cells)
        np.savez(layer_dir / "per_cell_sae_features.npz",
                 features=X_sae_cell, labels=y)

        # ---- PCA-probe + SAE-probe ----
        pca_probe = agg(probe(X_pca_cell, y, splits))
        sae_probe = agg(probe(X_sae_cell, y, splits))
        (layer_dir / "pca_probe.json").write_text(
            json.dumps({"layer": layer, "n_features": int(X_pca_cell.shape[1]),
                        "results": pca_probe}, indent=2))
        (layer_dir / "sae_probe.json").write_text(
            json.dumps({"layer": layer, "n_features": int(X_sae_cell.shape[1]),
                        "k_active": args.sae_k, "results": sae_probe}, indent=2))
        print(f"[evo2]   PCA probe acc = {pca_probe['accuracy_mean']:.4f} "
              f"± {pca_probe['accuracy_std']:.4f}")
        print(f"[evo2]   SAE probe acc = {sae_probe['accuracy_mean']:.4f} "
              f"± {sae_probe['accuracy_std']:.4f}")

        # ---- SAE - PCA ablation gap + random null ----
        # Truncate K-grid at k99 from SVD to avoid past-effective-rank artifacts
        k99 = svd_results[layer]["k99"]
        K_grid = [K for K in args.ablation_K_grid if K <= max(k99, 8)]
        if not K_grid:
            K_grid = [1, 2, 4, 8]
        print(f"[evo2]   ablation K-grid (truncated at k99={k99}): {K_grid}")
        gap = sae_pca_ablation_gap(
            X_sae_cell, X_pca_cell, y, splits,
            K_grid=K_grid, include_random_null=True,
        )
        (layer_dir / "ablation_gap.json").write_text(json.dumps(gap, indent=2))
        print(f"[evo2]   any K significant (2σ above null) = "
              f"{gap.get('any_K_significant', False)}")

        sae_summary[layer] = {
            "d_model": d_model,
            "n_tokens": int(token_acts.shape[0]),
            "var_explained_final": train_log["epoch_var_explained"][-1],
            "dead_features_ever": train_log["dead_features_ever"],
            "pca_probe": pca_probe,
            "sae_probe": sae_probe,
            "k99": k99,
            **extract_summary_fields(gap),
        }
        del token_acts, X_pca_cell, X_sae_cell, tok_codes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Cross-layer outputs
    (out_dir / "svd_diag.json").write_text(json.dumps(svd_results, indent=2))
    (out_dir / "phase7_summary.json").write_text(
        json.dumps({"model": "evo2_7b", "dataset": "genomic_benchmarks",
                    "d_model": d_model, "n_layers_total": n_layers,
                    "sae_layers_probed": sae_layers,
                    "sae_dict": sae_dict, "sae_k": args.sae_k,
                    "sae_expansion": args.sae_expansion,
                    "pca_dim": pca_dim,
                    "seeds": args.seeds,
                    "n_samples": len(seqs),
                    "per_layer": sae_summary},
                   indent=2))

    # Plot + REPORT (shared helpers in common_sae.py — phase 7.1 protocol-aligned)
    plot_phase7_summary(out_dir, sae_layers, sae_summary,
                        model_name="Evo-2 7B", dataset_name="genomic_benchmarks")
    write_phase7_report(out_dir, "evo2_7b", "genomic_benchmarks",
                        d_model, n_layers, sae_layers, sae_dict, args.sae_k,
                        args.sae_expansion, pca_dim, args.seeds, len(seqs),
                        sae_summary, svd_results)

    print(f"\n[evo2] === DONE === outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
