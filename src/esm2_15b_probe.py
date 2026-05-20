"""esm2_15b_probe.py — ESM-2 15B end-to-end PCA + SAE probing on DeepLoc.

Flat script per phase 7 design (no bio_fm_probe toolkit). Mirrors evo2_probe.py
but for ESM-2 15B on the protein DeepLoc subcellular-localization benchmark.

Pipeline (same flow as evo2_probe.py):
  1. Load ESM-2 15B (facebook/esm2_t48_15B_UR50D) via HF EsmModel + fp16.
  2. Load DeepLoc FASTA, parse header `|Location|` per ESM convention.
  3. Tokenize + per-batch forward with hooks on target layers; collect
     token-level activations.
  4. Per layer:
       (a) SVD spectrum diagnostic
       (b) PCA fit on tokens → mean-pool per sample → PCA-probe (5 seeds)
       (c) TopK SAE training (expansion=32× by default) → encode →
           mean-pool per sample → SAE-probe (5 seeds)
       (d) SAE - PCA ablation gap + random-feature ablation null
  5. Write per-layer JSON outputs + per-recipe REPORT.md + summary plot.

Output dir: results/esm2_15b__deeploc/ (default).

Memory notes (A6000 48 GB):
  - ESM-2 15B fp16 weights ≈ 30 GB
  - Forward batch=2 seq=1024 ≈ 5-10 GB activations
  - Use batch_size=2 by default; reduce to 1 if OOM
"""
from __future__ import annotations

import argparse
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
ROOT = THIS.parent.parent
sys.path.insert(0, str(THIS.parent))

from common_sae import (  # noqa: E402
    aggregate_per_cell, agg, encode_batched, extract_summary_fields,
    extract_tokens_from_iter,
    make_splits, plot_phase7_summary, probe, sae_pca_ablation_gap,
    svd_spectrum_diagnostic, train_topk_sae, write_phase7_report,
)


MAX_LEN_DEFAULT = 1024


# Standard DeepLoc 2.0 single-location set
DEEPLOC_CLASSES = [
    "Cytoplasm", "Nucleus", "Extracellular", "Cell.membrane",
    "Mitochondrion", "Plastid", "Endoplasmic.reticulum", "Lysosome/Vacuole",
    "Golgi.apparatus", "Peroxisome",
]


# ============================================================================
# ESM-2 model loader and forward hook
# ============================================================================
def load_esm2_15b(model_dir: str, device: str = "cuda", fp16: bool = True):
    """Load ESM-2 15B from HF or local dir. Returns
    (model, tokenizer, d_model, n_layers)."""
    from transformers import AutoTokenizer, EsmModel

    print(f"[esm2-15b] loading {model_dir} (fp16={fp16})")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = EsmModel.from_pretrained(
        model_dir,
        torch_dtype=torch.float16 if fp16 else torch.float32,
        add_pooling_layer=False,
    )
    model.to(device).eval()
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    print(f"[esm2-15b] loaded: n_layers={n_layers}, d_model={d_model}")
    return model, tok, int(d_model), int(n_layers)


def iter_esm2_activations(
    model, input_ids: np.ndarray, attn_mask: np.ndarray,
    target_layers: List[str], batch_size: int, device: str,
) -> Iterator[Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
    """Forward all samples, hook target_layers, yield per-batch
    (captured, valid_mask).

    valid_mask = True for real AA tokens (excludes pad AND CLS at position 0).
    """
    encoder = model.encoder
    n_layers = len(encoder.layer)

    captured: Dict[str, torch.Tensor] = {}
    handles = []

    if "layer_00_input" in target_layers:
        def pre_hook(_m, args, kwargs):
            captured["layer_00_input"] = args[0].detach().float()
        handles.append(
            encoder.register_forward_pre_hook(pre_hook, with_kwargs=True)
        )

    target_block_idxs = []
    for name in target_layers:
        if name == "layer_00_input":
            continue
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
        handles.append(encoder.layer[idx].register_forward_hook(make_hook(name)))

    n_samples = input_ids.shape[0]
    try:
        for start in tqdm(range(0, n_samples, batch_size),
                          desc="esm2-15b forward", unit="batch"):
            end = min(start + batch_size, n_samples)
            ids = torch.from_numpy(input_ids[start:end]).to(device)
            mask = torch.from_numpy(attn_mask[start:end]).to(device)
            with torch.no_grad():
                model(input_ids=ids, attention_mask=mask)
            valid = (mask == 1).clone()
            valid[:, 0] = False   # exclude CLS at position 0
            yield dict(captured), valid
            captured.clear()
    finally:
        for h in handles:
            h.remove()


# ============================================================================
# DeepLoc loader (flat, no bio_fm_probe imports)
# ============================================================================
def load_deeploc(path: Path) -> Tuple[List[str], np.ndarray]:
    seqs: List[str] = []
    loc_strs: List[str] = []
    with path.open() as f:
        cur_seq: List[str] = []
        cur_loc = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_loc is not None and cur_seq:
                    seqs.append("".join(cur_seq))
                    loc_strs.append(cur_loc)
                parts = line[1:].split("|")
                cur_loc = parts[1] if len(parts) > 1 else "Unknown"
                cur_seq = []
            else:
                cur_seq.append(line)
        if cur_loc is not None and cur_seq:
            seqs.append("".join(cur_seq))
            loc_strs.append(cur_loc)
    keep = [i for i, loc in enumerate(loc_strs) if loc in DEEPLOC_CLASSES]
    seqs = [seqs[i] for i in keep]
    loc_strs = [loc_strs[i] for i in keep]
    cls_to_idx = {c: i for i, c in enumerate(DEEPLOC_CLASSES)}
    labels = np.array([cls_to_idx[l] for l in loc_strs], dtype=np.int64)
    return seqs, labels


# ============================================================================
# Main
# ============================================================================
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="checkpoints/esm2_t48_15B_UR50D")
    p.add_argument("--data_path", default="data/deeploc/deeploc_data.fasta")
    p.add_argument("--sae_layers", nargs="+", default=None,
                   help="default: input, mid, final")
    p.add_argument("--sae_expansion", type=float, default=32.0)
    p.add_argument("--sae_k", type=int, default=32)
    p.add_argument("--sae_epochs", type=int, default=20)
    p.add_argument("--sae_batch", type=int, default=4096)
    p.add_argument("--sae_lr", type=float, default=1e-3)
    p.add_argument("--pca_dim", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=14000)
    p.add_argument("--max_len", type=int, default=MAX_LEN_DEFAULT)
    p.add_argument("--extract_batch_size", type=int, default=2)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--ablation_K_grid", type=int, nargs="+",
                   default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    p.add_argument("--no_fp16", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None,
                   help="default: results/esm2_15b__deeploc/")
    args = p.parse_args()

    out_dir = (Path(args.out) if args.out
               else ROOT / "results" / "esm2_15b__deeploc")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[esm2-15b] output -> {out_dir}")

    data_path = (Path(args.data_path) if Path(args.data_path).is_absolute()
                 else ROOT / args.data_path)
    seqs, labels = load_deeploc(data_path)
    print(f"[esm2-15b] loaded {len(seqs)} sequences from {data_path}, "
          f"n_classes={len(np.unique(labels))}, "
          f"mean_len={int(np.mean([len(s) for s in seqs]))}")

    # Truncate sequences over MAX_LEN
    seqs = [s[:args.max_len] for s in seqs]

    if len(seqs) > args.max_samples:
        rng = np.random.default_rng(args.seeds[0])
        idx = np.sort(rng.choice(len(seqs), args.max_samples, replace=False))
        seqs = [seqs[i] for i in idx]
        labels = labels[idx]
        print(f"[esm2-15b] subsampled to {len(seqs)}")

    y_classes, y = np.unique(labels, return_inverse=True)

    model, tok, d_model, n_layers = load_esm2_15b(
        args.model_dir, device=args.device, fp16=not args.no_fp16,
    )

    # Tokenize whole dataset
    print(f"[esm2-15b] tokenizing (max_len={args.max_len}) ...")
    encoded = tok(
        seqs, padding="max_length", truncation=True,
        max_length=args.max_len, return_tensors="np",
    )
    input_ids = encoded["input_ids"].astype(np.int64)
    attn_mask = encoded["attention_mask"].astype(np.int64)
    print(f"[esm2-15b] tokenized: input_ids={input_ids.shape}")

    if args.sae_layers is None:
        mid = max(1, (n_layers + 1) // 2)
        sae_layers = ["layer_00_input", f"layer_{mid:02d}", f"layer_{n_layers:02d}"]
    else:
        sae_layers = args.sae_layers
    print(f"[esm2-15b] SAE layers: {sae_layers}")

    sae_dict = max(64, int(round(args.sae_expansion * d_model)))
    pca_dim = args.pca_dim or min(sae_dict, d_model)
    print(f"[esm2-15b] SAE dict_size = {sae_dict} "
          f"(expansion = {sae_dict / d_model:.2f}x over d={d_model}); "
          f"PCA dim = {pca_dim}")

    splits = make_splits(y, args.seeds)

    svd_results = {}
    sae_summary = {}

    for layer in sae_layers:
        print(f"\n{'=' * 60}\n[esm2-15b] === layer {layer} ===\n{'=' * 60}")
        layer_dir = out_dir / layer
        layer_dir.mkdir(exist_ok=True)
        acts_cache = layer_dir / "token_activations.npz"
        sae_ckpt = layer_dir / "sae.pt"

        # ---- Extract token activations ----
        if acts_cache.exists() and not args.force:
            print(f"[esm2-15b] loading cached activations from {acts_cache}")
            d = np.load(acts_cache, allow_pickle=False)
            token_acts, token_cell = d["acts"], d["cell_idx"]
        else:
            t0 = time.time()
            token_acts, token_cell = extract_tokens_from_iter(
                iter_esm2_activations(
                    model, input_ids, attn_mask,
                    target_layers=[layer],
                    batch_size=args.extract_batch_size, device=args.device,
                ),
                target_layer=layer, n_samples=len(seqs),
            )
            np.savez(acts_cache, acts=token_acts, cell_idx=token_cell)
            print(f"[esm2-15b] extracted {token_acts.shape} in "
                  f"{time.time() - t0:.0f}s")

        n_cells = len(seqs)

        # ---- SVD diagnostic ----
        svd_results[layer] = svd_spectrum_diagnostic(token_acts)
        print(f"[esm2-15b] SVD: PR="
              f"{svd_results[layer]['participation_ratio']:.1f}, "
              f"k95={svd_results[layer]['k95']}, "
              f"k99={svd_results[layer]['k99']}")

        # ---- PCA fit ----
        print(f"[esm2-15b] fitting PCA(n={pca_dim}) on "
              f"{token_acts.shape[0]} tokens ...")
        n_pca = min(pca_dim, token_acts.shape[1], token_acts.shape[0] - 1)
        pca = PCA(n_components=n_pca, random_state=0)
        tok_pca = pca.fit_transform(token_acts).astype(np.float32)
        X_pca_cell = aggregate_per_cell(tok_pca, token_cell, n_cells=n_cells)
        np.savez(layer_dir / "per_cell_pca_features.npz",
                 features=X_pca_cell, labels=y,
                 evr_sum=pca.explained_variance_ratio_.sum())

        # ---- SAE train ----
        if sae_ckpt.exists() and not args.force:
            print(f"[esm2-15b] loading cached SAE from {sae_ckpt}")
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

        # ---- SAE encode ----
        print(f"[esm2-15b] encoding SAE features ...")
        tok_codes = encode_batched(sae, token_acts,
                                   batch_size=args.sae_batch,
                                   device=args.device)
        X_sae_cell = aggregate_per_cell(tok_codes, token_cell, n_cells=n_cells)
        np.savez(layer_dir / "per_cell_sae_features.npz",
                 features=X_sae_cell, labels=y)

        # ---- Probes ----
        pca_probe = agg(probe(X_pca_cell, y, splits))
        sae_probe = agg(probe(X_sae_cell, y, splits))
        (layer_dir / "pca_probe.json").write_text(
            json.dumps({"layer": layer, "n_features": int(X_pca_cell.shape[1]),
                        "results": pca_probe}, indent=2))
        (layer_dir / "sae_probe.json").write_text(
            json.dumps({"layer": layer, "n_features": int(X_sae_cell.shape[1]),
                        "k_active": args.sae_k, "results": sae_probe}, indent=2))
        print(f"[esm2-15b]   PCA probe acc = {pca_probe['accuracy_mean']:.4f} "
              f"± {pca_probe['accuracy_std']:.4f}")
        print(f"[esm2-15b]   SAE probe acc = {sae_probe['accuracy_mean']:.4f} "
              f"± {sae_probe['accuracy_std']:.4f}")

        # ---- Ablation gap + null ----
        k99 = svd_results[layer]["k99"]
        K_grid = [K for K in args.ablation_K_grid if K <= max(k99, 8)]
        if not K_grid:
            K_grid = [1, 2, 4, 8]
        print(f"[esm2-15b]   ablation K-grid (truncated at k99={k99}): {K_grid}")
        gap = sae_pca_ablation_gap(
            X_sae_cell, X_pca_cell, y, splits,
            K_grid=K_grid, include_random_null=True,
        )
        (layer_dir / "ablation_gap.json").write_text(json.dumps(gap, indent=2))
        print(f"[esm2-15b]   any K significant = "
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

    (out_dir / "svd_diag.json").write_text(json.dumps(svd_results, indent=2))
    (out_dir / "phase7_summary.json").write_text(
        json.dumps({"model": "esm2_t48_15B_UR50D", "dataset": "deeploc",
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
                        model_name="ESM-2 15B", dataset_name="DeepLoc")
    write_phase7_report(out_dir, "esm2_t48_15B_UR50D", "deeploc",
                        d_model, n_layers, sae_layers, sae_dict, args.sae_k,
                        args.sae_expansion, pca_dim, args.seeds, len(seqs),
                        sae_summary, svd_results)

    print(f"\n[esm2-15b] === DONE === outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
