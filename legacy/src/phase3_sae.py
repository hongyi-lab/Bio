"""phase3: TopK sparse autoencoder on scGPT gene-token activations.

Motivation from phase 2:
  - PCA-50 on raw log1p genes hits acc = 0.940 ± 0.008.
  - scGPT's best CLS layer (layer 3) hits 0.909 ± 0.007 (-3.1 pp vs PCA-50).
  - scGPT's mean-pool peaks at layer_00_input (0.932) and decays monotonically
    to layer 12 (0.819).
  -> scGPT's transformer blocks DO NOT add cell-type-discriminative capacity
     above the input embedding lookup. They actively erode it as depth grows.

  So scGPT IS computing something, but it is not (just) cell-type. Phase 3 asks:
  what is it, and can we recover interpretable, sparse features from it?

For each target layer we:
  1. Forward all cells through scGPT, collect every non-pad, non-CLS gene-token
     activation at that layer -> flat (N_tokens, d_model) array.
  2. Train a TopK SAE (Anthropic-style) on those activations.
  3. Sanity: reconstruction variance-explained, sparsity, dead-feature count.
  4. Per-cell mean SAE activation -> LR probe on cell type, 5 seeds, mean +/- std.
     Compare to phase2 dense probe at the same layer.
  5. (optional) Gene-set correlations: per cell, score known gene sets (HALLMARK_*,
     KEGG_*, etc.) on raw log1p expression; for each SAE feature correlate its
     per-cell activation with each gene-set score. Top-K features per set are
     candidate "gene-set neurons".
  6. (optional) Feature ablation: zero out the top-K cell-type-correlated SAE
     features in reconstruction space, re-probe cell type, measure drop.

Outputs (under results/phase3/<layer>/):
  sae.pt                       state_dict + config
  training_log.json            loss + var_explained + dead_features per epoch
  per_cell_features.npz        (n_cells, n_features) mean SAE activation
  cell_type_probe.json         LR probe on SAE features, 5 seeds
  gene_set_correlations.json   if --gene_sets_dir provided
  feature_ablation.json        if --do_ablation
  curves.png                   training curves
  SUMMARY.md                   per-layer digest

Usage:
  python src/phase3_sae.py                                       # default layers
  python src/phase3_sae.py --layers layer_00_input layer_03 layer_12
  python src/phase3_sae.py --dict_size 2048 --k 32 --epochs 20 --batch_size 4096
  python src/phase3_sae.py --gene_sets_dir gene_sets/
  python src/phase3_sae.py --do_ablation --ablation_k 10
  python src/phase3_sae.py --force_extract --force_retrain      # nuke caches
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(THIS.parent))

from load_scgpt import (  # noqa: E402
    MAX_LEN, N_BINS, PAD_TOKEN, PAD_VALUE,
    load_scgpt_model, preprocess_adata_for_scgpt,
)


# ============================================================================
# TopK SAE
# ============================================================================
class TopKSAE(nn.Module):
    """Anthropic-style TopK sparse autoencoder.

    - encoder: linear, then keep top-k pre-activations, ReLU
    - decoder: linear with unit-norm columns (rows here since W_dec is (n_feat, d_in))
    - loss: pure MSE; sparsity enforced exactly by TopK -> no L1 coefficient to tune

    Conventions:
      x  shape (B, d_in)
      pre = (x - b_dec) @ W_enc + b_enc     shape (B, n_feat)
      z   = ReLU(TopK(pre))                  shape (B, n_feat)
      x_recon = z @ W_dec + b_dec            shape (B, d_in)
    """

    def __init__(self, d_in: int, n_features: int, k: int):
        super().__init__()
        self.d_in = d_in
        self.n_features = n_features
        self.k = k
        self.W_enc = nn.Parameter(torch.randn(d_in, n_features) / (d_in ** 0.5))
        W_dec = self.W_enc.detach().clone().T.contiguous()       # (n_feat, d_in)
        W_dec = W_dec / W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec = nn.Parameter(W_dec)
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = self.encode_pre(x)
        topk_vals, topk_idx = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, topk_idx, F.relu(topk_vals))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        norms = self.W_dec.data.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec.data /= norms


def train_topk_sae(
    activations: np.ndarray,
    d_in: int,
    n_features: int,
    k: int,
    batch_size: int,
    epochs: int,
    lr: float,
    device: str,
    log_every: int = 100,
) -> Tuple[TopKSAE, Dict]:
    """Train TopK SAE on a flat (N_tokens, d_in) activation buffer.

    Loss: pure MSE. Decoder column-normalized after every step (Anthropic recipe).
    Returns (model, training_log).
    """
    sae = TopKSAE(d_in, n_features, k).to(device)
    sae.normalize_decoder()  # safety: ensure decoder unit-norm at init
    opt = torch.optim.Adam(sae.parameters(), lr=lr)

    n_tokens = activations.shape[0]
    n_batches = n_tokens // batch_size
    print(f"[sae] training: {n_tokens} tokens, {n_batches} batches/epoch, "
          f"dict={n_features}, k={k}, epochs={epochs}, bs={batch_size}")

    # Keep activations on CPU; index per batch onto GPU.
    # For 1.6M x 512 fp32 = ~3.2 GB. Fits in RAM, transfer per batch.
    act_t = torch.from_numpy(activations).float().contiguous()

    log: Dict[str, list] = {
        "epoch_loss": [], "epoch_var_explained": [],
        "epoch_dead_features": [], "epoch_l0_mean": [],
        "epoch_time_s": [], "config": {
            "d_in": d_in, "n_features": n_features, "k": k,
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "n_tokens": int(n_tokens),
        },
    }

    # Cumulative "ever-activated" counter for dead-feature accounting.
    ever_active = torch.zeros(n_features, dtype=torch.bool, device=device)
    var_total = float(act_t.var().item())

    for epoch in range(epochs):
        t0 = time.time()
        perm = torch.randperm(n_tokens)
        epoch_losses, epoch_l0 = [], []
        epoch_active = torch.zeros(n_features, dtype=torch.bool, device=device)

        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            x = act_t[idx].to(device, non_blocking=True)
            x_recon, z = sae(x)
            loss = F.mse_loss(x_recon, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sae.normalize_decoder()
            epoch_losses.append(loss.item())
            with torch.no_grad():
                active_this = (z > 0).any(dim=0)
                epoch_active |= active_this
                ever_active |= active_this
                epoch_l0.append(float((z > 0).float().sum(dim=-1).mean().item()))

        avg_loss = float(np.mean(epoch_losses))
        var_explained = 1.0 - avg_loss / max(var_total, 1e-12)
        alive_epoch = int(epoch_active.sum().item())
        dead_epoch = n_features - alive_epoch
        avg_l0 = float(np.mean(epoch_l0))
        log["epoch_loss"].append(avg_loss)
        log["epoch_var_explained"].append(var_explained)
        log["epoch_dead_features"].append(dead_epoch)
        log["epoch_l0_mean"].append(avg_l0)
        log["epoch_time_s"].append(time.time() - t0)
        print(f"[sae] epoch {epoch + 1:>2}/{epochs}  loss={avg_loss:.6f}  "
              f"var_exp={var_explained:.4f}  L0={avg_l0:.1f}  "
              f"dead(epoch)={dead_epoch}/{n_features}  "
              f"dt={log['epoch_time_s'][-1]:.1f}s")

    log["dead_features_ever"] = int(n_features - ever_active.sum().item())
    return sae, log


@torch.no_grad()
def encode_in_batches(sae: TopKSAE, activations: np.ndarray,
                      batch_size: int, device: str) -> np.ndarray:
    """Encode (N_tokens, d_in) -> (N_tokens, n_features) sparse codes."""
    sae.eval()
    out = np.zeros((activations.shape[0], sae.n_features), dtype=np.float32)
    for b in range(0, activations.shape[0], batch_size):
        x = torch.from_numpy(activations[b:b + batch_size]).float().to(device)
        z = sae.encode(x).cpu().numpy()
        out[b:b + batch_size] = z
    return out


# ============================================================================
# Activation extraction at a single target layer
# ============================================================================
def _is_valid_layer(name: str, n_blocks: int) -> bool:
    return name == "layer_00_input" or name in {
        f"layer_{i + 1:02d}" for i in range(n_blocks)
    }


def extract_token_activations(
    adata, model, vocab, target_layer: str,
    device: str, batch_size: int, gene_col: str = "gene_name",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward all cells, hook target_layer, collect every non-pad non-CLS
    gene-token activation.

    Returns:
        token_acts:     (N_tokens, d_model) float32
        token_cell_idx: (N_tokens,) int64 cell index
        token_gene_id:  (N_tokens,) int64 gene vocab id
    """
    from scgpt.tokenizer import tokenize_and_pad_batch

    captured: Dict[str, torch.Tensor] = {}
    handles = []
    encoder = model.transformer_encoder
    for attr in ("enable_nested_tensor", "use_nested_tensor"):
        if hasattr(encoder, attr):
            setattr(encoder, attr, False)
    n_blocks = len(encoder.layers)
    if not _is_valid_layer(target_layer, n_blocks):
        raise SystemExit(f"unknown layer name {target_layer!r}; expected "
                         f"layer_00_input or layer_01..layer_{n_blocks:02d}")

    def pre_hook(_module, inputs):
        captured["layer_00_input"] = inputs[0].detach()
    handles.append(encoder.register_forward_pre_hook(pre_hook))

    def make_hook(idx: int):
        name = f"layer_{idx + 1:02d}"
        def h(_m, _i, output):
            captured[name] = output.detach()
        return h
    for i, layer in enumerate(encoder.layers):
        handles.append(layer.register_forward_hook(make_hook(i)))

    if gene_col not in adata.var.columns:
        adata.var["gene_name"] = adata.var.index.astype(str)
        gene_col = "gene_name"
    genes = adata.var[gene_col].tolist()
    gene_ids_all = np.array(vocab(genes), dtype=int)
    binned = adata.layers["X_binned"]
    if hasattr(binned, "toarray"):
        binned = binned.toarray()
    binned = np.asarray(binned)
    pad_idx = vocab[PAD_TOKEN]

    acts_chunks, cell_chunks, gene_chunks = [], [], []

    try:
        for start in tqdm(range(0, adata.n_obs, batch_size),
                          desc=f"extract {target_layer}", unit="batch"):
            end = min(start + batch_size, adata.n_obs)
            batch_X = binned[start:end].astype(np.int64)
            tok = tokenize_and_pad_batch(
                batch_X, gene_ids_all, max_len=MAX_LEN,
                vocab=vocab, pad_token=PAD_TOKEN, pad_value=PAD_VALUE,
                append_cls=True, include_zero_gene=False,
            )
            input_gene_ids = tok["genes"].to(device)
            input_values = tok["values"].to(device).float()
            pad_mask = input_gene_ids.eq(pad_idx)

            with torch.no_grad():
                model._encode(input_gene_ids, input_values, src_key_padding_mask=pad_mask)

            acts = captured[target_layer]  # (B, seq, d)
            valid = (~pad_mask).clone()
            valid[:, 0] = False  # drop CLS

            cell_idx_full = torch.arange(start, end, device=device).unsqueeze(1).expand_as(valid)
            acts_chunks.append(acts[valid].cpu().numpy().astype(np.float32))
            cell_chunks.append(cell_idx_full[valid].cpu().numpy().astype(np.int64))
            gene_chunks.append(input_gene_ids[valid].cpu().numpy().astype(np.int64))
            captured.clear()
    finally:
        for h in handles:
            h.remove()

    return (
        np.concatenate(acts_chunks, axis=0),
        np.concatenate(cell_chunks, axis=0),
        np.concatenate(gene_chunks, axis=0),
    )


# ============================================================================
# Per-cell feature aggregation
# ============================================================================
def aggregate_per_cell(token_features: np.ndarray, token_cell_idx: np.ndarray,
                      n_cells: int) -> np.ndarray:
    """Mean token-level feature activation per cell.

    Returns: (n_cells, n_features) float32. Cells with zero tokens get zero row.
    """
    n_feat = token_features.shape[1]
    out = np.zeros((n_cells, n_feat), dtype=np.float32)
    counts = np.zeros(n_cells, dtype=np.int64)
    np.add.at(out, token_cell_idx, token_features)
    np.add.at(counts, token_cell_idx, 1)
    safe = np.maximum(counts, 1).reshape(-1, 1)
    out /= safe
    return out


# ============================================================================
# Linear probe with shared splits
# ============================================================================
def _lr() -> LogisticRegression:
    return LogisticRegression(max_iter=2000, solver="lbfgs", n_jobs=-1)


def probe(X: np.ndarray, y: np.ndarray, splits):
    out = []
    for s, (tr, te) in enumerate(splits):
        clf = _lr()
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        out.append({
            "seed_idx": s,
            "accuracy": float(accuracy_score(y[te], pred)),
            "macro_f1": float(f1_score(y[te], pred, average="macro")),
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


# ============================================================================
# Gene sets
# ============================================================================
def parse_grp_file(path: Path) -> List[str]:
    """Parse a .grp (GenePattern) gene-set file.

    Format: one gene symbol per line. Header lines starting with '>' or '#'
    (and obvious URL lines) are skipped.
    """
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith((">", "#")) or "http" in line.lower():
                continue
            out.append(line.split()[0])
    return out


def load_gene_sets(gene_sets_dir: Path) -> Dict[str, List[str]]:
    sets = {}
    for f in sorted(gene_sets_dir.glob("*.grp")):
        sets[f.stem] = parse_grp_file(f)
    for f in sorted(gene_sets_dir.glob("*.txt")):
        # txt fallback: same one-symbol-per-line format
        sets[f.stem] = parse_grp_file(f)
    return sets


def score_gene_sets_per_cell(
    adata, gene_sets: Dict[str, List[str]], gene_col: str = "gene_name",
) -> Dict[str, np.ndarray]:
    """Per-cell gene-set score: mean log1p expression over genes in the set
    that are present in adata.var. Returns dict[name] -> (n_cells,) float32.

    Drops sets with < 3 overlapping genes (too noisy).
    """
    if gene_col not in adata.var.columns:
        adata.var["gene_name"] = adata.var.index.astype(str)
        gene_col = "gene_name"
    gene_to_idx = {g: i for i, g in enumerate(adata.var[gene_col].astype(str).tolist())}

    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X).astype(np.float32)

    out = {}
    for name, genes in gene_sets.items():
        idx = [gene_to_idx[g] for g in genes if g in gene_to_idx]
        if len(idx) < 3:
            print(f"[gene_sets] skip {name}: only {len(idx)} genes overlap")
            continue
        out[name] = X[:, idx].mean(axis=1).astype(np.float32)
    return out


def correlate_features_with_gene_sets(
    cell_features: np.ndarray, gene_set_scores: Dict[str, np.ndarray],
    top_k: int = 10,
) -> Dict[str, Dict]:
    """For each gene set, return Pearson r with every SAE feature and the top-k
    feature indices ranked by |r|.
    """
    out = {}
    F_centered = cell_features - cell_features.mean(axis=0, keepdims=True)
    F_std = cell_features.std(axis=0)
    for name, score in gene_set_scores.items():
        s_centered = score - score.mean()
        s_std = score.std()
        denom = F_std * s_std + 1e-12
        r = (F_centered * s_centered[:, None]).mean(axis=0) / denom
        order = np.argsort(-np.abs(r))[:top_k]
        out[name] = {
            "top_features": order.tolist(),
            "top_r": r[order].tolist(),
            "n_features_with_nonzero_var": int((F_std > 1e-8).sum()),
        }
    return out


# ============================================================================
# Adata loader (matches phase2)
# ============================================================================
def _load_adata_for_probe(data_path: Path, label_col: str):
    adata = sc.read_h5ad(data_path)
    print(f"[phase3] loaded {data_path}: shape={adata.shape}")
    if adata.raw is not None and (adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()).min() < 0:
        print(f"[phase3] .X has negatives; swapping in .raw ({adata.raw.X.shape})")
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
    return adata, label_col


# ============================================================================
# Main
# ============================================================================
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/pbmc3k.h5ad")
    p.add_argument("--label_col", default="louvain")
    p.add_argument("--model_dir", default="checkpoints/scGPT_human")
    p.add_argument("--layers", nargs="+",
                   default=["layer_00_input", "layer_03", "layer_12"],
                   help="layer names to run SAE on")
    p.add_argument("--dict_size", type=int, default=2048,
                   help="SAE dictionary size (expansion = dict_size / d_model)")
    p.add_argument("--k", type=int, default=32, help="TopK active features per token")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--extract_batch_size", type=int, default=16,
                   help="cells per scGPT forward batch")
    p.add_argument("--probe_seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--max_cells", type=int, default=20000)
    p.add_argument("--gene_sets_dir", default=None,
                   help="directory with .grp gene-set files")
    p.add_argument("--do_ablation", action="store_true",
                   help="run top-k feature ablation on cell-type probe")
    p.add_argument("--ablation_k", type=int, default=10)
    p.add_argument("--force_extract", action="store_true")
    p.add_argument("--force_retrain", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    data_path = ROOT / args.data if not Path(args.data).is_absolute() else Path(args.data)
    model_dir = ROOT / args.model_dir if not Path(args.model_dir).is_absolute() else Path(args.model_dir)
    base_out = ROOT / "results" / "phase3"
    base_out.mkdir(parents=True, exist_ok=True)

    # ---------- Load data, fix labels, subsample if needed ----------
    adata, label_col = _load_adata_for_probe(data_path, args.label_col)
    if adata.n_obs > args.max_cells:
        rng = np.random.default_rng(args.probe_seeds[0])
        idx = rng.choice(adata.n_obs, size=args.max_cells, replace=False)
        adata = adata[np.sort(idx)].copy()
        print(f"[phase3] subsampled to {adata.n_obs} cells")

    y_str = adata.obs[label_col].astype(str).values
    classes, y = np.unique(y_str, return_inverse=True)
    print(f"[phase3] {len(classes)} classes, {adata.n_obs} cells")

    # Splits shared across all probes (same convention as phase2)
    splits = [
        train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=s)
        for s in args.probe_seeds
    ]

    # ---------- Preprocess once (vocab filtering + binning) ----------
    print(f"[phase3] loading scGPT from {model_dir}")
    model, vocab, _ = load_scgpt_model(model_dir, device=args.device)
    adata = preprocess_adata_for_scgpt(adata, vocab)
    print(f"[phase3] post-preprocess: {adata.shape}")

    # ---------- Optional gene-set scores (computed once) ----------
    gene_set_scores = None
    if args.gene_sets_dir is not None:
        gsd = ROOT / args.gene_sets_dir if not Path(args.gene_sets_dir).is_absolute() else Path(args.gene_sets_dir)
        if not gsd.exists():
            print(f"[phase3] WARNING: gene_sets_dir {gsd} does not exist — skipping gene set scoring")
        else:
            sets = load_gene_sets(gsd)
            print(f"[phase3] loaded {len(sets)} gene sets from {gsd}")
            # adata.X here is log1p-normed after preprocess; that's what we want
            gene_set_scores = score_gene_sets_per_cell(adata, sets)
            print(f"[phase3] scored {len(gene_set_scores)} gene sets ({len(sets) - len(gene_set_scores)} dropped)")

    # ---------- Per-layer loop ----------
    per_layer_summary = {}

    for target_layer in args.layers:
        print(f"\n{'=' * 60}\n[phase3] LAYER: {target_layer}\n{'=' * 60}")
        layer_dir = base_out / target_layer
        layer_dir.mkdir(parents=True, exist_ok=True)
        acts_cache = layer_dir / "token_activations.npz"
        sae_ckpt = layer_dir / "sae.pt"
        train_log_path = layer_dir / "training_log.json"
        per_cell_path = layer_dir / "per_cell_features.npz"
        probe_path = layer_dir / "cell_type_probe.json"
        gs_path = layer_dir / "gene_set_correlations.json"
        abl_path = layer_dir / "feature_ablation.json"
        curves_path = layer_dir / "curves.png"

        # 1) Extract token activations (cache to disk)
        if acts_cache.exists() and not args.force_extract:
            print(f"[phase3] loading cached activations from {acts_cache}")
            d = np.load(acts_cache, allow_pickle=False)
            token_acts = d["acts"]
            token_cell = d["cell_idx"]
            token_gene = d["gene_id"]
        else:
            token_acts, token_cell, token_gene = extract_token_activations(
                adata, model, vocab, target_layer,
                device=args.device, batch_size=args.extract_batch_size,
            )
            np.savez(acts_cache, acts=token_acts, cell_idx=token_cell, gene_id=token_gene)
            print(f"[phase3] cached activations: {token_acts.shape} -> {acts_cache}")
        d_model = token_acts.shape[1]

        # 2) Train SAE
        if sae_ckpt.exists() and not args.force_retrain:
            print(f"[phase3] loading cached SAE from {sae_ckpt}")
            ckpt = torch.load(sae_ckpt, map_location=args.device)
            sae = TopKSAE(ckpt["config"]["d_in"], ckpt["config"]["n_features"], ckpt["config"]["k"]).to(args.device)
            sae.load_state_dict(ckpt["state_dict"])
            train_log = json.loads(train_log_path.read_text())
        else:
            sae, train_log = train_topk_sae(
                token_acts, d_in=d_model, n_features=args.dict_size, k=args.k,
                batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
                device=args.device,
            )
            torch.save({
                "state_dict": sae.state_dict(),
                "config": {"d_in": d_model, "n_features": args.dict_size, "k": args.k},
                "layer": target_layer,
            }, sae_ckpt)
            train_log_path.write_text(json.dumps(train_log, indent=2))
            print(f"[phase3] saved SAE -> {sae_ckpt}")

        # 3) Encode every token (sparse code) and aggregate per cell
        print("[phase3] encoding all tokens and aggregating per cell...")
        token_codes = encode_in_batches(sae, token_acts, batch_size=args.batch_size,
                                        device=args.device)
        cell_features = aggregate_per_cell(token_codes, token_cell, n_cells=adata.n_obs)
        np.savez(per_cell_path, features=cell_features, labels=y)

        # 4) LR probe on cell type
        runs = probe(cell_features, y, splits)
        probe_res = _agg(runs)
        probe_path.write_text(json.dumps({
            "layer": target_layer,
            "n_features": int(cell_features.shape[1]),
            "k_active_per_token": args.k,
            "results": probe_res,
        }, indent=2))
        print(f"[phase3] cell-type probe on SAE features: "
              f"acc = {probe_res['accuracy_mean']:.4f} +/- {probe_res['accuracy_std']:.4f}, "
              f"F1 = {probe_res['macro_f1_mean']:.4f} +/- {probe_res['macro_f1_std']:.4f}")

        # 5) Optional: gene-set correlations
        gs_summary = None
        if gene_set_scores is not None:
            print(f"[phase3] correlating {cell_features.shape[1]} SAE features with "
                  f"{len(gene_set_scores)} gene sets...")
            gs_summary = correlate_features_with_gene_sets(cell_features, gene_set_scores, top_k=10)
            gs_path.write_text(json.dumps(gs_summary, indent=2))
            print(f"[phase3] wrote gene-set correlations -> {gs_path}")
            for name, info in list(gs_summary.items())[:5]:
                top_r = info["top_r"][0] if info["top_r"] else float("nan")
                print(f"[phase3]   {name}: top |r| = {abs(top_r):.3f} on feature {info['top_features'][0]}")

        # 6) Optional: feature ablation
        abl_summary = None
        if args.do_ablation:
            print(f"[phase3] feature-ablation: zero out top-{args.ablation_k} features by mean |w| onto cell-type LR")
            clf = _lr()
            clf.fit(cell_features, y)
            # importance = mean |coefficient| across classes (multinomial)
            importance = np.abs(clf.coef_).mean(axis=0)
            top_idx = np.argsort(-importance)[:args.ablation_k]
            mask = np.ones(cell_features.shape[1], dtype=np.float32)
            mask[top_idx] = 0.0
            X_ablated = cell_features * mask
            runs_abl = probe(X_ablated, y, splits)
            abl_res = _agg(runs_abl)
            abl_summary = {
                "ablation_k": args.ablation_k,
                "top_features_zeroed": top_idx.tolist(),
                "feature_importance_top_k": importance[top_idx].tolist(),
                "before": probe_res,
                "after": abl_res,
                "delta_acc": abl_res["accuracy_mean"] - probe_res["accuracy_mean"],
                "delta_f1": abl_res["macro_f1_mean"] - probe_res["macro_f1_mean"],
            }
            abl_path.write_text(json.dumps(abl_summary, indent=2))
            print(f"[phase3] ablation: acc {probe_res['accuracy_mean']:.4f} -> "
                  f"{abl_res['accuracy_mean']:.4f}  delta = {abl_summary['delta_acc']:+.4f}")

        # 7) Training curve plot
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        ep = np.arange(1, len(train_log["epoch_loss"]) + 1)
        axes[0].plot(ep, train_log["epoch_loss"])
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("MSE loss"); axes[0].set_title("training loss")
        axes[1].plot(ep, train_log["epoch_var_explained"])
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("var explained"); axes[1].set_title("reconstruction")
        axes[1].set_ylim(0, 1)
        axes[2].plot(ep, train_log["epoch_dead_features"], label="dead this epoch")
        axes[2].set_xlabel("epoch"); axes[2].set_ylabel("# dead features"); axes[2].set_title("sparsity")
        axes[2].axhline(args.dict_size, color="gray", linestyle=":", alpha=0.5, label="dict size")
        axes[2].legend(fontsize=8)
        for ax in axes:
            ax.grid(alpha=0.3)
        fig.suptitle(f"SAE training — {target_layer}  (d={d_model}, dict={args.dict_size}, k={args.k})")
        fig.tight_layout()
        fig.savefig(curves_path, dpi=140, bbox_inches="tight")
        plt.close(fig)

        per_layer_summary[target_layer] = {
            "d_model": d_model,
            "n_tokens": int(token_acts.shape[0]),
            "final_var_explained": float(train_log["epoch_var_explained"][-1]),
            "final_dead_features_epoch": int(train_log["epoch_dead_features"][-1]),
            "dead_features_ever": int(train_log.get("dead_features_ever",
                                                    train_log["epoch_dead_features"][-1])),
            "cell_type_probe": probe_res,
            "ablation": abl_summary,
            "gene_set_correlations_top1": (
                {name: info["top_r"][0] for name, info in gs_summary.items()}
                if gs_summary else None
            ),
        }

    # ---------- Cross-layer summary ----------
    summary_path = base_out / "SUMMARY.md"
    md = ["# Phase 3 — TopK SAE on scGPT activations\n"]
    md.append(f"- Data: `{data_path.name}` — {adata.n_obs} cells, {len(classes)} classes, label=`{label_col}`")
    md.append(f"- SAE config: dict={args.dict_size}, k={args.k}, epochs={args.epochs}, bs={args.batch_size}, lr={args.lr}")
    md.append(f"- Probe seeds: {args.probe_seeds} (cell-type probe results = mean ± std)")
    md.append(f"- Phase2 baselines for context: PCA-50 = 0.940±0.008, scGPT best CLS (layer 3) = 0.909±0.007\n")

    md.append("## Per-layer table\n")
    md.append("| layer | tokens | var_exp | dead_ever | cell-type acc (SAE) | cell-type F1 (SAE) | Δacc vs PCA-50 |")
    md.append("|---|---|---|---|---|---|---|")
    pca50 = 0.9398
    for name, s in per_layer_summary.items():
        pr = s["cell_type_probe"]
        delta = pr["accuracy_mean"] - pca50
        md.append(
            f"| {name} | {s['n_tokens']} | {s['final_var_explained']:.3f} "
            f"| {s['dead_features_ever']}/{args.dict_size} "
            f"| {pr['accuracy_mean']:.4f}±{pr['accuracy_std']:.4f} "
            f"| {pr['macro_f1_mean']:.4f}±{pr['macro_f1_std']:.4f} "
            f"| {delta:+.4f} |"
        )
    md.append("")

    if args.do_ablation:
        md.append("## Top-K feature ablation\n")
        md.append(f"For each layer: fit LR on SAE features, identify top-{args.ablation_k} by mean |coef|, zero them, re-probe.\n")
        md.append(f"| layer | acc before | acc after | Δacc | F1 before | F1 after | ΔF1 |")
        md.append(f"|---|---|---|---|---|---|---|")
        for name, s in per_layer_summary.items():
            a = s.get("ablation")
            if a is None:
                continue
            md.append(
                f"| {name} "
                f"| {a['before']['accuracy_mean']:.4f} "
                f"| {a['after']['accuracy_mean']:.4f} "
                f"| {a['delta_acc']:+.4f} "
                f"| {a['before']['macro_f1_mean']:.4f} "
                f"| {a['after']['macro_f1_mean']:.4f} "
                f"| {a['delta_f1']:+.4f} |"
            )
        md.append("")

    if gene_set_scores is not None:
        md.append("## Gene-set best |Pearson r| per layer\n")
        md.append("(Single best-correlated SAE feature per gene set, |r| reported.)\n")
        all_sets = sorted({n for s in per_layer_summary.values()
                            if s["gene_set_correlations_top1"]
                            for n in s["gene_set_correlations_top1"]})
        md.append("| gene_set | " + " | ".join(args.layers) + " |")
        md.append("|" + "---|" * (1 + len(args.layers)))
        for name in all_sets:
            row = [name]
            for layer in args.layers:
                top = per_layer_summary[layer]["gene_set_correlations_top1"]
                if top is None or name not in top:
                    row.append("—")
                else:
                    row.append(f"{abs(top[name]):.3f}")
            md.append("| " + " | ".join(row) + " |")
        md.append("")

    md.append("## Reading guide\n")
    md.append("- `var_exp` close to 1 means the SAE reconstructs well; below ~0.8 means TopK is too aggressive.")
    md.append("- `dead_ever` is features that never activated across the whole training set. Large fraction dead ⇒ dict too big or k too small.")
    md.append("- `Δacc vs PCA-50` < 0 means the SAE on this layer's activations cannot recover cell-type as well as raw-PCA on log1p — i.e., the layer's activations are not a better cell-type-encoding substrate than the raw expression matrix.")
    md.append("- High |r| with a gene set on layer L but not layer 0 = scGPT is computing that pathway-level signal in the transformer, not just lookup. Worth investigating.")

    summary_path.write_text("\n".join(md))
    print(f"\n[phase3] wrote {summary_path}")
    print("\n=== PHASE 3 SUMMARY ===")
    for name, s in per_layer_summary.items():
        pr = s["cell_type_probe"]
        print(f"{name:20s}  var_exp={s['final_var_explained']:.3f}  "
              f"dead_ever={s['dead_features_ever']}/{args.dict_size}  "
              f"probe acc = {pr['accuracy_mean']:.4f} ± {pr['accuracy_std']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
