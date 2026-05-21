"""evaluate_sae_jaspar.py — per-SAE-feature TFBS hit-rate against JASPAR.

This replaces the aggregate "SAE - PCA ablation gap" metric with the
methodology used in the published SAE-on-DNA papers (Evo 2 / Brixi et al.
Nature 2026; HyenaDNA-small SAE / arxiv 2507.07486):

  For each live SAE feature f:
    1. Find the top-K most-activating tokens across the corpus.
    2. For each top token, extract a sequence window of ±W bp around it.
    3. Score each window against every JASPAR PWM.
    4. Report the best-matching PWM and its p-value (against a sequence-
       shuffled null).
    5. A feature "hits" if its best PWM p-value < threshold (default 1e-3).

  Aggregate: report "fraction of live features with a TFBS hit". This is
  the metric you can compare directly to the published papers.

Inputs:
    trained SAE       (sae.pt produced by train_topk_sae)
    activations cache (token_activations.npz with `acts` + `cell_idx`)
    sequence corpus   (the raw DNA strings used for forward extraction)
    JASPAR motifs     (directory of MEME-format files or one combined file)

Notes on JASPAR data:
    JASPAR ships PWMs in several formats; this script accepts MEME format
    (one file with multiple motifs). Download:
        https://jaspar.genereg.net/download/data/2024/CORE/
        JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt
    Place at data/jaspar/JASPAR2024_CORE_vertebrates.meme and pass
    --jaspar_path data/jaspar/JASPAR2024_CORE_vertebrates.meme.

Usage:
    python src/phase7/evaluate_sae_jaspar.py \\
        --sae_ckpt results/.../layer_16/sae.pt \\
        --acts_cache results/.../layer_16/token_activations.npz \\
        --data_dir data/genomic_benchmarks/human_nontata_promoters \\
        --jaspar_path data/jaspar/JASPAR2024_CORE_vertebrates.meme \\
        --out results/.../layer_16/jaspar_hits.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent.parent
sys.path.insert(0, str(THIS.parent))


def _resolve(p: str) -> str:
    path = Path(p)
    return str(path if path.is_absolute() else ROOT / path)


from common_sae import TopKSAE  # noqa: E402


# ============================================================================
# JASPAR PWM loading — MEME format parser, no biopython dependency
# ============================================================================
def parse_meme_motifs(meme_path: Path) -> List[Dict]:
    """Parse a JASPAR-MEME file into a list of motifs.

    Returns list of dicts with keys:
        id (str), name (str), length (int), pwm (np.ndarray of shape (4, L))

    PWM columns are in order [A, C, G, T]; values are log-likelihoods over
    uniform 0.25 background (matching JASPAR's MEME export convention).
    """
    motifs: List[Dict] = []
    lines = meme_path.read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("MOTIF "):
            i += 1
            continue
        parts = line.split()
        mid = parts[1] if len(parts) > 1 else f"motif_{len(motifs)}"
        mname = parts[2] if len(parts) > 2 else mid
        # Skip ahead to "letter-probability matrix: ... w= N ..."
        w = None
        i += 1
        while i < len(lines):
            m = re.match(
                r"letter-probability matrix:.*?w=\s*(\d+)", lines[i].strip()
            )
            if m:
                w = int(m.group(1))
                i += 1
                break
            i += 1
        if w is None:
            continue
        # Read next w rows of 4 floats each, skipping blanks / non-data lines
        rows: List[List[float]] = []
        while i < len(lines) and len(rows) < w:
            toks = lines[i].split()
            i += 1
            if len(toks) != 4:
                if rows:
                    # We were inside the data block and hit a non-data line —
                    # abort this motif if the matrix is incomplete.
                    break
                continue
            try:
                rows.append([float(x) for x in toks])
            except ValueError:
                if rows:
                    break
                continue
        if len(rows) != w:
            continue
        probs = np.asarray(rows, dtype=np.float64).T          # shape (4, w)
        probs = np.clip(probs, 1e-6, None)
        ll = np.log2(probs / 0.25)
        motifs.append({"id": mid, "name": mname, "length": w, "pwm": ll})
    return motifs


# ============================================================================
# Sequence -> token-position mapping
# ============================================================================
def build_cell_position_map(cell_idx: np.ndarray) -> List[List[int]]:
    """For each cell c, return the list of (global) token indices that belong
    to it, in order. Assumes the forward pass preserved within-cell ordering
    (true for our pipeline — no NaN filter at layer_16 in bf16).

    The position within the cell of token t in cell c is `j` where
    `cell_token_indices[c][j] == t`.
    """
    n_cells = int(cell_idx.max()) + 1 if len(cell_idx) else 0
    out: List[List[int]] = [[] for _ in range(n_cells)]
    for t, c in enumerate(cell_idx):
        out[int(c)].append(t)
    return out


# ============================================================================
# PWM scoring
# ============================================================================
_BASE = {"A": 0, "C": 1, "G": 2, "T": 3, "a": 0, "c": 1, "g": 2, "t": 3}


def seq_to_indices(s: str) -> np.ndarray:
    return np.array([_BASE.get(c, -1) for c in s], dtype=np.int64)


def score_window(window_idx: np.ndarray, pwm: np.ndarray) -> float:
    """Best PWM score across all valid sliding positions of pwm over window."""
    w = pwm.shape[1]
    L = window_idx.shape[0]
    if L < w:
        return -np.inf
    best = -np.inf
    for s in range(L - w + 1):
        sub = window_idx[s:s + w]
        if (sub < 0).any():     # contains non-ACGT (e.g., pad)
            continue
        score = float(pwm[sub, np.arange(w)].sum())
        if score > best:
            best = score
    return best


def pvalue_via_shuffle(window: str, pwm: np.ndarray, n_shuffles: int = 200,
                       rng: np.random.Generator = None) -> float:
    """Empirical p-value: fraction of shuffled windows whose best PWM score
    is >= the observed best score."""
    if rng is None:
        rng = np.random.default_rng(0)
    win_idx = seq_to_indices(window)
    obs = score_window(win_idx, pwm)
    if not np.isfinite(obs):
        return 1.0
    chars = list(window)
    ge = 0
    for _ in range(n_shuffles):
        rng.shuffle(chars)
        if score_window(seq_to_indices("".join(chars)), pwm) >= obs:
            ge += 1
    return (ge + 1) / (n_shuffles + 1)


# ============================================================================
# Main eval: per-feature TFBS hit
# ============================================================================
def evaluate_feature_hits(
    sae: TopKSAE,
    activations: np.ndarray,
    cell_idx: np.ndarray,
    sequences: List[str],
    motifs: List[Dict],
    top_k_tokens: int = 50,
    window_half_width: int = 15,
    p_threshold: float = 1e-3,
    n_shuffles: int = 200,
    encode_batch_size: int = 4096,
    device: str = "cuda",
    max_features: int = 0,
) -> Dict:
    """For every live SAE feature: find top-activating tokens, extract
    sequence windows, score against every motif, report best motif + p-value.

    A feature is "live" if it activates on >= 1 token over the corpus.
    A feature "hits" a TFBS if best_motif_p < p_threshold.

    max_features > 0 restricts evaluation to the first N live features
    (useful for quick sanity runs).
    """
    sae.eval()
    n_tokens = activations.shape[0]
    n_feat = int(sae.n_features)
    print(f"[jaspar] {n_tokens} tokens, {n_feat} SAE features, "
          f"{len(motifs)} motifs, top_k={top_k_tokens}, window=±{window_half_width}")

    # 1. encode in chunks, keep top-K activations per feature with their token index
    print("[jaspar] encoding + tracking top-K per feature ...")
    top_vals = np.full((n_feat, top_k_tokens), -np.inf, dtype=np.float32)
    top_tok = np.full((n_feat, top_k_tokens), -1, dtype=np.int64)

    with torch.no_grad():
        for b in tqdm(range(0, n_tokens, encode_batch_size),
                      desc="encode", unit="batch"):
            j = min(b + encode_batch_size, n_tokens)
            x = torch.from_numpy(activations[b:j].astype(np.float32)).to(device, non_blocking=True)
            z = sae.encode(x).cpu().numpy()              # (B, n_feat) fp32
            # For each feature, merge this batch's max-K with running top-K
            for fi in range(n_feat):
                col = z[:, fi]
                if col.max() <= 0:
                    continue
                # top-K within this batch
                if (col > 0).sum() <= top_k_tokens:
                    cand_idx = np.where(col > 0)[0]
                else:
                    cand_idx = np.argpartition(-col, top_k_tokens)[:top_k_tokens]
                cand_vals = col[cand_idx]
                cand_global_tok = b + cand_idx
                # merge with current top-K
                merged_vals = np.concatenate([top_vals[fi], cand_vals])
                merged_tok = np.concatenate([top_tok[fi], cand_global_tok])
                keep = np.argpartition(-merged_vals, top_k_tokens - 1)[:top_k_tokens]
                top_vals[fi] = merged_vals[keep]
                top_tok[fi] = merged_tok[keep]

    # Find live features (any activation > 0)
    live = (top_vals.max(axis=1) > 0)
    live_idx = np.where(live)[0]
    print(f"[jaspar] live features: {live.sum()}/{n_feat} "
          f"({100*live.sum()/n_feat:.1f}%)")
    if max_features > 0 and len(live_idx) > max_features:
        live_idx = live_idx[:max_features]
        print(f"[jaspar] restricting to first {max_features} live features")

    # 2. build cell -> tokens map (so we can recover within-cell position)
    cell_tokens = build_cell_position_map(cell_idx)

    def window_for_token(t_global: int) -> str:
        # Find which cell + position-within-cell this token belongs to
        c = int(cell_idx[t_global])
        within = cell_tokens[c].index(t_global)
        seq = sequences[c]
        L = len(seq)
        lo = max(0, within - window_half_width)
        hi = min(L, within + window_half_width + 1)
        return seq[lo:hi].upper()

    # 3. per-feature: score each top-K window against every motif
    rng = np.random.default_rng(0)
    per_feature: List[Dict] = []
    print(f"[jaspar] scoring {len(live_idx)} live features against {len(motifs)} motifs ...")
    for fi in tqdm(live_idx, desc="features", unit="feat"):
        # Aggregate top-K windows for this feature
        kept = top_tok[fi][top_vals[fi] > 0]
        if len(kept) == 0:
            continue
        windows = [window_for_token(int(t)) for t in kept[:top_k_tokens]]
        # For each motif, take max score across this feature's windows
        best = {"motif_id": None, "motif_name": None, "max_score": -np.inf,
                "best_window": None}
        for m in motifs:
            pwm = m["pwm"]
            for w in windows:
                if len(w) < pwm.shape[1]:
                    continue
                s = score_window(seq_to_indices(w), pwm)
                if s > best["max_score"]:
                    best["max_score"] = s
                    best["motif_id"] = m["id"]
                    best["motif_name"] = m["name"]
                    best["best_window"] = w
        if best["motif_id"] is None:
            per_feature.append({
                "feature_idx": int(fi), "best_motif_id": None,
                "p_value": 1.0, "hit": False,
            })
            continue
        # Compute p-value of the best motif on its best window
        best_motif = next(m for m in motifs if m["id"] == best["motif_id"])
        pval = pvalue_via_shuffle(best["best_window"], best_motif["pwm"],
                                  n_shuffles=n_shuffles, rng=rng)
        per_feature.append({
            "feature_idx": int(fi),
            "best_motif_id": best["motif_id"],
            "best_motif_name": best["motif_name"],
            "best_score": float(best["max_score"]),
            "best_window": best["best_window"],
            "p_value": float(pval),
            "hit": bool(pval < p_threshold),
        })

    n_hits = sum(1 for r in per_feature if r["hit"])
    n_eval = len(per_feature)
    summary = {
        "n_features_total": int(n_feat),
        "n_features_live": int(live.sum()),
        "n_features_evaluated": int(n_eval),
        "n_hits": int(n_hits),
        "hit_rate_among_evaluated": float(n_hits / max(n_eval, 1)),
        "hit_rate_among_live": float(n_hits / max(live.sum(), 1)),
        "p_threshold": p_threshold,
        "top_k_tokens": top_k_tokens,
        "window_half_width": window_half_width,
        "n_shuffles": n_shuffles,
        "n_motifs_scored": len(motifs),
        "per_feature": per_feature,
    }
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sae_ckpt", required=True,
                   help="path to sae.pt produced by train_topk_sae")
    p.add_argument("--acts_cache", required=True,
                   help="path to token_activations.npz (the SAE training set)")
    p.add_argument("--data_dir",
                   default="data/genomic_benchmarks/human_nontata_promoters",
                   help="raw sequence corpus dir (for context windows)")
    p.add_argument("--jaspar_path", required=True,
                   help="MEME-format JASPAR file with PWMs")
    p.add_argument("--out", required=True,
                   help="output JSON path")
    p.add_argument("--top_k_tokens", type=int, default=50)
    p.add_argument("--window_half_width", type=int, default=15)
    p.add_argument("--p_threshold", type=float, default=1e-3)
    p.add_argument("--n_shuffles", type=int, default=200)
    p.add_argument("--encode_batch_size", type=int, default=4096)
    p.add_argument("--max_features", type=int, default=0,
                   help="0 = evaluate all live features; useful >0 for sanity runs")
    p.add_argument("--max_samples", type=int, default=20000,
                   help="must match the --max_samples used during extraction")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    sae_ckpt_path = Path(_resolve(args.sae_ckpt))
    acts_cache_path = Path(_resolve(args.acts_cache))
    data_dir = Path(_resolve(args.data_dir))
    jaspar_path = Path(_resolve(args.jaspar_path))
    out_path = Path(_resolve(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load JASPAR motifs
    print(f"[jaspar] loading motifs from {jaspar_path}")
    motifs = parse_meme_motifs(jaspar_path)
    print(f"[jaspar]   parsed {len(motifs)} motifs (lengths: "
          f"{min(m['length'] for m in motifs)} - "
          f"{max(m['length'] for m in motifs)})")
    if not motifs:
        raise SystemExit(f"no motifs parsed from {jaspar_path}")

    # 2. Load SAE
    print(f"[jaspar] loading SAE from {sae_ckpt_path}")
    ckpt = torch.load(sae_ckpt_path, map_location=args.device)
    cfg = ckpt["config"]
    sae = TopKSAE(cfg["d_in"], cfg["n_features"], cfg["k"]).to(args.device)
    sae.load_state_dict(ckpt["state_dict"])
    print(f"[jaspar]   d_in={cfg['d_in']}, n_features={cfg['n_features']}, "
          f"k={cfg['k']}")

    # 3. Load activations
    print(f"[jaspar] loading activations from {acts_cache_path}")
    d = np.load(acts_cache_path, allow_pickle=False, mmap_mode="r")
    acts, cell = d["acts"], d["cell_idx"]
    print(f"[jaspar]   acts={acts.shape} {acts.dtype}")

    # 4. Load sequences (must be the same subsampling as during extraction)
    print(f"[jaspar] loading sequences from {data_dir}")
    from evo_probe import load_genomic_benchmarks
    seqs, _ = load_genomic_benchmarks(data_dir)
    if len(seqs) > args.max_samples:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(len(seqs), args.max_samples, replace=False))
        seqs = [seqs[i] for i in idx]
    print(f"[jaspar]   using {len(seqs)} sequences "
          f"(mean_len={int(np.mean([len(s) for s in seqs]))})")

    # 5. Evaluate
    t0 = time.time()
    summary = evaluate_feature_hits(
        sae=sae, activations=acts, cell_idx=cell, sequences=seqs, motifs=motifs,
        top_k_tokens=args.top_k_tokens,
        window_half_width=args.window_half_width,
        p_threshold=args.p_threshold,
        n_shuffles=args.n_shuffles,
        encode_batch_size=args.encode_batch_size,
        device=args.device,
        max_features=args.max_features,
    )
    summary["wall_time_s"] = time.time() - t0
    summary["sae_ckpt"] = str(sae_ckpt_path)
    summary["acts_cache"] = str(acts_cache_path)
    summary["jaspar_path"] = str(jaspar_path)

    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[jaspar] === SUMMARY ===")
    print(f"  live features:        {summary['n_features_live']} / {summary['n_features_total']}")
    print(f"  evaluated:            {summary['n_features_evaluated']}")
    print(f"  TFBS hits (p<{args.p_threshold:.0e}): {summary['n_hits']} "
          f"({100*summary['hit_rate_among_evaluated']:.1f}% of evaluated, "
          f"{100*summary['hit_rate_among_live']:.1f}% of live)")
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
