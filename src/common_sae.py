"""common_sae.py — shared util module for phase 7 probe scripts.

The user opted out of the bio_fm_probe package layering. This file is the ONE
piece of shared code allowed across the per-model scripts (evo2_probe.py and
esm2_15b_probe.py): the TopK SAE class, its training loop, the linear
probe helpers, the SVD diagnostic, the ablation-gap metric, AND the shared
plot + REPORT renderers.

Phase 7.1 protocol-alignment updates (vs phase 5/6):
  - K-grid auto-truncated at k99 from SVD (callers pass that in).
  - sae_pca_ablation_gap stores decomposed drop_PCA and drop_SAE curves
    side-by-side with the derived gap — phase 5's missing diagnostic.
  - Fraction-ablated columns stored explicitly so cross-space (PCA vs SAE,
    different dims) comparisons can normalize.
  - Random-feature ablation null computed by default, with 2σ
    significance flags per (layer, K).
  - plot_summary uses a 2×3 layout: probe / decomposition / gap+null /
    SAE quality / significance heatmap / fraction-ablated comparison.

Public API:
  TopKSAE                          Anthropic-style TopK sparse autoencoder
  train_topk_sae(...)              MSE-only training with decoder unit-norm
  encode_batched(sae, X, ...)      batched sparse code extraction
  aggregate_per_cell(...)          token-level codes -> per-sample mean
  svd_spectrum_diagnostic(...)     PR / k50 / k95 / k99 stats
  make_splits(y, seeds, ...)       stratified train/test splits per seed
  probe(X, y, splits)              LR probe (5-seed)
  agg(runs)                        mean+/-std aggregation across seeds
  ablation_curve(X, y, ..., crit)  per-K ablation curve
  sae_pca_ablation_gap(...)        gap + decomposed drops + random null + sig
  plot_phase7_summary(out_dir, layers, summary, model, dataset)
                                    shared 2×3 summary plot
  write_phase7_report(out_dir, ...) shared REPORT.md renderer
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


# ============================================================================
# Linear probe + helpers
# ============================================================================
def _lr(high_dim: bool = False) -> LogisticRegression:
    return LogisticRegression(
        max_iter=5000 if high_dim else 2000,
        solver="lbfgs", n_jobs=-1, C=1.0,
    )


def make_splits(y: np.ndarray, seeds: List[int], test_size: float = 0.2):
    return [
        train_test_split(
            np.arange(len(y)), test_size=test_size, stratify=y, random_state=s,
        )
        for s in seeds
    ]


def probe(X: np.ndarray, y: np.ndarray, splits, high_dim: bool = False):
    runs = []
    for s, (tr, te) in enumerate(splits):
        clf = _lr(high_dim=high_dim)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        runs.append({
            "seed_idx": s,
            "accuracy": float(accuracy_score(y[te], pred)),
            "macro_f1": float(f1_score(y[te], pred, average="macro")),
        })
    return runs


def agg(runs):
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
# SVD spectrum diagnostic
# ============================================================================
def svd_spectrum_diagnostic(
    activations: np.ndarray, n_samples_cap: int = 200_000,
) -> dict:
    """Centered SVD on activations (or a uniform sample). Returns cumulative-
    variance thresholds k50/k90/k95/k99/k99.9 and participation ratio.
    """
    n, d = activations.shape
    if n > n_samples_cap:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, n_samples_cap, replace=False)
        X = activations[idx].astype(np.float64)
    else:
        X = activations.astype(np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    eigvals = s ** 2
    total = eigvals.sum()
    cum = np.cumsum(eigvals) / max(total, 1e-12)

    def k_at(thresh):
        return int(np.searchsorted(cum, thresh) + 1)

    pr = float((eigvals.sum() ** 2) / (eigvals ** 2).sum())
    return {
        "n_used": int(X.shape[0]),
        "d": int(d),
        "var_per_elem": float(activations.var()),
        "per_dim_std_mean": float(activations.std(axis=0).mean()),
        "k50": k_at(0.50),
        "k90": k_at(0.90),
        "k95": k_at(0.95),
        "k99": k_at(0.99),
        "k999": k_at(0.999),
        "participation_ratio": pr,
        "top_eigvals_normalized": (eigvals[: min(50, len(eigvals))] / total).tolist(),
    }


# ============================================================================
# TopK SAE
# ============================================================================
class TopKSAE(nn.Module):
    """Anthropic-style TopK sparse autoencoder. MSE-only loss, decoder rows
    L2-normalized after every step. No L1 coefficient to tune.
    """

    def __init__(self, d_in: int, n_features: int, k: int):
        super().__init__()
        self.d_in, self.n_features, self.k = d_in, n_features, k
        self.W_enc = nn.Parameter(torch.randn(d_in, n_features) / (d_in ** 0.5))
        W_dec = self.W_enc.detach().clone().T.contiguous()
        W_dec = W_dec / W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec = nn.Parameter(W_dec)
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        vals, idx = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, idx, F.relu(vals))
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
    activations: np.ndarray, d_in: int, n_features: int, k: int,
    batch_size: int, epochs: int, lr: float, device: str,
) -> Tuple[TopKSAE, dict]:
    """Train TopK SAE on a flat (N_tokens, d_in) activation buffer."""
    sae = TopKSAE(d_in, n_features, k).to(device)
    sae.normalize_decoder()
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    act_t = torch.from_numpy(activations).float().contiguous()
    n_tokens = act_t.shape[0]
    n_batches = n_tokens // batch_size
    var_total = float(act_t.var().item())
    print(f"[sae] training: {n_tokens} tokens, {n_batches} batches/epoch, "
          f"dict={n_features}, k={k}, epochs={epochs}, bs={batch_size}")

    log: Dict[str, list] = {
        "epoch_loss": [], "epoch_var_explained": [],
        "epoch_dead_features": [], "epoch_l0_mean": [], "epoch_time_s": [],
        "config": {"d_in": d_in, "n_features": n_features, "k": k,
                   "epochs": epochs, "batch_size": batch_size, "lr": lr,
                   "n_tokens": int(n_tokens)},
    }
    ever_active = torch.zeros(n_features, dtype=torch.bool, device=device)

    for epoch in range(epochs):
        t0 = time.time()
        perm = torch.randperm(n_tokens)
        losses, l0s = [], []
        active_this = torch.zeros(n_features, dtype=torch.bool, device=device)
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            x = act_t[idx].to(device, non_blocking=True)
            x_recon, z = sae(x)
            loss = F.mse_loss(x_recon, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sae.normalize_decoder()
            losses.append(loss.item())
            with torch.no_grad():
                a = (z > 0).any(dim=0)
                active_this |= a
                ever_active |= a
                l0s.append(float((z > 0).float().sum(dim=-1).mean().item()))
        log["epoch_loss"].append(float(np.mean(losses)))
        log["epoch_var_explained"].append(
            1.0 - log["epoch_loss"][-1] / max(var_total, 1e-12)
        )
        log["epoch_dead_features"].append(int(n_features - active_this.sum().item()))
        log["epoch_l0_mean"].append(float(np.mean(l0s)))
        log["epoch_time_s"].append(time.time() - t0)
        print(f"[sae] {epoch + 1:>2}/{epochs}  loss={log['epoch_loss'][-1]:.6f}  "
              f"var_exp={log['epoch_var_explained'][-1]:.4f}  "
              f"L0={log['epoch_l0_mean'][-1]:.1f}  "
              f"dead={log['epoch_dead_features'][-1]}/{n_features}  "
              f"dt={log['epoch_time_s'][-1]:.1f}s")

    log["dead_features_ever"] = int(n_features - ever_active.sum().item())
    return sae, log


@torch.no_grad()
def encode_batched(
    sae: TopKSAE, activations: np.ndarray, batch_size: int, device: str,
) -> np.ndarray:
    sae.eval()
    out = np.zeros((activations.shape[0], sae.n_features), dtype=np.float32)
    for b in range(0, activations.shape[0], batch_size):
        x = torch.from_numpy(activations[b:b + batch_size]).float().to(device)
        out[b:b + batch_size] = sae.encode(x).cpu().numpy()
    return out


def aggregate_per_cell(
    token_features: np.ndarray, token_cell_idx: np.ndarray, n_cells: int,
) -> np.ndarray:
    """Mean per-cell aggregation of token-level features. (N_tok, n_feat) ->
    (n_cells, n_feat)."""
    n_feat = token_features.shape[1]
    out = np.zeros((n_cells, n_feat), dtype=np.float32)
    counts = np.zeros(n_cells, dtype=np.int64)
    np.add.at(out, token_cell_idx, token_features)
    np.add.at(counts, token_cell_idx, 1)
    out /= np.maximum(counts, 1).reshape(-1, 1)
    return out


# ============================================================================
# Ablation gap (the headline phase-5 metric)
# ============================================================================
DEFAULT_K_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def _ablate_and_score(
    X: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray,
    ablate_idx: np.ndarray,
) -> float:
    X_ab = X.copy()
    if ablate_idx.size:
        X_ab[:, ablate_idx] = 0.0
    clf = _lr()
    clf.fit(X_ab[tr], y[tr])
    return float(accuracy_score(y[te], clf.predict(X_ab[te])))


def _ablation_curve_one_seed(
    X: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray,
    K_grid: List[int], criterion: str, rng: np.random.Generator,
):
    n_feat = X.shape[1]
    clf = _lr()
    clf.fit(X[tr], y[tr])
    base_acc = float(accuracy_score(y[te], clf.predict(X[te])))
    if criterion == "probe_l2":
        importance = np.linalg.norm(clf.coef_, axis=0)
        ranking = np.argsort(-importance)
    elif criterion == "random":
        ranking = rng.permutation(n_feat)
    else:
        raise ValueError(f"unknown criterion {criterion!r}")
    per_K = {}
    for K in K_grid:
        K_eff = min(K, n_feat)
        per_K[K] = _ablate_and_score(X, y, tr, te, ranking[:K_eff])
    return base_acc, per_K


def ablation_curve(
    X: np.ndarray, y: np.ndarray, splits, K_grid: List[int] = None,
    criterion: str = "probe_l2", seed_for_random: int = 0,
) -> dict:
    K_grid = list(K_grid) if K_grid is not None else list(DEFAULT_K_GRID)
    rng = np.random.default_rng(seed_for_random)
    per_seed = []
    for s, (tr, te) in enumerate(splits):
        base, per_K = _ablation_curve_one_seed(
            X, y, tr, te, K_grid, criterion, rng,
        )
        per_seed.append({"seed_idx": s, "base_acc": base, "ablated_acc": per_K})

    base_accs = np.array([r["base_acc"] for r in per_seed])
    ablated = np.array(
        [[r["ablated_acc"][K] for K in K_grid] for r in per_seed],
        dtype=np.float64,
    )
    drops = base_accs[:, None] - ablated
    return {
        "base_acc_mean": float(base_accs.mean()),
        "base_acc_std": float(base_accs.std(ddof=1) if len(base_accs) > 1 else 0.0),
        "K_grid": K_grid,
        "ablated_acc_mean": ablated.mean(axis=0).tolist(),
        "ablated_acc_std": (ablated.std(axis=0, ddof=1) if len(per_seed) > 1
                            else np.zeros(len(K_grid))).tolist(),
        "drop_mean": drops.mean(axis=0).tolist(),
        "drop_std": (drops.std(axis=0, ddof=1) if len(per_seed) > 1
                     else np.zeros(len(K_grid))).tolist(),
        "criterion": criterion,
        "n_features": int(X.shape[1]),
        "n_seeds": int(len(per_seed)),
        "per_seed": per_seed,
    }


def sae_pca_ablation_gap(
    X_sae: np.ndarray, X_pca: np.ndarray, y: np.ndarray, splits,
    K_grid: List[int] = None, include_random_null: bool = True,
) -> dict:
    """Headline metric: gap = drop_PCA - drop_SAE, per K in K_grid.

    Plus random-feature ablation null when include_random_null=True. Real gap
    should significantly exceed null gap to claim signal.
    """
    K_grid = list(K_grid) if K_grid is not None else list(DEFAULT_K_GRID)
    print(f"[ablation] SAE-PCA gap: X_sae={X_sae.shape}, X_pca={X_pca.shape}, "
          f"n_seeds={len(splits)}, K_grid={K_grid}")
    out: Dict = {}
    out["sae"] = ablation_curve(X_sae, y, splits, K_grid, criterion="probe_l2")
    out["pca"] = ablation_curve(X_pca, y, splits, K_grid, criterion="probe_l2")
    sae_d = np.array(
        [[r["ablated_acc"][K] for K in K_grid] for r in out["sae"]["per_seed"]],
        dtype=np.float64)
    pca_d = np.array(
        [[r["ablated_acc"][K] for K in K_grid] for r in out["pca"]["per_seed"]],
        dtype=np.float64)
    sae_b = np.array([r["base_acc"] for r in out["sae"]["per_seed"]])
    pca_b = np.array([r["base_acc"] for r in out["pca"]["per_seed"]])
    # Decomposed drops — needed to diagnose whether a gap signal is "PCA crashed"
    # vs "SAE crashed". Phase 5 explicitly identified this as the missing view.
    drop_pca = pca_b[:, None] - pca_d
    drop_sae = sae_b[:, None] - sae_d
    gaps = drop_pca - drop_sae

    # Fraction-ablated bookkeeping — phase 5 fix 1. The two bases have very
    # different total dims (PCA capped at d_model, SAE = expansion × d_model),
    # so the same absolute K means very different ablation fractions. Store
    # the fractions explicitly so plots / REPORT can normalize.
    n_sae = int(X_sae.shape[1])
    n_pca = int(X_pca.shape[1])
    K_arr = np.asarray(K_grid)
    sae_fraction = (K_arr / max(n_sae, 1)).tolist()
    pca_fraction = (K_arr / max(n_pca, 1)).tolist()

    out["gap"] = {
        "K_grid": K_grid,
        "gap_mean": gaps.mean(axis=0).tolist(),
        "gap_std": (gaps.std(axis=0, ddof=1) if len(gaps) > 1
                    else np.zeros(len(K_grid))).tolist(),
        # decomposition
        "drop_pca_mean": drop_pca.mean(axis=0).tolist(),
        "drop_pca_std": (drop_pca.std(axis=0, ddof=1) if len(drop_pca) > 1
                         else np.zeros(len(K_grid))).tolist(),
        "drop_sae_mean": drop_sae.mean(axis=0).tolist(),
        "drop_sae_std": (drop_sae.std(axis=0, ddof=1) if len(drop_sae) > 1
                         else np.zeros(len(K_grid))).tolist(),
        # fraction bookkeeping (phase 5 fix 1)
        "n_features_pca": n_pca,
        "n_features_sae": n_sae,
        "pca_fraction_ablated": pca_fraction,
        "sae_fraction_ablated": sae_fraction,
    }

    if include_random_null:
        out["sae_random"] = ablation_curve(
            X_sae, y, splits, K_grid, criterion="random", seed_for_random=0,
        )
        out["pca_random"] = ablation_curve(
            X_pca, y, splits, K_grid, criterion="random", seed_for_random=0,
        )
        sae_r = np.array(
            [[r["ablated_acc"][K] for K in K_grid]
             for r in out["sae_random"]["per_seed"]], dtype=np.float64)
        pca_r = np.array(
            [[r["ablated_acc"][K] for K in K_grid]
             for r in out["pca_random"]["per_seed"]], dtype=np.float64)
        gaps_r = (pca_b[:, None] - pca_r) - (sae_b[:, None] - sae_r)
        out["gap_random"] = {
            "K_grid": K_grid,
            "gap_mean": gaps_r.mean(axis=0).tolist(),
            "gap_std": (gaps_r.std(axis=0, ddof=1) if len(gaps_r) > 1
                        else np.zeros(len(K_grid))).tolist(),
        }
        # Significance: gap should exceed null by more than 2 sigma at some K
        sig_mask = []
        real_m = np.array(out["gap"]["gap_mean"])
        real_s = np.array(out["gap"]["gap_std"])
        null_m = np.array(out["gap_random"]["gap_mean"])
        null_s = np.array(out["gap_random"]["gap_std"])
        max_s = np.maximum(real_s, null_s)
        # Per-K significance: real - null > 2 * max(real_std, null_std)
        sig_mask = ((real_m - null_m) > 2 * max_s).tolist()
        out["significance_2sigma_per_K"] = sig_mask
        out["any_K_significant"] = bool(any(sig_mask))

    return out


# ============================================================================
# Token-level extraction helper (generic)
# ============================================================================
def extract_tokens_from_iter(
    layer_iter, target_layer: str, n_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Iterate over (captured_per_layer_dict, valid_mask) tuples (per-batch
    yields from a model-specific forward-hook generator); collect token-level
    activations at one target_layer.

    Each batch's captured[target_layer] is (B, seq, d); valid_mask is (B, seq).
    Returns:
        token_acts:     (N_tokens, d) float32
        token_cell_idx: (N_tokens,)   int64 — which sample each token came from
    """
    acts_chunks: List[np.ndarray] = []
    cell_chunks: List[np.ndarray] = []
    cells_processed = 0
    for captured, valid in layer_iter:
        if target_layer not in captured:
            raise KeyError(f"target_layer {target_layer!r} not in captured "
                           f"(have {list(captured.keys())})")
        x = captured[target_layer]  # (B, seq, d)
        B = x.shape[0]
        cell_idx_full = (
            torch.arange(cells_processed, cells_processed + B, device=x.device)
            .unsqueeze(1).expand_as(valid)
        )
        acts_chunks.append(x[valid].cpu().numpy().astype(np.float32))
        cell_chunks.append(cell_idx_full[valid].cpu().numpy().astype(np.int64))
        cells_processed += B
    return (
        np.concatenate(acts_chunks, axis=0),
        np.concatenate(cell_chunks, axis=0),
    )


# ============================================================================
# Shared plot + report renderers (phase 7.1)
# ============================================================================
def plot_phase7_summary(
    out_dir: Path, sae_layers: List[str], sae_summary: Dict,
    model_name: str, dataset_name: str,
) -> None:
    """2×3 plot — protocol-aligned phase 7 summary.

    Panels:
      (0,0) per-layer probe accuracy: PCA vs SAE (vs raw modality baseline if
            provided by caller in sae_summary[layer].get('baseline_acc'))
      (0,1) drop_PCA (dashed) vs drop_SAE (solid) per layer over K
            -- the phase-5 missing decomposition diagnostic
      (0,2) gap curve with 2σ random-null band (the derived metric, secondary)
      (1,0) SAE training quality: var explained + dead features
      (1,1) per-(layer, K) significance heatmap: real gap > 2σ random null?
      (1,2) same curves vs FRACTION ablated (K / dim_alive) — pca-frac and
            sae-frac on a shared axis. Resolves the dim-mismatch criticism.
    """
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    xs = np.arange(len(sae_layers))
    colors = ["C0", "C2", "C3", "C4", "C5", "C6"]

    # (0,0) probe acc
    ax = axes[0, 0]
    pca_m = [sae_summary[l]["pca_probe"]["accuracy_mean"] for l in sae_layers]
    pca_s = [sae_summary[l]["pca_probe"]["accuracy_std"] for l in sae_layers]
    sae_m = [sae_summary[l]["sae_probe"]["accuracy_mean"] for l in sae_layers]
    sae_s = [sae_summary[l]["sae_probe"]["accuracy_std"] for l in sae_layers]
    ax.errorbar(xs, pca_m, yerr=pca_s, marker="o", label="PCA-probe", color="C0")
    ax.errorbar(xs, sae_m, yerr=sae_s, marker="s", label="SAE-probe", color="C1")
    ax.set_xticks(xs); ax.set_xticklabels(sae_layers, rotation=15, fontsize=8)
    ax.set_xlabel("layer"); ax.set_ylabel("probe accuracy")
    ax.set_title("(A) Per-layer probe: PCA vs SAE features")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # (0,1) DECOMPOSITION — drop_PCA (dashed) vs drop_SAE (solid)
    ax = axes[0, 1]
    for i, l in enumerate(sae_layers):
        K = sae_summary[l]["gap_K_grid"]
        c = colors[i % len(colors)]
        drop_pca_m = sae_summary[l].get("drop_pca_mean")
        drop_sae_m = sae_summary[l].get("drop_sae_mean")
        if drop_pca_m is not None and drop_sae_m is not None:
            ax.plot(K, drop_pca_m, "--", marker="o", color=c, alpha=0.9,
                    label=f"{l} drop_PCA")
            ax.plot(K, drop_sae_m, "-", marker="s", color=c, alpha=0.7,
                    label=f"{l} drop_SAE")
    ax.axhline(0, color="black", ls=":", alpha=0.4)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("K (ablated features)")
    ax.set_ylabel("probe acc drop (base − ablated)")
    ax.set_title("(B) Decomposition: drop_PCA (---) vs drop_SAE (—)\n"
                 "(phase-5 missing diagnostic)")
    ax.grid(alpha=0.3); ax.legend(fontsize=6, loc="upper left")

    # (0,2) gap + 2σ null
    ax = axes[0, 2]
    for i, l in enumerate(sae_layers):
        K = sae_summary[l]["gap_K_grid"]
        m = np.array(sae_summary[l]["gap_mean"])
        s = np.array(sae_summary[l]["gap_std"])
        c = colors[i % len(colors)]
        ax.plot(K, m, marker="o", color=c, label=l, lw=1.5)
        ax.fill_between(K, m - s, m + s, color=c, alpha=0.18)
        nm = sae_summary[l].get("gap_null_mean")
        ns = sae_summary[l].get("gap_null_std")
        if nm is not None and ns is not None:
            nm = np.array(nm); ns = np.array(ns)
            ax.fill_between(K, nm - 2 * ns, nm + 2 * ns,
                            color=c, alpha=0.06, hatch="//")
    ax.axhline(0, color="black", ls=":", alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("K (ablated features)")
    ax.set_ylabel("gap = drop_PCA − drop_SAE")
    ax.set_title("(C) Gap (real curve) vs 2σ random null (hatched)")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    # (1,0) SAE training quality
    ax = axes[1, 0]
    ax2 = ax.twinx()
    ve = [sae_summary[l]["var_explained_final"] for l in sae_layers]
    dead = [sae_summary[l]["dead_features_ever"] for l in sae_layers]
    ax.bar(xs - 0.2, ve, width=0.4, color="C0", label="var explained")
    ax2.bar(xs + 0.2, dead, width=0.4, color="C3", alpha=0.7,
            label="dead features (ever)")
    ax.set_xticks(xs); ax.set_xticklabels(sae_layers, rotation=15, fontsize=8)
    ax.set_ylabel("var explained", color="C0")
    ax2.set_ylabel("# dead features", color="C3")
    ax.set_title("(D) SAE training quality")
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.3, axis="y")

    # (1,1) significance heatmap
    ax = axes[1, 1]
    K_grid_0 = sae_summary[sae_layers[0]]["gap_K_grid"]
    sig_grid = np.zeros((len(sae_layers), len(K_grid_0)))
    for i, l in enumerate(sae_layers):
        s = sae_summary[l].get("significance_per_K") or [False] * sig_grid.shape[1]
        sig_grid[i, :len(s)] = [float(b) for b in s]
    im = ax.imshow(sig_grid, aspect="auto", cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(np.arange(sig_grid.shape[1]))
    ax.set_xticklabels(K_grid_0, rotation=0, fontsize=8)
    ax.set_yticks(np.arange(len(sae_layers)))
    ax.set_yticklabels(sae_layers)
    ax.set_xlabel("K"); ax.set_ylabel("layer")
    ax.set_title("(E) (layer, K) where real gap > 2σ above random null")
    fig.colorbar(im, ax=ax, label="significant (0/1)")

    # (1,2) FRACTION-ablated comparison: drop curves vs fraction
    ax = axes[1, 2]
    for i, l in enumerate(sae_layers):
        c = colors[i % len(colors)]
        pca_frac = sae_summary[l].get("pca_fraction_ablated")
        sae_frac = sae_summary[l].get("sae_fraction_ablated")
        drop_pca_m = sae_summary[l].get("drop_pca_mean")
        drop_sae_m = sae_summary[l].get("drop_sae_mean")
        if pca_frac is not None and drop_pca_m is not None:
            ax.plot(pca_frac, drop_pca_m, "--", marker="o", color=c,
                    label=f"{l} PCA (frac of {sae_summary[l].get('n_features_pca')})",
                    alpha=0.9)
        if sae_frac is not None and drop_sae_m is not None:
            ax.plot(sae_frac, drop_sae_m, "-", marker="s", color=c,
                    label=f"{l} SAE (frac of {sae_summary[l].get('n_features_sae')})",
                    alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("fraction of features ablated (K / dim)")
    ax.set_ylabel("probe acc drop")
    ax.set_title("(F) drop vs FRACTION ablated (matched comparison)")
    ax.grid(alpha=0.3); ax.legend(fontsize=6, loc="upper left")

    fig.suptitle(
        f"{model_name} × {dataset_name} — phase 7 probe summary "
        f"(protocol-aligned: decomposition + fraction + null + significance)",
        y=0.995, fontsize=12, weight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = out_dir / "summary.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    fig.savefig(out_dir / "summary.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_png}")


def write_phase7_report(
    out_dir: Path, model_name: str, dataset_name: str,
    d_model: int, n_layers: int, sae_layers: List[str],
    sae_dict: int, sae_k: int, sae_expansion: float, pca_dim: int,
    seeds: List[int], n_samples: int,
    sae_summary: Dict, svd_results: Dict,
) -> None:
    md = [f"# Phase 7 — {model_name} × {dataset_name}\n"]
    md.append(f"- Model: `{model_name}` (d_model={d_model}, n_layers={n_layers})")
    md.append(f"- Dataset: `{dataset_name}`, n_samples={n_samples}")
    md.append(f"- SAE: expansion={sae_expansion}× (dict={sae_dict}), k={sae_k}")
    md.append(f"- PCA dim: {pca_dim} (capped at d_model={d_model})")
    md.append(f"- Seeds: {seeds}\n")

    md.append("## Per-layer probe results")
    md.append("| layer | PCA acc | SAE acc | Δ(SAE − PCA) | SAE var_exp | dead | k99 |")
    md.append("|---|---|---|---|---|---|---|")
    for l in sae_layers:
        s = sae_summary[l]
        pca_m = s["pca_probe"]["accuracy_mean"]
        pca_sd = s["pca_probe"]["accuracy_std"]
        sae_m = s["sae_probe"]["accuracy_mean"]
        sae_sd = s["sae_probe"]["accuracy_std"]
        md.append(
            f"| {l} | {pca_m:.4f}±{pca_sd:.4f} | {sae_m:.4f}±{sae_sd:.4f} "
            f"| {sae_m - pca_m:+.4f} | {s['var_explained_final']:.3f} "
            f"| {s['dead_features_ever']}/{sae_dict} | {s['k99']} |")
    md.append("")

    md.append("## drop_PCA and drop_SAE decomposition\n")
    md.append("Decomposed view — the phase-5 missing diagnostic. Same K-grid in "
              "raw absolute terms (also see fraction-of-features columns).\n")
    K_grids = sae_summary[sae_layers[0]]["gap_K_grid"]
    md.append("| layer | metric | " + " | ".join(f"K={k}" for k in K_grids) + " |")
    md.append("|---|---|" + "---|" * len(K_grids))
    for l in sae_layers:
        s = sae_summary[l]
        if s.get("drop_pca_mean") is None:
            continue
        md.append(f"| {l} | drop_PCA | " +
                  " | ".join(f"{v:+.3f}" for v in s["drop_pca_mean"]) + " |")
        md.append(f"| {l} | drop_SAE | " +
                  " | ".join(f"{v:+.3f}" for v in s["drop_sae_mean"]) + " |")
    md.append("")

    md.append("## SAE − PCA ablation gap (significant K marked ★)")
    md.append("Positive gap ⇒ drop_PCA > drop_SAE.")
    md.append("★ = real gap exceeds random-null gap by ≥ 2σ (per-K).\n")
    md.append("| layer | " + " | ".join(f"K={k}" for k in K_grids) + " |")
    md.append("|---|" + "---|" * len(K_grids))
    for l in sae_layers:
        s = sae_summary[l]
        row = [l]
        for i in range(len(s["gap_K_grid"])):
            sig = (s.get("significance_per_K") or [False] * len(s["gap_K_grid"]))[i]
            marker = "★" if sig else ""
            row.append(f"{s['gap_mean'][i]:+.3f}±{s['gap_std'][i]:.3f}{marker}")
        md.append("| " + " | ".join(row) + " |")
    md.append("")

    md.append("## Fraction-ablated bookkeeping (PCA dim ≠ SAE dim)")
    md.append("Same K means different ablation fractions of each basis. The "
              "phase-5 protocol fix.\n")
    md.append("| layer | n_pca | n_sae | K=" +
              " | K=".join(str(k) for k in K_grids) +
              " (PCA frac / SAE frac) |")
    md.append("|---|---|---|" + "---|" * len(K_grids))
    for l in sae_layers:
        s = sae_summary[l]
        n_pca = s.get("n_features_pca", "?")
        n_sae = s.get("n_features_sae", "?")
        pf = s.get("pca_fraction_ablated") or [None] * len(K_grids)
        sf = s.get("sae_fraction_ablated") or [None] * len(K_grids)
        cells = []
        for i in range(len(K_grids)):
            if pf[i] is not None and sf[i] is not None:
                cells.append(f"{pf[i]:.3%} / {sf[i]:.3%}")
            else:
                cells.append("—")
        md.append(f"| {l} | {n_pca} | {n_sae} | " + " | ".join(cells) + " |")
    md.append("")

    md.append("## SVD spectrum")
    md.append("| layer | PR | k50 | k95 | k99 | var_per_elem |")
    md.append("|---|---|---|---|---|---|")
    for l, ss in svd_results.items():
        md.append(f"| {l} | {ss['participation_ratio']:.1f} | "
                  f"{ss['k50']} | {ss['k95']} | {ss['k99']} | "
                  f"{ss['var_per_elem']:.3f} |")
    md.append("")

    md.append("## Significance digest")
    any_sig = [l for l in sae_layers if sae_summary[l].get("any_K_significant")]
    if any_sig:
        md.append(f"- **Signal detected**: layers with ≥1 K above 2σ random "
                  f"null: {', '.join(any_sig)}")
    else:
        md.append(f"- **No signal**: no layer has any K where real gap exceeds "
                  f"random-null by 2σ → consistent with 'bio FM at LLM scale "
                  f"still doesn't show ablation gap above noise'.")
    md.append("")

    (out_dir / "REPORT.md").write_text("\n".join(md))
    print(f"[report] wrote {out_dir / 'REPORT.md'}")


def extract_summary_fields(gap_output: dict) -> dict:
    """Helper: pull all the bookkeeping fields out of sae_pca_ablation_gap()
    return so callers can stash them in per-layer summary dicts uniformly.
    """
    g = gap_output["gap"]
    return {
        "gap_K_grid": g["K_grid"],
        "gap_mean": g["gap_mean"],
        "gap_std": g["gap_std"],
        "drop_pca_mean": g.get("drop_pca_mean"),
        "drop_pca_std": g.get("drop_pca_std"),
        "drop_sae_mean": g.get("drop_sae_mean"),
        "drop_sae_std": g.get("drop_sae_std"),
        "n_features_pca": g.get("n_features_pca"),
        "n_features_sae": g.get("n_features_sae"),
        "pca_fraction_ablated": g.get("pca_fraction_ablated"),
        "sae_fraction_ablated": g.get("sae_fraction_ablated"),
        "gap_null_mean": gap_output.get("gap_random", {}).get("gap_mean"),
        "gap_null_std": gap_output.get("gap_random", {}).get("gap_std"),
        "significance_per_K": gap_output.get("significance_2sigma_per_K"),
        "any_K_significant": gap_output.get("any_K_significant"),
    }
