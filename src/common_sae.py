"""common_sae.py — small shared util module for phase 7 probe scripts.

The user opted out of the bio_fm_probe package layering. This file is the ONE
piece of shared code allowed across the per-model scripts (evo2_probe.py and
esm2_15b_probe.py): the TopK SAE class, its training loop, the linear
probe helpers, the SVD diagnostic, and the ablation-gap metric.

Lifted verbatim from bio_fm_probe/core/probes.py and core/ablation.py
(phase 4 / 5 code) — algorithms unchanged, just removed package layering and
import paths.

Public API:
  TopKSAE                          Anthropic-style TopK sparse autoencoder
  train_topk_sae(...)              MSE-only training with decoder unit-norm
  encode_batched(sae, X, ...)      batched sparse code extraction
  aggregate_per_cell(...)          token-level codes -> per-sample mean
  svd_spectrum_diagnostic(...)     PR / k50 / k95 / k99 stats on activations
  make_splits(y, seeds, ...)       stratified train/test splits per seed
  probe(X, y, splits)              LR probe (5-seed)
  agg(runs)                        mean+/-std aggregation across seeds
  ablation_curve(X, y, ..., crit)  per-K ablation curve with two criteria:
                                    'probe_l2' or 'random'
  sae_pca_ablation_gap(...)        the headline metric:
                                    gap = drop_PCA - drop_SAE, plus random null
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

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
    gaps = (pca_b[:, None] - pca_d) - (sae_b[:, None] - sae_d)
    out["gap"] = {
        "K_grid": K_grid,
        "gap_mean": gaps.mean(axis=0).tolist(),
        "gap_std": (gaps.std(axis=0, ddof=1) if len(gaps) > 1
                    else np.zeros(len(K_grid))).tolist(),
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
