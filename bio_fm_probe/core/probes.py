"""Model-agnostic probes:

  - linear-probe + PCA baselines (phase 2 logic, refactored)
  - layer-0 CLS sanity (H1)
  - SVD spectrum diagnostic
  - TopK SAE + training (phase 3 logic, refactored)
  - per-cell aggregation of token-level codes
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
# Linear probe + baselines
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


def probe_with_pca(X: np.ndarray, y: np.ndarray, splits, n_components: int):
    runs = []
    for s, (tr, te) in enumerate(splits):
        n_c = min(n_components, X.shape[1], len(tr) - 1)
        pca = PCA(n_components=n_c, random_state=0)
        Xtr = pca.fit_transform(X[tr])
        Xte = pca.transform(X[te])
        clf = _lr()
        clf.fit(Xtr, y[tr])
        pred = clf.predict(Xte)
        runs.append({
            "seed_idx": s,
            "accuracy": float(accuracy_score(y[te], pred)),
            "macro_f1": float(f1_score(y[te], pred, average="macro")),
            "n_components_used": int(pca.n_components_),
            "evr_sum": float(pca.explained_variance_ratio_.sum()),
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
# H1 — layer 0 CLS constancy sanity
# ============================================================================
def layer0_cls_sanity(cls_dict: Dict[str, np.ndarray], ref_layer: str) -> dict:
    L0 = cls_dict.get("layer_00_input")
    Lref = cls_dict.get(ref_layer)
    if L0 is None or Lref is None:
        return {"skipped": True,
                "reason": "no layer_00_input or reference layer in CLS cache"}

    def stats(M: np.ndarray) -> dict:
        std = M.std(axis=0)
        return {
            "shape": list(M.shape),
            "mean_abs_value": float(np.abs(M).mean()),
            "std_per_dim_mean": float(std.mean()),
            "std_per_dim_median": float(np.median(std)),
            "std_per_dim_max": float(std.max()),
            "max_abs_deviation_from_mean": float(
                np.abs(M - M.mean(axis=0, keepdims=True)).max()
            ),
        }

    return {
        "layer_00_input_cls": stats(L0),
        f"{ref_layer}_cls_reference": stats(Lref),
        "ratio_l0_to_ref_std": float(
            L0.std(axis=0).mean() / max(1e-12, Lref.std(axis=0).mean())
        ),
        "interpretation_note": (
            "ratio_l0_to_ref_std reports per-dim std of the CLS slot at the "
            "input relative to a mid-layer CLS. << 1 means the CLS slot is "
            "near-constant at input ⇒ any layer-0 baseline using CLS-pool is "
            "degenerate. Numbers, not verdicts."
        ),
    }


# ============================================================================
# SVD spectrum diagnostic
# ============================================================================
def svd_spectrum_diagnostic(
    activations: np.ndarray, n_samples_cap: int = 200_000,
) -> dict:
    """Centered SVD on activations (or a sample, if too large). Returns
    cumulative-variance thresholds k50/k90/k95/k99/k99.9 and participation
    ratio (effective rank).
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
        active_this_epoch = torch.zeros(n_features, dtype=torch.bool, device=device)
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
                active_this_epoch |= a
                ever_active |= a
                l0s.append(float((z > 0).float().sum(dim=-1).mean().item()))
        log["epoch_loss"].append(float(np.mean(losses)))
        log["epoch_var_explained"].append(
            1.0 - log["epoch_loss"][-1] / max(var_total, 1e-12)
        )
        log["epoch_dead_features"].append(int(n_features - active_this_epoch.sum().item()))
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
    """Mean per-cell aggregation of token-level features."""
    n_feat = token_features.shape[1]
    out = np.zeros((n_cells, n_feat), dtype=np.float32)
    counts = np.zeros(n_cells, dtype=np.int64)
    np.add.at(out, token_cell_idx, token_features)
    np.add.at(counts, token_cell_idx, 1)
    out /= np.maximum(counts, 1).reshape(-1, 1)
    return out
