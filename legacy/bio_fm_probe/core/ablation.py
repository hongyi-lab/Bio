"""SAE − PCA ablation gap — the headline cross-FM knowledge-geometry metric.

Theoretical framing (user's note translated):
  SAE's interpretability comes from inverting two assumptions: linear
  representation hypothesis and sparse activation hypothesis. When these hold,
  sparse dictionary decomposition is essentially unique. SAE = basis change
  from polysemantic neuron coordinates to a basis where knowledge is sparse.

  Concentrated models (knowledge on few neurons): PCA top-K can locate it,
  SAE adds little.
  Distributed models (knowledge spread non-orthogonally across many neurons):
  PCA misses it, SAE recovers the hidden sparse dictionary.

  The SAE−PCA ablation gap is therefore an empirical measure of each model's
  knowledge-carrying *geometry*. Cross-FM, it's a curve over K (number of
  ablated features) reporting how distributed each model's representations are.

Protocol (user-confirmed):
  - Ablation level:   probe-level (zero columns in probe input, refit LR)
  - Selection:        probe coefficient L2-norm (per-feature ‖w‖₂ across classes)
  - Output:           gap = drop_PCA − drop_SAE, per K in K_grid, per seed

Plus a random-selection null baseline (gap should be ~0 under random ablation).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


DEFAULT_K_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def _lr() -> LogisticRegression:
    return LogisticRegression(
        max_iter=2000, solver="lbfgs", n_jobs=-1, C=1.0,
    )


def _probe_coef_l2(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """Train LR on X_train, return per-feature ‖coef‖₂.

    For multinomial LR with K classes, coef_ is (K, n_features); we take the
    L2-norm across classes per feature to get a single importance score per
    feature.
    """
    clf = _lr()
    clf.fit(X_train, y_train)
    coef = clf.coef_  # (n_classes, n_features) — or (1, n_features) for binary
    return np.linalg.norm(coef, axis=0)  # (n_features,)


def _ablate_and_score(
    X: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray,
    ablate_idx: np.ndarray,
) -> float:
    """Zero columns ablate_idx in X, refit LR on tr, score on te."""
    X_ab = X.copy()
    if ablate_idx.size:
        X_ab[:, ablate_idx] = 0.0
    clf = _lr()
    clf.fit(X_ab[tr], y[tr])
    return float(accuracy_score(y[te], clf.predict(X_ab[te])))


def _ablation_curve_one_seed(
    X: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray,
    K_grid: List[int], criterion: str, rng: np.random.Generator,
) -> Tuple[float, Dict[int, float]]:
    """For one (train, test) split, compute base acc + per-K ablated acc.

    Returns:
        base_acc: accuracy with no ablation
        per_K:    {K: ablated_acc} for K in K_grid (K capped at X.shape[1])
    """
    n_feat = X.shape[1]

    # Base accuracy (no ablation) — also gives us the ranking
    clf = _lr()
    clf.fit(X[tr], y[tr])
    base_acc = float(accuracy_score(y[te], clf.predict(X[te])))

    if criterion == "probe_l2":
        importance = np.linalg.norm(clf.coef_, axis=0)
        ranking = np.argsort(-importance)  # descending
    elif criterion == "random":
        ranking = rng.permutation(n_feat)
    else:
        raise ValueError(f"unknown criterion {criterion!r}")

    per_K: Dict[int, float] = {}
    for K in K_grid:
        K_eff = min(K, n_feat)
        ablate_idx = ranking[:K_eff]
        per_K[K] = _ablate_and_score(X, y, tr, te, ablate_idx)
    return base_acc, per_K


def ablation_curve(
    X: np.ndarray, y: np.ndarray, splits, K_grid: List[int] = None,
    criterion: str = "probe_l2", seed_for_random: int = 0,
) -> Dict:
    """Ablation curve over K, aggregated across splits.

    For each (train, test) split:
      1. Fit LR on train.
      2. Score on test → base_acc.
      3. Rank features by criterion ("probe_l2": ‖coef‖₂; "random": uniform).
      4. For each K in K_grid, zero top-K features, refit LR, score on test.

    Returns:
        {
            "base_acc_mean":   float,
            "base_acc_std":    float,
            "K_grid":          list[int],
            "ablated_acc_mean": list[float],
            "ablated_acc_std":  list[float],
            "drop_mean":       list[float],   # base - ablated
            "drop_std":        list[float],
            "criterion":       str,
            "n_features":      int,
            "n_seeds":         int,
            "per_seed": [...]  # full per-seed records
        }
    """
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
) -> Dict:
    """Headline metric — full per-K ablation curves for both SAE and PCA,
    plus the gap series gap = drop_PCA − drop_SAE.

    gap > 0 ⇒ ablating top-K PCA dims hurts more than ablating top-K SAE dims
              ⇒ knowledge is more *distributed* than PCA can capture
              ⇒ SAE recovers a hidden sparse dictionary that PCA misses.

    gap ≈ 0 ⇒ PCA and SAE find the same task-discriminative directions
              ⇒ knowledge is *concentrated* in a few PCA-aligned dims
              ⇒ no extra inversion space; SAE adds nothing PCA didn't already.

    Args:
        X_sae:  (n_samples, n_sae_features) per-sample SAE-aggregated features
        X_pca:  (n_samples, n_pca_features) per-sample PCA features (any K)
        y:      (n_samples,) labels
        splits: list of (train_idx, test_idx) tuples
        K_grid: ablation K values (default DEFAULT_K_GRID)
        include_random_null: also compute random-selection ablation as null

    Returns a dict with:
        sae:    ablation_curve(X_sae, ..., "probe_l2")
        pca:    ablation_curve(X_pca, ..., "probe_l2")
        gap:    per-K mean ± std of (drop_PCA - drop_SAE)
        sae_random:  (if include_random_null) random-selection null for SAE
        pca_random:  (if include_random_null) random-selection null for PCA
        gap_random:  (if include_random_null) random-null gap — should be ~0
    """
    K_grid = list(K_grid) if K_grid is not None else list(DEFAULT_K_GRID)
    print(f"[ablation] SAE-PCA gap: X_sae={X_sae.shape}, X_pca={X_pca.shape}, "
          f"n_seeds={len(splits)}, K_grid={K_grid}")

    out: Dict = {}
    out["sae"] = ablation_curve(X_sae, y, splits, K_grid, criterion="probe_l2")
    out["pca"] = ablation_curve(X_pca, y, splits, K_grid, criterion="probe_l2")

    # Per-seed gap (drop_PCA − drop_SAE) per K
    sae_drops = np.array(
        [[r["ablated_acc"][K] for K in K_grid] for r in out["sae"]["per_seed"]],
        dtype=np.float64,
    )
    pca_drops = np.array(
        [[r["ablated_acc"][K] for K in K_grid] for r in out["pca"]["per_seed"]],
        dtype=np.float64,
    )
    sae_base = np.array([r["base_acc"] for r in out["sae"]["per_seed"]])
    pca_base = np.array([r["base_acc"] for r in out["pca"]["per_seed"]])
    # drop_PCA - drop_SAE = (pca_base - pca_ablated) - (sae_base - sae_ablated)
    gaps = (pca_base[:, None] - pca_drops) - (sae_base[:, None] - sae_drops)
    out["gap"] = {
        "K_grid": K_grid,
        "gap_mean": gaps.mean(axis=0).tolist(),
        "gap_std": (gaps.std(axis=0, ddof=1) if len(gaps) > 1
                    else np.zeros(len(K_grid))).tolist(),
        "interpretation": (
            "gap > 0 ⇒ knowledge more distributed than PCA captures; "
            "gap ≈ 0 ⇒ knowledge concentrated in PCA-aligned dims."
        ),
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
             for r in out["sae_random"]["per_seed"]],
            dtype=np.float64,
        )
        pca_r = np.array(
            [[r["ablated_acc"][K] for K in K_grid]
             for r in out["pca_random"]["per_seed"]],
            dtype=np.float64,
        )
        gaps_r = (pca_base[:, None] - pca_r) - (sae_base[:, None] - sae_r)
        out["gap_random"] = {
            "K_grid": K_grid,
            "gap_mean": gaps_r.mean(axis=0).tolist(),
            "gap_std": (gaps_r.std(axis=0, ddof=1) if len(gaps_r) > 1
                        else np.zeros(len(K_grid))).tolist(),
            "interpretation": (
                "random-null gap; should be ≈ 0 within seed std. "
                "If real gap is not significantly above this, the metric is "
                "not informative."
            ),
        }

    return out
