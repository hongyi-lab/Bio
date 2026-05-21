"""SVD diagnostic for phase 3: distinguish low-rank collapse (Alt A) from
trivial-reconstruction (Alt B), and confirm scale parity across layers.

Inputs (all cached on disk from phase3_sae.py):
    results/phase3/<layer>/token_activations.npz   (acts: N_tokens x d_model)
    results/phase3/<layer>/training_log.json       (epoch_loss[0], epoch_var_explained[0])

Outputs:
    results/phase3/svd_diagnostic.json    raw stats + spectrum metrics per layer
    results/phase3/svd_spectrum.png       cumulative variance curves

What the numbers mean:
    Alt A (low-rank collapse) — layer 12 dead-feature count reflects manifold,
    not SAE config. Look at k99 / participation-ratio: if k99 << d_model only
    at layer 12, the activation lives in a small subspace; an SAE with k=32 and
    dict=2048 can't help but leave most of the dictionary dead.

    Alt B (TopK trivial-reconstruction) — var_exp=1.000 only because the input
    is near-zero noise. Look at raw_variance_per_elem: if layer 12 raw variance
    is comparable to layer 0, var_exp=1 is a real claim about reconstruction
    quality, not a scale artifact. Cross-check: epoch_loss[0] / raw_variance
    should equal (1 - epoch_var_explained[0]).

Usage:
    python src/phase3_svd_diag.py
    python src/phase3_svd_diag.py --layers layer_00_input layer_12
    python src/phase3_svd_diag.py --max_rows 500000   # subsample for speed
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PHASE3_DIR = ROOT / "results" / "phase3"


def _spectrum_via_gram(X: np.ndarray, center: bool = True) -> np.ndarray:
    """Return singular values of X (centered if requested) by eigendecomposing
    the d x d Gram matrix. Cheaper than full SVD when N >> d.

    eigvals(X^T X / N) == singular_values(X)^2 / N
    Returns singular values themselves, sorted descending.
    """
    N, d = X.shape
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    # X^T X computed in float64 for numerical stability of small eigvals
    G = (X.T.astype(np.float64) @ X.astype(np.float64))
    eig = np.linalg.eigvalsh(G)[::-1]              # descending
    eig = np.clip(eig, 0.0, None)
    sing = np.sqrt(eig)
    return sing


def _spectrum_metrics(singular_values: np.ndarray) -> Dict[str, float]:
    s = np.asarray(singular_values, dtype=np.float64)
    var = s ** 2
    total = var.sum()
    cum = np.cumsum(var) / max(total, 1e-30)

    def k_for(threshold: float) -> int:
        idx = int(np.searchsorted(cum, threshold))
        return int(min(idx + 1, len(s)))

    # Participation ratio = (sum lambda)^2 / sum lambda^2 — soft "effective rank"
    pr = float(total ** 2 / max((var ** 2).sum(), 1e-30))
    return {
        "d_model": int(len(s)),
        "total_variance": float(total),
        "k_for_var_0p50": k_for(0.50),
        "k_for_var_0p90": k_for(0.90),
        "k_for_var_0p95": k_for(0.95),
        "k_for_var_0p99": k_for(0.99),
        "k_for_var_0p999": k_for(0.999),
        "participation_ratio": pr,
        "top1_share":  float(var[0] / max(total, 1e-30)),
        "top10_share": float(var[:10].sum() / max(total, 1e-30)),
        "top50_share": float(var[:50].sum() / max(total, 1e-30)),
    }


def _raw_stats(X: np.ndarray) -> Dict[str, float]:
    """Per-element / per-dim stats. epoch_loss is mean-squared-error per element,
    so the relevant comparison is `var_per_element` (Frobenius_norm^2 / (N*d))."""
    N, d = X.shape
    per_dim_std = X.std(axis=0)
    var_per_elem = float((X.astype(np.float64) ** 2).mean()
                         - (X.astype(np.float64).mean()) ** 2)
    return {
        "n_tokens": int(N),
        "d_model": int(d),
        "abs_mean": float(np.abs(X).mean()),
        "per_dim_std_mean": float(per_dim_std.mean()),
        "per_dim_std_median": float(np.median(per_dim_std)),
        "per_dim_std_max": float(per_dim_std.max()),
        "per_dim_std_min": float(per_dim_std.min()),
        "frobenius_norm_sq_div_Nd": var_per_elem,  # this matches the MSE convention
        "var_per_element_centered": var_per_elem,
    }


def analyze_layer(layer_dir: Path, max_rows: int | None) -> dict:
    cache = layer_dir / "token_activations.npz"
    log = layer_dir / "training_log.json"
    if not cache.exists():
        raise FileNotFoundError(f"missing {cache}; run phase3_sae.py first")

    t0 = time.time()
    print(f"[svd] {layer_dir.name}: loading {cache.stat().st_size / 1e9:.2f} GB cache ...",
          flush=True)
    npz = np.load(cache, mmap_mode="r")
    X_full = np.asarray(npz["acts"])
    print(f"[svd]   acts shape = {X_full.shape}, dtype = {X_full.dtype}", flush=True)

    if max_rows is not None and X_full.shape[0] > max_rows:
        rng = np.random.default_rng(0)
        sel = rng.choice(X_full.shape[0], size=max_rows, replace=False)
        X = X_full[np.sort(sel)].astype(np.float32, copy=False)
        print(f"[svd]   subsampled to {X.shape[0]} rows for speed", flush=True)
    else:
        X = np.ascontiguousarray(X_full, dtype=np.float32)

    raw = _raw_stats(X)
    print(f"[svd]   raw: abs_mean={raw['abs_mean']:.4g}  "
          f"per_dim_std_mean={raw['per_dim_std_mean']:.4g}  "
          f"var_per_elem={raw['var_per_element_centered']:.4g}", flush=True)

    sing = _spectrum_via_gram(X, center=True)
    sm = _spectrum_metrics(sing)
    print(f"[svd]   spectrum: k99={sm['k_for_var_0p99']}/{sm['d_model']}  "
          f"k95={sm['k_for_var_0p95']}  participation_ratio={sm['participation_ratio']:.2f}",
          flush=True)

    # Cross-check Alt B: epoch_loss[0] vs raw variance
    out = {"raw_stats": raw, "spectrum": sm, "singular_values": sing.tolist()}
    if log.exists():
        tlog = json.loads(log.read_text())
        loss0 = float(tlog["epoch_loss"][0])
        ve0 = float(tlog["epoch_var_explained"][0])
        loss_final = float(tlog["epoch_loss"][-1])
        ve_final = float(tlog["epoch_var_explained"][-1])
        implied_v_init = loss0 / max(1.0 - ve0, 1e-12)
        implied_v_final = loss_final / max(1.0 - ve_final, 1e-12)
        out["cross_check_alt_B"] = {
            "epoch_loss_first": loss0,
            "epoch_var_explained_first": ve0,
            "implied_var_per_elem_from_first_epoch": implied_v_init,
            "epoch_loss_final": loss_final,
            "epoch_var_explained_final": ve_final,
            "implied_var_per_elem_from_final_epoch": implied_v_final,
            "measured_var_per_elem": raw["var_per_element_centered"],
            "ratio_implied_to_measured_final": (
                implied_v_final / max(raw["var_per_element_centered"], 1e-12)
            ),
            "note": (
                "var_per_elem implied by SAE training (loss / (1 - var_exp)) should "
                "match the directly measured per-element variance to within ~1%. "
                "A drastic mismatch would mean the var_explained metric in training "
                "is measuring something different from what we think; a match means "
                "the layer-12 'var_exp = 1.000' is a real reconstruction claim, not "
                "a scale artifact (Alt B refuted)."
            ),
        }
    out["wall_time_s"] = time.time() - t0
    return out


def plot_spectra(per_layer: Dict[str, dict], out_png: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    cmap = {"layer_00_input": "C0", "layer_03": "C2", "layer_12": "C3"}
    for layer_name, res in per_layer.items():
        sv = np.asarray(res["singular_values"], dtype=np.float64)
        var = sv ** 2
        cum = np.cumsum(var) / max(var.sum(), 1e-30)
        xs = np.arange(1, len(sv) + 1)
        c = cmap.get(layer_name, None)
        ax1.semilogy(xs, var / max(var.sum(), 1e-30), label=layer_name, color=c)
        ax2.plot(xs, cum, label=layer_name, color=c)
    ax1.set_xlabel("component rank"); ax1.set_ylabel("normalized variance (log)")
    ax1.set_title("activation spectrum"); ax1.grid(alpha=0.3); ax1.legend()
    ax2.set_xlabel("component rank"); ax2.set_ylabel("cumulative variance")
    ax2.set_title("cumulative variance explained")
    ax2.grid(alpha=0.3); ax2.legend(loc="lower right")
    for thr, ls in [(0.90, ":"), (0.95, "--"), (0.99, "-.")]:
        ax2.axhline(thr, color="gray", linestyle=ls, alpha=0.5)
    fig.tight_layout(); fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--layers", nargs="+",
                   default=["layer_00_input", "layer_03", "layer_12"])
    p.add_argument("--max_rows", type=int, default=None,
                   help="subsample N rows for speed (default: use all)")
    p.add_argument("--out_json", default="results/phase3/svd_diagnostic.json")
    p.add_argument("--out_png",  default="results/phase3/svd_spectrum.png")
    args = p.parse_args()

    out_json = ROOT / args.out_json if not Path(args.out_json).is_absolute() else Path(args.out_json)
    out_png  = ROOT / args.out_png  if not Path(args.out_png).is_absolute()  else Path(args.out_png)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    per_layer: Dict[str, dict] = {}
    for name in args.layers:
        layer_dir = PHASE3_DIR / name
        if not layer_dir.exists():
            print(f"[svd] WARN: {layer_dir} missing — skipping", flush=True)
            continue
        per_layer[name] = analyze_layer(layer_dir, args.max_rows)

    summary: List[str] = []
    summary.append("layer            | d  | k50 | k90 | k95 | k99  | k99.9 | PR    | var_per_elem")
    summary.append("-" * 82)
    for name, res in per_layer.items():
        sm = res["spectrum"]; raw = res["raw_stats"]
        summary.append(
            f"{name:16s} | {sm['d_model']:3d} | "
            f"{sm['k_for_var_0p50']:3d} | {sm['k_for_var_0p90']:3d} | {sm['k_for_var_0p95']:3d} | "
            f"{sm['k_for_var_0p99']:4d} | {sm['k_for_var_0p999']:5d} | "
            f"{sm['participation_ratio']:5.1f} | {raw['var_per_element_centered']:.4g}"
        )

    print("\n=== SVD DIAGNOSTIC ===")
    print("\n".join(summary))

    print("\n=== Alt-B cross-check (epoch_loss / (1 - var_exp) should ≈ measured var) ===")
    for name, res in per_layer.items():
        if "cross_check_alt_B" not in res:
            continue
        c = res["cross_check_alt_B"]
        print(f"{name:16s}  implied(final)={c['implied_var_per_elem_from_final_epoch']:.4g}  "
              f"measured={c['measured_var_per_elem']:.4g}  "
              f"ratio={c['ratio_implied_to_measured_final']:.3f}")

    out = {
        "per_layer": per_layer,
        "summary_table": summary,
        "interpretation": {
            "alt_A_low_rank_collapse": (
                "If k_for_var_0p99 is small only at layer 12 (e.g., <100/512), "
                "the manifold is genuinely low-rank, and the 844 dead SAE features "
                "are forced by the data rather than a config artifact. If all layers "
                "have similar k99, then dict_size=2048 was simply too generous and "
                "the layer 12 dead-feature signal is weak evidence for collapse."
            ),
            "alt_B_trivial_reconstruction": (
                "If raw var_per_element at layer 12 is comparable to layer 0 "
                "(same order of magnitude), and the cross-check ratio is ~1.0, then "
                "var_explained = 1.000 is a real reconstruction claim. If layer 12 "
                "raw variance is 100x smaller than layer 0, then var_exp = 1 just "
                "means 'reconstructed near-zero signal'."
            ),
        },
    }
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\n[svd] wrote {out_json}")
    if per_layer:
        plot_spectra(per_layer, out_png)
        print(f"[svd] wrote {out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
