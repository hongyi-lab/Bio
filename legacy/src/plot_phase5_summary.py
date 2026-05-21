"""Phase 5 big-picture figure.

Reads results/{scgpt__pbmc3k, hyenadna__genomic_benchmarks}/audit/ JSON outputs
and produces results/_phase5_summary.{png,pdf}.

Six panels (2 rows × 3 cols):
  Row 1 — model vs baseline
    (0,0) scGPT per-layer probe acc + log1p_pca baseline
    (0,1) HyenaDNA per-layer probe acc + kmer_pca baseline
    (0,2) SVD participation ratio per layer (overlay)

  Row 2 — SAE−PCA ablation gap and its decomposition
    (1,0) scGPT gap curves per layer (gap = drop_PCA − drop_SAE)
    (1,1) HyenaDNA gap curves per layer
    (1,2) drop_PCA vs drop_SAE decomposed for two "extreme" layers
          (scGPT L06 — big negative dip; HyenaDNA L00_input — large positive plateau)

The decomposition panel matters because it shows whether a given gap value
comes from PCA collapsing, SAE collapsing, or both — which the protocol
critique in our analysis hinges on.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

SCGPT = RESULTS / "scgpt__pbmc3k" / "audit"
HYENA = RESULTS / "hyenadna__genomic_benchmarks" / "audit"


def _load_json(p: Path) -> Dict:
    return json.loads(p.read_text())


def _per_layer_accs(per_layer_json: Dict):
    layers = per_layer_json["layers"]
    results = per_layer_json["results"]
    out = {}
    for pool in ("cls", "mean"):
        if all(pool in results[l] for l in layers):
            m = np.array([results[l][pool]["accuracy_mean"] for l in layers])
            s = np.array([results[l][pool]["accuracy_std"] for l in layers])
            out[pool] = (m, s)
    return layers, out


def _ablation_data(audit_dir: Path):
    """Return {layer: gap_json} for all SAE layers under this audit."""
    sae_dir = audit_dir / "sae"
    out = {}
    for layer_dir in sorted(sae_dir.iterdir()):
        ag = layer_dir / "ablation_gap.json"
        if ag.exists():
            out[layer_dir.name] = _load_json(ag)
    return out


def _svd_data(audit_dir: Path):
    return _load_json(audit_dir / "svd_diag.json")


def main() -> int:
    scgpt_pl = _load_json(SCGPT / "per_layer_probe.json")
    scgpt_bl = _load_json(SCGPT / "baselines.json")
    scgpt_svd = _svd_data(SCGPT)
    scgpt_ab = _ablation_data(SCGPT)

    hyena_pl = _load_json(HYENA / "per_layer_probe.json")
    hyena_bl = _load_json(HYENA / "baselines.json")
    hyena_svd = _svd_data(HYENA)
    hyena_ab = _ablation_data(HYENA)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # ---- (0,0) scGPT probe acc ----
    ax = axes[0, 0]
    layers, accs = _per_layer_accs(scgpt_pl)
    xs = np.arange(len(layers))
    xt = [l.replace("layer_", "").replace("_input", "in")
          .lstrip("0") if "_input" in l or l != "layer_00_input" else "in"
          for l in layers]
    xt = ["in" if "input" in l else l.replace("layer_", "").lstrip("0") or "0"
          for l in layers]
    for pool, color, marker, ls in (("cls", "C0", "o", "-"),
                                     ("mean", "C1", "s", "--")):
        if pool in accs:
            m, s = accs[pool]
            ax.plot(xs, m, marker=marker, ls=ls, color=color, label=f"{pool}-pool")
            ax.fill_between(xs, m - s, m + s, color=color, alpha=0.15)
    bl = scgpt_bl["probe"]
    ax.axhline(bl["accuracy_mean"], color="red", ls=":", lw=2,
               label=f"log1p+PCA-50 = {bl['accuracy_mean']:.3f}")
    ax.fill_between(xs,
                    bl["accuracy_mean"] - bl["accuracy_std"],
                    bl["accuracy_mean"] + bl["accuracy_std"],
                    color="red", alpha=0.08)
    ax.set_xticks(xs)
    ax.set_xticklabels(xt, rotation=0, fontsize=8)
    ax.set_xlabel("layer")
    ax.set_ylabel("probe accuracy (5-seed mean ± std)")
    ax.set_title("scGPT × pbmc3k\nprobe acc vs log1p+PCA-50 baseline")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")

    # ---- (0,1) HyenaDNA probe acc ----
    ax = axes[0, 1]
    layers, accs = _per_layer_accs(hyena_pl)
    xs = np.arange(len(layers))
    xt = ["in" if "input" in l else l.replace("layer_", "").lstrip("0") or "0"
          for l in layers]
    for pool, color, marker, ls in (("mean", "C1", "s", "--"),):
        if pool in accs:
            m, s = accs[pool]
            ax.plot(xs, m, marker=marker, ls=ls, color=color, label=f"{pool}-pool")
            ax.fill_between(xs, m - s, m + s, color=color, alpha=0.15)
    bl = hyena_bl["probe"]
    ax.axhline(bl["accuracy_mean"], color="red", ls=":", lw=2,
               label=f"kmer+PCA-50 = {bl['accuracy_mean']:.3f}")
    ax.fill_between(xs,
                    bl["accuracy_mean"] - bl["accuracy_std"],
                    bl["accuracy_mean"] + bl["accuracy_std"],
                    color="red", alpha=0.08)
    ax.set_xticks(xs)
    ax.set_xticklabels(xt, rotation=0, fontsize=8)
    ax.set_xlabel("layer")
    ax.set_ylabel("probe accuracy (5-seed mean ± std)")
    ax.set_title("HyenaDNA × genomic_benchmarks\nprobe acc vs kmer+PCA-50 baseline")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    # ---- (0,2) SVD participation ratio ----
    ax = axes[0, 2]
    # scGPT
    scgpt_layers = list(scgpt_svd.keys())
    scgpt_pr = [scgpt_svd[l]["participation_ratio"] for l in scgpt_layers]
    scgpt_k95 = [scgpt_svd[l]["k95"] for l in scgpt_layers]
    hyena_layers = list(hyena_svd.keys())
    hyena_pr = [hyena_svd[l]["participation_ratio"] for l in hyena_layers]
    hyena_k95 = [hyena_svd[l]["k95"] for l in hyena_layers]

    n_s = len(scgpt_layers)
    n_h = len(hyena_layers)
    xs_s = np.arange(n_s)
    xs_h = np.arange(n_s, n_s + n_h)
    ax.bar(xs_s, scgpt_pr, color="C0", alpha=0.7, label="scGPT PR (eff. rank)")
    ax.bar(xs_h, hyena_pr, color="C2", alpha=0.7, label="HyenaDNA PR (eff. rank)")
    # k95 on twin axis
    ax2 = ax.twinx()
    ax2.plot(xs_s, scgpt_k95, "o-", color="C0", alpha=0.9, lw=1.5,
             label="scGPT k95")
    ax2.plot(xs_h, hyena_k95, "s-", color="C2", alpha=0.9, lw=1.5,
             label="HyenaDNA k95")
    ax.set_xticks(np.concatenate([xs_s, xs_h]))
    ax.set_xticklabels(
        [l.replace("layer_", "").replace("_input", "in") for l in scgpt_layers + hyena_layers],
        rotation=0, fontsize=7,
    )
    ax.set_ylabel("participation ratio (bars)", color="black")
    ax2.set_ylabel("k95 (markers)", color="black")
    ax.set_xlabel("layer (scGPT first 3, HyenaDNA next 3)")
    ax.set_title("SVD spectrum — effective rank per layer")
    ax.grid(alpha=0.3, axis="y")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")
    # annotate scGPT vs HyenaDNA d_model
    ax.axvline(n_s - 0.5, color="gray", ls=":", alpha=0.5)
    ax.text(0.02, 0.92, f"scGPT d=512", transform=ax.transAxes, fontsize=8,
            color="C0", weight="bold")
    ax.text(0.55, 0.92, f"HyenaDNA d=256", transform=ax.transAxes, fontsize=8,
            color="C2", weight="bold")

    # ---- (1,0) scGPT ablation gap curves ----
    ax = axes[1, 0]
    colors = {"layer_00_input": "C0", "layer_06": "C3", "layer_12": "C2"}
    for layer in scgpt_ab.keys():
        gap = scgpt_ab[layer]["gap"]
        K = np.array(gap["K_grid"])
        m = np.array(gap["gap_mean"])
        s = np.array(gap["gap_std"])
        ax.plot(K, m, marker="o", color=colors.get(layer, "C5"),
                label=layer, lw=1.7)
        ax.fill_between(K, m - s, m + s, color=colors.get(layer, "C5"), alpha=0.2)
    ax.axhline(0, color="black", ls=":", alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("K (ablated features)")
    ax.set_ylabel("gap = drop_PCA − drop_SAE")
    ax.set_title("scGPT — ablation gap curves per layer")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")

    # ---- (1,1) HyenaDNA ablation gap curves ----
    ax = axes[1, 1]
    colors_h = {"layer_00_input": "C0", "layer_02": "C3", "layer_04": "C2"}
    for layer in hyena_ab.keys():
        gap = hyena_ab[layer]["gap"]
        K = np.array(gap["K_grid"])
        m = np.array(gap["gap_mean"])
        s = np.array(gap["gap_std"])
        ax.plot(K, m, marker="o", color=colors_h.get(layer, "C5"),
                label=layer, lw=1.7)
        ax.fill_between(K, m - s, m + s, color=colors_h.get(layer, "C5"), alpha=0.2)
    ax.axhline(0, color="black", ls=":", alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("K (ablated features)")
    ax.set_ylabel("gap = drop_PCA − drop_SAE")
    ax.set_title("HyenaDNA — ablation gap curves per layer")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    # ---- (1,2) drop decomposition for two "extreme" layers ----
    ax = axes[1, 2]
    # scGPT L06 (big negative dip): show drop_PCA and drop_SAE separately
    sg6 = scgpt_ab["layer_06"]
    sg6_pca_drop_mean = np.array(sg6["pca"]["base_acc_mean"]) - np.array(sg6["pca"]["ablated_acc_mean"])
    sg6_sae_drop_mean = np.array(sg6["sae"]["base_acc_mean"]) - np.array(sg6["sae"]["ablated_acc_mean"])
    K_sg = np.array(sg6["gap"]["K_grid"])
    ax.plot(K_sg, sg6_pca_drop_mean, "--", marker="o", color="C3",
            label="scGPT L06 — drop_PCA", lw=1.7)
    ax.plot(K_sg, sg6_sae_drop_mean, "-", marker="o", color="C3",
            label="scGPT L06 — drop_SAE", lw=1.7, alpha=0.7)

    # HyenaDNA L00_input (big positive plateau)
    hd0 = hyena_ab["layer_00_input"]
    hd0_pca_drop = np.array(hd0["pca"]["base_acc_mean"]) - np.array(hd0["pca"]["ablated_acc_mean"])
    hd0_sae_drop = np.array(hd0["sae"]["base_acc_mean"]) - np.array(hd0["sae"]["ablated_acc_mean"])
    K_hd = np.array(hd0["gap"]["K_grid"])
    ax.plot(K_hd, hd0_pca_drop, "--", marker="s", color="C0",
            label="HyenaDNA L00 — drop_PCA", lw=1.7)
    ax.plot(K_hd, hd0_sae_drop, "-", marker="s", color="C0",
            label="HyenaDNA L00 — drop_SAE", lw=1.7, alpha=0.7)

    ax.axhline(0, color="black", ls=":", alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("K (ablated features)")
    ax.set_ylabel("probe acc drop = base − ablated")
    ax.set_title("Decomposition: drop_PCA (dashed) vs drop_SAE (solid)\n"
                 "for two 'extreme' layers")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(
        "Phase 5 — bio FM probing summary: scGPT (scRNA, pbmc3k) × HyenaDNA (DNA, promoters)",
        y=0.995, fontsize=13, weight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_png = RESULTS / "_phase5_summary.png"
    out_pdf = RESULTS / "_phase5_summary.pdf"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_png}")
    print(f"[plot] wrote {out_pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
