"""plot_phase7_summary.py — cross-recipe overlay plot for phase 7.

Reads `results/<recipe>/phase7_summary.json` for each given recipe slug and
emits results/_phase7_cross_recipe.{png,pdf} + a markdown table.

The two main panels overlay Evo-2 7B and ESM-2 15B (LLM-scale) curves together
with phase 5's small-scale baselines (scGPT 51M, HyenaDNA 6.6M) when those
recipes are also passed in.

Usage:
    # default — show both LLM-scale runs alone
    python src/plot_phase7_summary.py evo2_7b__genomic_benchmarks esm2_15b__deeploc

    # with small-scale phase-5 context (different recipe slug naming)
    python src/plot_phase7_summary.py evo2_7b__genomic_benchmarks esm2_15b__deeploc \
        --phase5_recipes scgpt__pbmc3k hyenadna__genomic_benchmarks
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
RESULTS = ROOT / "results"


def _load_phase7_summary(slug: str) -> Dict:
    p = RESULTS / slug / "phase7_summary.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _load_phase5_recipe(slug: str) -> Dict:
    """Minimal loader for phase 5 audit/ outputs (different layout)."""
    audit = RESULTS / slug / "audit"
    if not audit.exists():
        return {}
    out = {"model": slug.split("__")[0], "dataset": slug.split("__")[1],
           "per_layer": {}}
    sae_dir = audit / "sae"
    if sae_dir.exists():
        for layer_dir in sorted(sae_dir.iterdir()):
            ag = layer_dir / "ablation_gap.json"
            if not ag.exists():
                continue
            gap = json.loads(ag.read_text())
            out["per_layer"][layer_dir.name] = {
                "gap_K_grid": gap["gap"]["K_grid"],
                "gap_mean": gap["gap"]["gap_mean"],
                "gap_std": gap["gap"]["gap_std"],
                "gap_null_mean": gap.get("gap_random", {}).get("gap_mean"),
                "gap_null_std": gap.get("gap_random", {}).get("gap_std"),
            }
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("recipes", nargs="+",
                   help="phase 7 recipe slugs, e.g. evo2_7b__genomic_benchmarks")
    p.add_argument("--phase5_recipes", nargs="+", default=[],
                   help="optional phase 5 recipe slugs for context")
    p.add_argument("--out_png", default="results/_phase7_cross_recipe.png")
    p.add_argument("--out_pdf", default="results/_phase7_cross_recipe.pdf")
    p.add_argument("--out_md", default="results/_phase7_cross_recipe.md")
    args = p.parse_args()

    phase7 = {}
    for slug in args.recipes:
        d = _load_phase7_summary(slug)
        if d:
            phase7[slug] = d
        else:
            print(f"[plot] WARN: no phase7_summary.json under results/{slug}/")

    phase5 = {}
    for slug in args.phase5_recipes:
        d = _load_phase5_recipe(slug)
        if d.get("per_layer"):
            phase5[slug] = d

    if not phase7:
        raise SystemExit("no phase 7 recipe data found")

    # ---------- plot ----------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel A: probe acc per layer per recipe (PCA + SAE)
    ax = axes[0]
    bar_x = 0
    bar_width = 0.35
    xticklabels = []
    for slug, d in phase7.items():
        layers = d["sae_layers_probed"]
        for l in layers:
            s = d["per_layer"][l]
            pca = s["pca_probe"]["accuracy_mean"]
            pca_sd = s["pca_probe"]["accuracy_std"]
            sae = s["sae_probe"]["accuracy_mean"]
            sae_sd = s["sae_probe"]["accuracy_std"]
            ax.bar(bar_x - bar_width / 2, pca, bar_width,
                   yerr=pca_sd, color="C0",
                   label="PCA-probe" if bar_x == 0 else None)
            ax.bar(bar_x + bar_width / 2, sae, bar_width,
                   yerr=sae_sd, color="C1",
                   label="SAE-probe" if bar_x == 0 else None)
            xticklabels.append(f"{slug.split('__')[0]}\n{l}")
            bar_x += 1
    ax.set_xticks(np.arange(bar_x))
    ax.set_xticklabels(xticklabels, rotation=45, fontsize=7, ha="right")
    ax.set_ylabel("probe accuracy")
    ax.set_title("Per-layer probe accuracy (PCA vs SAE features)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    # Panel B: ablation gap curves with random null
    ax = axes[1]
    colors = {"evo2_7b": "C0", "esm2_t48_15B_UR50D": "C3",
              "esm2_15b": "C3", "scgpt": "C2", "hyenadna": "C4"}
    style = {"layer_00_input": "-", "mid": "--", "final": ":"}

    for slug, d in phase7.items():
        layers = d["sae_layers_probed"]
        model_key = slug.split("__")[0]
        base_color = colors.get(model_key, "C5")
        for li, l in enumerate(layers):
            s = d["per_layer"][l]
            K = np.array(s["gap_K_grid"])
            m = np.array(s["gap_mean"])
            sd = np.array(s["gap_std"])
            ls = "-" if li == 0 else ("--" if li == 1 else ":")
            ax.plot(K, m, marker="o", color=base_color, ls=ls,
                    label=f"{model_key} {l}", lw=1.5)
            ax.fill_between(K, m - sd, m + sd, color=base_color, alpha=0.12)
            if s.get("gap_null_mean") is not None:
                nm = np.array(s["gap_null_mean"])
                ns = np.array(s["gap_null_std"])
                # Random-null 2σ band as faint shaded region
                ax.fill_between(K, nm - 2 * ns, nm + 2 * ns,
                                color=base_color, alpha=0.04, hatch="//")

    # phase 5 small-scale reference lines
    for slug, d in phase5.items():
        model_key = d["model"]
        base_color = colors.get(model_key, "gray")
        for l, s in d["per_layer"].items():
            K = np.array(s["gap_K_grid"])
            m = np.array(s["gap_mean"])
            sd = np.array(s["gap_std"])
            ax.plot(K, m, color=base_color, alpha=0.4, lw=1, ls=":",
                    label=f"(phase 5) {model_key} {l}")

    ax.axhline(0, color="black", ls=":", alpha=0.4)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("K (ablated features)")
    ax.set_ylabel("gap = drop_PCA − drop_SAE")
    ax.set_title("SAE−PCA ablation gap curves\n(hatched = 2σ random-null band)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, loc="best")

    fig.suptitle(
        "Phase 7 cross-recipe — Evo-2 7B (DNA) × ESM-2 15B (protein) LLM-scale audit",
        y=0.99, fontsize=12, weight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=140, bbox_inches="tight")
    fig.savefig(args.out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {args.out_png}")
    print(f"[plot] wrote {args.out_pdf}")

    # ---------- markdown digest ----------
    md = [f"# Phase 7 cross-recipe digest\n"]
    md.append("| recipe | best PCA acc | best SAE acc | Δ(SAE − PCA) | "
              "max gap (sig?) |")
    md.append("|---|---|---|---|---|")
    for slug, d in phase7.items():
        layers = d["sae_layers_probed"]
        pca_best = max(d["per_layer"][l]["pca_probe"]["accuracy_mean"] for l in layers)
        sae_best = max(d["per_layer"][l]["sae_probe"]["accuracy_mean"] for l in layers)
        any_sig = any(d["per_layer"][l].get("any_K_significant") for l in layers)
        max_gap = max(
            max(d["per_layer"][l]["gap_mean"]) if d["per_layer"][l]["gap_mean"] else 0
            for l in layers
        )
        sig_marker = "★" if any_sig else ""
        md.append(f"| {slug} | {pca_best:.4f} | {sae_best:.4f} | "
                  f"{sae_best - pca_best:+.4f} | {max_gap:+.3f}{sig_marker} |")
    md.append("")
    md.append("★ = at least one (layer, K) pair has real gap exceeding the "
              "random-null gap by 2σ.\n")

    md.append("## Per-layer detail")
    for slug, d in phase7.items():
        md.append(f"\n### {slug}")
        md.append("| layer | PCA acc | SAE acc | gap@best K | any K sig? |")
        md.append("|---|---|---|---|---|")
        for l in d["sae_layers_probed"]:
            s = d["per_layer"][l]
            pca = s["pca_probe"]["accuracy_mean"]
            sae = s["sae_probe"]["accuracy_mean"]
            if s["gap_mean"]:
                best_k_idx = int(np.argmax(s["gap_mean"]))
                best_gap = s["gap_mean"][best_k_idx]
                best_K = s["gap_K_grid"][best_k_idx]
            else:
                best_gap, best_K = 0, "-"
            sig = "★" if s.get("any_K_significant") else ""
            md.append(f"| {l} | {pca:.4f} | {sae:.4f} | "
                      f"{best_gap:+.3f} @ K={best_K} | {sig} |")

    Path(args.out_md).write_text("\n".join(md))
    print(f"[plot] wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
