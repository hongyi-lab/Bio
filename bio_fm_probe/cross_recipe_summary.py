"""Cross-recipe summary.

After running run_recipe.py on several (model × dataset) recipes, read each
recipe's audit/ JSONs and emit:
  - results/_master_table.md         summary table modality-grouped
  - results/_ablation_gap_curves.pdf one panel per modality, lines = (model × layer)

Usage:
    python -m bio_fm_probe.cross_recipe_summary \\
        scgpt__pbmc3k scgpt__immune_human \\
        geneformer__pbmc3k hyenadna__genomic_benchmarks esm2__deeploc
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


def _load_recipe(audit_dir: Path) -> Dict:
    out = {"audit_dir": str(audit_dir)}
    for name in ["baselines.json", "per_layer_probe.json", "svd_diag.json"]:
        f = audit_dir / name
        if f.exists():
            out[name.replace(".json", "")] = json.loads(f.read_text())
    out["sae"] = {}
    sae_dir = audit_dir / "sae"
    if sae_dir.exists():
        for layer_dir in sorted(sae_dir.iterdir()):
            if not layer_dir.is_dir():
                continue
            ag = layer_dir / "ablation_gap.json"
            tl = layer_dir / "training_log.json"
            entry = {}
            if ag.exists():
                entry["ablation_gap"] = json.loads(ag.read_text())
            if tl.exists():
                entry["training_log"] = json.loads(tl.read_text())
            if entry:
                out["sae"][layer_dir.name] = entry
    return out


def _best_layer_acc(per_layer_json: Dict) -> float:
    layers = per_layer_json["layers"]
    results = per_layer_json["results"]
    best = -1.0
    for n in layers:
        for pool in ("cls", "mean"):
            entry = results[n].get(pool)
            if entry is None:
                continue
            best = max(best, entry["accuracy_mean"])
    return best


def _modality_of(per_layer_json: Dict) -> str:
    # heuristic: if "cls" missing in any layer, model is CLS-less (HyenaDNA)
    layers = per_layer_json["layers"]
    results = per_layer_json["results"]
    has_cls = all("cls" in results[n] for n in layers)
    if not has_cls:
        return "dna"   # only DNA adapter we have without CLS in first push
    return "scrna_or_protein"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("recipes", nargs="+",
                   help="recipe slugs (e.g. scgpt__pbmc3k)")
    p.add_argument("--results_root", default="results")
    p.add_argument("--out_md", default="results/_master_table.md")
    p.add_argument("--out_pdf", default="results/_ablation_gap_curves.pdf")
    args = p.parse_args()

    root = Path(args.results_root)
    data: Dict[str, Dict] = {}
    for slug in args.recipes:
        audit = root / slug / "audit"
        if not audit.exists():
            print(f"[summary] WARN: no audit dir for {slug} ({audit})")
            continue
        data[slug] = _load_recipe(audit)
    if not data:
        raise SystemExit("no complete recipe outputs found")

    # ---------- master table ----------
    md = [f"# Cross-recipe master table — {len(data)} recipe(s)\n"]
    md.append("Each row = one (model, dataset) audit. `gap@K` is the headline "
              "SAE−PCA ablation gap at K ablated features in the SAE feature "
              "space (and matched K in PCA). Higher gap ⇒ knowledge more "
              "distributed; gap ≈ 0 ⇒ knowledge concentrated in PCA dims.\n")

    md.append("| recipe | baseline acc | best layer-probe acc | "
              "Δ vs baseline | gap@K=32 (best layer) | gap@K=128 (best) |")
    md.append("|---|---|---|---|---|---|")

    for slug, d in data.items():
        base = d.get("baselines", {}).get("probe", {})
        base_acc = base.get("accuracy_mean", float("nan"))
        best_acc = _best_layer_acc(d.get("per_layer_probe", {"layers": [], "results": {}}))

        # find best gap layer for the highlight
        best_gap32 = best_gap128 = float("nan")
        if d.get("sae"):
            best_gap32 = max(
                (info["ablation_gap"]["gap"]["gap_mean"][
                    info["ablation_gap"]["gap"]["K_grid"].index(32)
                ] if 32 in info["ablation_gap"]["gap"]["K_grid"] else float("nan"))
                for info in d["sae"].values()
                if "ablation_gap" in info
            )
            best_gap128 = max(
                (info["ablation_gap"]["gap"]["gap_mean"][
                    info["ablation_gap"]["gap"]["K_grid"].index(128)
                ] if 128 in info["ablation_gap"]["gap"]["K_grid"] else float("nan"))
                for info in d["sae"].values()
                if "ablation_gap" in info
            )
        md.append(
            f"| {slug} | {base_acc:.4f} | {best_acc:.4f} | "
            f"{best_acc - base_acc:+.4f} | "
            f"{best_gap32:+.4f} | {best_gap128:+.4f} |"
        )
    md.append("")

    md.append("## Per-layer gap curves\n")
    for slug, d in data.items():
        if not d.get("sae"):
            continue
        md.append(f"### {slug}")
        md.append("| layer | " +
                  " | ".join(f"gap@K={k}" for k in [1, 4, 16, 32, 64, 128, 256]) + " |")
        md.append("|---" + "|---" * 7 + "|")
        for layer, info in d["sae"].items():
            if "ablation_gap" not in info:
                continue
            gap = info["ablation_gap"]["gap"]
            K_grid = gap["K_grid"]
            row = [layer]
            for k in [1, 4, 16, 32, 64, 128, 256]:
                if k in K_grid:
                    row.append(f"{gap['gap_mean'][K_grid.index(k)]:+.3f}")
                else:
                    row.append("—")
            md.append("| " + " | ".join(row) + " |")
        md.append("")

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(md))
    print(f"[summary] wrote {args.out_md}")

    # ---------- plot ----------
    if any(d.get("sae") for d in data.values()):
        fig, ax = plt.subplots(figsize=(10, 6))
        for slug, d in data.items():
            if not d.get("sae"):
                continue
            for layer, info in d["sae"].items():
                if "ablation_gap" not in info:
                    continue
                gap = info["ablation_gap"]["gap"]
                K = gap["K_grid"]
                m = np.array(gap["gap_mean"])
                s = np.array(gap["gap_std"])
                label = f"{slug} :: {layer}"
                ax.plot(K, m, marker="o", label=label, alpha=0.8)
                ax.fill_between(K, m - s, m + s, alpha=0.15)
        ax.axhline(0, color="black", linestyle=":", alpha=0.4)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("K (ablated features)")
        ax.set_ylabel("gap = drop_PCA − drop_SAE")
        ax.set_title("Cross-recipe SAE−PCA ablation gap")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(args.out_pdf, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"[summary] wrote {args.out_pdf}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
