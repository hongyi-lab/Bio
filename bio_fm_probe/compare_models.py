"""Cross-model audit comparison.

After running run_audit.py on several models, this script reads each model's
audit JSONs and emits one comparison table.

Usage:
    python -m bio_fm_probe.compare_models scgpt scmamba geneformer

Output: results/_cross_model_summary.md  (and prints to stdout)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


def _load_model_summary(audit_dir: Path) -> Dict:
    paths = {
        "baselines":  audit_dir / "baselines.json",
        "per_layer":  audit_dir / "per_layer_probe.json",
        "svd":        audit_dir / "svd_diag.json",
        "layer0":     audit_dir / "layer0_sanity.json",
    }
    missing = [name for name, p in paths.items() if not p.exists()]
    if missing:
        raise SystemExit(f"missing {missing} under {audit_dir}")
    return {k: json.loads(p.read_text()) for k, p in paths.items()}


def _best_layer_acc(per_layer_json) -> Dict:
    """Best across all layers × {cls, mean} pooling."""
    layers = per_layer_json["layers"]
    results = per_layer_json["results"]
    best_acc, best_name, best_pool = -1.0, None, None
    for n in layers:
        for pool in ("cls", "mean"):
            acc = results[n][pool]["accuracy_mean"]
            if acc > best_acc:
                best_acc, best_name, best_pool = acc, n, pool
    return {"acc": best_acc, "layer": best_name, "pool": best_pool}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+",
                   help="model names (subdirs under <results_root>/<model>/audit/)")
    p.add_argument("--results_root", default="results")
    p.add_argument("--out", default="results/_cross_model_summary.md")
    args = p.parse_args()

    root = Path(args.results_root)
    summaries: Dict[str, Dict] = {}
    for m in args.models:
        audit = root / m / "audit"
        try:
            summaries[m] = _load_model_summary(audit)
        except SystemExit as e:
            print(f"[compare] skip {m}: {e}")
            continue

    if not summaries:
        raise SystemExit("no models had complete audit output")

    md = [f"# Cross-model audit — {len(summaries)} model(s)\n"]
    md.append("| metric | " + " | ".join(summaries) + " |")
    md.append("|---|" + "---|" * len(summaries))

    def row(label: str, fn):
        cells = [fn(s) for s in summaries.values()]
        md.append(f"| {label} | " + " | ".join(cells) + " |")

    row("raw log1p LR acc",
        lambda s: f"{s['baselines']['raw_log1p_LR']['accuracy_mean']:.4f}")
    row("PCA-50 LR acc (baseline)",
        lambda s: f"{s['baselines']['PCA50_LR']['accuracy_mean']:.4f}")
    row("PCA-512 LR acc",
        lambda s: f"{s['baselines']['PCA512_LR']['accuracy_mean']:.4f}")

    def _best(s):
        return _best_layer_acc(s["per_layer"])

    row("best model acc (any layer × pool)",
        lambda s: f"{_best(s)['acc']:.4f}")
    row("best at: layer",
        lambda s: _best(s)["layer"])
    row("best at: pool",
        lambda s: _best(s)["pool"])
    row("Δ best vs PCA-50",
        lambda s: f"{_best(s)['acc'] - s['baselines']['PCA50_LR']['accuracy_mean']:+.4f}")
    row("layer 0 std ratio (H1)",
        lambda s: f"{s['layer0'].get('ratio_l0_to_ref_std', float('nan')):.2e}")

    def first_layer(s):
        return next(iter(s["svd"].values())) if s["svd"] else {}

    def last_layer(s):
        return list(s["svd"].values())[-1] if s["svd"] else {}

    row("PR (input layer)",
        lambda s: (f"{first_layer(s)['participation_ratio']:.1f}"
                   if first_layer(s) else "—"))
    row("PR (final layer probed)",
        lambda s: (f"{last_layer(s)['participation_ratio']:.1f}"
                   if last_layer(s) else "—"))
    row("k95 (final layer probed)",
        lambda s: (str(last_layer(s)["k95"])
                   if last_layer(s) else "—"))

    # Optional: number of layers in the model
    row("n_layers",
        lambda s: str(len(s["per_layer"]["layers"]) - 1))  # minus the input pseudo-layer

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md))

    print("\n".join(md))
    print(f"\n[compare] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
