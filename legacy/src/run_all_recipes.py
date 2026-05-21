"""Python orchestrator for phase-5 recipes (run_recipe entry).

What it runs, in order:
    step 2: download HyenaDNA + GenomicBenchmarks → recipe(hyenadna, genomic_benchmarks)
    step 3: download ESM-2 + DeepLoc              → recipe(esm2, deeploc)
    step 4: cross_recipe_summary on all completed recipes
            (defaults to scgpt__pbmc3k + the two above, since step 1 is
             expected to already be on disk from `bio_fm_probe.run_recipe
             --model scgpt --dataset pbmc3k`).

Step 1 is intentionally not re-run here — the assumption is you've already
finished it. Pass --include_scgpt to re-run it.

Designed to be runnable from VS Code (F5 / Run Python File). Each step is a
separate subprocess so a crash in one doesn't poison the others; ctrl+C
cleanly cancels.

Usage:
    python src/run_all_recipes.py                       # step 2 + 3 + 4
    python src/run_all_recipes.py --skip_sae            # baselines + svd + ablation only (no SAE training)
    python src/run_all_recipes.py --only dna            # just the DNA recipe
    python src/run_all_recipes.py --only protein
    python src/run_all_recipes.py --include_scgpt       # also re-run scgpt+pbmc3k
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
os.chdir(ROOT)

PYTHON = sys.executable

# ---------------------------------------------------------------------------
# Recipes (model, dataset, checkpoint dir, downloaders)
# ---------------------------------------------------------------------------
RECIPES = {
    "scgpt": {
        "model": "scgpt",
        "model_dir": "checkpoints/scGPT_human",
        "dataset": "pbmc3k",
        "downloads": [
            ([PYTHON, "src/download_checkpoint.py"], "checkpoints/scGPT_human"),
            ([PYTHON, "src/download_data.py"], "data/pbmc3k.h5ad"),
        ],
        "recipe_tag": "scgpt__pbmc3k",
    },
    "dna": {
        "model": "hyenadna",
        "model_dir": "checkpoints/hyenadna-small-32k-seqlen-hf",
        "dataset": "genomic_benchmarks",
        "downloads": [
            ([PYTHON, "src/download_hyenadna.py"],
             "checkpoints/hyenadna-small-32k-seqlen-hf"),
            ([PYTHON, "src/download_genomic_benchmarks.py"],
             "data/genomic_benchmarks"),
        ],
        "recipe_tag": "hyenadna__genomic_benchmarks",
    },
    "protein": {
        "model": "esm2",
        "model_dir": "checkpoints/esm2_t12_35M_UR50D",
        "dataset": "deeploc",
        "downloads": [
            ([PYTHON, "src/download_esm2.py"], "checkpoints/esm2_t12_35M_UR50D"),
            ([PYTHON, "src/download_deeploc.py"], "data/deeploc"),
        ],
        "recipe_tag": "esm2__deeploc",
    },
}


def _section(msg: str) -> None:
    print("=" * 68)
    print(f"[run_all_recipes] {msg}")
    print("=" * 68, flush=True)


def _run(cmd: list, what: str) -> None:
    print(f"[run_all_recipes] $ {' '.join(cmd)}", flush=True)
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"\n[run_all_recipes] FAIL: {what} (exit {e.returncode})", file=sys.stderr)
        sys.exit(e.returncode)


def _exists(path: Path) -> bool:
    if path.is_file():
        return path.stat().st_size > 0
    if path.is_dir():
        return any(path.iterdir())
    return False


def download_if_needed(recipe_key: str) -> None:
    spec = RECIPES[recipe_key]
    for cmd, marker in spec["downloads"]:
        marker_path = ROOT / marker
        if _exists(marker_path):
            print(f"[run_all_recipes] [{recipe_key}] {marker} exists — skip download")
            continue
        print(f"[run_all_recipes] [{recipe_key}] downloading ({marker} missing)...")
        _run(cmd, what=f"{recipe_key} download")


def run_recipe(recipe_key: str, skip_sae: bool, force: bool) -> None:
    spec = RECIPES[recipe_key]
    audit_md = ROOT / "results" / spec["recipe_tag"] / "audit" / "AUDIT.md"
    if audit_md.exists() and not force:
        print(f"[run_all_recipes] [{recipe_key}] {audit_md} already exists — skip "
              f"(use --force to re-run)")
        return
    _section(f"[{recipe_key}] RECIPE START → {spec['recipe_tag']}")
    cmd = [
        PYTHON, "-m", "bio_fm_probe.run_recipe",
        "--model", spec["model"],
        "--model_dir", spec["model_dir"],
        "--dataset", spec["dataset"],
    ]
    if skip_sae:
        cmd.append("--skip_sae")
    _run(cmd, what=f"{recipe_key} recipe")
    _section(f"[{recipe_key}] RECIPE DONE")


def cross_summary(recipe_keys: list) -> None:
    tags = []
    for k in recipe_keys:
        tag = RECIPES[k]["recipe_tag"]
        audit_md = ROOT / "results" / tag / "audit" / "AUDIT.md"
        if audit_md.exists():
            tags.append(tag)
        else:
            print(f"[run_all_recipes] WARN: {tag} has no AUDIT.md — omitting from summary")
    if len(tags) < 2:
        print("[run_all_recipes] fewer than 2 recipes done — skipping cross-summary")
        return
    _section(f"CROSS-RECIPE SUMMARY → {tags}")
    _run([PYTHON, "-m", "bio_fm_probe.cross_recipe_summary", *tags],
         what="cross-recipe summary")
    print("[run_all_recipes] DONE. See results/_master_table.md and "
          "results/_ablation_gap_curves.pdf")


def main() -> int:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--only", choices=list(RECIPES) + ["all"], default="all",
                   help='subset to run: "dna", "protein", "scgpt", or "all"')
    p.add_argument("--include_scgpt", action="store_true",
                   help="also run scgpt__pbmc3k (assumes already done by default)")
    p.add_argument("--skip_sae", action="store_true",
                   help="run baselines + SVD + ablation only; no SAE training")
    p.add_argument("--force", action="store_true",
                   help="re-run a recipe even if its AUDIT.md exists")
    args = p.parse_args()

    if args.only == "all":
        to_run = ["dna", "protein"]
        if args.include_scgpt:
            to_run = ["scgpt"] + to_run
    else:
        to_run = [args.only]

    _section(f"plan: {to_run}  skip_sae={args.skip_sae}  force={args.force}")

    for key in to_run:
        download_if_needed(key)
        run_recipe(key, skip_sae=args.skip_sae, force=args.force)

    # Always summarize across the canonical 3 if their AUDIT.md exists.
    cross_summary(["scgpt", "dna", "protein"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
