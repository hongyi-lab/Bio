"""Python orchestrator for the cross-model audit.

Equivalent to src/run_all_audits.sh but runnable directly from VS Code
(F5 / Run Python File). Each model's audit is a separate subprocess so a
crash in one doesn't take down the others, and you can ctrl+C cleanly.

Usage from a terminal:
    python src/run_all_audits.py                                # defaults
    python src/run_all_audits.py --skip_sae
    python src/run_all_audits.py --models scgpt                 # subset
    python src/run_all_audits.py --models scgpt geneformer --skip_sae
    python src/run_all_audits.py --geneformer_variant v2-104m

Usage from VS Code:
    Open this file. Press F5 (or click "Run Python File"). If you want flags,
    add them via launch.json's "args" list, or change DEFAULTS below.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent  # /srv/hgao0864/Bio/scGPT_probing
os.chdir(ROOT)             # so relative paths in sub-scripts resolve

PYTHON = sys.executable    # use the same interpreter VS Code launched us with

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Each entry tells the orchestrator:
#   ckpt_dir    where the audit expects the checkpoint
#   download    list[str] command to fetch the checkpoint (None = no download)
#   variant_arg (optional) extra arg appended to the download cmd
MODEL_REGISTRY = {
    "scgpt": {
        "ckpt_dir": "checkpoints/scGPT_human",
        "download": [PYTHON, "src/download_checkpoint.py"],
    },
    "geneformer": {
        "ckpt_dir_template": "checkpoints/geneformer{suffix}",
        "download": [PYTHON, "src/download_geneformer.py"],
    },
    # Adapters below are stubs / placeholders. Wire them once their
    # adapter files are filled in and registered in run_audit.ADAPTER_REGISTRY.
    # "scfoundation": {...},
    # "scbert":       {...},
    # "uce":          {...},
}


def _section(msg: str) -> None:
    print("=" * 64)
    print(f"[run_all] {msg}")
    print("=" * 64, flush=True)


def _step(msg: str) -> None:
    print(f"[run_all] {msg}", flush=True)


def _run(cmd: list, what: str) -> None:
    _step(f"$ {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"\n[run_all] FAIL: {what} (exit code {e.returncode})", file=sys.stderr)
        sys.exit(e.returncode)


def _ckpt_present(d: Path) -> bool:
    return d.is_dir() and any(d.iterdir())


def _resolve_ckpt_dir(model: str, geneformer_variant: str) -> Path:
    info = MODEL_REGISTRY[model]
    if model == "geneformer":
        suffix = "" if geneformer_variant == "v1" else f"_{geneformer_variant.replace('-', '_')}"
        return ROOT / info["ckpt_dir_template"].format(suffix=suffix)
    return ROOT / info["ckpt_dir"]


def download_if_missing(model: str, ckpt_dir: Path, geneformer_variant: str) -> None:
    if _ckpt_present(ckpt_dir):
        _step(f"[{model}] checkpoint at {ckpt_dir.relative_to(ROOT)} — skip download")
        return
    info = MODEL_REGISTRY[model]
    cmd = list(info["download"])
    if model == "geneformer":
        cmd.extend(["--variant", geneformer_variant])
    _step(f"[{model}] downloading checkpoint ...")
    _run(cmd, what=f"{model} download")


def audit_model(model: str, ckpt_dir: Path, args: argparse.Namespace) -> None:
    _section(f"[{model}] AUDIT START")
    cmd = [
        PYTHON, "-m", "bio_fm_probe.run_audit",
        "--adapter", model,
        "--model_dir", str(ckpt_dir),
        "--data", args.data,
        "--label_col", args.label_col,
        "--device", args.device,
    ]
    if args.skip_sae:
        cmd.append("--skip_sae")
    if args.sae_layers:
        cmd.append("--sae_layers")
        cmd.extend(args.sae_layers)
    _run(cmd, what=f"{model} audit")
    _section(f"[{model}] AUDIT DONE")


def compare(models_done: list) -> None:
    if len(models_done) < 2:
        _step("fewer than 2 models produced AUDIT.md — skipping comparison")
        return
    _section("CROSS-MODEL SUMMARY")
    _run(
        [PYTHON, "-m", "bio_fm_probe.compare_models", *models_done],
        what="cross-model comparison",
    )
    _step("DONE. See results/_cross_model_summary.md")


def main() -> int:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--models", nargs="+", default=["scgpt", "geneformer"],
                   choices=list(MODEL_REGISTRY))
    p.add_argument("--data", default="data/pbmc3k.h5ad")
    p.add_argument("--label_col", default="louvain")
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip_sae", action="store_true",
                   help="layer probe + baselines + SVD only (saves ~2h/model)")
    p.add_argument("--geneformer_variant", default="v1",
                   choices=["v1", "v2-104m", "v2-313m"])
    p.add_argument("--sae_layers", nargs="+", default=None,
                   help="override the audit's default SAE layer choice")
    args = p.parse_args()

    _section(
        f"DATA={args.data}  LABEL_COL={args.label_col}  DEVICE={args.device}\n"
        f"[run_all] MODELS={args.models}  SKIP_SAE={args.skip_sae}  "
        f"GENEFORMER_VARIANT={args.geneformer_variant}"
    )

    # 1. download any missing checkpoints
    ckpt_dirs = {}
    for m in args.models:
        ckpt_dirs[m] = _resolve_ckpt_dir(m, args.geneformer_variant)
        download_if_missing(m, ckpt_dirs[m], args.geneformer_variant)

    # 2. audit each model
    for m in args.models:
        audit_model(m, ckpt_dirs[m], args)

    # 3. cross-model comparison (only models that produced an AUDIT.md)
    done = [m for m in args.models if (ROOT / f"results/{m}/audit/AUDIT.md").exists()]
    compare(done)
    return 0


if __name__ == "__main__":
    sys.exit(main())
