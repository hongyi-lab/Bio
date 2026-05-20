"""Download an Evo HF checkpoint.

Default = `togethercomputer/evo-1-8k-base` — 7B params, StripedHyena
architecture, 8k context. The 131k-context variant is also available but
heavier (`togethercomputer/evo-1-131k-base`).

Usage:
    python src/download_evo.py
    python src/download_evo.py --variant togethercomputer/evo-1-131k-base
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="togethercomputer/evo-1-8k-base")
    p.add_argument("--dest", default=None,
                   help="default: checkpoints/<variant_basename>/")
    args = p.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("huggingface_hub not installed. pip install huggingface_hub")

    basename = args.variant.split("/")[-1]
    dest = Path(args.dest) if args.dest else Path("checkpoints") / basename
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[evo] downloading {args.variant} -> {dest}")
    print(f"[evo] note: this checkpoint is ~14 GB (fp16); allow time + bandwidth")
    snapshot_download(
        repo_id=args.variant,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
    )
    print(f"[evo] done")
    print(f"[evo] use with: --model evo --model_dir {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
