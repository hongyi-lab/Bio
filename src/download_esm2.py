"""Download an ESM-2 HF checkpoint.

Default = `facebook/esm2_t12_35M_UR50D` — 12 layer, 480d, ~35M params. Big
enough to be informative, small enough to forward all of DeepLoc on one A6000
in minutes.

Larger variants:
  facebook/esm2_t30_150M_UR50D    (150M)
  facebook/esm2_t33_650M_UR50D    (650M)
  facebook/esm2_t36_3B_UR50D      (3B  — fp16 required)
  facebook/esm2_t48_15B_UR50D     (15B — multi-GPU required)

Usage:
    python src/download_esm2.py
    python src/download_esm2.py --variant facebook/esm2_t30_150M_UR50D
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="facebook/esm2_t12_35M_UR50D")
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
    print(f"[esm2] downloading {args.variant} -> {dest}")
    snapshot_download(
        repo_id=args.variant,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
    )
    print(f"[esm2] done")
    print(f"[esm2] use with: --adapter esm2 --model_dir {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
