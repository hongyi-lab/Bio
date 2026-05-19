"""Download a HyenaDNA HF checkpoint.

Default = `LongSafari/hyenadna-small-32k-seqlen-hf` — smallest reasonable
variant (~6 MB weights, 32k context).

Usage:
    python src/download_hyenadna.py
    python src/download_hyenadna.py --variant LongSafari/hyenadna-medium-160k-seqlen-hf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="LongSafari/hyenadna-small-32k-seqlen-hf")
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
    print(f"[hyenadna] downloading {args.variant} -> {dest}")
    snapshot_download(
        repo_id=args.variant,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
    )
    print(f"[hyenadna] done")
    print(f"[hyenadna] use with: --adapter hyenadna --model_dir {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
