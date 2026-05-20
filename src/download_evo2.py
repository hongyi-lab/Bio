"""Download Evo-2 7B from HuggingFace.

Default: `arcinstitute/evo2_7b` (7B params, StripedHyena-2, open-weights).

Usage:
    python src/download_evo2.py
    python src/download_evo2.py --variant arcinstitute/evo2_20b   # bigger
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="arcinstitute/evo2_7b")
    p.add_argument("--dest", default=None,
                   help="default: checkpoints/<variant_basename>/")
    args = p.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit(
            "huggingface_hub not installed. pip install huggingface_hub")

    basename = args.variant.split("/")[-1]
    dest = Path(args.dest) if args.dest else Path("checkpoints") / basename
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[evo2] downloading {args.variant} -> {dest}")
    print(f"[evo2] note: 7B checkpoint ~14 GB fp16; 20B ~40 GB. "
          f"Allow time + bandwidth.")
    snapshot_download(
        repo_id=args.variant,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
    )
    print(f"[evo2] done")
    print(f"[evo2] use with: python src/evo2_probe.py --model_dir {dest} ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
