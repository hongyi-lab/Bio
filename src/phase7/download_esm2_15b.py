"""Download ESM-2 15B from HuggingFace.

Default: `facebook/esm2_t48_15B_UR50D` (15B params, 48 layers × d=5120).
Released by Meta FAIR; fully open-weights. ~30 GB fp16.

Usage:
    python src/download_esm2_15b.py
    # Smaller variants for testing:
    python src/download_esm2_15b.py --variant facebook/esm2_t36_3B_UR50D
    python src/download_esm2_15b.py --variant facebook/esm2_t33_650M_UR50D
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="facebook/esm2_t48_15B_UR50D")
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
    print(f"[esm2-15b] downloading {args.variant} -> {dest}")
    print(f"[esm2-15b] note: 15B checkpoint is ~30 GB fp16; expect 30-60 min on "
          f"a fast connection.")
    snapshot_download(
        repo_id=args.variant,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
    )
    print(f"[esm2-15b] done")
    print(f"[esm2-15b] use with: python src/esm2_15b_probe.py "
          f"--model_dir {dest} ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
