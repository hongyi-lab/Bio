"""Download the DeepLoc 2.0 dataset (Thumuluri et al. 2022).

DeepLoc 2.0 is a public protein-sequence benchmark for subcellular
localization (10 classes per protein, sequence-only, ~14k proteins).

The official static release is hosted at:
  https://services.healthtech.dtu.dk/services/DeepLoc-2.0/

Because static-URL changes break gracefully but unpredictably, this script
supports either:
  (1) automatic download via the official URL (default), or
  (2) a user-supplied local FASTA file (--from_local).

If automatic download fails (URL changed or rate-limited), pass --from_local
with a path you've already downloaded by hand from the DeepLoc page.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path


DEFAULT_URL = (
    # Static URL last verified 2024. May change — pass --url override or
    # --from_local if download fails.
    "https://services.healthtech.dtu.dk/services/DeepLoc-2.0/data/"
    "Swissprot_Train_Validation_dataset.fasta"
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dest", default="data/deeploc/deeploc_data.fasta")
    p.add_argument("--url", default=DEFAULT_URL,
                   help="public FASTA URL (override if upstream moved)")
    p.add_argument("--from_local", default=None,
                   help="copy a local FASTA into dest instead of downloading")
    args = p.parse_args()

    dest = Path(args.dest)
    if dest.exists():
        print(f"[deeploc] {dest} already exists — skip.")
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)

    if args.from_local:
        src = Path(args.from_local)
        if not src.exists():
            raise SystemExit(f"--from_local path does not exist: {src}")
        shutil.copy2(src, dest)
        print(f"[deeploc] copied {src} -> {dest}")
        return 0

    print(f"[deeploc] downloading {args.url} ...")
    try:
        urllib.request.urlretrieve(args.url, dest)
    except Exception as e:
        raise SystemExit(
            f"[deeploc] download failed: {e}\n"
            f"Manually download from https://services.healthtech.dtu.dk/services/DeepLoc-2.0/\n"
            f"and pass --from_local <path>."
        )
    print(f"[deeploc] saved {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
