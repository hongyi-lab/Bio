"""Download a Genomic Benchmarks dataset and dump train/test CSVs.

Uses the `genomic-benchmarks` pip package (Grešová et al. 2023) which exposes
several DNA classification tasks. Default: `human_nontata_promoters` (binary,
~36k train / 9k test sequences, ~250 bp each).

Usage:
    python src/download_genomic_benchmarks.py
    python src/download_genomic_benchmarks.py --task human_enhancers_cohn
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="human_nontata_promoters")
    p.add_argument("--dest", default=None,
                   help="default: data/genomic_benchmarks/<task>/")
    args = p.parse_args()

    dest = Path(args.dest) if args.dest else Path(f"data/genomic_benchmarks/{args.task}")
    if (dest / "train.csv").exists() and (dest / "test.csv").exists():
        print(f"[gb] {dest} already populated — skip.")
        return 0

    try:
        from genomic_benchmarks.dataset_getters.pytorch_datasets import get_dataset
        from genomic_benchmarks.loc2seq import download_dataset
    except ImportError:
        raise SystemExit(
            "[gb] genomic-benchmarks not installed.\n"
            "  pip install genomic-benchmarks\n"
        )

    print(f"[gb] downloading task {args.task} ...")
    download_dataset(args.task, version=0)
    train_ds = get_dataset(args.task, "train", version=0)
    test_ds = get_dataset(args.task, "test", version=0)

    dest.mkdir(parents=True, exist_ok=True)
    for split, ds in (("train", train_ds), ("test", test_ds)):
        out = dest / f"{split}.csv"
        with out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sequence", "label"])
            for seq, label in ds:
                w.writerow([str(seq), int(label)])
        print(f"[gb] wrote {out}: {len(ds)} samples")

    print(f"[gb] ready. Adapter expects: --data {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
