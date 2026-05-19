"""Download a small public PBMC dataset for probing experiments.

Default: scanpy's pbmc3k_processed (~6 MB, includes louvain cell-type labels).
Optional: --multiome flag to grab 10X PBMC Multiome (much larger, ~1 GB).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlopen

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def _download_with_progress(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    start_bytes = tmp.stat().st_size if tmp.exists() else 0

    req = urlopen(url)
    total = int(req.headers.get("Content-Length", 0))
    if start_bytes and start_bytes == total:
        tmp.rename(dst)
        return

    mode = "ab" if start_bytes else "wb"
    with open(tmp, mode) as f, tqdm(
        total=total, initial=start_bytes, unit="B", unit_scale=True,
        unit_divisor=1024, desc=dst.name,
    ) as pbar:
        while True:
            chunk = req.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
            pbar.update(len(chunk))
    tmp.rename(dst)


def fetch_pbmc3k() -> Path:
    """Use scanpy's bundled downloader so progress + caching are handled for us."""
    import scanpy as sc

    dst = DATA_DIR / "pbmc3k.h5ad"
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[skip] {dst} already exists ({dst.stat().st_size/1e6:.2f} MB)")
        return dst

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading pbmc3k_processed (scanpy datasets, ~6 MB)...")
    adata = sc.datasets.pbmc3k_processed()
    adata.write_h5ad(dst)
    print(f"Saved {dst}  shape={adata.shape}")
    print(f"  obs columns: {list(adata.obs.columns)}")
    return dst


def fetch_pbmc_multiome() -> Path:
    """10X PBMC Multiome (RNA + ATAC), ~12k cells. Large download."""
    url = (
        "https://cf.10xgenomics.com/samples/cell-arc/2.0.0/"
        "pbmc_granulocyte_sorted_10k/"
        "pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5"
    )
    dst = DATA_DIR / "pbmc_multiome_10k.h5"
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[skip] {dst} already exists ({dst.stat().st_size/1e6:.2f} MB)")
        return dst
    print(f"Downloading {url}")
    _download_with_progress(url, dst)
    print(f"Saved {dst}")
    return dst


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--multiome", action="store_true",
                   help="Also fetch 10X PBMC Multiome (RNA+ATAC, ~1 GB)")
    args = p.parse_args()

    fetch_pbmc3k()
    if args.multiome:
        fetch_pbmc_multiome()
    return 0


if __name__ == "__main__":
    sys.exit(main())
