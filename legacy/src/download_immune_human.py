"""Download an immune-human PBMC-style benchmark from cellxgene census.

Default pulls ~30k 10x Chromium PBMCs from the cellxgene census API and saves
as a single h5ad with raw counts in .X plus cell_type labels in .obs.

Why cellxgene census: stable URL, paper-cited labels, no GEO accession dance.
Alternative: a static h5ad URL if cellxgene is too heavy — see --static_url.

Usage:
    python src/download_immune_human.py
    python src/download_immune_human.py --max_cells 10000
    python src/download_immune_human.py --static_url <url>      # bypass census
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def via_census(dest: Path, max_cells: int) -> None:
    try:
        import cellxgene_census
    except ImportError:
        raise SystemExit(
            "[immune_human] cellxgene_census not installed.\n"
            "  pip install cellxgene-census\n"
            "Or use --static_url to download a pre-prepared h5ad instead."
        )
    print("[immune_human] opening cellxgene census ...")
    with cellxgene_census.open_soma(census_version="stable") as census:
        # Filter: human, 10x Chromium v3, PBMC-style tissue, healthy.
        # This pulls a fairly large multi-source PBMC slice; subsampled to max_cells.
        adata = cellxgene_census.get_anndata(
            census=census,
            organism="Homo sapiens",
            obs_value_filter=(
                "tissue_general == 'blood' "
                "and assay == '10x 3\\' v3' "
                "and disease == 'normal'"
            ),
            obs_column_names=[
                "cell_type", "donor_id", "assay", "tissue", "disease",
                "sex", "self_reported_ethnicity",
            ],
        )
    print(f"[immune_human] fetched {adata.shape}")

    if adata.n_obs > max_cells:
        import numpy as np
        rng = np.random.default_rng(0)
        idx = rng.choice(adata.n_obs, size=max_cells, replace=False)
        adata = adata[np.sort(idx)].copy()
        print(f"[immune_human] subsampled to {adata.n_obs} cells")

    dest.parent.mkdir(parents=True, exist_ok=True)
    adata.write(dest)
    print(f"[immune_human] saved {dest} (shape={adata.shape})")


def via_url(dest: Path, url: str) -> None:
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[immune_human] downloading {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"[immune_human] saved {dest}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dest", default="data/immune_human.h5ad")
    p.add_argument("--max_cells", type=int, default=30000)
    p.add_argument("--static_url", default=None,
                   help="bypass cellxgene_census and download a pre-prepared h5ad")
    args = p.parse_args()

    dest = Path(args.dest)
    if dest.exists():
        print(f"[immune_human] {dest} already exists — skip.")
        return 0

    if args.static_url:
        via_url(dest, args.static_url)
    else:
        via_census(dest, args.max_cells)
    return 0


if __name__ == "__main__":
    sys.exit(main())
