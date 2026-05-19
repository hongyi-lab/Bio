"""pbmc3k dataset adapter — wraps the existing data/pbmc3k.h5ad path under
the standard DatasetAdapter interface so phase 1-3 experiments are reproducible
via the recipe interface.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import scanpy as sc

from ..core.dataset import DatasetAdapter, Sample


class Pbmc3kDataset(DatasetAdapter):
    name = "pbmc3k"
    modality = "scrna"

    DEFAULT_H5AD = "data/pbmc3k.h5ad"
    LABEL_FALLBACK = ("louvain", "leiden", "cell_type", "celltype", "CellType")

    def load(self, data_dir: Optional[str] = None) -> Sample:
        path = Path(data_dir) if data_dir else Path(self.DEFAULT_H5AD)
        if not path.exists():
            raise SystemExit(
                f"pbmc3k not found at {path}. Run: python src/download_data.py"
            )
        adata = sc.read_h5ad(path)
        print(f"[pbmc3k] loaded {path}: shape={adata.shape}")

        # phase 1-3 convention: pbmc3k_processed stores scaled matrix in .X
        # (has negatives), but log1p-normed matrix in .raw — swap if so.
        X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
        if adata.raw is not None and X.min() < 0:
            print(f"[pbmc3k] .X has negatives; swapping in .raw "
                  f"({adata.raw.X.shape})")
            import anndata as ad
            adata = ad.AnnData(
                X=adata.raw.X, obs=adata.obs.copy(),
                var=adata.raw.var.copy(), obsm=dict(adata.obsm),
            )

        # pick label column
        label_col = None
        for c in self.LABEL_FALLBACK:
            if c in adata.obs.columns:
                label_col = c
                break
        if label_col is None:
            raise SystemExit(
                f"no label column in obs; have {list(adata.obs.columns)}"
            )
        y_str = adata.obs[label_col].astype(str).values
        classes, y = np.unique(y_str, return_inverse=True)
        print(f"[pbmc3k] label={label_col!r}, {len(classes)} classes")

        return Sample(
            inputs=adata,
            labels=y,
            modality=self.modality,
            baseline_kind="log1p_pca",
            task_kind="classification",
            meta={
                "n_classes": int(len(classes)),
                "class_names": classes.tolist(),
                "label_col": label_col,
            },
        )
