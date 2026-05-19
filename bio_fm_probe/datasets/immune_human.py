"""immune_human dataset adapter — a larger PBMC-style benchmark for cross-FM
audit. Downloaded from cellxgene census or a static h5ad URL by
src/download_immune_human.py.

This adapter expects a single h5ad file under `data_dir` with raw counts in .X
(or log-normalized; auto-detected) and a 'cell_type' column in obs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import scanpy as sc

from ..core.dataset import DatasetAdapter, Sample


class ImmuneHumanDataset(DatasetAdapter):
    name = "immune_human"
    modality = "scrna"

    DEFAULT_H5AD = "data/immune_human.h5ad"

    def load(self, data_dir: Optional[str] = None) -> Sample:
        path = Path(data_dir) if data_dir else Path(self.DEFAULT_H5AD)
        if not path.exists():
            raise SystemExit(
                f"immune_human not found at {path}. "
                f"Run: python src/download_immune_human.py"
            )
        adata = sc.read_h5ad(path)
        print(f"[immune_human] loaded {path}: shape={adata.shape}")

        # pick label column (prefer fine-grained cell_type; fall back if absent)
        for cand in ("cell_type", "celltype", "CellType",
                     "louvain", "leiden", "cell_type_ontology_term_id"):
            if cand in adata.obs.columns:
                label_col = cand
                break
        else:
            raise SystemExit(
                f"no label column; have {list(adata.obs.columns)}"
            )
        y_str = adata.obs[label_col].astype(str).values

        # Filter classes with too few cells (LR can't stratify-split with <5)
        from collections import Counter
        counts = Counter(y_str)
        keep_classes = {c for c, n in counts.items() if n >= 10}
        if len(keep_classes) < len(counts):
            mask = np.array([c in keep_classes for c in y_str])
            adata = adata[mask].copy()
            y_str = y_str[mask]
            print(f"[immune_human] kept {mask.sum()}/{len(mask)} cells "
                  f"after dropping classes with <10 cells "
                  f"({len(counts) - len(keep_classes)} dropped)")

        classes, y = np.unique(y_str, return_inverse=True)
        print(f"[immune_human] label={label_col!r}, {len(classes)} classes")

        # Make sure the matrix in .X is non-negative (raw counts or log1p)
        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        if X.min() < 0:
            if adata.raw is not None:
                import anndata as ad
                adata = ad.AnnData(
                    X=adata.raw.X, obs=adata.obs.copy(),
                    var=adata.raw.var.copy(), obsm=dict(adata.obsm),
                )
                print(f"[immune_human] swapped .X for .raw (.X had negatives)")

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
