"""Template for a new DatasetAdapter — copy this to bio_fm_probe/datasets/<name>.py.

Steps to add a new benchmark:
  1. Copy this file, rename the class.
  2. Fill in load(data_dir) — read from disk, return Sample.
  3. Set modality / baseline_kind / task_kind / meta appropriately.
  4. Add a download script under src/download_<name>.py (also user-runnable).
  5. Register in bio_fm_probe/datasets/__init__.py:
       register_dataset("<name>", "bio_fm_probe.datasets.<name>:<Class>Dataset")
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.dataset import DatasetAdapter, Sample


class TemplateDataset(DatasetAdapter):
    name = "template"
    modality = "scrna"   # "scrna" | "dna" | "protein"

    def load(self, data_dir: str) -> Sample:
        raise NotImplementedError(
            "Fill in load() — read your benchmark from data_dir, return Sample. "
            "See pbmc3k.py / genomic_benchmarks.py / deeploc.py for examples."
        )
        # Example shape:
        # return Sample(
        #     inputs=adata,                  # or list[str] for sequences
        #     labels=labels_array,
        #     modality=self.modality,
        #     baseline_kind="log1p_pca",     # or kmer_pca / onehot_aa_pca
        #     task_kind="classification",
        #     meta={"n_classes": n_classes, "class_names": [...]},
        # )
