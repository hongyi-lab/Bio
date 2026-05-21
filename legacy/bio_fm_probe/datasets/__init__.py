"""Dataset adapters — one (modality, benchmark) pair per file.

Each adapter calls register_dataset(name, "module:ClassName") at import time.
The run_recipe entry point imports this package to populate DATASET_REGISTRY.
"""
from __future__ import annotations

from ..core.dataset import register_dataset


# scRNA
register_dataset("pbmc3k",       "bio_fm_probe.datasets.pbmc3k:Pbmc3kDataset")
register_dataset("immune_human", "bio_fm_probe.datasets.immune_human:ImmuneHumanDataset")
# DNA
register_dataset("genomic_benchmarks",
                 "bio_fm_probe.datasets.genomic_benchmarks:GenomicBenchmarksDataset")
# protein
register_dataset("deeploc",      "bio_fm_probe.datasets.deeploc:DeeplocDataset")
