"""genomic_benchmarks dataset adapter — DNA classification benchmark from
Grešová et al. 2023 (the `genomic-benchmarks` pip package).

Default task: `human_nontata_promoters` — binary classification of human
non-TATA promoter sequences vs random genomic windows. Sequences are typically
~250 bp, ~36k train / 9k test samples; small enough to forward through
HyenaDNA / Nucleotide Transformer / Caduceus on a single A6000 in minutes.

Expected `data_dir`: a directory containing two CSVs (or fasta files) of
sequence + label, prepared by src/download_genomic_benchmarks.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from ..core.dataset import DatasetAdapter, Sample


class GenomicBenchmarksDataset(DatasetAdapter):
    name = "genomic_benchmarks"
    modality = "dna"

    DEFAULT_DIR = "data/genomic_benchmarks/human_nontata_promoters"
    DEFAULT_TASK = "human_nontata_promoters"

    def load(self, data_dir: Optional[str] = None) -> Sample:
        root = Path(data_dir) if data_dir else Path(self.DEFAULT_DIR)
        if not root.exists():
            raise SystemExit(
                f"genomic_benchmarks data not found at {root}. "
                f"Run: python src/download_genomic_benchmarks.py"
            )

        # The download script saves CSVs with columns: sequence,label
        seqs: List[str] = []
        labels: List[int] = []
        for split in ("train", "test"):
            csv = root / f"{split}.csv"
            if not csv.exists():
                raise SystemExit(
                    f"missing {csv}; run src/download_genomic_benchmarks.py"
                )
            import csv as _csv
            with csv.open() as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    seqs.append(row["sequence"])
                    labels.append(int(row["label"]))

        labels_arr = np.asarray(labels, dtype=np.int64)
        classes = np.unique(labels_arr)
        print(f"[genomic_benchmarks] task={root.name}, n_samples={len(seqs)}, "
              f"n_classes={len(classes)}, mean_len="
              f"{int(np.mean([len(s) for s in seqs]))}")

        return Sample(
            inputs=seqs,
            labels=labels_arr,
            modality=self.modality,
            baseline_kind="kmer_pca",
            task_kind="classification",
            meta={
                "task": root.name,
                "n_classes": int(len(classes)),
                "class_names": [str(c) for c in classes.tolist()],
                "mean_seq_len": int(np.mean([len(s) for s in seqs])),
            },
        )
