"""DeepLoc dataset adapter — protein subcellular localization.

DeepLoc 2.0 (Thumuluri et al. 2022) is a standard protein-sequence
classification benchmark. 10 subcellular locations, ~14k proteins,
sequence-only (no structure required), well-suited to ESM-2 / ProtTrans
embeddings.

Expected `data_dir`: a fasta-with-header file (or pre-parsed CSV) prepared by
src/download_deeploc.py.

DeepLoc 2.0 release format (public): a single FASTA-style file with
identifiers like  >ID|Location|...  where Location is one of the 10 classes.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from ..core.dataset import DatasetAdapter, Sample


# Standard DeepLoc 2.0 single-location set (10-class)
DEEPLOC_CLASSES = [
    "Cytoplasm", "Nucleus", "Extracellular", "Cell.membrane",
    "Mitochondrion", "Plastid", "Endoplasmic.reticulum", "Lysosome/Vacuole",
    "Golgi.apparatus", "Peroxisome",
]


class DeeplocDataset(DatasetAdapter):
    name = "deeploc"
    modality = "protein"

    DEFAULT_FASTA = "data/deeploc/deeploc_data.fasta"

    def load(self, data_dir: Optional[str] = None) -> Sample:
        path = Path(data_dir) if data_dir else Path(self.DEFAULT_FASTA)
        if not path.exists():
            raise SystemExit(
                f"deeploc not found at {path}. "
                f"Run: python src/download_deeploc.py"
            )

        seqs: List[str] = []
        loc_strs: List[str] = []
        with path.open() as f:
            cur_seq: List[str] = []
            cur_loc: Optional[str] = None
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if cur_loc is not None and cur_seq:
                        seqs.append("".join(cur_seq))
                        loc_strs.append(cur_loc)
                    # Parse header: >ID|Location|...   (DeepLoc 2.0 convention)
                    parts = line[1:].split("|")
                    cur_loc = parts[1] if len(parts) > 1 else "Unknown"
                    cur_seq = []
                else:
                    cur_seq.append(line)
            if cur_loc is not None and cur_seq:
                seqs.append("".join(cur_seq))
                loc_strs.append(cur_loc)

        # Filter to classes present in the standard DeepLoc 2.0 set
        keep = [i for i, loc in enumerate(loc_strs) if loc in DEEPLOC_CLASSES]
        if len(keep) < len(seqs):
            print(f"[deeploc] kept {len(keep)}/{len(seqs)} sequences after "
                  f"filtering to known classes")
            seqs = [seqs[i] for i in keep]
            loc_strs = [loc_strs[i] for i in keep]

        # Length cap — protein models have context limits; cap at 1024 for ESM-2-small
        MAX_LEN = 1024
        for i, s in enumerate(seqs):
            if len(s) > MAX_LEN:
                seqs[i] = s[:MAX_LEN]

        classes = np.array(DEEPLOC_CLASSES)
        cls_to_idx = {c: i for i, c in enumerate(DEEPLOC_CLASSES)}
        labels = np.array([cls_to_idx[l] for l in loc_strs], dtype=np.int64)
        print(f"[deeploc] loaded {len(seqs)} sequences, {len(classes)} classes, "
              f"mean_len={int(np.mean([len(s) for s in seqs]))}, "
              f"max_len_after_cap={MAX_LEN}")

        return Sample(
            inputs=seqs,
            labels=labels,
            modality=self.modality,
            baseline_kind="onehot_aa_pca",
            task_kind="classification",
            meta={
                "n_classes": int(len(classes)),
                "class_names": classes.tolist(),
                "max_seq_len_cap": MAX_LEN,
                "mean_seq_len": int(np.mean([len(s) for s in seqs])),
            },
        )
