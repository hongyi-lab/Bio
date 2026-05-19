"""Dataset abstraction — symmetric to BioFMAdapter.

A `Sample` is the uniform return of any dataset adapter:
  - inputs: modality-typed payload (AnnData for scRNA, list[str] for DNA/protein)
  - labels: np.ndarray of shape (n_samples,)
  - modality: one of "scrna", "dna", "protein"
  - baseline_kind: which baseline feature builder to use for this sample
  - task_kind: "classification" or "regression"
  - meta: free-form dict (n_classes, class_names, dataset_version, etc.)

A `DatasetAdapter` is a thin ABC. Concrete subclasses register in DATASET_REGISTRY.

The contract is intentionally minimal — modality-specific quirks live in the
adapter's load() (e.g. cellxgene queries for scRNA, fasta parsing for protein).
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


VALID_MODALITIES = {"scrna", "dna", "protein"}
VALID_BASELINES = {"log1p_pca", "kmer_pca", "onehot_aa_pca"}
VALID_TASK_KINDS = {"classification", "regression"}


@dataclass
class Sample:
    """Uniform return of any DatasetAdapter.load()."""

    inputs: Any
    labels: np.ndarray
    modality: str                       # "scrna" | "dna" | "protein"
    baseline_kind: str                  # which baseline builder to use
    task_kind: str = "classification"
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.modality not in VALID_MODALITIES:
            raise ValueError(
                f"modality {self.modality!r} not in {VALID_MODALITIES}"
            )
        if self.baseline_kind not in VALID_BASELINES:
            raise ValueError(
                f"baseline_kind {self.baseline_kind!r} not in {VALID_BASELINES}"
            )
        if self.task_kind not in VALID_TASK_KINDS:
            raise ValueError(
                f"task_kind {self.task_kind!r} not in {VALID_TASK_KINDS}"
            )
        if not isinstance(self.labels, np.ndarray):
            self.labels = np.asarray(self.labels)

    @property
    def n_samples(self) -> int:
        return int(len(self.labels))


class DatasetAdapter(abc.ABC):
    """Abstract base for any (modality, benchmark) loader.

    Subclasses must implement:
        name: str
        modality: str
        load(data_dir) -> Sample
    """

    name: str
    modality: str

    @abc.abstractmethod
    def load(self, data_dir: str) -> Sample:
        """Return a Sample for this benchmark. Should be deterministic
        given the same data_dir contents.
        """


# Populated lazily by adapter files (each calls register_dataset() at import).
DATASET_REGISTRY: Dict[str, str] = {
    # name -> "module.path:ClassName"
}


def register_dataset(name: str, class_path: str) -> None:
    DATASET_REGISTRY[name] = class_path


def load_dataset_adapter(name: str) -> DatasetAdapter:
    if name not in DATASET_REGISTRY:
        raise SystemExit(
            f"unknown dataset {name!r}; registered: {list(DATASET_REGISTRY)}"
        )
    import importlib
    mod_path, cls_name = DATASET_REGISTRY[name].split(":")
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)()
