"""AuditRecipe — (Model adapter, Dataset adapter) tuple.

The recipe resolves a model adapter + dataset adapter from their respective
registries, runs a validity check (modality compatibility), and provides a
deterministic output directory.

This is the smallest unit a user can `python -m bio_fm_probe.run_recipe` on.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class AuditRecipe:
    model_name: str
    dataset_name: str

    def output_dir(self, results_root: str = "results") -> Path:
        return Path(results_root) / f"{self.model_name}__{self.dataset_name}" / "audit"

    def slug(self) -> str:
        return f"{self.model_name}__{self.dataset_name}"


def validate_modality_match(model_modality: str, dataset_modality: str) -> None:
    """Raise SystemExit if model and dataset modalities don't match.

    This is the rubbish-in-rubbish-out guard: don't feed DNA to scGPT or
    scRNA to ESM-2.
    """
    if model_modality != dataset_modality:
        raise SystemExit(
            f"modality mismatch: model expects {model_modality!r}, "
            f"dataset is {dataset_modality!r}. Refusing to run — this would "
            f"be rubbish-in-rubbish-out."
        )
