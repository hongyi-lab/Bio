"""bio_fm_probe — model-agnostic probing toolkit for biological foundation models.

Drop-in workflow:
  1. Add a model adapter under `adapters/<name>.py` implementing BioFMAdapter.
  2. Add a dataset adapter under `datasets/<name>.py` implementing DatasetAdapter.
  3. Register both in the respective registries.
  4. Run:
       python -m bio_fm_probe.run_recipe --model <m> --dataset <d>

The recipe layer enforces modality compatibility — feeding DNA to scGPT, or
scRNA to ESM-2, is refused as rubbish-in-rubbish-out.

The headline cross-FM metric is the **SAE − PCA ablation gap** (see
`core/ablation.py`): an empirical measure of how distributed each model's
knowledge-carrying geometry is.
"""

from .core.adapter import BioFMAdapter
from .core.dataset import (
    DatasetAdapter,
    Sample,
    DATASET_REGISTRY,
    register_dataset,
    load_dataset_adapter,
)
from .core.recipe import AuditRecipe, validate_modality_match
from .core.ablation import sae_pca_ablation_gap, ablation_curve, DEFAULT_K_GRID

__all__ = [
    "BioFMAdapter",
    "DatasetAdapter",
    "Sample",
    "DATASET_REGISTRY",
    "register_dataset",
    "load_dataset_adapter",
    "AuditRecipe",
    "validate_modality_match",
    "sae_pca_ablation_gap",
    "ablation_curve",
    "DEFAULT_K_GRID",
]
