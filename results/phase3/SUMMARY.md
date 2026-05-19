# Phase 3 — TopK SAE on scGPT activations

- Data: `pbmc3k.h5ad` — 2638 cells, 8 classes, label=`louvain`
- SAE config: dict=2048, k=32, epochs=20, bs=4096, lr=0.001
- Probe seeds: [0, 1, 2, 3, 4] (cell-type probe results = mean ± std)
- Phase2 baselines for context: PCA-50 = 0.940±0.008, scGPT best CLS (layer 3) = 0.909±0.007

## Per-layer table

| layer | tokens | var_exp | dead_ever | cell-type acc (SAE) | cell-type F1 (SAE) | Δacc vs PCA-50 |
|---|---|---|---|---|---|---|
| layer_00_input | 2055164 | 0.919 | 0/2048 | 0.9258±0.0043 | 0.8448±0.0363 | -0.0140 |
| layer_03 | 2055164 | 0.975 | 118/2048 | 0.8655±0.0075 | 0.7200±0.0488 | -0.0743 |
| layer_12 | 2055164 | 1.000 | 844/2048 | 0.8390±0.0139 | 0.6366±0.0544 | -0.1008 |

## Reading guide

- `var_exp` close to 1 means the SAE reconstructs well; below ~0.8 means TopK is too aggressive.
- `dead_ever` is features that never activated across the whole training set. Large fraction dead ⇒ dict too big or k too small.
- `Δacc vs PCA-50` < 0 means the SAE on this layer's activations cannot recover cell-type as well as raw-PCA on log1p — i.e., the layer's activations are not a better cell-type-encoding substrate than the raw expression matrix.
- High |r| with a gene set on layer L but not layer 0 = scGPT is computing that pathway-level signal in the transformer, not just lookup. Worth investigating.