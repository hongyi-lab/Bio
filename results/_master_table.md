# Cross-recipe master table — 2 recipe(s)

Each row = one (model, dataset) audit. `gap@K` is the headline SAE−PCA ablation gap at K ablated features in the SAE feature space (and matched K in PCA). Higher gap ⇒ knowledge more distributed; gap ≈ 0 ⇒ knowledge concentrated in PCA dims.

| recipe | baseline acc | best layer-probe acc | Δ vs baseline | gap@K=32 (best layer) | gap@K=128 (best) |
|---|---|---|---|---|---|
| scgpt__pbmc3k | 0.9348 | 0.9352 | +0.0004 | +0.0314 | +0.1273 |
| hyenadna__genomic_benchmarks | 0.8188 | 0.8155 | -0.0033 | +0.1731 | +0.0004 |

## Per-layer gap curves

### scgpt__pbmc3k
| layer | gap@K=1 | gap@K=4 | gap@K=16 | gap@K=32 | gap@K=64 | gap@K=128 | gap@K=256 |
|---|---|---|---|---|---|---|---|
| layer_00_input | +0.003 | -0.006 | -0.008 | +0.002 | +0.034 | +0.127 | +0.228 |
| layer_06 | -0.005 | -0.009 | -0.053 | -0.332 | -0.284 | -0.117 | +0.027 |
| layer_12 | +0.009 | +0.066 | +0.201 | +0.031 | +0.031 | +0.031 | +0.031 |

### hyenadna__genomic_benchmarks
| layer | gap@K=1 | gap@K=4 | gap@K=16 | gap@K=32 | gap@K=64 | gap@K=128 | gap@K=256 |
|---|---|---|---|---|---|---|---|
| layer_00_input | +0.064 | +0.182 | +0.179 | +0.173 | +0.000 | +0.000 | +0.000 |
| layer_02 | -0.001 | -0.003 | -0.007 | -0.004 | -0.006 | -0.013 | +0.178 |
| layer_04 | -0.000 | -0.001 | -0.005 | -0.005 | -0.007 | -0.005 | +0.227 |
