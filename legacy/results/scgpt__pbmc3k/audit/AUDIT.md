# Audit — scgpt__pbmc3k

- Model: `scgpt` (d_model=512, n_layers=12, modality=scrna)
- Dataset: `pbmc3k` (n_samples=2638, classes=8, baseline=log1p_pca)
- Seeds: [0, 1, 2, 3, 4]

## Baseline
- **log1p_pca**: acc = 0.9348 ± 0.0061, F1 = 0.9006 ± 0.0427
- baseline PCA dim: 50

## Per-layer probe (5-seed mean ± std)
| layer | CLS acc | mean-pool acc |
|---|---|---|
| layer_00_input | 0.4337±0.0000 | 0.9352±0.0063 |
| layer_01 | 0.9027±0.0082 | 0.9269±0.0088 |
| layer_02 | 0.9121±0.0103 | 0.9280±0.0107 |
| layer_03 | 0.9231±0.0097 | 0.9250±0.0114 |
| layer_04 | 0.9220±0.0112 | 0.9216±0.0116 |
| layer_05 | 0.9170±0.0120 | 0.9189±0.0142 |
| layer_06 | 0.9140±0.0137 | 0.9110±0.0146 |
| layer_07 | 0.9072±0.0121 | 0.9091±0.0115 |
| layer_08 | 0.9068±0.0090 | 0.9049±0.0101 |
| layer_09 | 0.9030±0.0086 | 0.9027±0.0087 |
| layer_10 | 0.8943±0.0136 | 0.8936±0.0140 |
| layer_11 | 0.8799±0.0133 | 0.8811±0.0154 |
| layer_12 | 0.8413±0.0096 | 0.8458±0.0090 |

## SVD spectrum
| layer | PR | k50 | k95 | k99 | var_per_elem |
|---|---|---|---|---|---|
| layer_00_input | 44.2 | 51 | 405 | 483 | 1.867 |
| layer_06 | 13.0 | 7 | 238 | 398 | 0.975 |
| layer_12 | 10.8 | 4 | 88 | 240 | 0.959 |

## SAE per selected layer
| layer | var_exp | dead_ever | SAE probe acc | PCA probe acc |
|---|---|---|---|---|
| layer_00_input | 0.918 | 0/2048 | 0.9292±0.0074 | 0.9371±0.0049 |
| layer_06 | 0.993 | 499/2048 | 0.8780±0.0081 | 0.9125±0.0113 |
| layer_12 | 1.000 | 936/2048 | 0.8095±0.0086 | 0.8409±0.0114 |

## SAE − PCA ablation gap (the headline)
Positive gap ⇒ knowledge more **distributed** than PCA captures ⇒ SAE recovers a hidden sparse dictionary PCA misses.
Gap ≈ 0 ⇒ knowledge **concentrated** in PCA-aligned dims ⇒ no extra inversion space.

| layer | gap@K=1 | gap@K=2 | gap@K=4 | gap@K=8 | gap@K=16 | gap@K=32 | gap@K=64 | gap@K=128 | gap@K=256 |
|---|---|---|---|---|---|---|---|---|---|
| layer_00_input | +0.003±0.004 | +0.001±0.003 | -0.006±0.008 | -0.007±0.007 | -0.008±0.010 | +0.002±0.009 | +0.034±0.016 | +0.127±0.014 | +0.228±0.011 |
| layer_06 | -0.005±0.007 | -0.004±0.003 | -0.009±0.003 | -0.017±0.008 | -0.053±0.009 | -0.332±0.013 | -0.284±0.010 | -0.117±0.014 | +0.027±0.008 |
| layer_12 | +0.009±0.006 | -0.002±0.011 | +0.066±0.010 | +0.262±0.017 | +0.201±0.015 | +0.031±0.008 | +0.031±0.008 | +0.031±0.008 | +0.031±0.008 |
