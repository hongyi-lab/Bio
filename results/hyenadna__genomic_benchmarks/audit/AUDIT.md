# Audit — hyenadna__genomic_benchmarks

- Model: `hyenadna` (d_model=256, n_layers=4, modality=dna)
- Dataset: `genomic_benchmarks` (n_samples=20000, classes=2, baseline=kmer_pca)
- Seeds: [0, 1, 2, 3, 4]

## Baseline
- **kmer_pca**: acc = 0.8188 ± 0.0050, F1 = 0.8187 ± 0.0050
- baseline PCA dim: 50

## Per-layer probe (5-seed mean ± std)
| layer | mean-pool acc |
|---|---|
| layer_00_input | 0.7278±0.0069 |
| layer_01 | 0.8033±0.0069 |
| layer_02 | 0.8124±0.0055 |
| layer_03 | 0.8155±0.0068 |
| layer_04 | 0.8143±0.0077 |

## SVD spectrum
| layer | PR | k50 | k95 | k99 | var_per_elem |
|---|---|---|---|---|---|
| layer_00_input | 3.1 | 2 | 3 | 4 | 0.001 |
| layer_02 | 3.5 | 1 | 15 | 41 | 1.078 |
| layer_04 | 4.9 | 2 | 24 | 66 | 1.217 |

## SAE per selected layer
| layer | var_exp | dead_ever | SAE probe acc | PCA probe acc |
|---|---|---|---|---|
| layer_00_input | 1.000 | 10/2048 | 0.7276±0.0071 | 0.7279±0.0073 |
| layer_02 | 0.999 | 0/2048 | 0.8124±0.0073 | 0.8116±0.0060 |
| layer_04 | 0.998 | 0/2048 | 0.8185±0.0068 | 0.8147±0.0083 |

## SAE − PCA ablation gap (the headline)
Positive gap ⇒ knowledge more **distributed** than PCA captures ⇒ SAE recovers a hidden sparse dictionary PCA misses.
Gap ≈ 0 ⇒ knowledge **concentrated** in PCA-aligned dims ⇒ no extra inversion space.

| layer | gap@K=1 | gap@K=2 | gap@K=4 | gap@K=8 | gap@K=16 | gap@K=32 | gap@K=64 | gap@K=128 | gap@K=256 |
|---|---|---|---|---|---|---|---|---|---|
| layer_00_input | +0.064±0.011 | +0.084±0.008 | +0.182±0.007 | +0.181±0.008 | +0.179±0.007 | +0.173±0.007 | +0.000±0.002 | +0.000±0.002 | +0.000±0.002 |
| layer_02 | -0.001±0.001 | -0.002±0.004 | -0.003±0.002 | -0.005±0.002 | -0.007±0.002 | -0.004±0.006 | -0.006±0.006 | -0.013±0.010 | +0.178±0.012 |
| layer_04 | -0.000±0.001 | -0.002±0.003 | -0.001±0.003 | -0.004±0.003 | -0.005±0.002 | -0.005±0.006 | -0.007±0.003 | -0.005±0.004 | +0.227±0.010 |
