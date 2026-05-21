# Phase 2 — scGPT layer probe with controls

- Data: `pbmc3k.h5ad` — 2638 cells, 8 classes, label=`louvain`
- Seeds: [0, 1, 2, 3, 4] (all numbers reported as mean ± std across seeds)
- d_model: 512

## H1 — layer 0 CLS variance
- `layer_00_input` CLS std / `layer_05` CLS std = **0.0002308**
- `layer_00_input` max |dev from mean| across cells = **0.0002251**
- (ratio ≪ 1 ⇒ CLS at layer 0 is near-constant across cells, so phase1 layer-0 baseline is degenerate and the 0→1 'jump' is artifact)

## H3 — baselines
- **raw_log1p_LR**: acc = 0.9428 ± 0.0099, macro-F1 = 0.9079 ± 0.0436
- **PCA50_LR**: acc = 0.9398 ± 0.0077, macro-F1 = 0.9106 ± 0.0433
- **PCA512_LR**: acc = 0.9428 ± 0.0083, macro-F1 = 0.9054 ± 0.0399

- Δ(best CLS layer − PCA-50 LR) = **-0.0311**
- Δ(layer 12 CLS − peak CLS)    = **-0.0947**

## H2 + H4 — per-layer, multi-seed, CLS vs mean-pool
- Best CLS layer:       **layer_03**  acc = 0.9087 ± 0.0070
- Best mean-pool layer: **layer_00_input**  acc = 0.9322 ± 0.0070
- Range across layers 1..N (CLS, acc): min = 0.8140, max = 0.9087, spread = 0.0947
- Median seed-std across layers 1..N (CLS): 0.0069

(If layer-1..N spread is within ~2× the median seed-std, the 'inverted-U' is not statistically distinguishable from a flat plateau with noise.)

## Full per-layer table (mean ± std across seeds)

| layer | CLS acc | CLS F1 | mean-pool acc | mean-pool F1 |
|---|---|---|---|---|
| layer_00_input | 0.4337±0.0000 | 0.0756±0.0000 | 0.9322±0.0070 | 0.8715±0.0342 |
| layer_01 | 0.8890±0.0078 | 0.7941±0.0488 | 0.9235±0.0093 | 0.8545±0.0426 |
| layer_02 | 0.9034±0.0092 | 0.8245±0.0467 | 0.9170±0.0059 | 0.8475±0.0383 |
| layer_03 | 0.9087±0.0070 | 0.8333±0.0379 | 0.9114±0.0063 | 0.8359±0.0450 |
| layer_04 | 0.9004±0.0051 | 0.8188±0.0454 | 0.8981±0.0059 | 0.8131±0.0395 |
| layer_05 | 0.8992±0.0066 | 0.8173±0.0408 | 0.8920±0.0064 | 0.8038±0.0405 |
| layer_06 | 0.8996±0.0048 | 0.8170±0.0381 | 0.8898±0.0059 | 0.7952±0.0479 |
| layer_07 | 0.8917±0.0074 | 0.7901±0.0382 | 0.8864±0.0086 | 0.7775±0.0505 |
| layer_08 | 0.8867±0.0070 | 0.7784±0.0468 | 0.8811±0.0066 | 0.7724±0.0493 |
| layer_09 | 0.8814±0.0064 | 0.7717±0.0468 | 0.8807±0.0058 | 0.7769±0.0369 |
| layer_10 | 0.8754±0.0069 | 0.7634±0.0543 | 0.8742±0.0094 | 0.7685±0.0439 |
| layer_11 | 0.8587±0.0068 | 0.7330±0.0452 | 0.8625±0.0074 | 0.7377±0.0474 |
| layer_12 | 0.8140±0.0121 | 0.6386±0.0346 | 0.8186±0.0112 | 0.6466±0.0307 |