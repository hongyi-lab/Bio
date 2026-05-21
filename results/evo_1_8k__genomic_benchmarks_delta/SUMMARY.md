# Phase 8 — SAE-on-delta × JASPAR hit rate (DRY RUN)

Pairs evaluated: 1 (2 SAEs reported)

## Per-layer table

| layer | kind | live / total | var_exp | dead_ever | resample | hits | hit rate (live) |
|---|---|---|---|---|---|---|---|
| layer_16 | **hyena** | 4096/4096 (100.0%) | 0.1561 | 0 | 0 | 85 | **2.1%** |
| layer_17 | **attention** | 516/4096 (12.6%) | -1.6619 | 678 | 0 | 75 | **14.5%** |

## Headline comparison (Hyena vs Attention)

Within each pair, compare the two kinds. If Hyena hit-rate ≈ Attention hit-rate across all pairs, the residual-stream linear-interface argument dominates and the 'operator-specific geometry' claim is unsupported. If they diverge, that's real evidence for an operator effect.

| pair | hyena hit rate | attention hit rate | Δ (att − hyena) |
|---|---|---|---|
| mid | 2.1% (layer_16) | 14.5% (layer_17) | +12.5pp |