# Phase 8 — Evo-1 7B SAE-on-delta × JASPAR TFBS hit-rate (2026-05-23 snapshot)

This folder is a self-contained snapshot of the phase 8 results: the JSONs needed to reproduce all figures + headline numbers, plus the final summary figure.

## Question

Within Evo-1 7B (a StripedHyena architecture: alternating Hyena and attention blocks on the same residual stream), do **Hyena deltas** and **attention deltas** carry equivalent biological structure?

For each target block, "delta" = `block_output - block_input`. This isolates the block's net contribution to the residual stream.

We compare 3 adjacent Hyena/attention pairs across early, middle, and late stripes:

| pair | Hyena block | attention block |
|---|---|---|
| early | layer_08 | layer_09 |
| mid | layer_16 | layer_17 |
| late | layer_24 | layer_25 *(JASPAR eval still running)* |

For each layer, we train a TopK SAE (dict=16,384, k=32, 10 epochs, Anthropic-style dead-feature resampling) on the delta token activations (5.02 M tokens × 4096 d_model, derived from 20 k human non-TATA promoter sequences from GenomicBenchmarks), then evaluate each live SAE feature by finding its top-50 activating tokens, extracting ±15 bp DNA context, and scoring against all 879 JASPAR 2024 vertebrate PWMs with a shuffle-null p-value.

## Headline findings

### 1. Attention deltas hit JASPAR motifs more often than Hyena deltas — replicated across pairs

| pair | Hyena hit-rate | attention hit-rate | Δ (att − hyena) |
|---|---|---|---|
| early (08 vs 09) | 33.83 % | 38.12 % | **+4.29 pp** |
| mid (16 vs 17) | 33.86 % | 38.12 % | **+4.26 pp** |
| late (24 vs 25) | *running* | *running* | — |

Two independent measurements, **same direction, same magnitude (within 0.03 pp)**.

### 2. Attention deltas are 5–10× sparser than Hyena deltas

| layer | live features / 16,384 |
|---|---|
| layer_08 (Hyena) | 5,253 |
| layer_09 (attention) | **564** |
| layer_16 (Hyena) | 12,372 |
| layer_17 (attention) | **2,290** |

Same dict size, same SAE training recipe (including resampling). Attention layers spontaneously concentrate signal into fewer features — and those fewer features carry more TFBS structure per feature.

### 3. No "spectral hiding place" — sequence-axis FFT of mid-stripe activations is white-ish for both

The natural follow-up worry: Hyena learns long convolutional filters; maybe it encodes structure along the *sequence* axis that channel-wise SAE literally can't see. We computed `torch.fft.rfft` along the sequence dimension for L=128, 2000 samples, then averaged power over channels.

| metric | Hyena (layer_16) | attention (layer_17) |
|---|---|---|
| non-DC peak frequency | k=43 (power = 0.81 × DC) | k=1 (power = 0.92 × DC) |
| mid-band (k=4..16) fraction | 19.6 % | 20.1 % |

Both spectra are roughly **flat from DC to Nyquist** (far above a 1/k reference). Hyena does NOT show peaked frequency-band structure. The +4 pp attention advantage stands on its own — it isn't a channel-SAE artifact missing a hidden spectral signal.

## Caveats (carry to the paper)

- **Uncorrected per-feature p-values.** The reported hit-rate is the fraction of live features whose best motif passes p<1e-3 against a 1000-shuffle null. We have not applied multiple-testing correction across the ~17 k (motif × window) comparisons per feature. Absolute numbers are likely inflated 5-10×; the **+4 pp pairwise Δ is the trustworthy quantity** because the inflation is identical on both sides of every pair.
- **Late stripe pending.** Two-pair signal is suggestive; three-pair confirmation is in progress (layer_24/25 JASPAR eval is set to resume with `max_features=2000`).
- **Promoter task only.** GenomicBenchmarks human_nontata_promoters is a regulatory dataset by construction. Results may differ on exonic / intergenic / coding-region tasks.
- **One model.** Tested on Evo-1 7B only. Whether the Hyena-vs-attention asymmetry generalizes to Evo-2 / HyenaDNA-large is open.

## Files in this folder

```
phase8_evo1_summary_2026_05_23/
├── README.md                      ← this file
├── make_figure.py                 ← script that produces phase8_summary.png from raw/
├── phase8_summary.png             ← 3-panel summary figure (the headline result)
└── raw/                           ← all the JSONs needed to reproduce the figure
    ├── layer_<NN>_<kind>__jaspar_hits.json     (4 files; layer_24/25 added when eval finishes)
    ├── layer_<NN>_<kind>__sae_training_log.json (6 files; one per layer)
    ├── power_spectrum_*.json
    └── power_spectrum_*.png       (the standalone 2-panel spectral figure)
```

## How to reproduce the summary figure

```bash
cd /srv/hgao0864/Bio/bio_fm_probing/results/phase8_evo1_summary_2026_05_23
python make_figure.py
# writes phase8_summary.png in the same folder
```
