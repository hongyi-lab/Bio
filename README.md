# bio_fm_probing

Probing biological foundation models. Started on [scGPT](https://github.com/bowang-lab/scGPT) (whole-human) as phases 1-3; now generalized into a model-agnostic toolkit under `bio_fm_probe/` so the same audit can be run on scMamba, Geneformer, scFoundation, UCE, and friends without rewriting probes.

```
bio_fm_probing/
├── checkpoints/<model>/        # weights / args / vocab per model (gitignored)
├── data/                       # pbmc3k.h5ad and other AnnData datasets (gitignored)
├── gene_sets/                  # public reference gene sets (MSigDB, KEGG, REACTOME, ...)
├── src/                        # legacy scGPT-specific scripts (phase 1-3)
│   ├── download_checkpoint.py  # gdown the scGPT whole-human folder
│   ├── download_data.py        # fetch pbmc3k (and optional 10X multiome)
│   ├── load_scgpt.py           # original scGPT loader / embedder
│   ├── sanity_check.py         # 50-cell UMAP sanity
│   ├── layer_probe.py          # phase 1: per-layer CLS probe, single seed
│   ├── phase2_probe.py         # phase 2: + multi-seed, baselines, mean-pool, H1
│   └── phase3_sae.py           # phase 3: TopK SAE on token-level activations
├── bio_fm_probe/               # model-agnostic refactor (phase 4)
│   ├── core/                   # adapter ABC, extraction, probes (LR, PCA, SVD, SAE)
│   ├── adapters/scgpt.py       # scGPT under the standard interface
│   ├── adapters/_template.py   # copy this to add a new model
│   ├── run_audit.py            # one-command end-to-end audit
│   ├── compare_models.py       # cross-model summary table
│   └── README.md               # toolkit usage
├── results/                    # audit outputs (small JSON+PNG tracked; NPZ gitignored)
└── README.md                   # this file
```

The new toolkit (`bio_fm_probe/`) reproduces phase 2/3 results when run on scGPT, and is what we use going forward for other foundation models. Phase 1-3 scripts in `src/` are kept for reproducibility of the original numbers.

> **Note on notebooks**: the repo intentionally has no `.ipynb` files. Sanity-checking is done with `src/sanity_check.py`. Use Jupyter locally if you want to explore interactively.

---

## 1. Environment

Conda env lives at `/srv/hgao0864/conda_envs/scgpt_env` (kept off `/home` since `/home` is near full). Python 3.10.

```bash
# create
source ~/miniconda3/etc/profile.d/conda.sh
conda create --prefix /srv/hgao0864/conda_envs/scgpt_env python=3.10 -y
conda activate /srv/hgao0864/conda_envs/scgpt_env

# torch with CUDA 11.8 (driver supports it on the A6000 here)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118

# scgpt + supporting libs (numpy pinned <2 for torch 2.1.2 ABI)
pip install 'numpy<2' 'scanpy>=1.9' anndata scgpt gdown tqdm \
    matplotlib seaborn umap-learn jupyterlab ipykernel
```

`flash-attn` is **not** installed (optional; failure to build is common). scGPT auto-falls back to the standard transformer.

Activate before every step below:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /srv/hgao0864/conda_envs/scgpt_env
cd /srv/hgao0864/Bio/scGPT_probing
```

## 2. Download the pretrained checkpoint

The whole-human checkpoint is hosted on Google Drive (linked from the official repo). The script uses `gdown` and shows a progress bar.

```bash
python src/download_checkpoint.py
```

Lands in `checkpoints/scGPT_human/`:

| file          | purpose                       | approx size |
|---------------|-------------------------------|-------------|
| `args.json`   | model hyperparameters         | <1 KB       |
| `vocab.json`  | gene-symbol vocabulary        | ~1 MB       |
| `best_model.pt` | model weights              | ~1.5 GB     |

The script is idempotent — if the three files are already present, it skips re-downloading.

If gdown fails (Google Drive quotas, redirects), re-run; gdown resumes partial downloads.

## 3. Download a PBMC dataset

```bash
python src/download_data.py            # pbmc3k.h5ad (~6 MB, has 'louvain' labels)
python src/download_data.py --multiome # additionally fetch 10X PBMC Multiome (~1 GB)
```

Output: `data/pbmc3k.h5ad` (and optionally `data/pbmc_multiome_10k.h5`).

## 4. Extract embeddings (`src/load_scgpt.py`)

```bash
python src/load_scgpt.py \
    --model_dir checkpoints/scGPT_human \
    --data data/pbmc3k.h5ad \
    --n_cells 50 \
    --batch_size 16 \
    --save results/embeddings.npz
```

Prints the shapes of:

- `cell_embeddings`: `[n_cells, d_model]` — the CLS-token output, one vector per cell.
- `per_gene_embeddings_last_batch`: `[batch, seq_len, d_model]` — per-gene token embeddings for the last batch (full sequence; index 0 is the CLS token, the rest are tokenized genes for that cell).

### Resumable inference

Per-batch cell embeddings are written to `--save` after every batch. Re-running the same command picks up at the last saved batch via `next_idx` stored in the NPZ. To restart from scratch, delete the save file.

## 5. End-to-end sanity check

```bash
python src/sanity_check.py
```

Loads pbmc3k → loads scGPT → samples 50 cells → extracts embeddings → UMAP. Writes:

- `results/sanity_embeddings.npz` (checkpoint of embeddings)
- `results/sanity_check_umap.png` (UMAP colored by cell type)
- `results/sanity_check_summary.json` (shapes, dims, device)

## 6. Phase 2 — controls for the layer probe

Phase 1 produced an inverted-U curve (layer 0 ≈ 0.43, peak at layer 5 ≈ 0.92, layer 12 ≈ 0.88) on a single seed, CLS-only, with no baseline. Before reading hierarchy into that shape, phase 2 tests four alternative explanations in one pass:

| Hypothesis | Test |
|---|---|
| **H1**: layer 0 CLS is a constant slot across cells (so the 0→1 jump is artifact) | per-dim std of `layer_00_input[:, 0, :]` vs. a mid-layer CLS reference |
| **H2**: "layer 5 peak" is within seed noise on a 528-cell test set | re-run with N seeds (default 5), report mean ± std bands |
| **H3**: pbmc3k cell type is recoverable from any reasonable feature set | PCA-50, PCA-512, raw log1p LR baselines on the same splits |
| **H4**: CLS-only under-represents late layers if information diffuses to gene tokens | parallel mean-pool curve over non-pad non-CLS gene tokens |

```bash
python src/phase2_probe.py                                  # default: pbmc3k, seeds 0..4
python src/phase2_probe.py --force_extract                  # ignore cached activations
python src/phase2_probe.py --seeds 0 1 2 3 4 5 6 7 8 9      # custom seed list
python src/phase2_probe.py --skip_baselines                 # H1 + H2 + H4 only
```

Outputs land in `results/phase2/`:

| file | content |
|---|---|
| `layer_activations.npz` | per-layer CLS + mean-pool cache (gitignored, regenerable) |
| `X_log1p_for_baselines.npy` | preprocessed log1p matrix (gitignored) |
| `layer0_sanity.json` | std / max-deviation stats for `layer_00_input` vs reference |
| `baselines.json` | PCA-50, PCA-512, raw log1p × N seeds |
| `per_layer_probe.json` | per layer × {CLS, mean-pool} × N seeds |
| `layer_probe_curve_v2.png` | two subplots (accuracy, macro-F1) with mean ± std bands + baseline lines |
| `SUMMARY.md` | digest with per-layer table and key deltas |

The script reuses `load_scgpt.py` (`load_scgpt_model` + `preprocess_adata_for_scgpt`) so the model side is identical to phase 1; only the hook (full `(B, seq, d)` capture for pooling) and the probe setup differ.

## 7. Phase 3 — TopK SAE on scGPT activations

Phase 2 result, condensed: **scGPT's transformer blocks do not add cell-type-discriminative capacity above the input embedding lookup, and PCA-50 on raw log1p genes beats every scGPT layer by ~3 pp.** Mean-pool peaks at `layer_00_input` (0.932) and decays monotonically to layer 12 (0.819). The earlier "inverted-U" was an artifact of (a) the layer-0 CLS slot being constant before attention, and (b) the missing baseline.

This shifts the question. scGPT is computing *something* during its forward pass — that something is just not cell-type per se. Phase 3 trains a TopK sparse autoencoder on gene-token activations at selected layers and asks:

1. **Reconstruction**: can a sparse code with ~32 active features per token explain the layer's activations?
2. **Cell-type via sparse code**: after compressing each cell to a mean-SAE-activation vector, does cell-type recover linearly? If yes, the cell-type signal is *there*, just in a sparse / non-linearly-readable layout.
3. **Gene-set correspondence** *(if `gene_sets/` provided)*: does any SAE feature correlate with a known pathway score (HALLMARK_*, KEGG_*, REACTOME_*)?
4. **Causal ablation** *(optional)*: zero out the top-k cell-type-correlated features and re-probe — measure the drop.

```bash
python src/phase3_sae.py                                        # default: layer_00_input, layer_03, layer_12
python src/phase3_sae.py --layers layer_03                      # one layer
python src/phase3_sae.py --dict_size 2048 --k 32 --epochs 20    # SAE hyperparams
python src/phase3_sae.py --gene_sets_dir gene_sets/             # turn on gene-set correlations
python src/phase3_sae.py --do_ablation --ablation_k 10          # top-k feature ablation
python src/phase3_sae.py --force_extract --force_retrain        # nuke caches
```

**Defaults**: `dict_size=2048` (4× expansion of `d_model=512`), `k=32`, `epochs=20`, `batch_size=4096`, `lr=1e-3`. Each layer caches its token activations (~3 GB for pbmc3k) and SAE weights, so re-runs skip the slow steps.

Outputs land in `results/phase3/<layer>/`:

| file | content |
|---|---|
| `token_activations.npz` | (N_tokens, d_model) + cell/gene indices — gitignored, regenerable |
| `sae.pt` | TopKSAE state_dict + config |
| `training_log.json` | per-epoch loss, var_explained, dead-feature count, mean L0 |
| `curves.png` | training-curve panel (loss / var_exp / dead features) |
| `per_cell_features.npz` | (n_cells, n_features) mean SAE activation per cell |
| `cell_type_probe.json` | 5-seed LR probe on SAE features (compare to phase2 dense) |
| `gene_set_correlations.json` | Pearson r between each feature and each gene-set score, top-10 per set (when gene sets are loaded) |
| `feature_ablation.json` | before/after probe with top-k features zeroed (when `--do_ablation`) |

A cross-layer `results/phase3/SUMMARY.md` is also written, with one row per layer and a Δacc-vs-PCA-50 column.

**Gene sets (`gene_sets/`)**: ten public reference sets (HALLMARK MTORC1 / MYC / UPR, KEGG PROTEASOME / RIBOSOME, REACTOME TRANSLATION, GO RIBOSOMAL_SUBUNIT, ER_STRESS, IGARASHI_ATF4, HK_genes). One gene symbol per line; comment lines starting with `#` or `>` and URL lines are skipped. Genes are matched against `adata.var.gene_name`; sets with fewer than 3 overlapping genes are dropped automatically.

## 8. Phase 4 — model-agnostic toolkit (`bio_fm_probe/`)

Phases 1-3 worked but were three separate scripts hardcoded for scGPT. Phase 4 turns them into a toolkit so testing the next bio foundation model (scMamba, Geneformer, scFoundation, UCE, …) is "write a ~50-line adapter, run one command, read `AUDIT.md`" rather than "rewrite three scripts".

```
bio_fm_probe/
├── core/                       # model-agnostic
│   ├── adapter.py              # BioFMAdapter ABC (3 methods to implement)
│   ├── extract.py              # generic forward+hook loop using the adapter
│   └── probes.py               # LR probe, PCA-50/512 baseline, SVD diag, TopK SAE
├── adapters/
│   ├── scgpt.py                # scGPT under the standard interface (refactor of src/)
│   └── _template.py            # copy to add a new model
├── run_audit.py                # end-to-end audit → results/<model>/audit/
└── compare_models.py           # cross-model summary table
```

Run the full audit on scGPT (equivalent to phase 2 + phase 3 + SVD in one shot):

```bash
python -m bio_fm_probe.run_audit \
    --adapter scgpt \
    --model_dir checkpoints/scGPT_human \
    --data data/pbmc3k.h5ad \
    --label_col louvain
```

Output goes to `results/scgpt/audit/`:
- `AUDIT.md` — one-page digest (baselines, layer 0 sanity, per-layer table, SVD spectrum, SAE summary)
- `baselines.json`, `layer0_sanity.json`, `per_layer_probe.json`, `svd_diag.json`
- `sae/<layer>/` — SAE weights, training log, per-cell sparse features, cell-type probe
- `cls_mean_per_layer.npz` — extraction cache (gitignored)

Compare several models after each has been audited:

```bash
python -m bio_fm_probe.compare_models scgpt scmamba geneformer
# writes results/_cross_model_summary.md
```

The comparison table answers, side by side, the same four questions the scGPT audit answered: does it beat PCA-50, is layer 0 CLS degenerate, where is the best layer, and does the representation collapse to low rank with depth.

Adding a new model is documented in `bio_fm_probe/README.md` — copy `adapters/_template.py` and fill in `load`, `preprocess`, `iter_layer_activations`.

## 9. Phase 5 — recipe layer + SAE−PCA ablation gap (`bio_fm_probe/run_recipe.py`)

Phase 4 toolkit covered "single model × layer probe + SAE" but kept dataset and task hardcoded to pbmc3k + cell-type. Phase 5 fills that gap with a `Dataset` adapter abstraction symmetric to `Model`, plus the headline cross-FM metric: **SAE − PCA ablation gap**.

The ablation gap empirically measures **how distributed each model's knowledge-carrying geometry is**:

- **Concentrated** models (knowledge on few neurons): gap ≈ 0; PCA top-K already locates the knowledge, SAE adds nothing.
- **Distributed** models (knowledge spread non-orthogonally across many neurons): gap > 0; PCA misses it, SAE recovers the hidden sparse dictionary.

The gap curve over K is the cross-FM benchmark axis: an empirical readout of each model's "knowledge-carrying geometry" rather than another method comparison.

```bash
# scRNA — scGPT on pbmc3k (reproduce phase 1-3 via recipe)
python -m bio_fm_probe.run_recipe --model scgpt --model_dir checkpoints/scGPT_human --dataset pbmc3k

# DNA — HyenaDNA on human_nontata_promoters
python src/download_hyenadna.py && python src/download_genomic_benchmarks.py
python -m bio_fm_probe.run_recipe --model hyenadna \
    --model_dir checkpoints/hyenadna-small-32k-seqlen-hf --dataset genomic_benchmarks

# Protein — ESM-2 on DeepLoc
python src/download_esm2.py && python src/download_deeploc.py
python -m bio_fm_probe.run_recipe --model esm2 \
    --model_dir checkpoints/esm2_t12_35M_UR50D --dataset deeploc

# Cross-recipe master table + gap-curve plot
python -m bio_fm_probe.cross_recipe_summary \
    scgpt__pbmc3k hyenadna__genomic_benchmarks esm2__deeploc
```

Outputs land in `results/<model>__<dataset>/audit/`:

| file | content |
|---|---|
| `AUDIT.md` | digest (baselines, per-layer probe, SVD, SAE, ablation gap table) |
| `baselines.json` | modality-appropriate baseline (log1p+PCA / kmer+PCA / onehot_aa+PCA) |
| `per_layer_probe.json` | per-layer LR probe (CLS + mean-pool, 5 seeds) |
| `svd_diag.json` | SVD spectrum per SAE layer (PR / k50 / k95 / k99) |
| `sae/<layer>/sae.pt` | TopK SAE weights |
| `sae/<layer>/training_log.json` | per-epoch loss / var_exp / dead-features |
| `sae/<layer>/per_cell_features.npz` | per-cell SAE-aggregated features |
| `sae/<layer>/per_cell_pca_features.npz` | per-cell PCA-aggregated features (matched dim) |
| `sae/<layer>/ablation_gap.json` | **the headline** — SAE and PCA ablation curves + gap series |
| `cls_mean_per_layer.npz` (gitignored) | extraction cache |

Cross-recipe summary writes:

- `results/_master_table.md` — gap@K table modality-grouped
- `results/_ablation_gap_curves.pdf` — overlay plot of all (model × layer) gap curves

**Modality validation**: `validate_modality_match` refuses to run a (model, dataset) pair where modalities don't match (e.g. scGPT on DNA). This is the explicit guard against rubbish-in-rubbish-out.

The phase 4 entry `run_audit.py` is kept as a legacy path for scRNA-only audits. New work goes through `run_recipe.py`.

## Troubleshooting

- **`flash_attn` import errors inside scgpt**: ignore — scgpt's transformer falls back to the stock PyTorch implementation when `use_fast_transformer=False`, which is what `load_scgpt.py` sets.
- **`RuntimeError: size mismatch` on `load_state_dict`**: `load_scgpt_model` strips mismatched keys (`strict=False`) and prints how many were kept. As long as encoder/embedding weights load, embeddings will work.
- **`numpy ABI` errors from torch**: confirm `pip show numpy` reports a `1.x` version. Reinstall with `pip install 'numpy<2'` if drifted.
- **gdown quota / "Cannot retrieve folder"**: open the folder URL in a browser once to clear the quota, then re-run the script.
