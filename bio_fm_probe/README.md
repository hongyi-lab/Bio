# bio_fm_probe — auditing toolkit for biological foundation models

A model-agnostic probing toolkit. Drop a model adapter and a dataset adapter in,
run one command, get a standardized audit (layer probe + baselines + SVD + SAE
+ **SAE − PCA ablation gap**) with an `AUDIT.md` digest. Then run
`cross_recipe_summary.py` to put several recipes side by side.

## What the audit answers

For each `(model, dataset)` recipe — same data, same seeds, modality-appropriate
baseline:

1. **Does the model beat the modality-appropriate baseline?** (`baselines.json`)
2. **Is layer-0 CLS slot constant before attention?** (`layer0_sanity.json`, scrna only)
3. **Where is the model's best layer, and how much does pooling matter?** (`per_layer_probe.json`)
4. **Does the representation collapse to low rank with depth?** (`svd_diag.json`)
5. **Can a TopK SAE reconstruct activations, and do its sparse features recover the task label?** (`sae/<layer>/...`)
6. **Headline: SAE − PCA ablation gap.** Empirically measures how distributed (vs concentrated) the model's knowledge geometry is. (`sae/<layer>/ablation_gap.json`)

The ablation gap is the cross-FM benchmark axis: concentrated models (LLaMA-like) → small gap, PCA suffices. Distributed models (Qwen-like) → large gap, only SAE recovers the hidden sparse dictionary PCA misses.

## Layout

```
bio_fm_probe/
├── core/
│   ├── adapter.py          # BioFMAdapter ABC (model side)
│   ├── dataset.py          # DatasetAdapter ABC + Sample dataclass
│   ├── extract.py          # generic per-layer / per-token extraction
│   ├── probes.py           # LR probe, PCA baselines, SVD, TopK SAE
│   ├── baselines.py        # modality-specific baselines (log1p / kmer / aa)
│   ├── ablation.py         # the SAE − PCA ablation gap
│   └── recipe.py           # AuditRecipe = (model, dataset) tuple
├── adapters/
│   ├── scgpt.py            # scRNA
│   ├── geneformer.py       # scRNA
│   ├── hyenadna.py         # DNA
│   ├── esm2.py             # protein
│   ├── scfoundation.py     # stub
│   ├── scbert.py           # stub
│   ├── uce.py              # stub
│   ├── scmamba.py          # placeholder
│   ├── scarf.py            # placeholder (no public checkpoint)
│   └── _template.py        # copy to add a new model
├── datasets/
│   ├── pbmc3k.py           # scRNA — legacy benchmark
│   ├── immune_human.py     # scRNA — cellxgene PBMC
│   ├── genomic_benchmarks.py  # DNA — human_nontata_promoters
│   ├── deeploc.py          # protein — subcellular localization
│   └── _template.py        # copy to add a new dataset
├── run_recipe.py           # **new headline entry**: one (model, dataset) -> AUDIT.md
├── run_audit.py            # legacy entry (scRNA + AnnData only)
└── cross_recipe_summary.py # several recipes -> master table + gap-curve plot
```

## Quickstart: run a recipe

```bash
# scRNA — scGPT on pbmc3k
python -m bio_fm_probe.run_recipe \
    --model scgpt --model_dir checkpoints/scGPT_human \
    --dataset pbmc3k

# scRNA — Geneformer on immune_human (after running both download scripts)
python src/download_geneformer.py
python src/download_immune_human.py
python -m bio_fm_probe.run_recipe \
    --model geneformer --model_dir checkpoints/geneformer \
    --dataset immune_human

# DNA — HyenaDNA on human_nontata_promoters
python src/download_hyenadna.py
python src/download_genomic_benchmarks.py
python -m bio_fm_probe.run_recipe \
    --model hyenadna --model_dir checkpoints/hyenadna-small-32k-seqlen-hf \
    --dataset genomic_benchmarks

# Protein — ESM-2 on DeepLoc
python src/download_esm2.py
python src/download_deeploc.py
python -m bio_fm_probe.run_recipe \
    --model esm2 --model_dir checkpoints/esm2_t12_35M_UR50D \
    --dataset deeploc

# Cross-recipe summary
python -m bio_fm_probe.cross_recipe_summary \
    scgpt__pbmc3k geneformer__immune_human \
    hyenadna__genomic_benchmarks esm2__deeploc
```

Output:

```
results/scgpt/audit/
├── AUDIT.md                  # readable digest
├── baselines.json
├── cls_mean_per_layer.npz    # gitignored
├── layer0_sanity.json
├── per_layer_probe.json
├── svd_diag.json
└── sae/
    ├── layer_00_input/
    │   ├── sae.pt
    │   ├── training_log.json
    │   ├── token_activations.npz   # gitignored, ~3 GB
    │   ├── per_cell_features.npz
    │   └── cell_type_probe.json
    ├── layer_06/...
    └── layer_12/...
```

To skip SAE (much faster — useful for first contact with a new model):
```bash
python -m bio_fm_probe.run_audit --skip_sae --adapter scgpt ...
```

## SAE − PCA ablation gap (the headline metric)

For each layer where we train an SAE, we also compute the **ablation gap**:

1. Build two sample-level feature matrices on the same layer's token-level activations:
   - **PCA features** — PCA of layer activations (default 512 dim), mean-pooled per sample
   - **SAE features** — TopK SAE codes (default 2048 dim, k=32), mean-pooled per sample
2. For each K ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256}:
   - Train LR on the full feature space
   - Rank features by ‖probe coefficient‖₂ across classes
   - Zero out top-K features (matched K in both spaces)
   - Refit LR, measure accuracy drop
3. **gap@K = drop_PCA − drop_SAE**

Interpretation:
- **gap > 0**: Knowledge is more distributed than PCA captures. The SAE recovers a hidden sparse dictionary PCA misses. Many small features each carrying a piece of the signal.
- **gap ≈ 0**: Knowledge is concentrated in PCA-aligned directions. SAE adds nothing extra over PCA. A few large features carry the signal.

A random-selection null is also computed; the real gap should significantly exceed the random-null gap.

This is what we use to compare models cross-modality. The gap **curve over K** (not a single number) is the cross-FM benchmark axis the toolkit is built around.

## Available adapters

| Adapter | Modality | Status | Architecture | Pretrained | Checkpoint source |
|---|---|---|---|---|---|
| `scgpt`        | scrna   | ✅ working | Transformer (12L / 512) | ~30M cells, whole-human | Google Drive (via `src/download_checkpoint.py`) |
| `geneformer`   | scrna   | ✅ working | BERT (V1: 6L / 256, V2: 12-20L / 512-768) | 30M / 95M cells | HF `ctheodoris/Geneformer` (via `src/download_geneformer.py`) |
| `hyenadna`     | dna     | ✅ working | Hyena state-space blocks | human reference genome | HF `LongSafari/hyenadna-small-32k-seqlen-hf` (via `src/download_hyenadna.py`) |
| `esm2`         | protein | ✅ working | Transformer (12L / 480 for 35M variant) | UniRef50 | HF `facebook/esm2_t12_35M_UR50D` (via `src/download_esm2.py`) |
| `scfoundation` | scrna   | 🟡 stub | xTrimoGene + Performer (100M params)       | ~50M cells              | biomap-research/scFoundation (custom format) |
| `scbert`       | scrna   | 🟡 stub | Performer (~5M params)                     | PanglaoDB               | github.com/TencentAILabHealthcare/scBERT |
| `uce`          | scrna   | 🟡 stub | 33L transformer + ESM gene-emb             | 36M cells (multi-species) | github.com/snap-stanford/UCE |
| `scmamba`      | scrna   | ⚠️ placeholder | Mamba state-space blocks               | unknown | checkpoint availability **unverified** |
| `scarf`        | scrna   | ⚠️ placeholder | Mamba × CLIP RNA+ATAC                  | 270M cells | not publicly released — contact authors |

## Available datasets

| Dataset | Modality | Baseline | Task | Source |
|---|---|---|---|---|
| `pbmc3k`              | scrna   | log1p_pca   | cell-type 8-class | bundled with scanpy |
| `immune_human`        | scrna   | log1p_pca   | cell-type (cellxgene labels) | cellxgene census (via `src/download_immune_human.py`) |
| `genomic_benchmarks`  | dna     | kmer_pca    | promoter classification (binary) | Grešová et al. 2023 (via `src/download_genomic_benchmarks.py`) |
| `deeploc`             | protein | onehot_aa_pca | subcellular localization (10-class) | DeepLoc 2.0 (via `src/download_deeploc.py`) |

**Stub** = file exists with the right class skeleton + TODO blocks explaining what's needed. Look at the stub's docstring to know what to fill in for each model.

**Placeholder** = no checkpoint known to be public; the file documents what action is needed to unblock the adapter.

To activate a stub once the model loads cleanly: un-comment its line in `ADAPTER_REGISTRY` inside `run_audit.py`.

## Adding a new model

1. Copy `adapters/_template.py` to `adapters/<your_model>.py`.
2. Implement three methods (~50-80 lines if the model is well-behaved):

   - `load(model_dir, device)` — build model, set `self.n_layers`, `self.d_model`
   - `preprocess(adata)` — return adata with model-specific input layers/var
   - `iter_layer_activations(adata, batch_size, device)` — generator that yields,
     per batch, `(captured: {layer_name: (B, seq, d)}, valid_mask: (B, seq) bool)`

3. Register your class in `run_audit.ADAPTER_REGISTRY`:

   ```python
   ADAPTER_REGISTRY["your_model"] = "bio_fm_probe.adapters.your_model:YourModelAdapter"
   ```

4. Run:

   ```bash
   python -m bio_fm_probe.run_audit --adapter your_model \
       --model_dir checkpoints/your_model --data data/pbmc3k.h5ad
   ```

If your model has no CLS token, set `cls_position = None` in the adapter; CLS
columns will be omitted automatically.

If the model has a fast-path (NestedTensor, FlashAttention fused kernels) that
breaks `register_forward_hook`, disable it in `load()` — see `adapters/scgpt.py`
for the pattern with `enable_nested_tensor`.

## Comparing models

After running the audit on several models:

```bash
python -m bio_fm_probe.compare_models scgpt scmamba geneformer
```

Writes `results/_cross_model_summary.md` with one row per metric:

```
| metric                          | scgpt    | scmamba  | geneformer |
|---|---|---|---|
| raw log1p LR acc                | 0.9428   | ...      | ...        |
| PCA-50 LR acc                   | 0.9398   | ...      | ...        |
| best model acc                  | 0.9322   | ...      | ...        |
| Δ best vs PCA-50                | -0.0076  | ...      | ...        |
| layer 0 std ratio (H1)          | 2.31e-04 | ...      | ...        |
| PR (input)                      | 44.2     | ...      | ...        |
| PR (final)                      |  8.7     | ...      | ...        |
| k95 (final)                     | 64       | ...      | ...        |
```

That table is what tells you whether the trend you saw in scGPT (transformer
blocks compress to low-rank manifold, fail to beat PCA on cell type) is
architecture-specific or universal.

## Notes / conventions

- Layer naming is fixed: `layer_00_input` (pre-encoder, post embeddings),
  `layer_01` through `layer_NN` for transformer/Mamba/etc blocks.
- All probes use the same 5 stratified splits per seed list (default
  `seeds=[0,1,2,3,4]`) so deltas across pooling / layers / models are
  directly comparable.
- Baselines (raw log1p, PCA-50, PCA-512) use the same splits as the model
  probes, so Δ vs PCA-50 is a clean number.
- The `_template.py` adapter raises `NotImplementedError` until you fill it
  in — copy it, don't import it.
