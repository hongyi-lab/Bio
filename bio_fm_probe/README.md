# bio_fm_probe — auditing toolkit for biological foundation models

A model-agnostic version of the scGPT phase 1-3 probes. Drop a new adapter in,
run one command, get a standardized audit (layer probe + baselines + SVD +
SAE) with an `AUDIT.md` digest. Then run `compare_models.py` to put several
models side by side.

What every audit answers, on the same data, with the same seeds:

1. **Does the model's best representation beat raw PCA-50?** (`baselines.json`)
2. **Is the layer-0 CLS slot constant before attention?** (`layer0_sanity.json`)
3. **Where is the model's best layer for cell-type, and how much does pooling matter?** (`per_layer_probe.json`)
4. **Does the representation collapse to low rank with depth?** (`svd_diag.json`)
5. **Can a TopK SAE on selected layers reconstruct the activations, and do its sparse features recover cell type?** (`sae/<layer>/...`)

These are the questions phases 1-3 answered for scGPT; the toolkit just makes
them mechanical to ask of any model with the same interface.

## Layout

```
bio_fm_probe/
├── core/
│   ├── adapter.py          # BioFMAdapter ABC
│   ├── extract.py          # generic per-layer / per-token extraction
│   └── probes.py           # LR probe, PCA baselines, SVD, TopK SAE
├── adapters/
│   ├── scgpt.py            # scGPT whole-human adapter
│   └── _template.py        # copy this to add a new model
├── run_audit.py            # entry point: one model -> AUDIT.md
└── compare_models.py       # several models -> cross-model table
```

## Quickstart: run on scGPT

```bash
python -m bio_fm_probe.run_audit \
    --adapter scgpt \
    --model_dir checkpoints/scGPT_human \
    --data data/pbmc3k.h5ad \
    --label_col louvain
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

## Available adapters

| Adapter | Status | Architecture | Pretrained | Checkpoint source |
|---|---|---|---|---|
| `scgpt`        | ✅ working | Transformer (12L / 512) | ~30M cells, whole-human | Google Drive (via `src/download_checkpoint.py`) |
| `geneformer`   | ✅ working | BERT (V1: 6L / 256, V2: 12-20L / 512-768) | 30M / 95M cells | HF `ctheodoris/Geneformer` (via `src/download_geneformer.py`) |
| `scfoundation` | 🟡 stub    | xTrimoGene + Performer (100M params)       | ~50M cells              | biomap-research/scFoundation (custom format) |
| `scbert`       | 🟡 stub    | Performer (~5M params)                     | PanglaoDB               | github.com/TencentAILabHealthcare/scBERT |
| `uce`          | 🟡 stub    | 33L transformer + ESM gene-emb             | 36M cells (multi-species) | github.com/snap-stanford/UCE |
| `scmamba`      | ⚠️ placeholder | Mamba state-space blocks               | unknown (paper claims 270M paired) | checkpoint availability **unverified** |
| `scarf`        | ⚠️ placeholder | Mamba × CLIP RNA+ATAC                  | 270M cells              | not publicly released — contact authors |

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
