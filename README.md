# bio FM probing — phase 7 (LLM-scale)

Active focus: **LLM-scale biological foundation models** (Evo-1 7B, ESM-2 15B).
Earlier small-scale phases (scGPT 512×12, HyenaDNA 256×4) are archived in `legacy/`
and are not the comparison target.

> The conda env is still `scgpt_env/` — historical artifact from phase 1.
> The env holds the carefully pinned dependency set for every model used here
> (torch 2.1.2+cu118, transformers 4.36.2, flash_attn 2.5.8, scgpt 0.2.4,
> ...). Don't rename it; nothing would benefit and many things would break.
> The project root was renamed from `scGPT_probing/` → `bio_fm_probing/`
> because we no longer run scGPT — Evo-1 / ESM-2 15B are the active models.

## Layout

```
bio_fm_probing/
├── src/phase7/                    ← all active code
│   ├── evo_probe.py               Evo-1 end-to-end probe (full forward + SAE + ablation)
│   ├── evo_resume.py              Same as evo_probe but resumes from cached intermediates
│   ├── esm2_15b_probe.py          ESM-2 15B end-to-end probe (DeepLoc; protein arm)
│   ├── common_sae.py              shared helpers: TopK SAE, PCA, SVD, ablation gap, plots
│   ├── plot_phase7_summary.py     REPORT.md + summary plot helper
│   ├── download_evo.py            HF download for togethercomputer/evo-1-8k-base
│   ├── download_evo2.py           HF download for arcinstitute/evo2_7b (raw, no HF loader)
│   ├── download_esm2_15b.py       HF download for facebook/esm2_t48_15B_UR50D
│   ├── evo2_probe.py              early Evo-2 attempt — kept until vortex/evo2 is installable
│   └── BUGS_AND_FIXES.md          full catalogue of LLM-scale engineering pitfalls
│
├── checkpoints/                   model weights (gitignored)
│   ├── evo-1-8k-base/             13 GB, safetensors (Evo-1 7B; bf16 model)
│   ├── evo2_7b/                   13 GB, raw .pt (Evo-2 7B — needs Arc Institute's vortex)
│   ├── esm2_t48_15B_UR50D/        57 GB, fp32 shards (ESM-2 15B; load with torch_dtype=fp16)
│   └── …phase 1-5 leftovers also live here; ignore them
│
├── data/                          datasets (gitignored)
│   ├── genomic_benchmarks/human_nontata_promoters/   DNA, used by Evo-1 probe
│   └── deeploc/                                       protein — MISSING (DTU URL 404)
│
├── results/                       active phase 7 outputs only
│   └── evo_1_8k__genomic_benchmarks/     per-layer JSON + REPORT.md when run completes
│
├── logs/                          stdout/stderr of background runs
└── legacy/                        phase 1-5 detritus (do not touch)
    ├── bio_fm_probe/              old model-agnostic toolkit (small-scale)
    ├── src/                       19 stale scripts (load_scgpt, phase 2/3 probes, etc.)
    ├── results/                   old phase 1-5 results
    ├── gene_sets/                 phase 3 gene-set files
    └── notebooks/                 empty
```

## Environment

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /srv/hgao0864/conda_envs/scgpt_env
```

Pinned deps that matter at LLM scale (see BUGS_AND_FIXES.md for why):

| package | version | why pinned |
|---|---|---|
| `torch` | `2.1.2+cu118` | scGPT compat (small-scale leftover); blocks newer transformers |
| `transformers` | `4.36.2` | 5.x needs torch ≥2.4; 4.36 uses `_register_pytree_node` |
| `flash_attn` | `2.5.8` | Dao-AILab prebuilt wheel `cu118torch2.1cxx11abiFALSE`; source build fails |
| `numpy` | `<2` | torch 2.1 ABI |
| `einops` | latest | Evo-1 modeling code |
| `huggingface_hub` | latest | HF downloads |

## Active runs

### Evo-1 7B on GenomicBenchmarks promoters (DNA)

```bash
cd /srv/hgao0864/Bio/bio_fm_probing
nohup python src/phase7/evo_resume.py --sae_layers layer_16 layer_32 \
    > logs/evo_resume_bf16.log 2>&1 &
echo "PID: $!"
```

`--sae_layers layer_16 layer_32` is required — it explicitly skips
`layer_00_input` because (1) all its artifacts are already cached from a
previous good run and (2) re-processing it triggers a CPU-bound hang on the
SVD step that I haven't root-caused.

ETA ~5-7 h. Output lands in `results/evo_1_8k__genomic_benchmarks/`:
- `REPORT.md` (the readable digest)
- `phase7_summary.json`, `phase7_summary.png`
- `layer_16/`, `layer_32/` subdirs with per-layer JSONs

Tail the log with `tail -f logs/evo_resume_bf16.log`.

### ESM-2 15B on DeepLoc (protein) — BLOCKED

Weights are downloaded (`checkpoints/esm2_t48_15B_UR50D/`, 57 GB), code is
patched, but DeepLoc data is missing (DTU URL returns 404). When you have a
working FASTA at `data/deeploc/deeploc_data.fasta`:

```bash
python src/phase7/esm2_15b_probe.py
```

## Defaults baked into the phase 7 scripts

These came from many painful evenings, see BUGS_AND_FIXES.md:

- **bf16 weights** + dtype-aware fp32 disk cache (fp16 NaN's at deep StripedHyena layers)
- `--max_len 512` (StripedHyena FFT OOMs at 8192)
- `--extract_batch_size 2` (bs=4 OOMs on a 48 GB A6000)
- `mmap_mode='r'` on all 39+ GB cache loads (cold-disk read used to block for 5+ min)
- Chunked PCA fit + projection (never materialize the (n_tokens, n_components) fp64 matrix)
- Chunked SAE encode + per-cell aggregate (never materialize the (n_tokens, n_features) sparse code)
- Chunked NaN/Inf filter with memmapped output
- All paths anchored to project root via `_resolve()` (scripts work from any cwd)
- `tqdm` on every multi-step loop (SAE epochs/batches, PCA chunks, ablation seeds, NaN filter)

## If something goes wrong

`src/phase7/BUGS_AND_FIXES.md` has the ~17 engineering pitfalls catalogued
with symptoms + fixes. Quick triage:

| symptom | likely cause | fix |
|---|---|---|
| `Unrecognized model in checkpoints/evo2_7b` | Evo-2 not HF-native | use Evo-1 instead |
| `Mean of empty slice` / `SVD did not converge` | activations are all NaN/Inf | check that model loaded with `dtype=bf16`, not fp16 |
| `OOM` in PCA / encode_batched | naive sklearn / numpy allocation | confirm script imports `fit_pca_and_aggregate_per_cell` and `encode_and_aggregate_per_cell` from common_sae |
| Stuck for hours on layer_00_input | unknown CPU-bound hang in SVD on fp16 mmap | pass `--sae_layers layer_16 layer_32` to skip it |
| GPU idle while CPU at 99% | model never loaded; stuck in Python code | check log for the last printed step; usually a memory or sklearn issue |
