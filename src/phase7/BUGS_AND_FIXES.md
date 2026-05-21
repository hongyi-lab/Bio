# Phase 7 — bugs encountered and fixes

Catalogue of every issue hit while scaling the probing pipeline from
phase-5 small models (scGPT 512×12, HyenaDNA 256×4) up to LLM-scale
foundation models (Evo-1 7B, ESM-2 15B).

The recurring root cause: code written for the small-scale memory profile
fails when activation matrices grow 8-30×. Almost every bug here is "fine at
d=512, breaks at d=4096" or "fine in fp32, NaNs in fp16."

For future LLM-scale runs: assume streaming everywhere from the start,
default to fp16 throughout, do not reuse the small-scale `bio_fm_probe/`
toolkit without re-auditing it for these failure modes.

---

## 1. Model loading & environment

### 1.1 Evo-2 is not HF-native
- **Symptom:** `AutoConfig.from_pretrained("checkpoints/evo2_7b")` →
  `ValueError: Unrecognized model in ... Should have a 'model_type' key`.
- **Cause:** `arcinstitute/evo2_7b` ships the raw Arc Institute checkpoint
  (`evo2_7b.pt`, one config field `architecture: StripedHyena2`). It needs Arc's
  own `vortex`/`evo2` package, which requires torch ≥ 2.5 and custom CUDA kernels.
- **Fix:** Pivot to **Evo-1** (`togethercomputer/evo-1-8k-base`). Same 7B scale,
  HuggingFace-native via `auto_map` + `trust_remote_code=True`. Created
  `src/phase7/evo_probe.py` (forked from the broken `evo2_probe.py`).

### 1.2 `transformers 5.x` incompatible with torch 2.1.2
- **Symptom:** `transformers.AutoModel` raises
  `ImportError: AutoModel requires the PyTorch library` even though torch is installed.
  Followed by `AttributeError: module 'torch.utils._pytree' has no attribute
  'register_pytree_node'` in `transformers.utils.generic`.
- **Cause:** `transformers 5.8.1` declares torch ≥ 2.4; it also uses
  `torch.utils._pytree.register_pytree_node` which only became public in torch 2.2.
  Our torch is pinned 2.1.2+cu118 (because scGPT needs it).
- **Fix:** `pip install transformers==4.36.2` — last 4.x line that still uses
  the private `_register_pytree_node`. Pin transformers explicitly when
  recreating the env.

### 1.3 Evo-1 modeling code requires `flash_attn` (no metadata-build wheel)
- **Symptom:** `pip install flash_attn` fails at metadata-generation step;
  modeling file refuses to import without it.
- **Cause:** flash-attn's source build needs torch importable inside a clean
  build env + specific CUDA assumptions our 11.8 env doesn't satisfy.
- **Fix:** Install a Dao-AILab pre-built wheel that matches the env exactly:
  ```
  pip install \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
  ```
  Wheel selection: `cu118` (our CUDA) + `torch2.1` + `cxx11abiFALSE` (matches
  `torch._C._GLIBCXX_USE_CXX11_ABI`).

### 1.4 `AutoModel` doesn't load Evo-1
- **Symptom:** `AutoModel.from_pretrained(..., trust_remote_code=True)` returns
  None or raises — Evo-1's `auto_map` registers only `AutoModelForCausalLM`.
- **Fix:** Use `AutoModelForCausalLM.from_pretrained(...)`. LM head is unused;
  only backbone activations are extracted via forward hooks.

### 1.5 ByteTokenizer has no `pad_token`
- **Symptom:** `ValueError: Asking to pad but the tokenizer does not have a
  padding token` when calling `tok(seqs, padding="max_length", ...)`.
- **Cause:** Evo-1's `ByteTokenizer` is a literal byte-level tokenizer (vocab=512,
  no special tokens defined). A=65, C=67, G=71, T=84 — all the special-token
  slots are None.
- **Fix:** After loading the tokenizer, set `tok.pad_token = chr(0)` (byte 0
  never appears in DNA strings; attention_mask handles position-wise masking).

---

## 2. Memory / OOM at LLM scale

### 2.1 CUDA OOM at default `max_len`
- **Symptom:** `OutOfMemoryError: Tried to allocate 16.00 GiB` inside
  `engine.prefill_via_modal_fft` on the first forward batch.
- **Cause:** StripedHyena's parallel Hyena filter (`parallel_iir`) computes a
  long convolution via FFT. At seq length 8192, the FFT intermediate is ~16 GB
  for one batch.
- **Fix:** Lower `MAX_LEN_DEFAULT` from 8192 → 512. GenomicBenchmarks promoter
  sequences are ~250 nt so 512 covers them with margin.

### 2.2 CUDA OOM at default `batch_size`
- **Symptom:** Same FFT path, still OOMs at max_len=512 with
  `--extract_batch_size 4` (process holds 45+ GB).
- **Fix:** Lower default `--extract_batch_size` 4 → 2. Peak GPU memory drops
  ~45 → ~32 GB. Throughput drops only ~30% (forward is FFT-bound, not bs-bound).

### 2.3 PCA fit_transform OOM (the big one)
- **Symptom:** Long swap thrashing then `OutOfMemoryError` inside
  `sklearn.PCA.fit_transform` on 5M × 4096 fp16 activations.
- **Cause:** sklearn's PCA internally casts input to fp64 and makes a centered
  copy → ~328 GB working memory for a (5M, 4096) input. Machine has 62 GB RAM
  + 127 GB swap, so it swaps until either OOM or the heat death of the universe.
- **Fix:** Added `fit_pca_and_aggregate_per_cell()` to `common_sae.py`:
  1. Sample 200k tokens, fit PCA on those (~6 GB peak)
  2. Project the full token set in 100k-row chunks (~1.6 GB per chunk)
  3. Aggregate per-cell on the fly (never materialize the (n_tokens, n_comp)
     fp64 matrix)

  Peak RAM: ~10 GB (vs 328 GB).

### 2.4 SAE `encode_batched` OOM (the *second* big one)
- **Symptom:** `np.zeros((5020000, 16384), dtype=np.float32)` →
  `numpy.core._exceptions._ArrayMemoryError: Unable to allocate 306. GiB`
  after SAE training already completed (2 hours of work nearly lost).
- **Cause:** `encode_batched` pre-allocated the full sparse-code matrix at the
  start, then filled it batch by batch. The pre-allocation is 306 GB.
- **Fix:** Added `encode_and_aggregate_per_cell()` to `common_sae.py` — streams
  encode + per-cell mean aggregation in one fused pass. Only a per-chunk
  `(batch, n_features)` array lives in memory. Peak: ~3 GB.

### 2.5 fp32 token-activation cache wastes disk
- **Symptom:** Per-layer cache file = 80 GB (× 3 layers = 240 GB).
- **Cause:** `extract_tokens_from_iter` cast hidden states `.float()` (→ fp32)
  before writing to `.npz`. Activations came out of an fp16 model so this
  cast doubles bytes with zero precision gain.
- **Fix:** `common_sae.py` line 464 — cast to `np.float16` before `np.savez`.
  Halves disk + halves I/O time. Downstream consumers promote to fp32/fp64 on
  load via `torch.from_numpy().float()` / `.astype(np.float64)` so this is
  transparent to them.

### 2.6 Cold-disk cache load blocks for minutes
- **Symptom:** After a swap-thrashing event evicted the page cache, the next
  run's `np.load(token_activations.npz)` hung at "loading cached activations"
  for 3+ minutes (vs 5-20 s when the file was hot in cache).
- **Cause:** Reading 39 GB from `/srv` HDD at ~150 MB/s is ~4 min just for raw
  I/O. The script's `np.load` materializes the whole array eagerly.
- **Fix:** `np.load(acts_cache, allow_pickle=False, mmap_mode='r')` —
  memory-maps the file instantly. Each downstream operation reads only the
  bytes it actually touches. All downstream consumers (SVD subsamples 200k
  rows, chunked PCA, chunked SAE-encode) work fine on mmap'd input.

---

## 3. Numerical instability

### 3.1 NaN/Inf in deep-layer activations (fp16 StripedHyena)
- **Symptom:** After 99 minutes of layer_16 forward extraction, `SVD did not
  converge` and `RuntimeWarning: invalid value encountered in subtract`.
- **Cause:** StripedHyena's Hyena filter does long convolutions via FFT. At
  deep layers in fp16, intermediate magnitudes can saturate to ±Inf or produce
  NaN. Known fp16 instability for StripedHyena/Mamba-family architectures.
- **Fix:** Added `_nan_filter()` helper in `evo_resume.py` —
  chunked NaN/Inf scan + persistent filtered cache (`filtered_acts.npy`,
  `filtered_cell_idx.npy`). Filtered arrays are written via
  `np.lib.format.open_memmap` so the full filtered array is never in RAM.
  Drop fraction is logged so the report can flag it.
- **Note for next time:** if drop fraction > 5% the layer's results need a
  caveat in the analysis. Alternative is full-fp32 inference but that doubles
  GPU memory and runtime.

---

## 4. Workflow / ergonomics

### 4.1 Relative paths resolved against caller's cwd
- **Symptom:** `python /srv/.../src/phase7/evo_probe.py` from `~/hgao0864/Bio/`
  → `is not a local folder` because `checkpoints/evo-1-8k-base` resolved
  against the wrong directory.
- **Fix:** Added `_resolve(p)` helper to both probe scripts. Path-using sites
  (`--model_dir`, `--data_dir`, `--out`) all anchor to `ROOT` derived from
  `__file__`. Script now works from any cwd.

### 4.2 Download scripts had the same cwd problem
- **Symptom:** `python download_evo.py` (run from harness cwd) saved the
  checkpoint to `/srv/hgao0864/Bio/checkpoints/...` instead of
  `/srv/hgao0864/Bio/scGPT_probing/checkpoints/...`.
- **Fix:** Manual `mv` + future fix is to add the same `_resolve()` pattern to
  the download scripts. **Still TODO.**

### 4.3 No progress bars on long-running steps
- **Symptom:** Multi-hour SAE training, 30-minute PCA, multi-stage ablation
  loops all silent. Looked indistinguishable from a hang.
- **Fix:** Added `tqdm` to: SAE training (outer epoch loop + inner batch loop
  with `leave=False, mininterval=1.0`), ablation seed/K loops, PCA
  project+aggregate chunk loop, NaN filter scan and write phases.

### 4.4 Misleading tqdm description
- **Symptom:** `evo_probe.py` (running Evo-1) printed `evo2 forward` in the
  tqdm desc.
- **Cause:** Earlier `replace_all "[evo2]" → "[evo]"` only caught bracketed
  strings, not bare ones.
- **Fix:** Fixed string in `iter_evo2_activations`. Cosmetic only.

---

## 5. Dependency conflicts (cosmetic)

### 5.1 `fsspec` / `datasets` version conflict
- **Symptom:** `pip install cellxgene_census` upgraded `fsspec` to 2026.4.0,
  breaking `datasets 2.21.0` (requires fsspec ≤ 2024.6.1).
- **Impact:** None on phase 7 — neither probe nor any phase-7 helper imports
  HF `datasets`. The pip resolver warning is cosmetic.
- **Fix:** Not needed currently. If a future protein dataset uses HF
  `datasets`, downgrade: `pip install 'fsspec[http]==2024.6.1'`.

---

## 6. Open / deferred

### 6.1 DeepLoc dataset URL 404
- **Symptom:** `https://services.healthtech.dtu.dk/services/DeepLoc-2.0/data/Swissprot_Train_Validation_dataset.fasta`
  returns 404.
- **Status:** Blocks ESM-2-15B probe pipeline (data side missing). ESM-2 15B
  weights are downloaded and the probe script is patched. Options when
  needed:
  - Manually fetch DeepLoc 2.0 from a working mirror (HF
    `Synthyra/Deeploc2` has CSVs but needs format conversion), drop at
    `data/deeploc/deeploc_data.fasta`, pass `--data_path`.
  - Swap to a different protein-localization dataset and write a new dataset
    loader (~50 lines).

### 6.2 Some SAE training steps still swap-thrash
- **Symptom:** `train_topk_sae` does `torch.from_numpy(activations).float()` —
  for 5M × 4096 fp16 input, this is an 82 GB fp32 in RAM. We have 62 GB.
- **Impact:** Layer_00_input SAE training took ~2 h (with swap; vs ~30 min
  expected). Layer_16 expected similar.
- **Possible future fix:** Stream activations from disk per-batch (mmap +
  per-batch fp32 cast) instead of upfront materialization. Not done yet —
  current behavior is slow-but-correct.

---

## TL;DR for next LLM-scale run

1. **Pin transformers to 4.36.2** if torch is 2.1.x.
2. **Install flash_attn from a pre-built wheel** matching exact torch/CUDA/ABI.
3. **mmap everything** that lives on disk above ~1 GB.
4. **Never call `sklearn.PCA.fit_transform` on (n_tokens, d) data** at LLM
   scale — fit on a 200k subsample, project in chunks, aggregate per-cell on
   the fly. Same idea applies to any "project the full token set into a new
   basis" step.
5. **Never pre-allocate the full (n_tokens, n_features) sparse code** for an
   SAE — stream encode + aggregate.
6. **Save activation caches in fp16** (precision of the source forward pass)
   not fp32.
7. **Expect NaN/Inf at deep layers in fp16** for Hyena/Mamba models. Plan a
   filter pass.
8. **Forward-pass batch_size 2** at d_model=4096 / 32 layers on a 48 GB GPU
   is the sweet spot. bs=4 OOMs.
9. **Add tqdm to every loop that takes more than 10 seconds.** Silent loops
   are indistinguishable from hangs.
10. **`_resolve()` all relative paths** against a script-anchored `ROOT`.
