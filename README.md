# scGPT_probing

Probing experiments on top of the [scGPT](https://github.com/bowang-lab/scGPT) whole-human foundation model. Goal: extract cell- and gene-level embeddings from pretrained scGPT and use them as features for downstream probing.

```
scGPT_probing/
├── checkpoints/scGPT_human/   # args.json, vocab.json, best_model.pt
├── data/                       # pbmc3k.h5ad (and optional multiome .h5)
├── notebooks/                  # local-only; not stored on the server
├── src/
│   ├── download_checkpoint.py  # gdown the scGPT whole-human folder
│   ├── download_data.py        # fetch pbmc3k (and optional 10X multiome)
│   ├── load_scgpt.py           # load checkpoint, extract embeddings
│   └── sanity_check.py         # 50-cell pipeline → UMAP png
├── results/                    # embeddings.npz, UMAP png, summary.json
└── README.md
```

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

## Troubleshooting

- **`flash_attn` import errors inside scgpt**: ignore — scgpt's transformer falls back to the stock PyTorch implementation when `use_fast_transformer=False`, which is what `load_scgpt.py` sets.
- **`RuntimeError: size mismatch` on `load_state_dict`**: `load_scgpt_model` strips mismatched keys (`strict=False`) and prints how many were kept. As long as encoder/embedding weights load, embeddings will work.
- **`numpy ABI` errors from torch**: confirm `pip show numpy` reports a `1.x` version. Reinstall with `pip install 'numpy<2'` if drifted.
- **gdown quota / "Cannot retrieve folder"**: open the folder URL in a browser once to clear the quota, then re-run the script.
