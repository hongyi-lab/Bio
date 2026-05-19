"""Load the scGPT whole-human checkpoint and extract embeddings.

Public functions:
    load_scgpt_model(model_dir, device) -> (model, vocab, model_args)
    preprocess_adata_for_scgpt(adata, vocab, gene_col) -> adata (with X_binned layer)
    embed_cells(adata, model, vocab, ...) -> dict with cell_embeddings + per_gene info

Resumable inference: pass save_path=... and the per-batch cell embeddings are flushed
to disk after every batch. Re-running picks up from the last saved batch.

Run directly:
    python src/load_scgpt.py --n_cells 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import scanpy as sc
import torch
from tqdm import tqdm

# scGPT imports — kept inside functions so the file at least imports without scgpt
PAD_TOKEN = "<pad>"
SPECIAL_TOKENS = [PAD_TOKEN, "<cls>", "<eoc>"]
PAD_VALUE = -2
N_BINS = 51
MAX_LEN = 1200


def load_scgpt_model(model_dir, device: str = "cuda"):
    """Build TransformerModel and load the pretrained weights.

    Returns (model, vocab, args_dict).
    """
    from scgpt.model import TransformerModel
    from scgpt.tokenizer.gene_tokenizer import GeneVocab

    model_dir = Path(model_dir)
    vocab_file = model_dir / "vocab.json"
    args_file = model_dir / "args.json"
    weights_file = model_dir / "best_model.pt"

    for f in [vocab_file, args_file, weights_file]:
        if not f.exists():
            raise FileNotFoundError(f"missing checkpoint file: {f}")

    vocab = GeneVocab.from_file(vocab_file)
    for tok in SPECIAL_TOKENS:
        if tok not in vocab:
            vocab.append_token(tok)
    vocab.set_default_index(vocab[PAD_TOKEN])

    with open(args_file) as f:
        model_args = json.load(f)

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=model_args["embsize"],
        nhead=model_args["nheads"],
        d_hid=model_args["d_hid"],
        nlayers=model_args["nlayers"],
        nlayers_cls=model_args.get("n_layers_cls", 3),
        n_cls=1,
        vocab=vocab,
        dropout=model_args.get("dropout", 0.0),
        pad_token=PAD_TOKEN,
        pad_value=PAD_VALUE,
        do_mvc=False,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        input_emb_style="continuous",
        n_input_bins=N_BINS,
        pre_norm=False,
        use_fast_transformer=False,
    )

    state = torch.load(weights_file, map_location=device)
    try:
        model.load_state_dict(state)
    except RuntimeError:
        own = model.state_dict()
        kept = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        print(f"[load_scgpt] loaded {len(kept)}/{len(state)} matched keys (strict=False)")
        model.load_state_dict(kept, strict=False)

    model.to(device)
    model.eval()
    return model, vocab, model_args


def _looks_like_raw_counts(X) -> bool:
    """Heuristic: raw counts are non-negative integers; processed data is float/has negatives."""
    sample = X[:10].toarray() if hasattr(X, "toarray") else np.asarray(X[:10])
    if sample.size == 0:
        return False
    if (sample < 0).any():
        return False
    return float(np.abs(sample - np.round(sample)).max()) < 1e-6


def preprocess_adata_for_scgpt(adata, vocab, gene_col: str = "gene_name"):
    """Filter to vocab genes, normalize, log1p, and bin expression to N_BINS bins.

    Adds adata.layers['X_binned'] which is what the model consumes. If `adata.X` is
    already log-normalized, normalize_total + log1p are skipped (scgpt's Preprocessor
    would otherwise produce NaNs on negative values).
    """
    from scgpt.preprocess import Preprocessor

    if gene_col not in adata.var.columns:
        adata.var["gene_name"] = adata.var.index.astype(str)
        gene_col = "gene_name"

    keep = adata.var[gene_col].apply(lambda g: g in vocab).values
    n_keep = int(keep.sum())
    print(f"[preprocess] genes in scGPT vocab: {n_keep}/{len(keep)}")
    if n_keep == 0:
        raise ValueError("no overlap between adata.var and scGPT vocab — check gene_col / symbols")
    adata = adata[:, keep].copy()

    is_raw = _looks_like_raw_counts(adata.X)
    print(f"[preprocess] detected adata.X as {'raw counts' if is_raw else 'already log-normalized'}")

    pre = Preprocessor(
        use_key="X",
        filter_gene_by_counts=False,
        filter_cell_by_counts=False,
        normalize_total=1e4 if is_raw else False,
        result_normed_key="X_normed",
        log1p=is_raw,
        result_log1p_key="X_log1p",
        subset_hvg=False,
        binning=N_BINS,
        result_binned_key="X_binned",
    )
    pre(adata, batch_key=None)
    return adata


def _save_checkpoint(path: Path, cell_embs: list, next_idx: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # np.savez auto-appends .npz if missing; pass an absolute base + suffix to make rename predictable
    tmp_base = path.with_suffix("")
    tmp_npz = Path(str(tmp_base) + ".tmp.npz")
    np.savez(
        str(tmp_base) + ".tmp",
        cell_embs=np.concatenate(cell_embs, axis=0) if cell_embs else np.zeros((0, 0)),
        next_idx=np.int64(next_idx),
    )
    tmp_npz.replace(path)


def _load_checkpoint(path: Path):
    if not path.exists():
        return [], 0
    data = np.load(path, allow_pickle=False)
    next_idx = int(data["next_idx"])
    arr = data["cell_embs"]
    return ([arr] if arr.size else []), next_idx


def embed_cells(
    adata,
    model,
    vocab,
    device: str = "cuda",
    batch_size: int = 32,
    save_path: Optional[Path] = None,
    gene_col: str = "gene_name",
    return_last_per_gene: bool = True,
):
    """Run model over all cells and return cell embeddings (+ optional per-gene embeddings).

    The CLS-token output is the cell embedding. The remaining sequence positions are
    per-gene embeddings for the genes that were tokenized for that cell.

    If save_path is given, intermediate batches are checkpointed to disk so an
    interrupted run can resume.
    """
    from scgpt.tokenizer import tokenize_and_pad_batch

    if gene_col not in adata.var.columns:
        adata.var["gene_name"] = adata.var.index.astype(str)
        gene_col = "gene_name"

    genes = adata.var[gene_col].tolist()
    gene_ids_all = np.array(vocab(genes), dtype=int)

    binned = adata.layers["X_binned"]
    if hasattr(binned, "toarray"):
        binned = binned.toarray()
    binned = np.asarray(binned)

    if save_path is not None:
        save_path = Path(save_path)
        cell_embs, start_idx = _load_checkpoint(save_path)
        if start_idx:
            print(f"[embed_cells] resuming from cell {start_idx}/{adata.n_obs}")
    else:
        cell_embs, start_idx = [], 0

    last_per_gene = None
    last_gene_ids = None
    n_cells = adata.n_obs
    pad_idx = vocab[PAD_TOKEN]

    pbar = tqdm(range(start_idx, n_cells, batch_size), desc="embed", unit="batch")
    for start in pbar:
        end = min(start + batch_size, n_cells)
        batch_X = binned[start:end].astype(np.int64)

        tokenized = tokenize_and_pad_batch(
            batch_X,
            gene_ids_all,
            max_len=MAX_LEN,
            vocab=vocab,
            pad_token=PAD_TOKEN,
            pad_value=PAD_VALUE,
            append_cls=True,
            include_zero_gene=False,
        )
        input_gene_ids = tokenized["genes"].to(device)
        input_values = tokenized["values"].to(device).float()
        pad_mask = input_gene_ids.eq(pad_idx)

        with torch.no_grad():
            enc = model._encode(input_gene_ids, input_values, src_key_padding_mask=pad_mask)

        cell_embs.append(enc[:, 0, :].cpu().numpy())
        if return_last_per_gene:
            last_per_gene = enc.cpu().numpy()
            last_gene_ids = input_gene_ids.cpu().numpy()

        if save_path is not None:
            _save_checkpoint(save_path, cell_embs, end)

    cell_emb_arr = np.concatenate(cell_embs, axis=0) if cell_embs else np.zeros((0,))
    return {
        "cell_embeddings": cell_emb_arr,
        "per_gene_embeddings_last_batch": last_per_gene,
        "gene_ids_last_batch": last_gene_ids,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="checkpoints/scGPT_human")
    p.add_argument("--data", default="data/pbmc3k.h5ad")
    p.add_argument("--n_cells", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--save", default="results/embeddings.npz")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--gene_col", default="gene_name")
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    model_dir = root / args.model_dir if not Path(args.model_dir).is_absolute() else Path(args.model_dir)
    data_path = root / args.data if not Path(args.data).is_absolute() else Path(args.data)
    save_path = root / args.save if not Path(args.save).is_absolute() else Path(args.save)

    print(f"[load_scgpt] device={args.device}")
    adata = sc.read_h5ad(data_path)
    if args.n_cells > 0 and args.n_cells < adata.n_obs:
        adata = adata[: args.n_cells].copy()
    print(f"[load_scgpt] adata: {adata.shape}")

    model, vocab, _ = load_scgpt_model(model_dir, device=args.device)
    adata = preprocess_adata_for_scgpt(adata, vocab, gene_col=args.gene_col)

    out = embed_cells(
        adata, model, vocab,
        device=args.device, batch_size=args.batch_size,
        save_path=save_path, gene_col=args.gene_col,
    )

    print(f"[load_scgpt] cell_embeddings shape:           {out['cell_embeddings'].shape}")
    if out["per_gene_embeddings_last_batch"] is not None:
        print(f"[load_scgpt] per-gene embeddings (last batch): {out['per_gene_embeddings_last_batch'].shape}")
    print(f"[load_scgpt] saved to {save_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
