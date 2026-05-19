"""Geneformer adapter — BERT-style transformer on rank-value-encoded gene sequences.

Geneformer (Theodoris et al., Nature 2023) is a BERT-style model pretrained on
30M (V1) / 95M (V2) human single-cell transcriptomes. Cells are represented as
sequences of gene tokens (Ensembl IDs) ranked by per-cell normalized expression.

Public checkpoint: huggingface.co/ctheodoris/Geneformer
  - V1 (default): 6 layers, d_model=256, max_input=2048
  - V2 104M:      12 layers, d_model=512, max_input=4096
  - V2 313M:      20 layers, d_model=768, max_input=4096

Expected files under model_dir (download via src/download_geneformer.py):
  config.json                           BertConfig
  pytorch_model.bin (or *.safetensors)  weights
  token_dictionary.pkl                  {Ensembl_ID: token_id} mapping

If pbmc3k has gene symbols in var.index, this adapter reads Ensembl IDs from
adata.var.gene_ids (the column scanpy's pbmc3k provides) — see preprocess().
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np
import torch
from tqdm import tqdm

from ..core.adapter import BioFMAdapter


CLS_TOKEN = "<cls>"
PAD_TOKEN = "<pad>"
MASK_TOKEN = "<mask>"


class GeneformerAdapter(BioFMAdapter):
    name = "geneformer"
    cls_position = 0   # BertModel + Geneformer prepend <cls>

    def __init__(self):
        self.model = None
        self.token_dict = None       # Ensembl_ID -> token_id
        self.pad_token_id = None
        self.cls_token_id = None
        self.max_input_size = 2048
        self.device = None
        # cached by preprocess()
        self._input_ids = None       # (n_cells, max_input)
        self._attn_mask = None       # (n_cells, max_input)

    # -------------------------------------------------------------------------
    def load(self, model_dir: str, device: str = "cuda") -> None:
        from transformers import BertModel

        model_dir = Path(model_dir)
        # Try common token-dict locations
        candidates = [
            model_dir / "token_dictionary.pkl",
            model_dir / "geneformer" / "token_dictionary.pkl",
            model_dir / "tokenizer" / "token_dictionary.pkl",
        ]
        td_path = next((p for p in candidates if p.exists()), None)
        if td_path is None:
            raise SystemExit(
                f"could not find token_dictionary.pkl under {model_dir}. "
                f"Run src/download_geneformer.py first."
            )
        with open(td_path, "rb") as f:
            self.token_dict = pickle.load(f)
        print(f"[geneformer] token vocab: {len(self.token_dict)}")

        self.pad_token_id = self.token_dict[PAD_TOKEN]
        self.cls_token_id = self.token_dict[CLS_TOKEN]

        model = BertModel.from_pretrained(
            str(model_dir), output_hidden_states=False, add_pooling_layer=False,
        )
        model.to(device).eval()

        self.model = model
        self.n_layers = model.config.num_hidden_layers
        self.d_model = model.config.hidden_size
        self.max_input_size = getattr(model.config, "max_position_embeddings", 2048)
        self.device = device
        print(f"[geneformer] loaded: n_layers={self.n_layers}, d_model={self.d_model}, "
              f"max_input={self.max_input_size}")

    # -------------------------------------------------------------------------
    def preprocess(self, adata, gene_col: str = None):
        """Rank-value-encode each cell into a token sequence.

        Geneformer's tokenization recipe:
          1. Use raw counts (NOT log-normalized).
          2. Per-cell, divide all expression values by the per-gene median over
             all cells (gene_median_dictionary; we approximate by per-cell
             median of nonzero values when no global dict is provided).
          3. Rank genes by normalized value (descending), keep nonzero genes.
          4. Take top max_input_size-1 (leave 1 slot for <cls>).
          5. Map gene Ensembl ID -> token ID, drop genes not in vocab.
          6. Prepend <cls>, pad with <pad>.

        For the baseline (PCA-50 / raw log1p), this method also adds an
        X_log1p layer derived from total-count normalization + log1p, on the
        gene set that overlaps with Geneformer's vocab — so the comparison is
        fair (same gene set, just different representation).
        """
        # ---- locate Ensembl IDs -------------------------------------------
        if gene_col is None:
            if "gene_ids" in adata.var.columns:
                gene_col = "gene_ids"          # scanpy pbmc3k convention
            elif "ensembl_id" in adata.var.columns:
                gene_col = "ensembl_id"
            elif str(adata.var.index[0]).startswith("ENSG"):
                gene_col = "_index"
            else:
                raise SystemExit(
                    "[geneformer] need Ensembl IDs. Add adata.var['gene_ids'] "
                    "or pass gene_col=... ; pbmc3k from scanpy already has "
                    "'gene_ids' so this should usually 'just work'."
                )
        if gene_col == "_index":
            ensembl_ids = adata.var.index.astype(str).tolist()
        else:
            ensembl_ids = adata.var[gene_col].astype(str).tolist()

        gene_token_ids = [self.token_dict.get(e) for e in ensembl_ids]
        keep_col_idx = np.array(
            [i for i, t in enumerate(gene_token_ids) if t is not None],
            dtype=np.int64,
        )
        kept_token_ids = np.array(
            [gene_token_ids[i] for i in keep_col_idx], dtype=np.int64,
        )
        n_keep = len(keep_col_idx)
        print(f"[geneformer] gene-vocab overlap: {n_keep}/{len(ensembl_ids)}")
        if n_keep == 0:
            raise SystemExit(
                "no overlap with Geneformer vocab — wrong gene_col? "
                "Geneformer uses Ensembl IDs (ENSG...) not gene symbols."
            )

        # ---- get raw counts ------------------------------------------------
        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X).astype(np.float32)
        if (X < 0).any():
            raise SystemExit(
                "[geneformer] adata.X has negative values; expected raw counts "
                "or non-negative normalized counts. Pass raw counts in adata.X."
            )
        X_kept = X[:, keep_col_idx]  # (n_cells, n_kept_genes)

        # ---- rank-encode each cell ----------------------------------------
        n_cells = X_kept.shape[0]
        max_g = self.max_input_size - 1   # reserve 1 slot for CLS
        input_ids = np.full(
            (n_cells, self.max_input_size), self.pad_token_id, dtype=np.int64,
        )
        attn_mask = np.zeros((n_cells, self.max_input_size), dtype=np.int64)

        for i in tqdm(range(n_cells), desc="[geneformer] tokenize", unit="cell"):
            row = X_kept[i]
            nz = row > 0
            if not nz.any():
                input_ids[i, 0] = self.cls_token_id
                attn_mask[i, 0] = 1
                continue
            nz_vals = row[nz]
            med = float(np.median(nz_vals)) or 1.0
            normed = row / med
            order = np.argsort(-normed)
            order = order[normed[order] > 0][:max_g]
            seq = np.concatenate([[self.cls_token_id], kept_token_ids[order]])
            input_ids[i, : len(seq)] = seq
            attn_mask[i, : len(seq)] = 1

        self._input_ids = input_ids
        self._attn_mask = attn_mask

        # ---- baseline layer ------------------------------------------------
        # Match what run_audit reads: adata.layers["X_log1p"] on the same gene
        # set Geneformer sees, log1p-normalized.
        import scanpy as sc
        adata_b = adata[:, keep_col_idx].copy()
        sc.pp.normalize_total(adata_b, target_sum=1e4)
        sc.pp.log1p(adata_b)
        if hasattr(adata_b.X, "toarray"):
            adata_b.layers["X_log1p"] = adata_b.X.toarray().astype(np.float32)
        else:
            adata_b.layers["X_log1p"] = np.asarray(adata_b.X).astype(np.float32)
        return adata_b

    # -------------------------------------------------------------------------
    def iter_layer_activations(self, adata, batch_size, device):
        encoder = self.model.encoder
        captured: Dict[str, torch.Tensor] = {}
        handles = []

        # input to encoder = post-embedding hidden_states
        def pre_hook(_m, args, kwargs):
            # BertEncoder.forward(hidden_states, attention_mask=..., ...)
            captured["layer_00_input"] = args[0].detach()
        handles.append(encoder.register_forward_pre_hook(pre_hook, with_kwargs=True))

        def make_hook(i: int):
            name = f"layer_{i + 1:02d}"
            def h(_m, _args, output):
                acts = output[0] if isinstance(output, tuple) else output
                captured[name] = acts.detach()
            return h

        for i, layer in enumerate(encoder.layer):
            handles.append(layer.register_forward_hook(make_hook(i)))

        n_cells = adata.n_obs
        try:
            for start in tqdm(range(0, n_cells, batch_size),
                              desc=f"{self.name} forward", unit="batch"):
                end = min(start + batch_size, n_cells)
                input_ids = torch.from_numpy(self._input_ids[start:end]).to(device)
                attn_mask = torch.from_numpy(self._attn_mask[start:end]).to(device)
                with torch.no_grad():
                    self.model(input_ids=input_ids, attention_mask=attn_mask)

                # valid: real gene tokens only (excludes pad AND CLS at position 0)
                valid = (attn_mask == 1).clone()
                valid[:, 0] = False

                yield dict(captured), valid
                captured.clear()
        finally:
            for h in handles:
                h.remove()
