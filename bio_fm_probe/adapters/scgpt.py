"""scGPT adapter — wraps the whole-human (or any scGPT) checkpoint behind
the standard BioFMAdapter interface.

Refactored from the phase 1-3 src/ scripts. Numerically identical results
should fall out when run with the same data/seeds.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np
import torch
from tqdm import tqdm

from ..core.adapter import BioFMAdapter


# scGPT tokenizer constants (must match pretraining)
PAD_TOKEN = "<pad>"
SPECIAL_TOKENS = [PAD_TOKEN, "<cls>", "<eoc>"]
PAD_VALUE = -2
N_BINS = 51
MAX_LEN = 1200


class ScGPTAdapter(BioFMAdapter):
    name = "scgpt"
    modality = "scrna"
    cls_position = 0  # scGPT prepends <cls> at sequence position 0

    def __init__(self):
        self.model = None
        self.vocab = None
        self.device = None
        self._gene_ids_all = None   # populated in preprocess

    # -------------------------------------------------------------------------
    def load(self, model_dir: str, device: str = "cuda") -> None:
        from scgpt.model import TransformerModel
        from scgpt.tokenizer.gene_tokenizer import GeneVocab

        model_dir = Path(model_dir)
        vocab = GeneVocab.from_file(model_dir / "vocab.json")
        for tok in SPECIAL_TOKENS:
            if tok not in vocab:
                vocab.append_token(tok)
        vocab.set_default_index(vocab[PAD_TOKEN])

        with open(model_dir / "args.json") as f:
            margs = json.load(f)

        model = TransformerModel(
            ntoken=len(vocab),
            d_model=margs["embsize"],
            nhead=margs["nheads"],
            d_hid=margs["d_hid"],
            nlayers=margs["nlayers"],
            nlayers_cls=margs.get("n_layers_cls", 3),
            n_cls=1,
            vocab=vocab,
            dropout=margs.get("dropout", 0.0),
            pad_token=PAD_TOKEN,
            pad_value=PAD_VALUE,
            do_mvc=False, do_dab=False,
            use_batch_labels=False, domain_spec_batchnorm=False,
            input_emb_style="continuous",
            n_input_bins=N_BINS,
            pre_norm=False,
            use_fast_transformer=False,
        )
        state = torch.load(model_dir / "best_model.pt", map_location=device)
        try:
            model.load_state_dict(state)
        except RuntimeError:
            own = model.state_dict()
            kept = {k: v for k, v in state.items()
                    if k in own and own[k].shape == v.shape}
            print(f"[scgpt] strict load failed; kept {len(kept)}/{len(state)} keys")
            model.load_state_dict(kept, strict=False)
        model.to(device).eval()

        # Disable nested-tensor fast path so hooks see regular tensors
        enc = model.transformer_encoder
        for attr in ("enable_nested_tensor", "use_nested_tensor"):
            if hasattr(enc, attr):
                setattr(enc, attr, False)

        self.model = model
        self.vocab = vocab
        self.n_layers = len(enc.layers)
        self.d_model = margs["embsize"]
        self.device = device
        print(f"[scgpt] loaded: n_layers={self.n_layers}, d_model={self.d_model}, "
              f"vocab={len(vocab)}")

    # -------------------------------------------------------------------------
    def preprocess(self, adata, gene_col: str = "gene_name"):
        from scgpt.preprocess import Preprocessor

        if gene_col not in adata.var.columns:
            adata.var["gene_name"] = adata.var.index.astype(str)
            gene_col = "gene_name"

        keep = adata.var[gene_col].apply(lambda g: g in self.vocab).values
        n_keep = int(keep.sum())
        if n_keep == 0:
            raise SystemExit("no overlap between adata.var and scGPT vocab")
        adata = adata[:, keep].copy()
        print(f"[scgpt] vocab overlap: {n_keep}/{len(keep)} genes")

        is_raw = self._is_raw_counts(adata.X)
        pre = Preprocessor(
            use_key="X",
            filter_gene_by_counts=False, filter_cell_by_counts=False,
            normalize_total=1e4 if is_raw else False,
            result_normed_key="X_normed",
            log1p=is_raw,
            result_log1p_key="X_log1p",
            subset_hvg=False,
            binning=N_BINS,
            result_binned_key="X_binned",
        )
        pre(adata, batch_key=None)

        self._gene_ids_all = np.array(
            self.vocab(adata.var[gene_col].tolist()), dtype=int,
        )
        return adata

    # -------------------------------------------------------------------------
    def iter_layer_activations(
        self, adata, batch_size: int, device: str = "cuda",
    ) -> Iterator[Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
        from scgpt.tokenizer import tokenize_and_pad_batch

        encoder = self.model.transformer_encoder
        captured: Dict[str, torch.Tensor] = {}
        handles = []

        def pre_hook(_module, inputs):
            captured["layer_00_input"] = inputs[0].detach()

        handles.append(encoder.register_forward_pre_hook(pre_hook))

        def make_hook(i: int):
            name = f"layer_{i + 1:02d}"

            def h(_m, _i, output):
                captured[name] = output.detach()

            return h

        for i, layer in enumerate(encoder.layers):
            handles.append(layer.register_forward_hook(make_hook(i)))

        binned = adata.layers["X_binned"]
        if hasattr(binned, "toarray"):
            binned = binned.toarray()
        binned = np.asarray(binned)
        pad_idx = self.vocab[PAD_TOKEN]

        try:
            for start in tqdm(range(0, adata.n_obs, batch_size),
                              desc=f"{self.name} forward", unit="batch"):
                end = min(start + batch_size, adata.n_obs)
                batch_X = binned[start:end].astype(np.int64)
                tok = tokenize_and_pad_batch(
                    batch_X, self._gene_ids_all, max_len=MAX_LEN,
                    vocab=self.vocab, pad_token=PAD_TOKEN, pad_value=PAD_VALUE,
                    append_cls=True, include_zero_gene=False,
                )
                input_gene_ids = tok["genes"].to(device)
                input_values = tok["values"].to(device).float()
                pad_mask = input_gene_ids.eq(pad_idx)

                with torch.no_grad():
                    self.model._encode(
                        input_gene_ids, input_values,
                        src_key_padding_mask=pad_mask,
                    )

                # valid: real gene tokens only (not pad, not CLS at pos 0)
                valid = (~pad_mask).clone()
                valid[:, 0] = False

                yield dict(captured), valid
                captured.clear()
        finally:
            for h in handles:
                h.remove()

    # -------------------------------------------------------------------------
    @staticmethod
    def _is_raw_counts(X) -> bool:
        sample = X[:10].toarray() if hasattr(X, "toarray") else np.asarray(X[:10])
        if sample.size == 0 or (sample < 0).any():
            return False
        return float(np.abs(sample - np.round(sample)).max()) < 1e-6
