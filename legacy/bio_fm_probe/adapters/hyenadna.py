"""HyenaDNA adapter — DNA sequence foundation model (Nguyen et al., NeurIPS 2023).

Architecture: Hyena state-space-style blocks over nucleotide tokens. Released
in several context lengths (1k, 16k, 32k, 160k, 450k, 1M). For first-pass
audit, we use the small 32k-context variant from HuggingFace:
  `LongSafari/hyenadna-small-32k-seqlen-hf`

This loads via HuggingFace `AutoModel` with `trust_remote_code=True`. The model
has its own tokenizer that maps nucleotides A/C/G/T/N to token ids.

Audit pipeline expectations:
  - inputs: list[str] DNA sequences from a DatasetAdapter
  - layer naming: layer_00_input (post-embed) + layer_01..layer_NN per block
  - valid_mask: True for real nucleotide tokens (excludes padding)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from ..core.adapter import BioFMAdapter


class HyenaDNAAdapter(BioFMAdapter):
    name = "hyenadna"
    modality = "dna"
    cls_position = None   # HyenaDNA has no CLS — sequence is bare nucleotides

    DEFAULT_HF_REPO = "LongSafari/hyenadna-small-32k-seqlen-hf"
    MAX_LEN = 8192   # cap context length — sufficient for short DNA tasks

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self._input_ids = None        # (n_samples, max_len)
        self._attn_mask = None        # (n_samples, max_len) bool

    def load(self, model_dir: str, device: str = "cuda") -> None:
        from transformers import AutoModel, AutoTokenizer

        model_dir = str(model_dir) if Path(model_dir).exists() else self.DEFAULT_HF_REPO
        tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_dir, trust_remote_code=True, output_hidden_states=False,
        )
        model.to(device).eval()

        self.model = model
        self.tokenizer = tok
        self.device = device

        # Probe the model for block count and d_model.
        # HyenaDNA's HF wrapper exposes `backbone.layers` (list of HyenaBlock-like).
        # Fallbacks for slightly different attr names.
        blocks = None
        for attr_path in (("backbone", "layers"), ("model", "layers"), ("layers",)):
            obj = model
            ok = True
            for a in attr_path:
                obj = getattr(obj, a, None)
                if obj is None:
                    ok = False
                    break
            if ok:
                blocks = obj
                self._blocks_attr = attr_path
                break
        if blocks is None:
            raise SystemExit(
                "[hyenadna] could not locate transformer/hyena blocks list on "
                "the model; check the HF repo's modeling_*.py for the right path "
                "and update HyenaDNAAdapter._blocks_attr."
            )
        self.n_layers = len(blocks)
        # Probe d_model from a parameter shape
        d_model = None
        for p in model.parameters():
            if p.dim() == 2:
                d_model = p.shape[-1]
                break
        # Try the model config directly
        d_model = getattr(model.config, "d_model",
                          getattr(model.config, "hidden_size", d_model))
        self.d_model = int(d_model)
        print(f"[hyenadna] loaded: n_layers={self.n_layers}, "
              f"d_model={self.d_model}, max_len={self.MAX_LEN}")

    def preprocess(self, adata_or_seqs):
        """Tokenize sequences and stash input_ids/attn_mask on self.

        Argument is either a list[str] (preferred, from a DNA DatasetAdapter)
        or a `Sample` object. Returns the same object back unmodified — for
        DNA we don't need an adata-like return because baselines build from
        the raw sequences.
        """
        if hasattr(adata_or_seqs, "inputs"):
            seqs = adata_or_seqs.inputs
        else:
            seqs = adata_or_seqs
        assert isinstance(seqs, list) and isinstance(seqs[0], str), \
            "[hyenadna] preprocess expects list[str] DNA sequences"

        encoded = self.tokenizer(
            seqs,
            padding="max_length", truncation=True, max_length=self.MAX_LEN,
            return_tensors="np",
        )
        self._input_ids = encoded["input_ids"].astype(np.int64)
        if "attention_mask" in encoded:
            self._attn_mask = encoded["attention_mask"].astype(np.int64)
        else:
            # HyenaDNA tokenizer doesn't always emit attention_mask;
            # treat all non-pad tokens as valid.
            pad_id = getattr(self.tokenizer, "pad_token_id", 0)
            self._attn_mask = (self._input_ids != pad_id).astype(np.int64)
        print(f"[hyenadna] tokenized: input_ids={self._input_ids.shape}")
        return adata_or_seqs

    def iter_layer_activations(self, sample_or_seqs, batch_size, device):
        # Find the blocks list (set in load())
        blocks = self.model
        for a in self._blocks_attr:
            blocks = getattr(blocks, a)

        captured: Dict[str, torch.Tensor] = {}
        handles = []

        # Pre-hook on the first block — capture its input as "layer_00_input"
        def pre_hook(_m, args):
            if not args:
                return
            captured["layer_00_input"] = args[0].detach()

        handles.append(blocks[0].register_forward_pre_hook(pre_hook))

        def make_hook(i: int):
            name = f"layer_{i + 1:02d}"

            def h(_m, _args, output):
                acts = output[0] if isinstance(output, tuple) else output
                captured[name] = acts.detach()

            return h

        for i, block in enumerate(blocks):
            handles.append(block.register_forward_hook(make_hook(i)))

        n_samples = self._input_ids.shape[0]
        try:
            for start in tqdm(range(0, n_samples, batch_size),
                              desc=f"{self.name} forward", unit="batch"):
                end = min(start + batch_size, n_samples)
                input_ids = torch.from_numpy(self._input_ids[start:end]).to(device)
                attn_mask = torch.from_numpy(self._attn_mask[start:end]).to(device)
                # HyenaDNA is a state-space model — no attention layer, so
                # forward() doesn't accept attention_mask. We still keep
                # attn_mask around to drive the `valid` pooling mask below.
                with torch.no_grad():
                    self.model(input_ids=input_ids)

                valid = (attn_mask == 1)
                # no CLS to exclude — HyenaDNA emits bare nucleotide tokens
                yield dict(captured), valid
                captured.clear()
        finally:
            for h in handles:
                h.remove()
