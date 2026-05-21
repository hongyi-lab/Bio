"""Evo adapter — DNA sequence foundation model (Nguyen et al., Science 2024).

Architecture: StripedHyena — interleaved Hyena (state-space-like) and attention
blocks. ~7B parameters. Released in two context-length variants:
  togethercomputer/evo-1-8k-base    (7B, 8k context)
  togethercomputer/evo-1-131k-base  (7B, 131k context)

This is the only LLM-scale DNA foundation model we test. Compared to
HyenaDNA-small (6.6M, d=256), Evo is ~1000× larger (d=4096, 32 layers).

Loading: AutoModel + trust_remote_code=True (custom StripedHyena code).
Tokenizer: char-level over A/C/G/T/N.

Compute notes (A6000 48 GB):
  - fp16 weights ≈ 14 GB
  - Inference activations at batch=4, seq=1024 ≈ 5-10 GB additional
  - Forward speed ≈ 0.2-0.5 s/sample at fp16
  - For SAE training at 16× expansion: dict_size = 16 × 4096 = 65,536
    SAE weights ≈ 2 GB fp32 (1 GB fp16), trainable batch=4096 fits comfortably.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from ..core.adapter import BioFMAdapter


class EvoAdapter(BioFMAdapter):
    name = "evo"
    modality = "dna"
    cls_position = None   # Evo has no CLS — bare nucleotide sequence

    DEFAULT_HF_REPO = "togethercomputer/evo-1-8k-base"
    MAX_LEN = 8192   # cap context; Evo-1-8k natively supports up to 8192

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self._input_ids = None
        self._attn_mask = None
        self._blocks_attr = None
        self._fp16 = True   # default to fp16 to fit weights on A6000

    def load(self, model_dir: str, device: str = "cuda", fp16: bool = True) -> None:
        from transformers import AutoModel, AutoTokenizer

        self._fp16 = fp16
        model_dir = (str(model_dir) if Path(model_dir).exists()
                     else self.DEFAULT_HF_REPO)
        tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_dir, trust_remote_code=True,
            torch_dtype=torch.float16 if fp16 else torch.float32,
            output_hidden_states=False,
        )
        model.to(device).eval()

        self.model = model
        self.tokenizer = tok
        self.device = device

        # Locate the block list. StripedHyena HF wrapper exposes either
        # `backbone.blocks` (most likely) or `model.blocks` depending on
        # the release. Fall back to attribute-walking.
        blocks = None
        for attr_path in (("backbone", "blocks"), ("backbone", "layers"),
                          ("model", "blocks"), ("model", "layers"),
                          ("blocks",), ("layers",)):
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
                "[evo] could not locate block list on the model; check "
                "modeling_stripedhyena.py for the right path and update "
                "EvoAdapter._blocks_attr."
            )
        self.n_layers = len(blocks)

        # d_model from config
        d_model = getattr(model.config, "d_model",
                          getattr(model.config, "hidden_size", None))
        if d_model is None:
            # last fallback: probe a parameter
            for p in model.parameters():
                if p.dim() == 2:
                    d_model = p.shape[-1]
                    break
        self.d_model = int(d_model)
        print(f"[evo] loaded: n_layers={self.n_layers}, d_model={self.d_model}, "
              f"max_len={self.MAX_LEN}, fp16={fp16}")

    def preprocess(self, sample_or_seqs):
        if hasattr(sample_or_seqs, "inputs"):
            seqs = sample_or_seqs.inputs
        else:
            seqs = sample_or_seqs
        assert isinstance(seqs, list) and isinstance(seqs[0], str), \
            "[evo] preprocess expects list[str] DNA sequences"

        encoded = self.tokenizer(
            seqs,
            padding="max_length", truncation=True, max_length=self.MAX_LEN,
            return_tensors="np",
        )
        self._input_ids = encoded["input_ids"].astype(np.int64)
        if "attention_mask" in encoded:
            self._attn_mask = encoded["attention_mask"].astype(np.int64)
        else:
            pad_id = getattr(self.tokenizer, "pad_token_id", 0)
            if pad_id is None:
                pad_id = 0
            self._attn_mask = (self._input_ids != pad_id).astype(np.int64)
        print(f"[evo] tokenized: input_ids={self._input_ids.shape}")
        return sample_or_seqs

    def iter_layer_activations(self, sample_or_seqs, batch_size, device):
        # Find blocks list
        blocks = self.model
        for a in self._blocks_attr:
            blocks = getattr(blocks, a)

        captured: Dict[str, torch.Tensor] = {}
        handles = []

        def pre_hook(_m, args):
            if not args:
                return
            captured["layer_00_input"] = args[0].detach().float()

        handles.append(blocks[0].register_forward_pre_hook(pre_hook))

        def make_hook(i: int):
            name = f"layer_{i + 1:02d}"

            def h(_m, _args, output):
                acts = output[0] if isinstance(output, tuple) else output
                # Cast fp16 activations back to fp32 for SAE / probe stability
                captured[name] = acts.detach().float()

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
                # StripedHyena.forward doesn't accept attention_mask (state-space
                # path doesn't use it); we drive valid-mask from tokenizer.
                with torch.no_grad():
                    self.model(input_ids=input_ids)
                valid = (attn_mask == 1)
                yield dict(captured), valid
                captured.clear()
        finally:
            for h in handles:
                h.remove()
