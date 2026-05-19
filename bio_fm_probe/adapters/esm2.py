"""ESM-2 adapter — protein sequence foundation model (Lin et al. 2023).

Architecture: encoder-only Transformer over amino acid tokens. Released in
five sizes (8M, 35M, 150M, 650M, 3B, 15B params). For first-pass audit, we use
the 35M variant from HuggingFace:
  `facebook/esm2_t12_35M_UR50D`
which gives 12 transformer layers, d_model=480.

Loads via HuggingFace `EsmModel`. Tokenizer prepends a `<cls>` token at
position 0 and appends `<eos>`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from ..core.adapter import BioFMAdapter


class ESM2Adapter(BioFMAdapter):
    name = "esm2"
    modality = "protein"
    cls_position = 0   # ESM-2 prepends a CLS-like token at position 0

    DEFAULT_HF_REPO = "facebook/esm2_t12_35M_UR50D"
    MAX_LEN = 1024

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self._input_ids = None
        self._attn_mask = None

    def load(self, model_dir: str, device: str = "cuda") -> None:
        from transformers import AutoTokenizer, EsmModel

        model_dir = str(model_dir) if Path(model_dir).exists() else self.DEFAULT_HF_REPO
        tok = AutoTokenizer.from_pretrained(model_dir)
        model = EsmModel.from_pretrained(model_dir, add_pooling_layer=False)
        model.to(device).eval()

        self.model = model
        self.tokenizer = tok
        self.device = device
        self.n_layers = model.config.num_hidden_layers
        self.d_model = model.config.hidden_size
        print(f"[esm2] loaded: n_layers={self.n_layers}, d_model={self.d_model}, "
              f"max_len={self.MAX_LEN}")

    def preprocess(self, sample_or_seqs):
        if hasattr(sample_or_seqs, "inputs"):
            seqs = sample_or_seqs.inputs
        else:
            seqs = sample_or_seqs
        assert isinstance(seqs, list) and isinstance(seqs[0], str), \
            "[esm2] preprocess expects list[str] protein sequences"

        encoded = self.tokenizer(
            seqs,
            padding="max_length", truncation=True, max_length=self.MAX_LEN,
            return_tensors="np",
        )
        self._input_ids = encoded["input_ids"].astype(np.int64)
        self._attn_mask = encoded["attention_mask"].astype(np.int64)
        print(f"[esm2] tokenized: input_ids={self._input_ids.shape}")
        return sample_or_seqs

    def iter_layer_activations(self, sample_or_seqs, batch_size, device):
        encoder = self.model.encoder
        captured: Dict[str, torch.Tensor] = {}
        handles = []

        def pre_hook(_m, args, kwargs):
            # EsmEncoder.forward(hidden_states, attention_mask, ...)
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

        n_samples = self._input_ids.shape[0]
        try:
            for start in tqdm(range(0, n_samples, batch_size),
                              desc=f"{self.name} forward", unit="batch"):
                end = min(start + batch_size, n_samples)
                input_ids = torch.from_numpy(self._input_ids[start:end]).to(device)
                attn_mask = torch.from_numpy(self._attn_mask[start:end]).to(device)
                with torch.no_grad():
                    self.model(input_ids=input_ids, attention_mask=attn_mask)

                # valid: non-pad and not CLS at position 0
                valid = (attn_mask == 1).clone()
                valid[:, 0] = False
                yield dict(captured), valid
                captured.clear()
        finally:
            for h in handles:
                h.remove()
