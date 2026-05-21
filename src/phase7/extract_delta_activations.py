"""extract_delta_activations.py — per-block "delta" extraction for SAE-on-delta.

Why this exists:
    Probing the residual stream (block output) mixes the contributions of every
    earlier block, so an SAE found there can't be attributed to any single
    operator. Within StripedHyena, Hyena blocks and attention blocks alternate
    on the SAME residual stream — same model, same data, same training
    objective, same scale. To isolate "what does this one block compute?",
    extract its delta:

        delta_i = block_i.output - block_i.input

    where input is the residual just before this block and output is the
    residual just after. The delta is the block's net contribution.

    If SAEs on Hyena deltas look structurally similar to SAEs on attention
    deltas (feature count, sparsity profile, downstream concept hit rate),
    the residual-stream linear-interface argument explains everything and
    "Hyena has a different geometry" is unsupported. If they look different,
    THAT is real evidence of an operator effect.

Caveat acknowledged:
    Delta = output - input still passes through the block's output projection
    (a linear map back to d_model). So delta is structurally constrained to
    expose linear directions. To test the "Hyena has Fourier-basis internal
    state" hypothesis you'd have to hook INSIDE the Hyena op (after FFT,
    before the output projection). That's a separate, more invasive
    experiment — not in scope here.

Pair selection for Evo-1 7B:
    attn_layer_idxs = [8, 16, 24] (0-indexed) — so blocks[8], blocks[16],
    blocks[24] are attention, every other block is Hyena. Adjacent
    Hyena/attention pairs (using 1-indexed layer_NN names):

        Pair early:  layer_08 (Hyena) ↔ layer_09 (attention)
        Pair mid:    layer_16 (Hyena) ↔ layer_17 (attention)
        Pair late:   layer_24 (Hyena) ↔ layer_25 (attention)

Outputs (under results/evo_1_8k__genomic_benchmarks_delta/<layer>/):
    token_activations.npz   the delta activations: shape (n_tokens, d_model)
                            with a `kind` field in metadata indicating
                            "hyena" or "attention" (saved as filename suffix
                            in the layer dir name).

Usage:
    # default: extract all 3 mid/late pairs
    python src/phase7/extract_delta_activations.py
    # custom pair set
    python src/phase7/extract_delta_activations.py --layers layer_16 layer_17
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent.parent
sys.path.insert(0, str(THIS.parent))


def _resolve(p: str) -> str:
    path = Path(p)
    return str(path if path.is_absolute() else ROOT / path)


from evo_probe import load_evo2, load_genomic_benchmarks  # noqa: E402


# Evo-1 attention block positions (0-indexed). All others are Hyena.
ATTN_LAYER_IDXS_0BASED = {8, 16, 24}

# Default pairs — adjacent Hyena/attention blocks across early/mid/late.
# layer_NN is 1-indexed (layer_01 = blocks[0], layer_32 = blocks[31]).
DEFAULT_PAIRS = [
    ("layer_08", "layer_09"),   # early: blocks[7]=Hyena, blocks[8]=attention
    ("layer_16", "layer_17"),   # mid:   blocks[15]=Hyena, blocks[16]=attention
    ("layer_24", "layer_25"),   # late:  blocks[23]=Hyena, blocks[24]=attention
]


def kind_of(layer_name: str) -> str:
    """'hyena' or 'attention' for a layer_NN name (1-indexed)."""
    idx_1based = int(layer_name.split("_")[1])
    return "attention" if (idx_1based - 1) in ATTN_LAYER_IDXS_0BASED else "hyena"


def iter_evo_delta_activations(
    model, tok, blocks_attr, input_ids: np.ndarray, attn_mask: np.ndarray,
    target_layers: List[str], batch_size: int, device: str,
):
    """Forward all samples; for each target layer hook (pre_input, output) and
    yield per-batch (captured_deltas: {layer_name: (B, seq, d)}, valid_mask: (B, seq)).
    """
    blocks = model
    for a in blocks_attr:
        blocks = getattr(blocks, a)
    n_layers = len(blocks)

    captured_pre: Dict[str, torch.Tensor] = {}
    captured_post: Dict[str, torch.Tensor] = {}
    handles = []

    def make_pre_hook(name):
        def h(_m, args):
            if not args:
                return
            captured_pre[name] = args[0].detach().float()
        return h

    def make_post_hook(name):
        def h(_m, _a, out):
            captured_post[name] = (out[0] if isinstance(out, tuple) else out).detach().float()
        return h

    target_block_idxs = []
    for name in target_layers:
        idx = int(name.split("_")[1]) - 1
        if not (0 <= idx < n_layers):
            raise ValueError(f"layer {name!r} out of range (n_layers={n_layers})")
        target_block_idxs.append((idx, name))
        handles.append(blocks[idx].register_forward_pre_hook(make_pre_hook(name)))
        handles.append(blocks[idx].register_forward_hook(make_post_hook(name)))

    n_samples = input_ids.shape[0]
    try:
        for start in tqdm(range(0, n_samples, batch_size),
                          desc="delta forward", unit="batch"):
            end = min(start + batch_size, n_samples)
            ids = torch.from_numpy(input_ids[start:end]).to(device)
            mask = torch.from_numpy(attn_mask[start:end]).to(device)
            with torch.no_grad():
                model(input_ids=ids)
            deltas: Dict[str, torch.Tensor] = {}
            for name in target_layers:
                pre = captured_pre.get(name)
                post = captured_post.get(name)
                if pre is None or post is None:
                    raise RuntimeError(f"missing pre/post capture for {name}")
                # post and pre may have different ordering (tuple unpack), but
                # both should be (B, seq, d). Subtract directly.
                deltas[name] = post - pre
            valid = (mask == 1)
            yield deltas, valid
            captured_pre.clear()
            captured_post.clear()
    finally:
        for h in handles:
            h.remove()


def extract_delta_tokens(
    delta_iter, target_layer: str, n_samples: int,
):
    """Same shape as common_sae.extract_tokens_from_iter, but consumes
    delta_iter (one layer at a time)."""
    acts_chunks: List[np.ndarray] = []
    cell_chunks: List[np.ndarray] = []
    cells_processed = 0
    for deltas, valid in delta_iter:
        x = deltas[target_layer]                   # (B, seq, d)
        B = x.shape[0]
        cell_idx_full = (
            torch.arange(cells_processed, cells_processed + B, device=x.device)
            .unsqueeze(1).expand_as(valid)
        )
        x_cpu = x[valid].cpu()
        # Same dtype policy as common_sae.extract_tokens_from_iter — bf16 needs fp32.
        if x_cpu.dtype == torch.bfloat16:
            arr = x_cpu.float().numpy()
        elif x_cpu.dtype == torch.float16:
            arr = x_cpu.numpy().astype(np.float16)
        else:
            arr = x_cpu.numpy().astype(np.float32)
        acts_chunks.append(arr)
        cell_chunks.append(cell_idx_full[valid].cpu().numpy().astype(np.int64))
        cells_processed += B
    return (
        np.concatenate(acts_chunks, axis=0) if acts_chunks else np.zeros((0,)),
        np.concatenate(cell_chunks, axis=0) if cell_chunks else np.zeros((0,)),
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="checkpoints/evo-1-8k-base")
    p.add_argument("--data_dir",
                   default="data/genomic_benchmarks/human_nontata_promoters")
    p.add_argument("--layers", nargs="+", default=None,
                   help="default: all 3 hyena/attention pairs")
    p.add_argument("--max_samples", type=int, default=20000)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--extract_batch_size", type=int, default=2)
    p.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None,
                   help="default: results/evo_1_8k__genomic_benchmarks_delta/")
    p.add_argument("--force", action="store_true",
                   help="ignore cache and re-extract")
    args = p.parse_args()

    if args.layers is None:
        args.layers = [name for pair in DEFAULT_PAIRS for name in pair]
    print(f"[delta] target layers: {args.layers}")
    for name in args.layers:
        print(f"[delta]   {name} = {kind_of(name)}")

    out_dir = (Path(_resolve(args.out)) if args.out
               else ROOT / "results" / "evo_1_8k__genomic_benchmarks_delta")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[delta] output -> {out_dir}")

    # Dataset
    data_dir = Path(_resolve(args.data_dir))
    seqs, labels = load_genomic_benchmarks(data_dir)
    if len(seqs) > args.max_samples:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(len(seqs), args.max_samples, replace=False))
        seqs = [seqs[i] for i in idx]
        labels = labels[idx]
    print(f"[delta] {len(seqs)} samples")

    # Skip layers that already have a cache (unless --force)
    layers_to_run = []
    for name in args.layers:
        ld = out_dir / f"{name}_{kind_of(name)}"
        cache_file = ld / "token_activations.npz"
        if cache_file.exists() and not args.force:
            sz = cache_file.stat().st_size / 1e9
            print(f"[delta] {name}: cache exists ({sz:.2f} GB) — skip "
                  f"(--force to re-extract)")
            continue
        ld.mkdir(parents=True, exist_ok=True)
        layers_to_run.append(name)
    if not layers_to_run:
        print("[delta] all target layers already cached. Done.")
        return 0

    # Load model
    model, tok, d_model, n_layers, blocks_attr = load_evo2(
        _resolve(args.model_dir), device=args.device, dtype=args.dtype,
    )

    # Tokenize once
    print(f"[delta] tokenizing (max_len={args.max_len}) ...")
    enc = tok(seqs, padding="max_length", truncation=True,
              max_length=args.max_len, return_tensors="np")
    input_ids = enc["input_ids"].astype(np.int64)
    if "attention_mask" in enc:
        attn_mask = enc["attention_mask"].astype(np.int64)
    else:
        pad_id = getattr(tok, "pad_token_id", 0) or 0
        attn_mask = (input_ids != pad_id).astype(np.int64)

    # Extract one layer at a time (memory-safe — each pass holds at most one
    # layer's pre+post activations on GPU).
    for layer in layers_to_run:
        ld = out_dir / f"{layer}_{kind_of(layer)}"
        cache_file = ld / "token_activations.npz"
        print(f"\n[delta] === {layer} ({kind_of(layer)}) ===")
        t0 = time.time()
        tok_acts, tok_cell = extract_delta_tokens(
            iter_evo_delta_activations(
                model, tok, blocks_attr, input_ids, attn_mask,
                target_layers=[layer],
                batch_size=args.extract_batch_size, device=args.device,
            ),
            target_layer=layer, n_samples=len(seqs),
        )
        np.savez(cache_file, acts=tok_acts, cell_idx=tok_cell)
        elapsed = time.time() - t0
        print(f"[delta]   {layer}: extracted {tok_acts.shape} "
              f"{tok_acts.dtype} in {elapsed:.0f}s -> {cache_file}")

    print(f"\n[delta] DONE — caches in {out_dir}")
    print(f"[delta] next: train SAE on each cache and compare with "
          f"src/phase7/evaluate_sae_jaspar.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
