"""power_spectrum_compare.py — sequence-axis FFT power spectrum, Hyena vs attention.

Motivation: the SAE-on-delta hit-rate measures structure along the *channel*
axis. If Hyena learns long convolutional filters, structure can also live
along the *sequence* axis — frequency-domain peaks that the channel SAE
literally can't see (its features are linear combinations of channels at a
single position).

This script:
  1. For one Hyena layer (default layer_16) and one attention layer (default
     layer_17), reconstructs per-sample sequence-axis activation tensors
     (L_target, d) from the flat (n_tokens, d) cache via cell_idx.
  2. Runs torch.fft.rfft along the L axis.
  3. Computes the power = |fft|^2 per (sample, frequency, channel),
     averages over channel and over sample, leaving one power vs frequency
     curve per layer.
  4. Plots both curves on the same log-y axis + saves the raw numbers as JSON.

If Hyena's curve has peaks at specific frequency bands while attention's is
flat-or-1/f, that's direct evidence of channel-SAE-invisible structure —
no need to train a spectral SAE to make the point.

Output:
  results/.../analyses/power_spectrum_layer16_vs_layer17.png
  results/.../analyses/power_spectrum_layer16_vs_layer17.json

Usage:
  python src/phase7/power_spectrum_compare.py
  # custom layer pair:
  python src/phase7/power_spectrum_compare.py --hyena_layer layer_08_hyena --attn_layer layer_09_attention
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent.parent


def load_cache(layer_dir: Path):
    """Returns (acts, cell_idx, fmt) supporting both new .npy and legacy .npz."""
    acts_npy = layer_dir / "acts.npy"
    cell_npy = layer_dir / "cell_idx.npy"
    if acts_npy.exists() and cell_npy.exists():
        # mmap_mode=r — we walk through the array sequentially in cell order,
        # so we don't need a full RAM load
        return np.load(acts_npy, mmap_mode="r"), np.load(cell_npy), "npy"
    legacy = layer_dir / "token_activations.npz"
    if legacy.exists():
        # npz doesn't support mmap; this fully loads ~80 GB into RAM. Slow
        # start (~5-10 min) but downstream sequential walk is fast.
        d = np.load(legacy, allow_pickle=False)
        return d["acts"], d["cell_idx"], "npz"
    raise FileNotFoundError(f"no acts cache in {layer_dir}")


def power_spectrum(acts, cell_idx, L_target: int, n_samples_cap: int, device: str):
    """Compute mean-over-sample-and-channel power spectrum.

    For each sample (run of identical cell_idx), if it has ≥ L_target valid
    tokens, take its first L_target tokens, FFT along L, compute power, mean
    over the channel dim, accumulate.
    """
    cell_idx_arr = np.asarray(cell_idx)
    # Run boundaries — tokens for cell c form a contiguous run because the
    # extraction loop processed samples in order.
    diff = np.diff(cell_idx_arr)
    boundaries = np.concatenate([[0], np.where(diff != 0)[0] + 1, [len(cell_idx_arr)]])
    n_cells = len(boundaries) - 1
    n_freqs = L_target // 2 + 1
    power_sum = torch.zeros(n_freqs, dtype=torch.float64, device=device)
    n_used = 0

    pbar = tqdm(range(n_cells), desc=f"FFT (L={L_target})", unit="sample")
    for i in pbar:
        start = int(boundaries[i])
        end = int(boundaries[i + 1])
        if end - start < L_target:
            continue
        x = np.ascontiguousarray(acts[start:start + L_target]).astype(np.float32)
        x_t = torch.from_numpy(x).to(device, non_blocking=True)        # (L, d)
        fft = torch.fft.rfft(x_t, dim=0)                                # (n_freqs, d)
        power = (fft.real.double() ** 2 + fft.imag.double() ** 2).mean(dim=1)
        power_sum += power
        n_used += 1
        if n_used >= n_samples_cap:
            break
    pbar.close()
    if n_used == 0:
        raise RuntimeError(
            f"no sample had ≥ {L_target} valid tokens. lower --L_target."
        )
    return (power_sum / n_used).cpu().numpy(), n_used


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hyena_layer", default="layer_16_hyena")
    p.add_argument("--attn_layer",  default="layer_17_attention")
    p.add_argument("--delta_root",
                   default="results/evo_1_8k__genomic_benchmarks_delta")
    p.add_argument("--L_target", type=int, default=128,
                   help="sequence length for FFT (skips samples shorter than this)")
    p.add_argument("--n_samples_cap", type=int, default=2000,
                   help="how many samples per layer to average over")
    p.add_argument("--out", default=None,
                   help="output dir; default: <delta_root>/analyses/")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    delta_root = Path(args.delta_root)
    if not delta_root.is_absolute():
        delta_root = ROOT / delta_root
    out_dir = Path(args.out) if args.out else delta_root / "analyses"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fft] hyena layer dir: {delta_root / args.hyena_layer}")
    print(f"[fft] attn  layer dir: {delta_root / args.attn_layer}")
    print(f"[fft] L_target={args.L_target}, n_samples_cap={args.n_samples_cap}")

    t0 = time.time()
    print(f"[fft] loading Hyena cache ({args.hyena_layer}) ...")
    h_acts, h_cell, h_fmt = load_cache(delta_root / args.hyena_layer)
    print(f"[fft]   {h_fmt}: acts={h_acts.shape} {h_acts.dtype}, cell={h_cell.shape}")
    h_power, h_used = power_spectrum(
        h_acts, h_cell, args.L_target, args.n_samples_cap, args.device,
    )
    print(f"[fft]   Hyena: averaged over {h_used} samples")
    del h_acts, h_cell

    print(f"[fft] loading attention cache ({args.attn_layer}) ...")
    a_acts, a_cell, a_fmt = load_cache(delta_root / args.attn_layer)
    print(f"[fft]   {a_fmt}: acts={a_acts.shape} {a_acts.dtype}, cell={a_cell.shape}")
    a_power, a_used = power_spectrum(
        a_acts, a_cell, args.L_target, args.n_samples_cap, args.device,
    )
    print(f"[fft]   attention: averaged over {a_used} samples")
    del a_acts, a_cell

    elapsed = time.time() - t0
    print(f"[fft] total wall time: {elapsed:.1f}s")

    # Normalize so DC component aligns — makes spectral *shape* the focus,
    # not absolute amplitude (which depends on the layer's overall scale).
    h_norm = h_power / h_power[0]
    a_norm = a_power / a_power[0]

    # --- Plot ---
    freqs = np.arange(args.L_target // 2 + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # left: absolute power (log-y)
    axes[0].semilogy(freqs, h_power, label=f"Hyena ({args.hyena_layer})", color="C0", lw=2)
    axes[0].semilogy(freqs, a_power, label=f"attention ({args.attn_layer})", color="C1", lw=2)
    axes[0].set_xlabel("frequency index k (0 = DC, L/2 = Nyquist)")
    axes[0].set_ylabel("mean |F(x)|² over channel & sample (log)")
    axes[0].set_title("absolute power")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    # right: shape-normalized (each curve / its DC)
    axes[1].semilogy(freqs, h_norm, label=f"Hyena (normalized by DC)", color="C0", lw=2)
    axes[1].semilogy(freqs, a_norm, label=f"attention (normalized by DC)", color="C1", lw=2)
    # 1/f reference for visual sanity
    ref = (freqs.astype(float) + 1) ** -1
    ref = ref / ref[0]
    axes[1].semilogy(freqs, ref, "k--", lw=1, alpha=0.4, label="1/k reference")
    axes[1].set_xlabel("frequency index k")
    axes[1].set_ylabel("power / DC")
    axes[1].set_title("shape (each curve / its DC component)")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    fig.suptitle(
        f"Sequence-axis power spectrum: {args.hyena_layer} vs {args.attn_layer}  "
        f"(L={args.L_target}, n_samples={min(h_used, a_used)})"
    )
    fig.tight_layout()
    png_path = out_dir / f"power_spectrum_{args.hyena_layer}_vs_{args.attn_layer}.png"
    fig.savefig(png_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[fft] wrote {png_path}")

    # --- JSON ---
    json_path = out_dir / f"power_spectrum_{args.hyena_layer}_vs_{args.attn_layer}.json"
    json_path.write_text(json.dumps({
        "config": {
            "hyena_layer": args.hyena_layer,
            "attn_layer": args.attn_layer,
            "L_target": args.L_target,
            "n_samples_cap": args.n_samples_cap,
            "n_freqs": int(len(freqs)),
            "n_samples_used_hyena": int(h_used),
            "n_samples_used_attn": int(a_used),
        },
        "freqs": freqs.tolist(),
        "hyena_power_abs": h_power.tolist(),
        "attn_power_abs": a_power.tolist(),
        "hyena_power_normalized_by_dc": h_norm.tolist(),
        "attn_power_normalized_by_dc": a_norm.tolist(),
        "wall_time_s": float(elapsed),
    }, indent=2))
    print(f"[fft] wrote {json_path}")

    # Quick numerical read for the user
    print("\n=== quick read ===")
    # Peak frequency (excluding DC) per layer
    h_peak_k = int(np.argmax(h_norm[1:]) + 1)
    a_peak_k = int(np.argmax(a_norm[1:]) + 1)
    print(f"Hyena non-DC peak: k={h_peak_k} (power_norm={h_norm[h_peak_k]:.3g})")
    print(f"Attn  non-DC peak: k={a_peak_k} (power_norm={a_norm[a_peak_k]:.3g})")
    # Mid-band concentration: power in k ∈ [4, 16] vs total
    h_midband = h_norm[4:17].sum() / h_norm.sum()
    a_midband = a_norm[4:17].sum() / a_norm.sum()
    print(f"Hyena mid-band (k=4..16) fraction: {h_midband:.3%}")
    print(f"Attn  mid-band (k=4..16) fraction: {a_midband:.3%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
