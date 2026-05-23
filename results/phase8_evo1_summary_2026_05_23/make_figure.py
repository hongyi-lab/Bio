"""make_figure.py — one summary figure for phase 8 (Evo-1 SAE-on-delta + TFBS).

Renders 3 panels into a single PNG that tells the whole story so far:

  A) JASPAR TFBS hit-rate per layer — grouped by pair, Hyena vs attention.
     Shows the +4pp attention-over-Hyena advantage replicated across the
     two completed pairs (early + mid).
  B) Live feature count per layer — same grouping. Shows that attention
     layers are 5-10× sparser than their adjacent Hyena layers (despite
     identical dictionary size + identical resampling).
  C) Sequence-axis power spectrum, mid-stripe (layer_16 vs 17), normalized
     by DC. Both curves are nearly flat → "Hyena has channel-SAE-invisible
     spectral structure" hypothesis NOT supported.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
OUT = HERE / "phase8_summary.png"

PAIRS = [
    ("early", "layer_08_hyena", "layer_09_attention"),
    ("mid",   "layer_16_hyena", "layer_17_attention"),
    ("late",  "layer_24_hyena", "layer_25_attention"),
]


def load_jaspar(name):
    p = RAW / f"{name}__jaspar_hits.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main():
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    # ------------------------------------------------------------------ #
    # Panel A — JASPAR TFBS hit-rate per layer
    # ------------------------------------------------------------------ #
    ax = axes[0]
    pair_labels, hyena_rates, attn_rates, h_n, a_n = [], [], [], [], []
    for label, h, a in PAIRS:
        dh = load_jaspar(h); da = load_jaspar(a)
        if dh is None or da is None:
            continue
        pair_labels.append(label)
        hyena_rates.append(dh["hit_rate_among_live"] * 100)
        attn_rates.append(da["hit_rate_among_live"] * 100)
        h_n.append(dh["n_features_live"])
        a_n.append(da["n_features_live"])
    x = np.arange(len(pair_labels))
    w = 0.36
    bars_h = ax.bar(x - w/2, hyena_rates, w, color="#4477AA", label="Hyena delta")
    bars_a = ax.bar(x + w/2, attn_rates,  w, color="#EE6677", label="attention delta")
    for i, (rh, ra) in enumerate(zip(hyena_rates, attn_rates)):
        ax.text(x[i] - w/2, rh + 0.4, f"{rh:.1f}%", ha="center", fontsize=9)
        ax.text(x[i] + w/2, ra + 0.4, f"{ra:.1f}%", ha="center", fontsize=9)
        delta = ra - rh
        ax.text(x[i], max(rh, ra) + 3, f"Δ = {delta:+.1f}pp",
                ha="center", fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="#FFFAE6", ec="gray", alpha=0.9))
    ax.set_xticks(x); ax.set_xticklabels(pair_labels)
    ax.set_ylabel("JASPAR TFBS hit-rate among live features (%)")
    ax.set_title("(A) Attention deltas hit motifs more often\nthan adjacent Hyena deltas")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ymax = max(max(hyena_rates), max(attn_rates)) + 8
    ax.set_ylim(0, ymax)

    # ------------------------------------------------------------------ #
    # Panel B — live-feature count per layer
    # ------------------------------------------------------------------ #
    ax = axes[1]
    bars_h = ax.bar(x - w/2, h_n, w, color="#4477AA", label="Hyena delta")
    bars_a = ax.bar(x + w/2, a_n, w, color="#EE6677", label="attention delta")
    for i, (nh, na) in enumerate(zip(h_n, a_n)):
        ax.text(x[i] - w/2, nh + 200, f"{nh}", ha="center", fontsize=9)
        ax.text(x[i] + w/2, na + 200, f"{na}", ha="center", fontsize=9)
        ratio = nh / max(na, 1)
        ax.text(x[i], max(nh, na) + 1500, f"{ratio:.1f}× sparser",
                ha="center", fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="#FFFAE6", ec="gray", alpha=0.9))
    ax.set_xticks(x); ax.set_xticklabels(pair_labels)
    ax.set_ylabel("# live SAE features (out of 16,384)")
    ax.set_title("(B) Attention deltas use ~5-10× fewer\nfeatures than Hyena deltas")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 17500)
    ax.axhline(16384, color="gray", linestyle=":", alpha=0.5)
    ax.text(0.5, 16500, "dict_size = 16,384", fontsize=8, color="gray", alpha=0.8)

    # ------------------------------------------------------------------ #
    # Panel C — sequence-axis power spectrum, normalized
    # ------------------------------------------------------------------ #
    ax = axes[2]
    spec = json.loads((RAW / "power_spectrum_layer_16_hyena_vs_layer_17_attention.json").read_text())
    freqs = np.asarray(spec["freqs"])
    h_norm = np.asarray(spec["hyena_power_normalized_by_dc"])
    a_norm = np.asarray(spec["attn_power_normalized_by_dc"])
    ax.semilogy(freqs, h_norm, color="#4477AA", lw=2, label="Hyena (layer_16)")
    ax.semilogy(freqs, a_norm, color="#EE6677", lw=2, label="attention (layer_17)")
    ref = 1.0 / (freqs.astype(float) + 1.0)
    ref = ref / ref[0]
    ax.semilogy(freqs, ref, "k--", lw=1, alpha=0.4, label="1/k reference\n(would be peak-free)")
    ax.set_xlabel("sequence-axis frequency index k\n(0=DC, L/2=Nyquist)")
    ax.set_ylabel("power / DC component (log)")
    ax.set_title("(C) No spectral hiding place — both spectra\nare flat-and-white, not peaked")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "Phase 8 — Evo-1 7B on GenomicBenchmarks promoters: "
        "SAE on per-block delta activations vs JASPAR TFBS motifs",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
