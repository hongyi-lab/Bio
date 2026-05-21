"""run_phase8.py — overnight orchestrator for SAE-on-delta + JASPAR TFBS hit rate.

Pipeline:
    1. Download JASPAR motifs (~2 MB, skipped if file already present)
    2. Extract delta activations for selected (Hyena, attention) layer pairs
    3. Train a TopK SAE on each delta cache (with dead-feature resampling)
    4. Evaluate each SAE against JASPAR motifs (per-feature TFBS hit rate)
    5. Aggregate a cross-layer SUMMARY.md

Resume: each step's per-layer outputs are checked. If the expected output
file already exists the step is skipped for that layer. Re-run with FORCE=1.

Expected wall time on A6000 (full run, 3 pairs = 6 SAEs):
    Step 1 (JASPAR download):        < 30 s
    Step 2 (delta extraction):       ~9 h total (~1.5 h per layer × 6)
    Step 3 (SAE training):           ~3 h total (~30 min per layer × 6)
    Step 4 (JASPAR eval):            ~3 h total (~30 min per layer × 6)
    Step 5 (summary):                < 10 s
    Total:                           ~15 h overnight + morning

Disk cost:
    Step 1: 2 MB
    Step 2: ~480 GB (80 GB fp32 cache × 6 layers)
    Step 3: ~3.2 GB (~530 MB sae.pt × 6)
    Step 4: ~15 MB (jaspar_hits.json × 6)

Two modes:
    Default (full):
        python src/phase7/run_phase8.py
    Dry-run (~30-60 min, ONE pair + tiny config):
        python src/phase7/run_phase8.py --dry_run

Recommended workflow:
    1. python src/phase7/run_phase8.py --dry_run
       Look for "[orchestrator] DRY RUN PASSED" at the end.
    2. If dry-run failed: STOP. Debug. Do not launch the full run.
    3. If dry-run passed: detach the full run:
         nohup python src/phase7/run_phase8.py \
             > logs/phase8_full.log 2>&1 &
         echo "PID: $!"

Outputs:
    data/jaspar/JASPAR2024_CORE_vertebrates.meme
    results/evo_1_8k__genomic_benchmarks_delta/<layer>_<kind>/
        ├── token_activations.npz    (gitignored)
        ├── sae.pt                   (gitignored)
        ├── sae_training_log.json
        ├── jaspar_hits.json
    results/evo_1_8k__genomic_benchmarks_delta/SUMMARY.md
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import List, Tuple

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent.parent
os.chdir(ROOT)
PYTHON = sys.executable


# ============================================================================
# Layer config — adjacent Hyena/attention pairs (see extract_delta_activations.py)
# ============================================================================
ATTN_LAYER_IDXS_0BASED = {8, 16, 24}

FULL_PAIRS: List[Tuple[str, str]] = [
    ("layer_08", "layer_09"),
    ("layer_16", "layer_17"),
    ("layer_24", "layer_25"),
]
DRY_PAIRS: List[Tuple[str, str]] = [
    ("layer_16", "layer_17"),
]


def kind_of(layer_name: str) -> str:
    idx_1based = int(layer_name.split("_")[1])
    return "attention" if (idx_1based - 1) in ATTN_LAYER_IDXS_0BASED else "hyena"


# ============================================================================
# JASPAR motif download
# ============================================================================
JASPAR_URL = (
    "https://jaspar.genereg.net/download/data/2024/CORE/"
    "JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt"
)
JASPAR_DEST = "data/jaspar/JASPAR2024_CORE_vertebrates.meme"


def download_jaspar(force: bool = False) -> Path:
    dst = ROOT / JASPAR_DEST
    if dst.exists() and dst.stat().st_size > 0 and not force:
        sz_kb = dst.stat().st_size / 1024
        print(f"[step 1/5] [jaspar] {dst} exists ({sz_kb:.0f} KB) — skip download")
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[step 1/5] [jaspar] downloading {JASPAR_URL} ...")
    try:
        urllib.request.urlretrieve(JASPAR_URL, dst)
    except Exception as e:
        raise SystemExit(f"[step 1/5] JASPAR download failed: {e}\n"
                         f"Manually save the MEME file to {dst} and re-run.")
    sz_kb = dst.stat().st_size / 1024
    print(f"[step 1/5] [jaspar] saved {dst} ({sz_kb:.0f} KB)")
    return dst


# ============================================================================
# Subprocess helper (streams stdout/stderr through so tqdm bars stay visible)
# ============================================================================
def _section(msg: str) -> None:
    print("=" * 72)
    print(f"[orchestrator] {msg}")
    print("=" * 72, flush=True)


def _run(cmd: list, what: str) -> None:
    print(f"[orchestrator] $ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"\n[orchestrator] FAIL: {what} (exit code {e.returncode}, "
              f"after {(time.time()-t0)/60:.1f} min)", file=sys.stderr)
        sys.exit(e.returncode)
    print(f"[orchestrator] OK: {what} ({(time.time()-t0)/60:.1f} min)\n", flush=True)


# ============================================================================
# Step 2 — delta extraction (one call, handles per-layer caching internally)
# ============================================================================
def step_extract(pairs: List[Tuple[str, str]], cfg: dict, force: bool) -> None:
    _section(f"step 2/5: delta extraction for {len(pairs)} pair(s) "
             f"= {2*len(pairs)} layers")
    all_layers = [l for pair in pairs for l in pair]
    cmd = [
        PYTHON, "src/phase7/extract_delta_activations.py",
        "--layers", *all_layers,
        "--max_samples", str(cfg["max_samples"]),
        "--max_len", str(cfg["max_len"]),
        "--extract_batch_size", str(cfg["extract_batch_size"]),
        "--dtype", cfg["dtype"],
    ]
    if force:
        cmd.append("--force")
    _run(cmd, "delta extraction")


# ============================================================================
# Step 3 — train SAE on each delta cache
# ============================================================================
def step_train_saes(pairs: List[Tuple[str, str]], cfg: dict, force: bool) -> None:
    _section(f"step 3/5: train SAE on {2*len(pairs)} delta caches")
    delta_root = ROOT / "results" / "evo_1_8k__genomic_benchmarks_delta"
    for i, pair in enumerate(pairs, 1):
        for layer in pair:
            kind = kind_of(layer)
            cache = delta_root / f"{layer}_{kind}" / "token_activations.npz"
            out_dir = delta_root / f"{layer}_{kind}"
            sae_ckpt = out_dir / "sae.pt"
            label = f"pair {i}/{len(pairs)}: {layer} ({kind})"
            if not cache.exists():
                print(f"[orchestrator] ❌ missing cache {cache} — step 2 didn't produce it. "
                      f"Halting.", file=sys.stderr)
                sys.exit(1)
            if sae_ckpt.exists() and not force:
                print(f"[orchestrator] skip SAE train for {label} (sae.pt exists)")
                continue
            cmd = [
                PYTHON, "src/phase7/train_sae_on_cache.py",
                "--cache_path", str(cache),
                "--out_dir", str(out_dir),
                "--sae_dict", str(cfg["sae_dict"]),
                "--sae_k", str(cfg["sae_k"]),
                "--sae_epochs", str(cfg["sae_epochs"]),
                "--sae_batch", str(cfg["sae_batch"]),
                "--resample_every", str(cfg["resample_every"]),
            ]
            if force:
                cmd.append("--force")
            _run(cmd, f"SAE train [{label}]")


# ============================================================================
# Step 4 — JASPAR evaluation on each trained SAE
# ============================================================================
def step_eval_jaspar(pairs: List[Tuple[str, str]], cfg: dict, jaspar_path: Path,
                     force: bool) -> None:
    _section(f"step 4/5: JASPAR TFBS eval on {2*len(pairs)} SAEs")
    delta_root = ROOT / "results" / "evo_1_8k__genomic_benchmarks_delta"
    for i, pair in enumerate(pairs, 1):
        for layer in pair:
            kind = kind_of(layer)
            out_dir = delta_root / f"{layer}_{kind}"
            sae_ckpt = out_dir / "sae.pt"
            cache = out_dir / "token_activations.npz"
            hits_json = out_dir / "jaspar_hits.json"
            label = f"pair {i}/{len(pairs)}: {layer} ({kind})"
            if not sae_ckpt.exists():
                print(f"[orchestrator] ❌ missing SAE {sae_ckpt} — step 3 didn't produce it. "
                      f"Halting.", file=sys.stderr)
                sys.exit(1)
            if hits_json.exists() and not force:
                print(f"[orchestrator] skip JASPAR eval for {label} (jaspar_hits.json exists)")
                continue
            cmd = [
                PYTHON, "src/phase7/evaluate_sae_jaspar.py",
                "--sae_ckpt", str(sae_ckpt),
                "--acts_cache", str(cache),
                "--jaspar_path", str(jaspar_path),
                "--out", str(hits_json),
                "--top_k_tokens", str(cfg["top_k_tokens"]),
                "--window_half_width", str(cfg["window_half_width"]),
                "--n_shuffles", str(cfg["n_shuffles"]),
                "--p_threshold", str(cfg["p_threshold"]),
                "--max_samples", str(cfg["max_samples"]),
                "--max_features", str(cfg["max_features"]),
            ]
            _run(cmd, f"JASPAR eval [{label}]")


# ============================================================================
# Step 5 — cross-layer SUMMARY.md
# ============================================================================
def step_summary(pairs: List[Tuple[str, str]], dry_run: bool) -> None:
    _section("step 5/5: cross-layer SUMMARY.md")
    delta_root = ROOT / "results" / "evo_1_8k__genomic_benchmarks_delta"
    rows = []
    for pair in pairs:
        for layer in pair:
            kind = kind_of(layer)
            out_dir = delta_root / f"{layer}_{kind}"
            sae_log = out_dir / "sae_training_log.json"
            hits = out_dir / "jaspar_hits.json"
            if not hits.exists():
                continue
            log = json.loads(sae_log.read_text()) if sae_log.exists() else {}
            h = json.loads(hits.read_text())
            rows.append({
                "layer": layer,
                "kind": kind,
                "live": h["n_features_live"],
                "n_total": h["n_features_total"],
                "live_frac": h["n_features_live"] / max(h["n_features_total"], 1),
                "hits": h["n_hits"],
                "hit_rate_live": h["hit_rate_among_live"],
                "var_exp": (log.get("epoch_var_explained") or [None])[-1],
                "dead_ever": log.get("dead_features_ever"),
                "resample_events": log.get("total_resample_events", 0),
            })

    md = [f"# Phase 8 — SAE-on-delta × JASPAR hit rate "
          f"({'DRY RUN' if dry_run else 'FULL'})\n"]
    md.append(f"Pairs evaluated: {len(pairs)} ({len(rows)} SAEs reported)\n")
    md.append("## Per-layer table\n")
    md.append("| layer | kind | live / total | var_exp | dead_ever | resample | hits | hit rate (live) |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        ve = r["var_exp"]
        ve_str = f"{ve:.4f}" if ve is not None else "n/a"
        md.append(
            f"| {r['layer']} | **{r['kind']}** "
            f"| {r['live']}/{r['n_total']} ({100*r['live_frac']:.1f}%) "
            f"| {ve_str} "
            f"| {r['dead_ever']} "
            f"| {r['resample_events']} "
            f"| {r['hits']} "
            f"| **{100*r['hit_rate_live']:.1f}%** |"
        )
    md.append("\n## Headline comparison (Hyena vs Attention)\n")
    md.append("Within each pair, compare the two kinds. If Hyena hit-rate ≈ Attention hit-rate "
              "across all pairs, the residual-stream linear-interface argument dominates and "
              "the 'operator-specific geometry' claim is unsupported. If they diverge, that's "
              "real evidence for an operator effect.\n")
    by_pair = {}
    for r in rows:
        idx = int(r["layer"].split("_")[1])
        pair_id = "early" if idx <= 9 else ("mid" if idx <= 17 else "late")
        by_pair.setdefault(pair_id, {})[r["kind"]] = r
    md.append("| pair | hyena hit rate | attention hit rate | Δ (att − hyena) |")
    md.append("|---|---|---|---|")
    for pair_id in ("early", "mid", "late"):
        if pair_id not in by_pair:
            continue
        h_r = by_pair[pair_id].get("hyena")
        a_r = by_pair[pair_id].get("attention")
        if not (h_r and a_r):
            continue
        delta = a_r["hit_rate_live"] - h_r["hit_rate_live"]
        md.append(
            f"| {pair_id} "
            f"| {100*h_r['hit_rate_live']:.1f}% ({h_r['layer']}) "
            f"| {100*a_r['hit_rate_live']:.1f}% ({a_r['layer']}) "
            f"| {100*delta:+.1f}pp |"
        )

    summary_path = delta_root / "SUMMARY.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(md))
    print(f"[orchestrator] wrote {summary_path}")
    print("\n".join(md[-15:]))


# ============================================================================
# Main
# ============================================================================
DRY_CONFIG = dict(
    max_samples=500,
    max_len=512,
    extract_batch_size=2,
    dtype="bf16",
    # SAE
    sae_dict=4096,                       # 4× smaller than full (16384/4)
    sae_k=32,
    sae_epochs=1,
    sae_batch=4096,
    resample_every=0,                    # disabled at 1-epoch scale
    # JASPAR eval — p-value floor = 1/(n_shuffles+1); threshold must be above
    # that to be achievable. Dry uses lenient threshold so we can verify the
    # plumbing detects hits if any exist.
    top_k_tokens=20,
    window_half_width=15,
    n_shuffles=200,
    p_threshold=0.01,                    # floor=0.005 with n_shuffles=200
    max_features=100,                    # cap so eval finishes fast
)

FULL_CONFIG = dict(
    max_samples=20000,
    max_len=512,
    extract_batch_size=2,
    dtype="bf16",
    # SAE
    sae_dict=16384,
    sae_k=32,
    sae_epochs=20,
    sae_batch=4096,
    resample_every=6000,
    # JASPAR eval — n_shuffles bumped to 1000 so we can use a strict
    # p<0.001 threshold (min achievable p = 1/1001 ≈ 0.001). Caveat: this
    # is the raw per-feature p-value; multiple-testing correction across
    # the (n_motifs × top_k_tokens) selection is a phase-9 problem.
    top_k_tokens=50,
    window_half_width=15,
    n_shuffles=1000,
    p_threshold=1e-3,
    max_features=0,                      # evaluate all live features
)


def main() -> int:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dry_run", action="store_true",
                   help="1 pair + tiny config, ~30-60 min. Use FIRST to verify pipeline.")
    p.add_argument("--force", action="store_true",
                   help="re-run every step, ignoring cached outputs")
    p.add_argument("--skip_extract", action="store_true",
                   help="skip step 2 (assume caches already on disk)")
    p.add_argument("--skip_train", action="store_true",
                   help="skip step 3 (assume sae.pt files already on disk)")
    p.add_argument("--skip_eval", action="store_true",
                   help="skip step 4 (just regenerate SUMMARY.md)")
    args = p.parse_args()

    cfg = DRY_CONFIG if args.dry_run else FULL_CONFIG
    pairs = DRY_PAIRS if args.dry_run else FULL_PAIRS

    mode = "DRY RUN" if args.dry_run else "FULL RUN"
    _section(f"phase 8 orchestrator: {mode}")
    print(f"[orchestrator] pairs = {pairs}")
    print(f"[orchestrator] config = {json.dumps(cfg, indent=2)}", flush=True)

    t_total = time.time()

    # Step 1: JASPAR
    jaspar_path = download_jaspar(force=args.force)

    # Step 2: delta extraction
    if not args.skip_extract:
        step_extract(pairs, cfg, args.force)
    else:
        print("[orchestrator] step 2 skipped via --skip_extract")

    # Step 3: SAE training
    if not args.skip_train:
        step_train_saes(pairs, cfg, args.force)
    else:
        print("[orchestrator] step 3 skipped via --skip_train")

    # Step 4: JASPAR eval
    if not args.skip_eval:
        step_eval_jaspar(pairs, cfg, jaspar_path, args.force)
    else:
        print("[orchestrator] step 4 skipped via --skip_eval")

    # Step 5: summary
    step_summary(pairs, dry_run=args.dry_run)

    elapsed = (time.time() - t_total) / 60
    print(f"\n[orchestrator] === ALL STEPS COMPLETED in {elapsed:.1f} min ===")
    if args.dry_run:
        print("[orchestrator] ✅ DRY RUN PASSED")
        print("[orchestrator] Next: launch the full run.")
        print("[orchestrator]    nohup python src/phase7/run_phase8.py "
              "> logs/phase8_full.log 2>&1 &")
    else:
        print("[orchestrator] ✅ FULL RUN COMPLETE — see SUMMARY.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
