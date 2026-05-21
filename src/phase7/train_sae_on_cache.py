"""train_sae_on_cache.py — train a TopK SAE on a single activation cache.

Standalone driver so the phase 8 orchestrator can call it once per
(layer, operator) cache via subprocess. Resume-aware: if sae.pt already
exists in --out_dir and --force isn't set, the script reports the cache
state and exits 0 without retraining.

Uses common_sae.train_topk_sae (includes dead-feature resampling).

Usage:
    python src/phase7/train_sae_on_cache.py \
        --cache_path results/.../layer_16_hyena/token_activations.npz \
        --out_dir    results/.../layer_16_hyena/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent.parent
sys.path.insert(0, str(THIS.parent))


def _resolve(p: str) -> str:
    path = Path(p)
    return str(path if path.is_absolute() else ROOT / path)


from common_sae import TopKSAE, train_topk_sae  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache_path", required=True,
                   help="token_activations.npz produced by extract_delta_activations.py "
                        "or evo_resume.py")
    p.add_argument("--out_dir", required=True,
                   help="where sae.pt + sae_training_log.json land")
    p.add_argument("--sae_dict", type=int, default=16384,
                   help="SAE dictionary size (4x default for d_model=4096)")
    p.add_argument("--sae_k", type=int, default=32)
    p.add_argument("--sae_epochs", type=int, default=20)
    p.add_argument("--sae_batch", type=int, default=4096)
    p.add_argument("--sae_lr", type=float, default=1e-3)
    p.add_argument("--resample_every", type=int, default=6000,
                   help="dead-feature resampling cadence in steps; 0 = disabled")
    p.add_argument("--resample_buffer", type=int, default=16384)
    p.add_argument("--force", action="store_true",
                   help="re-train even if sae.pt exists")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cache_path = Path(_resolve(args.cache_path))
    out_dir = Path(_resolve(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    sae_ckpt = out_dir / "sae.pt"
    log_path = out_dir / "sae_training_log.json"

    if sae_ckpt.exists() and not args.force:
        sz = sae_ckpt.stat().st_size / 1e6
        print(f"[sae-driver] {sae_ckpt} exists ({sz:.1f} MB) — skip "
              f"(--force to re-train)")
        if log_path.exists():
            log = json.loads(log_path.read_text())
            ve = log.get("epoch_var_explained", [None])[-1]
            dead_ever = log.get("dead_features_ever")
            n_res = log.get("total_features_resampled", 0)
            print(f"[sae-driver]   cached: final var_exp={ve}, "
                  f"dead_ever={dead_ever}, resampled={n_res}")
        return 0

    if not cache_path.exists():
        raise SystemExit(f"cache not found: {cache_path}")

    sz_gb = cache_path.stat().st_size / 1e9
    print(f"[sae-driver] mmap-loading {cache_path} ({sz_gb:.2f} GB)")
    d = np.load(cache_path, allow_pickle=False, mmap_mode="r")
    activations = d["acts"]
    n_tokens, d_in = activations.shape
    print(f"[sae-driver]   acts: shape={activations.shape}, dtype={activations.dtype}")
    print(f"[sae-driver] SAE config: d_in={d_in}, dict={args.sae_dict}, "
          f"k={args.sae_k}, epochs={args.sae_epochs}, "
          f"resample_every={args.resample_every}")

    t0 = time.time()
    sae, log = train_topk_sae(
        activations, d_in=d_in,
        n_features=args.sae_dict, k=args.sae_k,
        batch_size=args.sae_batch, epochs=args.sae_epochs,
        lr=args.sae_lr, device=args.device,
        resample_every=args.resample_every,
        resample_buffer_size=args.resample_buffer,
    )
    train_time = time.time() - t0

    # Save checkpoint + log
    torch.save({
        "state_dict": sae.state_dict(),
        "config": {"d_in": d_in, "n_features": args.sae_dict, "k": args.sae_k},
        "cache_path": str(cache_path),
        "training_log_summary": {
            "final_loss": log["epoch_loss"][-1],
            "final_var_explained": log["epoch_var_explained"][-1],
            "final_dead_features": log["epoch_dead_features"][-1],
            "dead_features_ever": log["dead_features_ever"],
            "total_resample_events": log.get("total_resample_events", 0),
            "total_features_resampled": log.get("total_features_resampled", 0),
            "train_time_s": train_time,
        },
    }, sae_ckpt)
    log_path.write_text(json.dumps(log, indent=2))

    print(f"\n[sae-driver] === DONE in {train_time/60:.1f} min ===")
    print(f"[sae-driver]   final var_exp = {log['epoch_var_explained'][-1]:.4f}")
    print(f"[sae-driver]   final L0      = {log['epoch_l0_mean'][-1]:.1f}")
    print(f"[sae-driver]   dead_ever     = {log['dead_features_ever']}/{args.sae_dict} "
          f"({100*log['dead_features_ever']/args.sae_dict:.1f}%)")
    print(f"[sae-driver]   resample events = {log.get('total_resample_events', 0)}, "
          f"total features resampled = {log.get('total_features_resampled', 0)}")
    print(f"[sae-driver]   saved {sae_ckpt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
