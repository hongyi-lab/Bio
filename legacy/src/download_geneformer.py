"""Download Geneformer model + token dictionary from Hugging Face.

Default = V1 (6 layers, d=256, max_input=2048) — most-tested, smallest.

V2 variants (12L/512 or 20L/768) live under fine_tuned_models/ in the repo;
pass --variant to download those.

Usage:
    python src/download_geneformer.py                       # V1, ~500 MB
    python src/download_geneformer.py --variant v2-104m     # ~400 MB
    python src/download_geneformer.py --dest checkpoints/geneformer_v2_104m
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

VARIANTS = {
    # V1 = main branch defaults; weights are at repo root
    "v1": {
        "repo_id": "ctheodoris/Geneformer",
        "allow_patterns": [
            "config.json",
            "pytorch_model.bin",
            "model.safetensors",
            "tokenizer*",
            "geneformer/token_dictionary.pkl",
            "geneformer/gene_median_dictionary.pkl",
            "geneformer/gene_name_id_dict.pkl",
        ],
    },
    # V2 sub-checkpoints live in subfolders; the user can rebind paths after download.
    "v2-104m": {
        "repo_id": "ctheodoris/Geneformer",
        "subfolder": "gf-12L-95M-i4096",
        "allow_patterns": [
            "gf-12L-95M-i4096/*",
            "geneformer/*.pkl",
        ],
    },
    "v2-313m": {
        "repo_id": "ctheodoris/Geneformer",
        "subfolder": "gf-20L-95M-i4096",
        "allow_patterns": [
            "gf-20L-95M-i4096/*",
            "geneformer/*.pkl",
        ],
    },
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="v1", choices=list(VARIANTS))
    p.add_argument("--dest", default=None,
                   help="destination dir; defaults to checkpoints/geneformer[_<variant>]")
    args = p.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit(
            "huggingface_hub not installed. Run: pip install huggingface_hub"
        )

    cfg = VARIANTS[args.variant]
    dest = Path(args.dest) if args.dest else (
        Path("checkpoints") / (
            "geneformer" if args.variant == "v1"
            else f"geneformer_{args.variant.replace('-', '_')}"
        )
    )
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[download] target dir: {dest}")
    print(f"[download] repo: {cfg['repo_id']}, variant: {args.variant}")

    local_dir = snapshot_download(
        repo_id=cfg["repo_id"],
        local_dir=str(dest),
        allow_patterns=cfg["allow_patterns"],
        local_dir_use_symlinks=False,
    )
    print(f"[download] cached to {local_dir}")

    # For V2, shove the weights up to the top level so the adapter's
    # BertModel.from_pretrained(model_dir) works without --subfolder.
    sub = cfg.get("subfolder")
    if sub is not None:
        sub_dir = dest / sub
        if sub_dir.exists():
            for f in sub_dir.iterdir():
                target = dest / f.name
                if not target.exists():
                    shutil.copy2(f, target)
            print(f"[download] copied {sub}/* up to top level")

    # token_dictionary.pkl: HF puts it under geneformer/ subdir; mirror to top
    td_sub = dest / "geneformer" / "token_dictionary.pkl"
    td_top = dest / "token_dictionary.pkl"
    if td_sub.exists() and not td_top.exists():
        shutil.copy2(td_sub, td_top)
        print(f"[download] mirrored {td_sub} -> {td_top}")

    # Quick verification: must have config.json + (pytorch_model.bin or safetensors) + token_dictionary.pkl
    must = ["config.json", "token_dictionary.pkl"]
    missing = [f for f in must if not (dest / f).exists()]
    weights_ok = (
        (dest / "pytorch_model.bin").exists() or (dest / "model.safetensors").exists()
    )
    if missing or not weights_ok:
        print(f"[download] WARNING: missing {missing}, weights={weights_ok}")
        print(f"[download] you may need to copy files manually from subfolders under {dest}")
        sys.exit(1)
    else:
        print(f"[download] OK — ready for: "
              f"python -m bio_fm_probe.run_audit --adapter geneformer --model_dir {dest} ...")


if __name__ == "__main__":
    main()
