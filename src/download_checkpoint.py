"""Download scGPT whole-human pretrained checkpoint to checkpoints/scGPT_human/.

Files: args.json, vocab.json, best_model.pt (~1.5 GB)
Source: official Google Drive folder linked from https://github.com/bowang-lab/scGPT
"""
from __future__ import annotations

import sys
from pathlib import Path

import gdown

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "checkpoints" / "scGPT_human"

WHOLE_HUMAN_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y"
)

REQUIRED_FILES = ["args.json", "vocab.json", "best_model.pt"]


def already_complete(out_dir: Path) -> bool:
    return all((out_dir / f).exists() and (out_dir / f).stat().st_size > 0 for f in REQUIRED_FILES)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if already_complete(OUT_DIR):
        print(f"[skip] checkpoint already complete at {OUT_DIR}")
        for f in REQUIRED_FILES:
            size_mb = (OUT_DIR / f).stat().st_size / 1e6
            print(f"  {f}: {size_mb:.2f} MB")
        return 0

    print(f"Downloading scGPT whole-human checkpoint to {OUT_DIR}")
    print(f"Source folder: {WHOLE_HUMAN_FOLDER_URL}")
    gdown.download_folder(
        url=WHOLE_HUMAN_FOLDER_URL,
        output=str(OUT_DIR),
        quiet=False,
        use_cookies=False,
    )

    missing = [f for f in REQUIRED_FILES if not (OUT_DIR / f).exists()]
    if missing:
        print(f"[ERROR] missing files after download: {missing}", file=sys.stderr)
        print(
            "Try downloading individually from the Google Drive folder, or use huggingface_hub:\n"
            "  python -c \"from huggingface_hub import snapshot_download; "
            "snapshot_download('your/scgpt-mirror', local_dir='checkpoints/scGPT_human')\"",
            file=sys.stderr,
        )
        return 1

    print("Download complete:")
    for f in REQUIRED_FILES:
        size_mb = (OUT_DIR / f).stat().st_size / 1e6
        print(f"  {f}: {size_mb:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
