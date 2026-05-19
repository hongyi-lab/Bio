#!/bin/bash
# Orchestrate the full cross-model audit.
#
# Steps:
#   1. Download each model's weights (skip if already on disk).
#   2. Run the standard audit pipeline on each.
#   3. Emit a cross-model comparison table.
#
# Models flagged as "stub" or "checkpoint not public" are skipped automatically
# until their adapter is wired in run_audit.ADAPTER_REGISTRY.
#
# Usage:
#   bash src/run_all_audits.sh                                # default models
#   MODELS="scgpt geneformer" bash src/run_all_audits.sh      # explicit subset
#   DATA=data/pbmc3k.h5ad bash src/run_all_audits.sh
#
# Env knobs:
#   DATA           default: data/pbmc3k.h5ad
#   LABEL_COL      default: louvain
#   MODELS         default: "scgpt geneformer"   (space-separated)
#   SKIP_SAE       default: 0 (set to 1 for a quick first-pass; saves ~2h/model)
#   DEVICE         default: cuda
#   GENEFORMER_VARIANT  default: v1 (v1 / v2-104m / v2-313m)

set -euo pipefail

DATA="${DATA:-data/pbmc3k.h5ad}"
LABEL_COL="${LABEL_COL:-louvain}"
MODELS="${MODELS:-scgpt geneformer}"
SKIP_SAE="${SKIP_SAE:-0}"
DEVICE="${DEVICE:-cuda}"
GENEFORMER_VARIANT="${GENEFORMER_VARIANT:-v1}"

EXTRA_ARGS=""
if [[ "$SKIP_SAE" == "1" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --skip_sae"
fi

echo "=================================================="
echo "[run_all] DATA=$DATA  LABEL_COL=$LABEL_COL  DEVICE=$DEVICE"
echo "[run_all] MODELS=$MODELS  SKIP_SAE=$SKIP_SAE"
echo "=================================================="

# Helper: download a model if its checkpoint dir is missing.
download_if_missing() {
    local model="$1"
    local ckpt_dir="$2"
    local download_cmd="$3"
    if [[ -d "$ckpt_dir" ]] && ls "$ckpt_dir" 2>/dev/null | grep -q .; then
        echo "[run_all] [$model] checkpoint exists at $ckpt_dir — skip download"
    else
        echo "[run_all] [$model] downloading checkpoint..."
        eval "$download_cmd"
    fi
}

# Helper: audit a model.
audit_model() {
    local model="$1"
    local ckpt_dir="$2"
    local extra="${3:-}"
    echo ""
    echo "[run_all] [$model] >>>>>>>>> AUDIT START <<<<<<<<<"
    python -m bio_fm_probe.run_audit \
        --adapter "$model" \
        --model_dir "$ckpt_dir" \
        --data "$DATA" \
        --label_col "$LABEL_COL" \
        --device "$DEVICE" \
        $extra $EXTRA_ARGS
    echo "[run_all] [$model] >>>>>>>>> AUDIT DONE  <<<<<<<<<"
}

# ----- scGPT -----
if [[ " $MODELS " == *" scgpt "* ]]; then
    download_if_missing \
        "scgpt" \
        "checkpoints/scGPT_human" \
        "python src/download_checkpoint.py"
    audit_model "scgpt" "checkpoints/scGPT_human"
fi

# ----- Geneformer -----
if [[ " $MODELS " == *" geneformer "* ]]; then
    if [[ "$GENEFORMER_VARIANT" == "v1" ]]; then
        GF_DIR="checkpoints/geneformer"
    else
        GF_DIR="checkpoints/geneformer_${GENEFORMER_VARIANT//-/_}"
    fi
    download_if_missing \
        "geneformer" \
        "$GF_DIR" \
        "python src/download_geneformer.py --variant $GENEFORMER_VARIANT"
    audit_model "geneformer" "$GF_DIR"
fi

# ----- scFoundation -----   (un-comment once adapter is wired)
# if [[ " $MODELS " == *" scfoundation "* ]]; then
#     download_if_missing \
#         "scfoundation" \
#         "checkpoints/scfoundation" \
#         "python src/download_scfoundation.py"
#     audit_model "scfoundation" "checkpoints/scfoundation"
# fi

# ----- scBERT -----   (un-comment once adapter is wired)
# if [[ " $MODELS " == *" scbert "* ]]; then
#     download_if_missing \
#         "scbert" \
#         "checkpoints/scbert" \
#         "python src/download_scbert.py"
#     audit_model "scbert" "checkpoints/scbert"
# fi

# ----- UCE -----   (un-comment once adapter is wired)
# if [[ " $MODELS " == *" uce "* ]]; then
#     download_if_missing \
#         "uce" \
#         "checkpoints/uce" \
#         "python src/download_uce.py"
#     audit_model "uce" "checkpoints/uce"
# fi

# ----- Cross-model comparison -----
echo ""
echo "[run_all] >>>>>>>>> CROSS-MODEL SUMMARY <<<<<<<<<"
# pick the models that successfully produced AUDIT.md
DONE_MODELS=""
for m in $MODELS; do
    if [[ -f "results/$m/audit/AUDIT.md" ]]; then
        DONE_MODELS="$DONE_MODELS $m"
    fi
done
DONE_MODELS=$(echo "$DONE_MODELS" | xargs)
if [[ -n "$DONE_MODELS" ]]; then
    python -m bio_fm_probe.compare_models $DONE_MODELS
    echo ""
    echo "[run_all] DONE. See results/_cross_model_summary.md"
else
    echo "[run_all] no model produced AUDIT.md; skipping comparison."
fi
