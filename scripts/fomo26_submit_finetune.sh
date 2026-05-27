#!/usr/bin/env bash
# Submit FOMO26 finetuning/eval array for one checkpoint.

set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/fomo26_submit_finetune.sh <checkpoint_label> <checkpoint_path> [sbatch args...]

Examples:
  scripts/fomo26_submit_finetune.sh nima_pdf1m /data/nima.ashjaee/fomo26/share/pt902_pdf1m_resenc6m_55204/last.ckpt
  scripts/fomo26_submit_finetune.sh official_amaes /data/nima.ashjaee/fomo26/share/official_amaes_resenc_b_fomo300k/last.ckpt --array=0-4
  scripts/fomo26_submit_finetune.sh nima_pdf1m "$NIMA_CKPT" --array=1,3 --qos=low
EOF
}

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

ckpt_label="$1"
ckpt_path="$2"
shift 2

export SMRI_PROJ="${SMRI_PROJ:-/admin/home/$USER/smri-proj}"
export FOMO_ROOT="${FOMO_ROOT:-/data/$USER/fomo26}"

script="$SMRI_PROJ/asparagus/scripts/fomo26_finetune_array.sbatch"

if [[ ! -f "$script" ]]; then
  echo "Missing sbatch script: $script" >&2
  exit 1
fi

if [[ ! -f "$ckpt_path" ]]; then
  echo "Checkpoint not found: $ckpt_path" >&2
  exit 1
fi

mkdir -p "$FOMO_ROOT/slurms"

sbatch \
  --export=ALL,CKPT_LABEL="$ckpt_label",CKPT_PATH="$ckpt_path" \
  "$@" \
  "$script"
