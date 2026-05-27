#!/usr/bin/env bash
# Run FOMO26 downstream finetuning/eval for one checkpoint.
#
# Submit all five tasks:
#   scripts/fomo26_submit_finetune.sh nima_pdf1m /data/nima.ashjaee/fomo26/share/pt902_pdf1m_resenc6m_55204/last.ckpt
#
# Submit a subset by overriding the array, for example Task 2 and Task 4 only:
#   scripts/fomo26_submit_finetune.sh nima_pdf1m "$NIMA_CKPT" --array=1,3
#
# Task indices:
#   0 CLS002_FOMO26_Infarct
#   1 SEG009_FOMO26_Meningioma
#   2 REGR002_FOMO26_BrainAge
#   3 SEG010_FOMO26_TrigeminalNeuralgia
#   4 CLS003_FOMO26_Polymicrogyria

#SBATCH --job-name=fomo26-ft
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=72:00:00
#SBATCH --array=0-4
#SBATCH --output=/data/%u/fomo26/slurms/%x-%A_%a.out
#SBATCH --error=/data/%u/fomo26/slurms/%x-%A_%a.err

set -euo pipefail

export SMRI_PROJ="${SMRI_PROJ:-/admin/home/$USER/smri-proj}"
export FOMO_ROOT="${FOMO_ROOT:-/data/$USER/fomo26}"
export ASPARAGUS_SOURCE="${ASPARAGUS_SOURCE:-$FOMO_ROOT/raw/hf_data/datasets_downloaded_from_hf}"
export ASPARAGUS_CONFIGS="${ASPARAGUS_CONFIGS:-$SMRI_PROJ/asparagus/configs}"
export ASPARAGUS_DATA="${ASPARAGUS_DATA:-$FOMO_ROOT/processed_data}"
export ASPARAGUS_MODELS="${ASPARAGUS_MODELS:-$FOMO_ROOT/models}"
export ASPARAGUS_RESULTS="${ASPARAGUS_RESULTS:-$FOMO_ROOT/results}"
export ASPARAGUS_RAW_LABELS="${ASPARAGUS_RAW_LABELS:-$FOMO_ROOT/raw_labels}"
export WANDB_ENTITY="${WANDB_ENTITY:-nimaashjaee-university-of-british-columbia}"
export WANDB_LOGGING="${WANDB_LOGGING:-true}"
export PYTHONUNBUFFERED=1

: "${CKPT_PATH:?Set CKPT_PATH=/path/to/pretrain.ckpt}"
: "${CKPT_LABEL:?Set CKPT_LABEL=short_checkpoint_label}"

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "Checkpoint not found: $CKPT_PATH" >&2
  exit 1
fi

mkdir -p "$FOMO_ROOT/slurms" "$ASPARAGUS_MODELS/fomo26_eval/$CKPT_LABEL"

source "$FOMO_ROOT/venvs/fomo_env/bin/activate"
cd "$SMRI_PROJ/asparagus"

tasks=(
  CLS002_FOMO26_Infarct
  SEG009_FOMO26_Meningioma
  REGR002_FOMO26_BrainAge
  SEG010_FOMO26_TrigeminalNeuralgia
  CLS003_FOMO26_Polymicrogyria
)

commands=(
  asp_finetune_cls
  asp_finetune_seg
  asp_finetune_reg
  asp_finetune_seg
  asp_finetune_cls
)

models=(
  resenc_unet_b_clsreg
  resenc_unet_b
  resenc_unet_b_clsreg
  resenc_unet_b
  resenc_unet_b_clsreg
)

i="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}"
task="${tasks[$i]}"
cmd="${commands[$i]}"
model="${models[$i]}"
run_dir="$ASPARAGUS_MODELS/fomo26_eval/$CKPT_LABEL/${task}/job_${SLURM_JOB_ID}_${i}"

echo "Running FOMO26 task index $i"
echo "  task:       $task"
echo "  command:    $cmd"
echo "  model:      $model"
echo "  checkpoint: $CKPT_PATH"
echo "  run_dir:    $run_dir"

common_overrides=(
  "task=${task}"
  "+model=${model}"
  "checkpoint_path=${CKPT_PATH}"
  "data.train_split=${TRAIN_SPLIT:-split_80_10_10}"
  "data.test_split=${TEST_SPLIT:-TEST_80_10_10}"
  "data.fold=${FOLD:-0}"
  "root=${ROOT:-fomo26_eval}"
  "stem=${CKPT_LABEL}"
  "hydra.run.dir=${run_dir}"
  "hardware.num_workers=${SLURM_CPUS_PER_TASK}"
  "training.batch_size=${BATCH_SIZE:-2}"
  "logger.wandb_entity=${WANDB_ENTITY}"
  "logger.wandb_logging=${WANDB_LOGGING}"
  "logger.log_to_stdout=true"
)

case "$cmd" in
  asp_finetune_seg)
    task_overrides=(
      "training.epochs=${SEG_EPOCHS:-150}"
      "training.warmup_epochs=${SEG_WARMUP_EPOCHS:-10}"
      "training.decoder_warmup_epochs=${SEG_DECODER_WARMUP_EPOCHS:-10}"
      "training.patch_size=${SEG_PATCH_SIZE:-[160,160,160]}"
    )
    ;;
  asp_finetune_cls)
    task_overrides=(
      "training.epochs=${CLS_EPOCHS:-50}"
    )
    ;;
  asp_finetune_reg)
    task_overrides=(
      "training.epochs=${REG_EPOCHS:-50}"
    )
    ;;
  *)
    echo "Unknown command: $cmd" >&2
    exit 1
    ;;
esac

srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  "$cmd" "${common_overrides[@]}" "${task_overrides[@]}"
