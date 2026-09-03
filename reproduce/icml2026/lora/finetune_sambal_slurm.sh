#!/bin/bash
#SBATCH --job-name=finetune_sambal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --output=/dev/null
#SBATCH --mem=16gb

set -euo pipefail

# Output goes to logs/<name>-<id>.out under the current directory: the slurm
# job name and id, or this script's name and a timestamp (under bash the
# output stays on the terminal as well).
mkdir -p logs
LOG_NAME=$(basename "$0")
LOG_NAME="${SLURM_JOB_NAME:-${LOG_NAME%.*}}"
LOG="logs/$LOG_NAME-${SLURM_JOB_ID:-$(date +%Y%m%d%H%M%S)}.out"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    exec >"$LOG" 2>&1
else
    exec > >(tee "$LOG") 2>&1
fi
export PYTHONUNBUFFERED=1   # the log fills as the run goes

if command -v module >/dev/null 2>&1; then  # clusters with environment modules; no-op elsewhere
    module purge
    module load "${SAMBAL_CUDA_MODULE:-cuda-12.9}"
fi

# Submit from the repository root, or export REPO_ROOT to the checkout path.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
GPTBERT="$REPO_ROOT/lm/gpt-bert"
DATA_DIR="${SAMBAL_DATA_DIR:-$REPO_ROOT/data}"
# Python environment: source $SAMBAL_ENV if set (a venv activate script),
# else a repo-root .venv/ if present, else run with the ambient python.
if [ -n "${SAMBAL_ENV:-}" ]; then
    source "$SAMBAL_ENV"
elif [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
fi
cd "$GPTBERT/pretraining"
export WANDB_MODE="${WANDB_MODE:-offline}"  # offline | online | disabled (see reproduce/icml2026/README.md)

LORAR="$1"
ALPHA="$2"
RUN=lotr_lora_r_${LORAR}_a_${ALPHA}_sambal
CKPT=../trained_models/gptbert_sambal_long_ema.bin
# Optional third argument: another base checkpoint (a relative path resolves
# from the repository root); the run directory is then suffixed with its name.
if [ $# -ge 3 ]; then
    CKPT="$3"
    case "$CKPT" in
        /*) ;;
        *) CKPT="$REPO_ROOT/$CKPT" ;;
    esac
    RUN=${RUN}_$(basename "$CKPT" .bin)
fi

python finetune_lora.py \
  --config_file ../configs/small.json \
  --tokenizer_path ../gpt-bert-babylm-small/tokenizer.json \
  --checkpoint $CKPT \
  --train_path $DATA_DIR/lotr/lotr_train.bin \
  --dev_path   $DATA_DIR/lotr/lotr_dev.bin \
  --test_path  $DATA_DIR/lotr/lotr_test.bin \
  --dataset_type causal \
  --seq_length 256 \
  --batch_size 32 \
  --epochs 100 \
  --lr 1e-3 \
  --use_lora \
  --lora_r $LORAR \
  --lora_alpha $ALPHA \
  --out_dir ft_out/$RUN \
  --train_embeddings \
  --enable_val_ppl_early_stop \
  --val_ppl_early_stop_patience 10 \
  --enable_blimp \
  --blimp_val_data_path $REPO_ROOT/evals/blimp/blimp_really_fast \
  --blimp_test_data_path $REPO_ROOT/evals/blimp/data \

