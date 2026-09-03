#!/bin/bash
#SBATCH --job-name=pubmed_exp
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --output=/dev/null
#SBATCH --mem=16G

# Single PUBMED training experiment
# Called by sweep_pubmed.sh with parameters passed via environment variables:
#   EXP_NAME, EXP_CONFIG, EXP_BATCH_SIZE, EXP_LR, EXP_EPOCHS

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

# Common paths
TRAIN_PATH=$DATA_DIR/pubmed/pubmed_abstracts_train.bin
VALID_PATH=$DATA_DIR/pubmed/pubmed_abstracts_dev.bin
TEST_PATH=$DATA_DIR/pubmed/pubmed_abstracts_test.bin
TOKENIZER=$GPTBERT/gpt-bert-babylm-small/tokenizer.json
OUTPUT_DIR=$GPTBERT/trained_models
BLIMP_VAL_PATH=$REPO_ROOT/evals/blimp/blimp_fast
BLIMP_TEST_PATH=$REPO_ROOT/evals/blimp/data

echo "========================================"
echo "Running PUBMED experiment: ${EXP_NAME}"
echo "  Config: ${EXP_CONFIG}"
echo "  Batch size: ${EXP_BATCH_SIZE}"
echo "  Learning rate: ${EXP_LR}"
echo "  Epochs: ${EXP_EPOCHS}"
echo "  Train: ${TRAIN_PATH}"
echo "  Valid: ${VALID_PATH}"
echo "  Test: ${TEST_PATH}"
echo "========================================"
date

python train_lotr.py \
    --train_path=$TRAIN_PATH \
    --valid_path=$VALID_PATH \
    --test_path=$TEST_PATH \
    --config_file=$EXP_CONFIG \
    --tokenizer_path=$TOKENIZER \
    --output_dir=$OUTPUT_DIR \
    --name=$EXP_NAME \
    --hybrid_numerator=1 \
    --hybrid_denominator=8 \
    --batch_size=$EXP_BATCH_SIZE \
    --seq_length=128 \
    --learning_rate=$EXP_LR \
    --epochs=$EXP_EPOCHS \
    --seed=42 \
    --seq_ramp_60_80 \
    --enable_blimp \
    --blimp_final_only \
    --blimp_data_path=$BLIMP_VAL_PATH \
    --blimp_test_data_path=$BLIMP_TEST_PATH \
    --blimp_eval_freq=10

echo "========================================"
echo "Completed: ${EXP_NAME}"
date
