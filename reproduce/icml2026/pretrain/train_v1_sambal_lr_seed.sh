#!/bin/bash
#SBATCH --job-name=train_v1_sambal_lr_seed
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --output=/dev/null
#SBATCH --mem=16G

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

if [ $# -lt 2 ]; then
    echo "Usage: $0 <learning_rate> <seed> [train_bin]"
    echo "Example: $0 0.007 42"
    echo "  train_bin defaults to the plain ablated corpus; pass the"
    echo "  reflexives+number-augmented bin to run the augmentation retrain"
    echo "  (reproduce/icml2026/TRAINING.md, 2.3 and 2.6)."
    echo "  A relative train_bin path resolves from the repository root."
    exit 1
fi

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

LEARNING_RATE="$1"
SEED="$2"
if [ $# -ge 3 ]; then
    DATA_PATH="$3"
    case "$DATA_PATH" in
        /*) ;;
        *) DATA_PATH="$REPO_ROOT/$DATA_PATH" ;;  # docs pass repo-root-relative paths
    esac
    DATA_NAME=_$(basename "$DATA_PATH" .bin)
else
    DATA_PATH="$DATA_DIR/babycosmofine_sambal/train_sambal_10M_tokenized.bin"
    DATA_NAME=""
fi

# The final test evaluation reads evals/blimp/data and
# evals/syntaxgym/data/syntaxgym — fetch both once first:
# python evals/blimp/fetch_data.py; python evals/syntaxgym/fetch_data.py
python train_v1.py \
    --train_path=$DATA_PATH \
    --valid_path=$DATA_PATH \
    --config_file=$GPTBERT/configs/small.json \
    --tokenizer_path=$GPTBERT/gpt-bert-babylm-small/tokenizer.json \
    --output_dir=$GPTBERT/trained_models \
    --name=gptbert_sambal_short${DATA_NAME}_lr_${LEARNING_RATE}_seed_${SEED} \
    --hybrid_numerator=1 \
    --hybrid_denominator=8 \
    --batch_size=128 \
    --learning_rate=$LEARNING_RATE \
    --epochs=10 \
    --seed=$SEED \
    --seq_ramp_60_80 \
    --blimp_data_path $REPO_ROOT/evals/blimp/blimp_fast \
    --blimp_eval_freq 1 \
    --syntaxgym_data_path $REPO_ROOT/evals/syntaxgym/syntaxgym_fast \
    --syntaxgym_test_data_path $REPO_ROOT/evals/syntaxgym/data/syntaxgym \
    --blimp_test_data_path $REPO_ROOT/evals/blimp/data \
    --enable_final_eval