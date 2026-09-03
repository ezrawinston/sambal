#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --time=8:00:00
#SBATCH --partition=preempt
#SBATCH --output=/dev/null
#SBATCH --mem=16G

set -euo pipefail

# Output goes to logs/<name>-<id>.out under the current directory: the slurm
# job name and id, or this script's name and a timestamp (under bash the
# output stays on the terminal as well).
mkdir -p logs
if [ -n "${SLURM_JOB_NAME:-}" ]; then
    LOG_NAME="$SLURM_JOB_NAME"
elif [ $# -eq 4 ]; then
    LOG_NAME="scaling_$1_$2_$3_seed$4"   # the launcher's --job-name for this cell
else
    LOG_NAME=$(basename "$0"); LOG_NAME="${LOG_NAME%.*}"
fi
LOG="logs/$LOG_NAME-${SLURM_JOB_ID:-$(date +%Y%m%d%H%M%S)}.out"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    exec >"$LOG" 2>&1
else
    exec > >(tee "$LOG") 2>&1
fi
export PYTHONUNBUFFERED=1   # the log fills as the run goes

# Usage: sbatch --job-name=<name> train_scaling_ablation.sh <data_type> <model_size> <data_fraction> <seed>
#   data_type: standard | sambal
#   model_size: 05m | 10m | 14m | 30m
#   data_fraction: 0.05 | 0.1 | 0.15 | 0.2 | 0.25 | 0.3 | 0.5 | 1.0
#   seed: random seed (e.g., 42, 123, 456)
#
# Use launch_scaling_ablation.sh to submit all experiment combinations

if [ $# -ne 4 ]; then
    echo "Usage: sbatch --job-name=<name> train_scaling_ablation.sh <data_type> <model_size> <data_fraction> <seed>"
    echo "  data_type: standard | sambal"
    echo "  model_size: 05m | 10m | 14m | 30m"
    echo "  data_fraction: 0.05 | 0.1 | 0.15 | 0.2 | 0.25 | 0.3 | 0.5 | 1.0"
    echo "  seed: random seed (e.g., 42, 123, 456)"
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

# Data paths
STANDARD_TRAIN=$DATA_DIR/babycosmofine/train_10M_tokenized.bin
STANDARD_VALID=$DATA_DIR/babycosmofine/train_10M_tokenized.bin
SAMBAL_TRAIN=$DATA_DIR/babycosmofine_sambal/train_sambal_10M_tokenized.bin
SAMBAL_VALID=$DATA_DIR/babycosmofine_sambal/train_sambal_10M_tokenized.bin

TOKENIZER=$GPTBERT/gpt-bert-babylm-small/tokenizer.json
OUTPUT_DIR=$GPTBERT/trained_models
BLIMP_PATH=$REPO_ROOT/evals/blimp/blimp_fast
SYNTAXGYM_PATH=$REPO_ROOT/evals/syntaxgym/data/syntaxgym

# Model configs directory
CONFIG_DIR=$GPTBERT/configs/scaling_configs

run_experiment() {
    local data_type=$1
    local model_size=$2
    local data_fraction=$3
    local seed=$4

    # Set data paths
    if [ "$data_type" == "standard" ]; then
        train_path=$STANDARD_TRAIN
        valid_path=$STANDARD_VALID
    else
        train_path=$SAMBAL_TRAIN
        valid_path=$SAMBAL_VALID
    fi

    # Set config
    local config="${CONFIG_DIR}/${model_size}.json"
    if [ ! -f "$config" ]; then
        echo "Config not found: $config"; exit 1
    fi

    # Create run name. model_size doubles as the zero-padded config stem
    # (e.g. 05m); the analysis parser normalizes the size token to its
    # unpadded form when aggregating results.
    local name="scaling_${data_type}_${model_size}_${data_fraction}_seed${seed}"

    echo "=========================================="
    echo "Running: $name"
    echo "  Data: $data_type, Model: $model_size, Fraction: $data_fraction, Seed: $seed"
    echo "=========================================="

    python train_v1.py \
        --train_path="$train_path" \
        --valid_path="$valid_path" \
        --config_file="$config" \
        --tokenizer_path="$TOKENIZER" \
        --output_dir="$OUTPUT_DIR" \
        --name="$name" \
        --hybrid_numerator=1 \
        --hybrid_denominator=8 \
        --batch_size=128 \
        --learning_rate=0.007 \
        --epochs=10 \
        --seed="$seed" \
        --data_fraction="$data_fraction" \
        --seq_ramp_60_80 \
        --eval_only_at_end \
        --blimp_data_path "$BLIMP_PATH" \
        --blimp_eval_freq 1 \
        --syntaxgym_data_path "$SYNTAXGYM_PATH" \
        --no_save

    echo "Completed: $name"
    echo ""
}

# Run single experiment with provided arguments
run_experiment "$1" "$2" "$3" "$4"
