#!/bin/bash
#SBATCH --job-name=train_long_gptbert
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --output=/dev/null
#SBATCH --mem=160G

# Long-regime pretraining, baseline arm — produces the headline baseline
# checkpoints (Tables 2-6, Figures 1-3, and the LoRA base weights).
# One 8-GPU node; the trainer requires the world size to be divisible by
# the hybrid denominator (8). Resubmitting after a preemption resumes from
# the last periodic save (--auto_resume).
#
# The protocol: 5400 optimizer steps (--steps) at sequence length 128,
# with the global batch ramping linearly upward from 8192 sequences. The
# learning-rate decay and the batch-size ramp are defined over a longer
# horizon (--schedule_horizon in train_long.py), so training ends inside
# the ramp with the learning rate still decaying. The released checkpoints
# (gptbert_babycosmofine_long.bin raw / gptbert_babycosmofine_long_ema.bin
# EMA) are this run's final save.

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

# Optional argument: another tokenized corpus (a relative path resolves from
# the repository root); the run is then named after it.
if [ $# -ge 1 ]; then
    DATA_PATH="$1"
    case "$DATA_PATH" in
        /*) ;;
        *) DATA_PATH="$REPO_ROOT/$DATA_PATH" ;;
    esac
    NAME=gptbert_babycosmofine_long_$(basename "$DATA_PATH" .bin)
else
    DATA_PATH="$DATA_DIR/babycosmofine/train_10M_tokenized.bin"
    NAME=gptbert_babycosmofine_long
fi

torchrun --nproc_per_node=8 train_long.py \
    --train_path=$DATA_PATH \
    --valid_path=$DATA_PATH \
    --config_file=$GPTBERT/configs/small.json \
    --tokenizer_path=$GPTBERT/gpt-bert-babylm-small/tokenizer.json \
    --output_dir=$GPTBERT/trained_models \
    --name=$NAME \
    --hybrid_numerator=7 \
    --hybrid_denominator=8 \
    --adaptive_validation \
    --save_every=200 \
    --steps=5400 \
    --auto_resume \
    --blimp_data_path $REPO_ROOT/evals/blimp/blimp_fast \
    --syntaxgym_data_path $REPO_ROOT/evals/syntaxgym/syntaxgym_fast

# The trainer stamps its output filenames; downstream eval and LoRA
# launchers reference the unstamped names — point them at this run:
cd "$GPTBERT/trained_models"
EMA=$(ls -t ${NAME}_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]_ema.bin | head -1)   # this run's own stamped saves
RAW="${EMA%_ema.bin}.bin"
ln -sf "$EMA" ${NAME}_ema.bin
ln -sf "$RAW" ${NAME}.bin
