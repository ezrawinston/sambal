#!/bin/bash
#SBATCH --job-name=build_suite
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
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
# Python environment: source $SAMBAL_ENV if set (a venv activate script),
# else a repo-root .venv/ if present, else run with the ambient python.
if [ -n "${SAMBAL_ENV:-}" ]; then
    source "$SAMBAL_ENV"
elif [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
fi
cd "$REPO_ROOT"

# The settings that built the shipped suite. Writes the candidate pool and
# controls over the committed files in place (git shows what changed; a
# rebuild differs if the oracle checkpoints differ). Afterwards refresh
# conflicts.jsonl from the new pool: derive_conflicts.py --force.
python evals/conflict/build_suite.py \
    --backend gptbert \
    --backend-arg "checkpoint=$GPTBERT/trained_models/gptbert_babycosmofine_long_ema.bin" \
    --backend-arg "config=$GPTBERT/configs/small.json" \
    --backend-arg "tokenizer=$GPTBERT/gpt-bert-babylm-small/tokenizer.json" \
    --backend-arg batch_size=256 \
    --other-backend-arg "checkpoint=$GPTBERT/trained_models/gptbert_sambal_long_ema.bin" \
    --other-backend-arg "config=$GPTBERT/configs/small.json" \
    --other-backend-arg "tokenizer=$GPTBERT/gpt-bert-babylm-small/tokenizer.json" \
    --other-backend-arg batch_size=256 \
    --out_conflicts "$REPO_ROOT/evals/conflict/pool/conflict_pool.jsonl" \
    --out_controls "$REPO_ROOT/evals/conflict/controls.jsonl" \
    --n_sva 200 --n_det 200 --n_passive 200 \
    --other_ctrl_margin 0.1 \
    --ctrl_margin 0.1 --conf_margin 0.1 \
    --batch_items 256
