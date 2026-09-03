#!/bin/bash
#SBATCH --job-name=eval_syntaxgym_ema_all
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
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

MODELS_DIR="${1:-$GPTBERT/trained_models}"
OUT_JSON="${2:-$REPO_ROOT/reproduce/icml2026/records/syntaxgym/short_regime_per_suite.json}"
mkdir -p "$(dirname "$OUT_JSON")"

CONFIG_FILE="$GPTBERT/configs/small.json"
TOKENIZER_PATH="$GPTBERT/gpt-bert-babylm-small/tokenizer.json"
SYNTAXGYM_TEST_DATA_PATH="$REPO_ROOT/evals/syntaxgym/data/syntaxgym"

# EMA checkpoints to evaluate. Default: the three seeds of each arm's main
# pretraining run (train_v1_gptbert_lr_seed.sh / train_v1_sambal_lr_seed.sh
# at lr 0.007), written over the committed record. Pass one or more
# checkpoint-name prefixes as extra arguments to evaluate a different set
# (then name your own output json):
#   sbatch eval_syntaxgym.sh <MODELS_DIR> <OUT_JSON> my_run_name ...
#
# For the LONG models (Table 2/6 long rows), invoke the scorer with the two
# checkpoints explicitly, writing the committed record in place. Per the
# weights note in reproduce/icml2026/RESULTS.md, the baseline is
# scored from its RAW weights and SAMBAL from EMA:
#   python evals/syntaxgym/syntaxgym_eval.py \
#     --input_path evals/syntaxgym/data/syntaxgym --suite_set core_only \
#     --output_json reproduce/icml2026/records/syntaxgym/long_models_per_suite.json \
#     --backend gptbert --backend-arg config=<config> --backend-arg tokenizer=<tokenizer> \
#     --models baseline:checkpoint=lm/gpt-bert/trained_models/gptbert_babycosmofine_long.bin \
#     --models sambal:checkpoint=lm/gpt-bert/trained_models/gptbert_sambal_long_ema.bin
if [ $# -ge 3 ]; then
    PREFIXES=("${@:3}")
    LABELS=()
else
    PREFIXES=(gptbert_sambal_short_lr_0.007_seed_0 gptbert_sambal_short_lr_0.007_seed_1 gptbert_sambal_short_lr_0.007_seed_2
              gptbert_babycosmofine_short_lr_0.007_seed_0 gptbert_babycosmofine_short_lr_0.007_seed_1 gptbert_babycosmofine_short_lr_0.007_seed_2)
    # keys matching the committed short_regime_per_suite.json record
    LABELS=(sambal_seed0 sambal_seed1 sambal_seed2
            baseline_seed0 baseline_seed1 baseline_seed2)
fi
MODEL_ARGS=()
i=0
for prefix in "${PREFIXES[@]}"; do
  for path in "$MODELS_DIR"/${prefix}*_ema.bin; do
    if [ "${#LABELS[@]}" -gt 0 ]; then
      label="${LABELS[$i]}"
    else
      label="$(basename "$path" .bin)"
    fi
    MODEL_ARGS+=(--models "${label}:checkpoint=${path}")
    i=$((i + 1))
  done
done

python evals/syntaxgym/syntaxgym_eval.py \
  --input_path "$SYNTAXGYM_TEST_DATA_PATH" \
  --suite_set core_only \
  --output_json "$OUT_JSON" \
  --backend gptbert \
  --backend-arg "config=$CONFIG_FILE" \
  --backend-arg "tokenizer=$TOKENIZER_PATH" \
  --backend-arg batch_size=32 \
  "${MODEL_ARGS[@]}"

echo "Wrote: $OUT_JSON"
