#!/bin/bash
#SBATCH --job-name=train_vocab_control
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
    echo "Usage: $0 <05m|14m|30m> <seed>"
    echo "Example: $0 05m 0"
    echo "  One Table 9 vocab-control cell (reproduce/icml2026/TRAINING.md, 2.2)."
    echo "  Prerequisite: tokenize the control corpus first (from the repository root):"
    echo "    python lm/gpt-bert/corpus_tokenization/tokenize_corpus.py \\"
    echo "        --data_folder=data/babycosmofine_top25k --train_file=train_top25k.jsonl --name=10M"
    echo "  (data/ is \$SAMBAL_DATA_DIR when that is set)"
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

SIZE="$1"
SEED="$2"
case "$SIZE" in
    05m) DISPLAY_SIZE=5M ;;
    14m) DISPLAY_SIZE=14M ;;
    30m) DISPLAY_SIZE=30M ;;
    *) echo "size must be one of: 05m 14m 30m (got '$SIZE')"; exit 1 ;;
esac

TRAIN_BIN="$DATA_DIR/babycosmofine_top25k/train_top25k_10M_tokenized.bin"
if [ ! -s "$TRAIN_BIN" ]; then
    echo "FATAL: $TRAIN_BIN is missing."
    echo "Tokenize the control corpus first (from the repository root):"
    echo "  python lm/gpt-bert/corpus_tokenization/tokenize_corpus.py \\"
    echo "      --data_folder=$DATA_DIR/babycosmofine_top25k --train_file=train_top25k.jsonl --name=10M"
    exit 1
fi

# The as-ran Table 9 configuration (reproduce/icml2026/TRAINING.md, 2.2).
python train_v1.py \
    --train_path=$TRAIN_BIN \
    --valid_path=$TRAIN_BIN \
    --config_file=$GPTBERT/configs/scaling_configs/$SIZE.json \
    --tokenizer_path=$GPTBERT/gpt-bert-babylm-small/tokenizer.json \
    --output_dir=$GPTBERT/trained_models \
    --name=vocab_control_${DISPLAY_SIZE}_seed_${SEED} \
    --hybrid_numerator=1 \
    --hybrid_denominator=8 \
    --batch_size=128 \
    --learning_rate=0.007 \
    --epochs=10 \
    --seed=$SEED \
    --data_fraction=0.202 \
    --no_save \
    --eval_only_at_end \
    --seq_ramp_60_80 \
    --blimp_data_path $REPO_ROOT/evals/blimp/blimp_fast \
    --blimp_test_data_path $REPO_ROOT/evals/blimp/data \
    --syntaxgym_test_data_path $REPO_ROOT/evals/syntaxgym/data/syntaxgym \
    --enable_final_eval
