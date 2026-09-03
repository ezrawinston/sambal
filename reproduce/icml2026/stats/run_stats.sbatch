#!/bin/bash
#SBATCH --job-name=sambal_stats
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/dev/null

# Context-statistics collection (resumable):
#   sbatch run_stats.sbatch [input.jsonl] [out.pkl]
# Defaults: the source corpus $SAMBAL_DATA_DIR/babycosmofine/train.jsonl and the
# committed sambal/resources/lemma_stats_top_25k.pkl. For a sharded run, split the
# input, submit one job per piece with its own out.pkl, and merge the pickles with
#   python -m sambal.stats_merge <dir> 'stats_shard_*.pkl' <out.pkl>

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

# Submit from the repository root, or export REPO_ROOT to the checkout path.
REPO="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
DATA_DIR="${SAMBAL_DATA_DIR:-$REPO/data}"
# Python environment: source $SAMBAL_ENV if set (a venv activate script),
# else a repo-root .venv/ if present, else run with the ambient python.
if [ -n "${SAMBAL_ENV:-}" ]; then
    source "$SAMBAL_ENV"
elif [ -f "$REPO/.venv/bin/activate" ]; then
    source "$REPO/.venv/bin/activate"
fi
INPUT="${1:-$DATA_DIR/babycosmofine/train.jsonl}"
OUT="${2:-$REPO/sambal/resources/lemma_stats_top_25k.pkl}"
case "$INPUT" in /*) ;; *) INPUT="$REPO/$INPUT" ;; esac   # relative paths resolve from the checkout
case "$OUT" in /*) ;; *) OUT="$REPO/$OUT" ;; esac
# Config staged under the checkout (one file per output): compute nodes may give
# jobs a private /tmp this script cannot write into.
mkdir -p "$REPO/logs"
OUT_STEM=$(basename "$OUT" .pkl)
CFG="$REPO/logs/sambal_stats_cfg_$OUT_STEM.json"

python - "$INPUT" "$OUT" "$CFG" <<'PYEOF'
import json, sys
inp, out, cfg_path = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = {
    "args": {
        "input": inp,
        "fmt": "jsonl",
        "out": out,
        "resume": True,
        "require_gpu": True,
        "no_chunk": True,
    },
    "config": {},
}
json.dump(cfg, open(cfg_path, "w"))
PYEOF
[ -s "$CFG" ] || { echo "FATAL: failed to write $CFG"; exit 1; }

python -m sambal.stats --config "$CFG"
