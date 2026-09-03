#!/bin/bash
#SBATCH --job-name=sambal_corpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/dev/null

# Corpus ablation, one shard per invocation:
#   sbatch run_sambal_corpus.sbatch <SHARD_ID> <NUM_SHARDS>
# Shard SHARD_ID of NUM_SHARDS processes the contiguous input-row range
# [ceil(N/NUM_SHARDS)*SHARD_ID, ...) of the source corpus, so the shards
# partition the input and concatenate in shard order. The released corpus used
# NUM_SHARDS=24 with seed = shard id. Requires: `pip install -e .` of this
# repo, the transformer spaCy model, and the fetched context statistics
# (sambal/resources/MANIFEST.md).
#
# Corpus VARIANTS (Table 7 ablations, benchmark-vocabulary replacements) run
# through the same script via two optional environment variables:
#   CONFIG_OVERRIDES  JSON object merged into the config block on top of the
#                     paper profile, e.g. '{"respect_gender": false}'
#   PATHS_OVERRIDES   JSON object merged into the paths block (resource-file
#                     keys such as "allowed_vocab_path"; values are used as
#                     given, so pass real paths)
#   OUT_DIR           output directory for the variant's shards (default:
#                     <data>/babycosmofine_sambal; a relative value resolves
#                     from the repository root). Give every variant its own.
# Per-variant invocations are listed in reproduce/icml2026/DATA.md (3.5).

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

SHARD_ID=${1:? "usage: sbatch run_sambal_corpus.sbatch <SHARD_ID> <NUM_SHARDS>"}
NUM_SHARDS=${2:? "usage: sbatch run_sambal_corpus.sbatch <SHARD_ID> <NUM_SHARDS>"}

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
# Config staged under the checkout: compute nodes may give jobs a private
# /tmp this script cannot write into.
mkdir -p "$REPO/logs"
CFG="$REPO/logs/sambal_corpus_shard_cfg_${SHARD_ID}.json"

python - "$REPO" "$DATA_DIR" "$SHARD_ID" "$NUM_SHARDS" "$CFG" <<'PYEOF'
import json, os, sys
repo, data_dir, cfg_path = sys.argv[1], sys.argv[2], sys.argv[5]
shard, num_shards = int(sys.argv[3]), int(sys.argv[4])
out_dir = os.environ.get("OUT_DIR") or f"{data_dir}/babycosmofine_sambal"
if not os.path.isabs(out_dir):
    out_dir = os.path.join(repo, out_dir)
overrides = json.loads(os.environ.get("CONFIG_OVERRIDES") or "{}")
paths_overrides = json.loads(os.environ.get("PATHS_OVERRIDES") or "{}")
if not isinstance(overrides, dict) or not isinstance(paths_overrides, dict):
    raise SystemExit("CONFIG_OVERRIDES / PATHS_OVERRIDES must be JSON objects")
cfg = {
    "profile": "icml2026",
    "args": {
        "input": f"{data_dir}/babycosmofine/train.jsonl",
        "fmt": "jsonl",
        "out": f"{out_dir}/shard_{shard:02d}.jsonl",
        "resume": True,
        # This job augments only its own contiguous slice of the input rows;
        # the shards partition the corpus and concatenate in shard order.
        "shard": shard,
        "num_shards": num_shards,
    },
    "config": {"seed": shard, **overrides},
}
if paths_overrides:
    cfg["paths"] = paths_overrides
os.makedirs(out_dir, exist_ok=True)
json.dump(cfg, open(cfg_path, "w"))
print("shard config:", json.dumps(cfg["config"]),
      "shard:", f"{shard}/{num_shards}",
      "paths:", json.dumps(paths_overrides), "->", cfg["args"]["out"])
PYEOF
[ -s "$CFG" ] || { echo "FATAL: failed to write $CFG"; exit 1; }

python -m sambal.augment --config "$CFG"
