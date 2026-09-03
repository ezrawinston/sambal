#!/bin/bash
# Launch all scaling ablation experiments: one slurm job per cell where
# sbatch exists, otherwise the cells run here one after another.
#
# Run from the repository root (worker jobs resolve paths via SLURM_SUBMIT_DIR).
# Usage: ./launch_scaling_ablation.sh [--dry-run]
#   --dry-run: Print the per-cell commands without executing
#
# Per-user submit caps are waited out: a refused cell is retried every
# SUBMIT_RETRY_SECONDS (default 120) until it lands. An interrupted wave
# resumes from a given 1-based cell with START_CELL=<n> (cells are numbered
# in the submission order --dry-run prints).

set -euo pipefail

DRY_RUN=false
if [ "${1:-}" == "--dry-run" ]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE ==="
    echo ""
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_SCRIPT="$SCRIPT_DIR/train_scaling_ablation.sh"
SUBMIT_RETRY_SECONDS="${SUBMIT_RETRY_SECONDS:-120}"
SUBMIT_MAX_ATTEMPTS="${SUBMIT_MAX_ATTEMPTS:-720}"   # 720 x 120 s = 24 h per cell
START_CELL="${START_CELL:-1}"
if command -v sbatch >/dev/null 2>&1; then HAVE_SBATCH=true; else HAVE_SBATCH=false; fi

DATA_TYPES=("standard" "sambal")
SEEDS=(42 123 456)

# (model_size, data_fraction) cells of the scaling grid, "size:fraction".
# 05m and 14m sweep all eight fractions; 10m the six smallest; 30m three
# (0.1, 0.5, 1.0 — the 30m:1.0 cells are the "short"-regime reference runs
# behind Tables 2/7 and Figures 4/6).
PAIRS=(
    "05m:0.05" "05m:0.1" "05m:0.15" "05m:0.2" "05m:0.25" "05m:0.3" "05m:0.5" "05m:1.0"
    "10m:0.05" "10m:0.1" "10m:0.15" "10m:0.2" "10m:0.25" "10m:0.3"
    "14m:0.05" "14m:0.1" "14m:0.15" "14m:0.2" "14m:0.25" "14m:0.3" "14m:0.5" "14m:1.0"
    "30m:0.1" "30m:0.5" "30m:1.0"
)

submitted=0
cell=0
total=$(( ${#DATA_TYPES[@]} * ${#PAIRS[@]} * ${#SEEDS[@]} ))

# Submit one cell, waiting out per-user submit caps: a refusal that names a
# submit/QOS limit is retried every SUBMIT_RETRY_SECONDS; any other sbatch
# error stops the wave.
submit_cell() {
    local job_name="$1"; shift
    local attempt=1 out
    while :; do
        if out=$("$@" 2>&1); then
            echo "$out"
            return 0
        fi
        case "$out" in
            *MaxSubmit*|*MaxJobs*|*"violates accounting/QOS policy"*|*"Socket timed out"*)
                if [ "$attempt" -ge "$SUBMIT_MAX_ATTEMPTS" ]; then
                    echo "giving up on $job_name after $attempt attempts: $out" >&2
                    return 1
                fi
                echo "queue cap reached at cell $cell/$total ($job_name); retrying in ${SUBMIT_RETRY_SECONDS}s (attempt $attempt)"
                sleep "$SUBMIT_RETRY_SECONDS"
                attempt=$((attempt + 1))
                ;;
            *)
                echo "sbatch failed for $job_name: $out" >&2
                return 1
                ;;
        esac
    done
}

for data_type in "${DATA_TYPES[@]}"; do
    for pair in "${PAIRS[@]}"; do
        # Parse the pair
        model_size="${pair%%:*}"
        data_fraction="${pair##*:}"

        for seed in "${SEEDS[@]}"; do
            cell=$((cell + 1))
            job_name="scaling_${data_type}_${model_size}_${data_fraction}_seed${seed}"

            if [ "$cell" -lt "$START_CELL" ]; then
                continue
            fi

            if [ "$DRY_RUN" == "true" ] && [ "$HAVE_SBATCH" == "true" ]; then
                echo "sbatch --job-name=$job_name $WORKER_SCRIPT $data_type $model_size $data_fraction $seed"
            elif [ "$DRY_RUN" == "true" ]; then
                echo "bash $WORKER_SCRIPT $data_type $model_size $data_fraction $seed"
            elif [ "$HAVE_SBATCH" == "true" ]; then
                echo "Submitting cell $cell/$total: $job_name"
                submit_cell "$job_name" sbatch --job-name="$job_name" "$WORKER_SCRIPT" "$data_type" "$model_size" "$data_fraction" "$seed"
                submitted=$((submitted + 1))
            else
                echo "Running cell $cell/$total: $job_name"
                bash "$WORKER_SCRIPT" "$data_type" "$model_size" "$data_fraction" "$seed"
                submitted=$((submitted + 1))
            fi
        done
    done
done

if [ "$DRY_RUN" == "false" ] && [ "$HAVE_SBATCH" == "true" ]; then
    echo ""
    echo "Submitted $submitted jobs. Monitor with: squeue -u \$USER"
    echo "If the wave was interrupted, resume it with START_CELL=<next cell> $0"
elif [ "$DRY_RUN" == "false" ]; then
    echo ""
    echo "Ran $submitted cells."
fi
