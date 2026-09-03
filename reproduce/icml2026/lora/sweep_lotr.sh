#!/bin/bash
# Hyperparameter sweep launcher for LOTR training from scratch: one slurm
# job per experiment where sbatch exists, otherwise the experiments run here
# one after another.
#
# Table 4's "from-scratch" row is the sweep configuration with the LOWEST
# FINAL DEV PERPLEXITY (the selection tables/table4_lotr.py applies to
# records/lora/from_scratch.json, which records each run's final metrics).

set -euo pipefail

# Run from the repository root, or export REPO_ROOT to the checkout path.
REPO_ROOT="${REPO_ROOT:-$PWD}"
export REPO_ROOT
if command -v sbatch >/dev/null 2>&1; then HAVE_SBATCH=true; else HAVE_SBATCH=false; fi
GPTBERT="$REPO_ROOT/lm/gpt-bert"

# Config paths
TINY_CONFIG="$GPTBERT/configs/tiny.json"
MINI_CONFIG="$GPTBERT/configs/mini.json"
SMALL_CONFIG="$GPTBERT/configs/small.json"

submit_experiment() {
    local name=$1
    local config=$2
    local batch_size=$3
    local lr=$4
    local epochs=$5

    if [ "$HAVE_SBATCH" == "true" ]; then
        echo "Submitting: $name"
        sbatch --job-name="lotr_${name}" \
               --export=ALL,EXP_NAME="$name",EXP_CONFIG="$config",EXP_BATCH_SIZE="$batch_size",EXP_LR="$lr",EXP_EPOCHS="$epochs" \
               "$REPO_ROOT/reproduce/icml2026/lora/run_lotr_experiment.sh"
    else
        echo "Running: $name"
        EXP_NAME="$name" EXP_CONFIG="$config" EXP_BATCH_SIZE="$batch_size" EXP_LR="$lr" EXP_EPOCHS="$epochs" \
            bash "$REPO_ROOT/reproduce/icml2026/lora/run_lotr_experiment.sh"
    fi
}

echo "=========================================="
echo "LOTR Hyperparameter Sweep - Submitting Jobs"
echo "=========================================="
date
echo ""

# =============================================================================
# TINY MODEL EXPERIMENTS
# =============================================================================
echo "--- Tiny model experiments ---"

submit_experiment "tiny_bs16_lr1e-3" $TINY_CONFIG 16 1e-3 200
submit_experiment "tiny_bs32_lr1e-3" $TINY_CONFIG 32 1e-3 200
submit_experiment "tiny_bs32_lr5e-4" $TINY_CONFIG 32 5e-4 200
submit_experiment "tiny_bs32_lr5e-3" $TINY_CONFIG 32 5e-3 200
submit_experiment "tiny_bs64_lr1e-3" $TINY_CONFIG 64 1e-3 150

# Tiny with very large batch
submit_experiment "tiny_bs256_lr5e-3" $TINY_CONFIG 256 5e-3 300

# =============================================================================
# MINI MODEL EXPERIMENTS
# =============================================================================
echo ""
echo "--- Mini model experiments ---"

submit_experiment "mini_bs32_lr1e-3" $MINI_CONFIG 32 1e-3 150
submit_experiment "mini_bs32_lr5e-4" $MINI_CONFIG 32 5e-4 150
submit_experiment "mini_bs128_lr1e-3" $MINI_CONFIG 128 1e-3 200

# Mini with very large batch
submit_experiment "mini_bs256_lr5e-3" $MINI_CONFIG 256 5e-3 300

# =============================================================================
# SMALL MODEL EXPERIMENTS (~30M params)
# =============================================================================
echo ""
echo "--- Small model experiments ---"

# Small model - will likely overfit but interesting to see
submit_experiment "small_bs32_lr5e-4" $SMALL_CONFIG 32 5e-4 100

# Small with lower LR
submit_experiment "small_bs32_lr1e-4" $SMALL_CONFIG 32 1e-4 100

# Small with large batch (full-batch style)
submit_experiment "small_bs256_lr1e-3" $SMALL_CONFIG 256 1e-3 200

# =============================================================================
# FULL-BATCH EXPERIMENTS (entire dataset per gradient update)
# =============================================================================
echo ""
echo "--- Full-batch experiments ---"

# These use very large batch sizes to approximate full-batch gradient descent
submit_experiment "tiny_fullbatch_lr1e-2" $TINY_CONFIG 512 1e-2 500
submit_experiment "mini_fullbatch_lr5e-3" $MINI_CONFIG 512 5e-3 500

echo ""
echo "=========================================="
if [ "$HAVE_SBATCH" == "true" ]; then echo "All jobs submitted. Use 'squeue -u \$USER' to monitor."; else echo "All experiments ran."; fi
echo "=========================================="