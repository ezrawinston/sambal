#!/bin/bash
# Hyperparameter sweep launcher for PUBMED training from scratch: one slurm
# job per experiment where sbatch exists, otherwise the experiments run here
# one after another.
#
# Table 11's "from-scratch" row is the sweep configuration with the LOWEST
# FINAL DEV PERPLEXITY (the selection tables/table11_pubmed.py applies to
# records/lora/from_scratch.json, which records each run's final metrics).

set -euo pipefail

# Run from the repository root, or export REPO_ROOT to the checkout path.
REPO_ROOT="${REPO_ROOT:-$PWD}"
export REPO_ROOT
if command -v sbatch >/dev/null 2>&1; then HAVE_SBATCH=true; else HAVE_SBATCH=false; fi
GPTBERT="$REPO_ROOT/lm/gpt-bert"

# Config path
TINY_CONFIG="$GPTBERT/configs/tiny.json"

submit_experiment() {
    local name=$1
    local config=$2
    local batch_size=$3
    local lr=$4
    local epochs=$5

    if [ "$HAVE_SBATCH" == "true" ]; then
        echo "Submitting: $name"
        sbatch --job-name="pubmed_${name}" \
               --export=ALL,EXP_NAME="$name",EXP_CONFIG="$config",EXP_BATCH_SIZE="$batch_size",EXP_LR="$lr",EXP_EPOCHS="$epochs" \
               "$REPO_ROOT/reproduce/icml2026/lora/run_pubmed_experiment.sh"
    else
        echo "Running: $name"
        EXP_NAME="$name" EXP_CONFIG="$config" EXP_BATCH_SIZE="$batch_size" EXP_LR="$lr" EXP_EPOCHS="$epochs" \
            bash "$REPO_ROOT/reproduce/icml2026/lora/run_pubmed_experiment.sh"
    fi
}

echo "=========================================="
echo "PUBMED Hyperparameter Sweep - Submitting Jobs"
echo "=========================================="
date
echo ""

# Tiny model experiments only (mirrors the tiny grid from sweep_lotr.sh)
submit_experiment "pubmed_tiny_bs16_lr1e-3"  $TINY_CONFIG 16  1e-3 200
submit_experiment "pubmed_tiny_bs32_lr1e-3"  $TINY_CONFIG 32  1e-3 200
submit_experiment "pubmed_tiny_bs64_lr1e-3"  $TINY_CONFIG 64  1e-3 150
submit_experiment "pubmed_tiny_bs32_lr5e-4"  $TINY_CONFIG 32  5e-4 200
submit_experiment "pubmed_tiny_bs32_lr5e-3"  $TINY_CONFIG 32  5e-3 200

echo ""
echo "=========================================="
if [ "$HAVE_SBATCH" == "true" ]; then echo "All jobs submitted. Use 'squeue -u \$USER' to monitor."; else echo "All experiments ran."; fi
echo "=========================================="
