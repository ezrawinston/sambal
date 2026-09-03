#!/usr/bin/env python3
"""Parse ablation-grid training logs and report mean/std over seeds.

Reads the slurm .out logs of the Table-7 corpus-variant retrains (short-regime
`train_v1.py` runs, one log per (variant, seed)) and prints per-variant
mean +/- std of the last logged best-temperature BLiMP and SyntaxGym numbers.

Log-to-run mapping: consecutive slurm job ids are assigned to (variant, seed)
pairs in nested submission order — outer loop over --ablations, inner loop
over --seeds — i.e. the order produced by:

    for v in <ablations>; do for s in <seeds>; do sbatch ... ; done; done

Give the id range of your submission batch; the defaults read the pretrain
launcher's logs/train_v1_sambal_lr_seed-<id>.out (point --log-dir/--log-pattern
at logs named otherwise):

    python reproduce/icml2026/assemble/parse_ablation_results.py \
        --slurm-id-start 123450 --slurm-id-end 123464
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict
import numpy as np

DEFAULT_ABLATIONS = ["no_ctx_buckets", "no_gender", "no_human", "no_toinf", "no_ud_roundtrip"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-dir", type=Path, default=Path("logs"),
                    help="Directory containing the .out logs.")
    ap.add_argument("--log-pattern", default="train_v1_sambal_lr_seed-{}.out",
                    help="Log filename pattern relative to --log-dir; '{}' is the slurm job id "
                         "(default: the launchers' logs/<job name>-<job id>.out naming).")
    ap.add_argument("--slurm-id-start", type=int, required=True,
                    help="First slurm job id of the submission batch.")
    ap.add_argument("--slurm-id-end", type=int, required=True,
                    help="Last slurm job id of the submission batch (inclusive).")
    ap.add_argument("--ablations", nargs="+", default=DEFAULT_ABLATIONS,
                    help="Variant slugs in submission order (outer loop).")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                    help="Seeds in submission order (inner loop).")
    return ap.parse_args()


def parse_log_file(filepath: Path):
    """Extract the last best-temperature SyntaxGym and BLiMP results from a log."""
    try:
        with open(filepath, "r") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    results = {}

    # SyntaxGym: "Best temperature: 0.65 -> Avg: 37.06%"
    syntaxgym_pattern = r"SyntaxGym Results:\s*\n\s*Best temperature: [\d.]+ -> Avg: ([\d.]+)%"
    syntaxgym_matches = list(re.finditer(syntaxgym_pattern, content))
    if syntaxgym_matches:
        results["syntaxgym"] = float(syntaxgym_matches[-1].group(1))

    # BLiMP: "Best temperature: 0.60 -> Avg UID: 62.14%"
    blimp_pattern = r"BLiMP Results:\s*\n\s*Best temperature: [\d.]+ -> Avg UID: ([\d.]+)%"
    blimp_matches = list(re.finditer(blimp_pattern, content))
    if blimp_matches:
        results["blimp"] = float(blimp_matches[-1].group(1))

    return results if results else None


def main():
    args = parse_args()

    slurm_ids = list(range(args.slurm_id_start, args.slurm_id_end + 1))

    # Expected order based on the nested submission loop: outer=ablation, inner=seed
    expected_mapping = []
    for ablation in args.ablations:
        for seed in args.seeds:
            expected_mapping.append((ablation, seed))

    print(f"Found {len(slurm_ids)} SLURM IDs, expecting {len(expected_mapping)} jobs")
    print(f"SLURM IDs: {slurm_ids[0]}-{slurm_ids[-1]}")
    print()

    # Collect results
    results = defaultdict(list)  # ablation -> list of {seed, syntaxgym, blimp}

    for i, slurm_id in enumerate(slurm_ids):
        log_file = args.log_dir / args.log_pattern.format(slurm_id)

        if i < len(expected_mapping):
            ablation, seed = expected_mapping[i]
        else:
            ablation, seed = f"unknown_{i}", i

        if not log_file.exists():
            print(f"Missing: {log_file} ({ablation} seed={seed})")
            continue

        scores = parse_log_file(log_file)
        if scores is None:
            print(f"No results in: {log_file} ({ablation} seed={seed})")
            continue

        entry = {"seed": seed, **scores}
        results[ablation].append(entry)
        print(f"Parsed: {log_file.name} -> {ablation} seed={seed}: blimp={scores.get('blimp', 'N/A')}, syntaxgym={scores.get('syntaxgym', 'N/A')}")

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Report results by ablation
    for ablation in args.ablations:
        if ablation not in results:
            print(f"\n{ablation}: NO DATA")
            continue

        entries = results[ablation]

        blimp_vals = [e["blimp"] for e in entries if "blimp" in e]
        syntaxgym_vals = [e["syntaxgym"] for e in entries if "syntaxgym" in e]

        print(f"\n{ablation} (n={len(entries)} seeds):")

        if blimp_vals:
            mean_b = np.mean(blimp_vals)
            std_b = np.std(blimp_vals)
            print(f"  BLiMP:     {mean_b:.2f} +/- {std_b:.2f}  (values: {blimp_vals})")
        else:
            print(f"  BLiMP:     N/A")

        if syntaxgym_vals:
            mean_s = np.mean(syntaxgym_vals)
            std_s = np.std(syntaxgym_vals)
            print(f"  SyntaxGym: {mean_s:.2f} +/- {std_s:.2f}  (values: {syntaxgym_vals})")
        else:
            print(f"  SyntaxGym: N/A")

    # Print table format
    print("\n" + "=" * 70)
    print("TABLE FORMAT")
    print("=" * 70)
    print(f"{'Ablation':<20} {'BLiMP':<20} {'SyntaxGym':<20}")
    print("-" * 60)

    for ablation in args.ablations:
        if ablation not in results:
            print(f"{ablation:<20} {'N/A':<20} {'N/A':<20}")
            continue

        entries = results[ablation]
        blimp_vals = [e["blimp"] for e in entries if "blimp" in e]
        syntaxgym_vals = [e["syntaxgym"] for e in entries if "syntaxgym" in e]

        if blimp_vals:
            blimp_str = f"{np.mean(blimp_vals):.2f} +/- {np.std(blimp_vals):.2f}"
        else:
            blimp_str = "N/A"

        if syntaxgym_vals:
            syntaxgym_str = f"{np.mean(syntaxgym_vals):.2f} +/- {np.std(syntaxgym_vals):.2f}"
        else:
            syntaxgym_str = "N/A"

        print(f"{ablation:<20} {blimp_str:<20} {syntaxgym_str:<20}")


if __name__ == "__main__":
    main()
