#!/usr/bin/env python3
"""Create a fast subset of SyntaxGym data by sampling 20% of items from each suite.

By default this reads the full suite JSON files from data/syntaxgym/ (populate
it with fetch_data.py) and writes sampled versions into syntaxgym_fast/ — the
committed frozen-subset location that the training launchers read. Uses a
fixed random seed; the committed subset is seed 42 and regenerates
byte-identically at the defaults.
"""

import json
import random
from pathlib import Path
import math


def create_fast_subset(input_dir: Path, output_dir: Path, sample_rate: float = 0.2, min_items: int = 2, seed: int = 42):
    """Sample items from each suite JSON file.

    Args:
        input_dir: Directory containing original suite JSON files
        output_dir: Directory to write sampled JSON files
        sample_rate: Proportion of items to sample (default 0.2 = 20%)
        min_items: Minimum number of items to sample per suite (default 2)
        seed: Random seed for reproducibility (default 42)
    """
    # Set random seed for reproducibility
    random.seed(seed)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each JSON file in the input directory
    json_files = sorted(input_dir.glob("*.json"))

    if not json_files:
        raise SystemExit(
            f"No JSON files found in {input_dir} — populate the full suite data "
            f"first (python evals/syntaxgym/fetch_data.py), or pass --input"
        )

    print(f"Processing {len(json_files)} suite files...")
    print(f"Sample rate: {sample_rate*100:.0f}% (minimum {min_items} items per suite)")
    print(f"Random seed: {seed}\n")

    for json_file in json_files:
        # Load the suite data
        with open(json_file, 'r') as f:
            suite_data = json.load(f)

        # Get original items
        original_items = suite_data.get("items", [])
        n_original = len(original_items)

        # Calculate number of items to sample

        n_sample = max(min_items, math.ceil(n_original * sample_rate))
        n_sample = min(n_sample, n_original)  # Don't sample more than available

        # Sample items (fixed random state for reproducibility)
        sampled_items = random.sample(original_items, n_sample)

        # Create new suite data with sampled items
        sampled_suite = suite_data.copy()
        sampled_suite["items"] = sampled_items

        # Write to output file
        output_file = output_dir / json_file.name
        with open(output_file, 'w') as f:
            json.dump(sampled_suite, f, indent=2)

        print(f"{json_file.name:30s}  {n_original:3d} -> {n_sample:3d} items ({n_sample/n_original*100:5.1f}%)")

    print(f"\nDone! Sampled suites written to {output_dir}")


if __name__ == "__main__":
    import argparse

    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Sample a fast SyntaxGym subset (seeded, per suite)."
    )
    parser.add_argument(
        "--input", type=Path, default=script_dir / "data" / "syntaxgym",
        help="directory of full suite JSON files (default: %(default)s; "
        "populate it with fetch_data.py)",
    )
    parser.add_argument(
        "--output", type=Path, default=script_dir / "syntaxgym_fast",
        help="output directory (default: %(default)s — the committed "
        "frozen-subset location the training launchers read)",
    )
    parser.add_argument(
        "--sample-rate", type=float, default=0.2,
        help="proportion of items to keep per suite (default: %(default)s)",
    )
    parser.add_argument(
        "--min-items", type=int, default=2,
        help="minimum items per suite (default: %(default)s)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="random seed (default: %(default)s, the committed subset's seed)",
    )
    cli = parser.parse_args()

    create_fast_subset(
        cli.input, cli.output,
        sample_rate=cli.sample_rate, min_items=cli.min_items, seed=cli.seed,
    )