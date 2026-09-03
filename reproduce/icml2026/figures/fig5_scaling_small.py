#!/usr/bin/env python3
"""Figure 5 — BLiMP vs model size and data fraction at the smallest token
budgets (blimp_grouped_bars_small). Same data source as Figure 4."""
import argparse
from pathlib import Path

import scaling_plots
from fig4_scaling import OUT, load_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", type=Path, default=None)
    args = ap.parse_args()

    results = load_results(args.log_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    scaling_plots.plot_grouped_bars_small_sizes(results, "blimp", OUT / "blimp_grouped_bars_small.png")


if __name__ == "__main__":
    main()
