#!/usr/bin/env python3
"""Figure 4 — BLiMP parameter/token efficiency (two panels).

Top panel: low-budget scaling curves (blimp_lines);
bottom panel: BLiMP vs model size and data fraction (blimp_grouped_bars).

Reads the committed parsed scaling results
(records/scaling/scaling_results.json). With --log_dir, re-parses raw
training logs instead (reproduce/icml2026/assemble/assemble_scaling_results.py).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from records_io import FIGURES_OUT, load_json  # noqa: E402
from assemble.assemble_scaling_results import collect_results  # noqa: E402
import scaling_plots  # noqa: E402

RESULTS_JSON = "scaling/scaling_results.json"
OUT = FIGURES_OUT


def load_results_json(rel):
    raw = load_json(rel)
    return {m: {s: {float(f): v for f, v in fr.items()}
                for s, fr in sizes.items()}
            for m, sizes in raw.items()}


def load_results(log_dir):
    if log_dir:
        return collect_results(log_dir)
    return load_results_json(RESULTS_JSON)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", type=Path, default=None,
                    help="Parse raw training logs instead of the committed record")
    args = ap.parse_args()

    results = load_results(args.log_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    scaling_plots.plot_line_by_size(results, "blimp", OUT / "blimp_lines.png")
    scaling_plots.plot_grouped_bars_by_size(results, "blimp", OUT / "blimp_grouped_bars.png")


if __name__ == "__main__":
    main()
