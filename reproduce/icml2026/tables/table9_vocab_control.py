#!/usr/bin/env python3
"""Table 9 — lexical-regularization control (BLiMP %).

Baseline and SAMBAL columns: per-seed end-of-training BLiMP at the 10% data
budget from records/scaling/scaling_results.json. Vocab-filtered column:
the control runs' end-of-training full-BLiMP best-temperature accuracy
(records/vocab_control/blimp.json), mean +/- sample standard deviation over
3 seeds.
"""
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from records_io import TABLES_OUT, load_json  # noqa: E402

OUT = TABLES_OUT

SIZES = [("5M", "5m"), ("14M", "14m"), ("30M", "30m")]


def main():
    scaling = load_json("scaling/scaling_results.json")
    control = load_json("vocab_control/blimp.json")["final_full_blimp_best_temp_avg"]

    print(f"{'Model size':<11} {'baseline':>9} {'Vocab-filtered':>17} {'SAMBAL':>8}")
    rows = []
    for label, key in SIZES:
        base = statistics.mean(r["blimp"] for r in scaling["Baseline"][key]["0.1"])
        samb = statistics.mean(r["blimp"] for r in scaling["SAMBAL"][key]["0.1"])
        ctrl = list(control[label].values())
        cm, cs = statistics.mean(ctrl), statistics.stdev(ctrl)
        print(f"{label:<11} {base:>9.1f} {cm:>11.2f} +/- {cs:.2f} {samb:>8.1f}")
        rows.append([label, f"{base:.1f}", f"{cm:.2f}", f"{cs:.2f}", f"{samb:.1f}"])

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "table9_vocab_control.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_size", "baseline", "vocab_filtered_mean", "vocab_filtered_sd", "sambal"])
        w.writerows(rows)
    print(f"\nwrote {OUT}/table9_vocab_control.csv")


if __name__ == "__main__":
    main()
