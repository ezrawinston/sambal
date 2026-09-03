#!/usr/bin/env python3
"""Table 7 — BLiMP ablation results.

Reads the committed ablation run records (records/ablations/blimp_finals.json:
final best-temperature accuracy over the full 67,000-item BLiMP set for 3
seeds per pipeline-component ablation) plus the SAMBAL (short) reference row from
records/scaling/scaling_results.json (30M, 100% data). Ablation rows report
mean +/- population standard deviation over the 3 seeds; the reference row
reports mean +/- sample standard deviation, matching how each set of runs was
summarized.
"""
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from records_io import TABLES_OUT, load_json  # noqa: E402

OUT = TABLES_OUT

LABELS = [
    ("no_toinf", "no to-inf"),
    ("no_gender", "no gender"),
    ("no_ud_roundtrip", "no UD-roundtrip"),
    ("no_humanness", "no humanness"),
    ("no_context_buckets", "no context buckets"),
]


def main():
    abl = load_json("ablations/blimp_finals.json")["final_blimp_best_temp_avg"]
    scaling = load_json("scaling/scaling_results.json")
    ref = [r["blimp"] for r in scaling["SAMBAL"]["30m"]["1.0"]]

    rows = [("SAMBAL (Short)", statistics.mean(ref), statistics.stdev(ref))]
    for key, label in LABELS:
        vals = list(abl[key].values())
        rows.append((label, statistics.mean(vals), statistics.pstdev(vals)))

    print(f"{'Ablation':<22} {'BLiMP':>16}")
    for name, m, s in rows:
        print(f"{name:<22} {m:>9.2f} +/- {s:.2f}")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "table7_ablations.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ablation", "blimp_mean", "blimp_sd"])
        for name, m, s in rows:
            w.writerow([name, f"{m:.2f}", f"{s:.2f}"])
    print(f"\nwrote {OUT}/table7_ablations.csv")


if __name__ == "__main__":
    main()
