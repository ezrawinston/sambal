#!/usr/bin/env python3
"""Table 2 — syntactic benchmark performance (BLiMP, SyntaxGym, SyntaxGym without
reflexive suites) for baseline vs. SAMBAL under the long and short training regimes.

Sources (all committed under records/):
- Long-regime BLiMP: full-BLiMP best-temperature accuracy of each long checkpoint
  (records/blimp/{baseline,sambal}_long_full_blimp.json).
- Long-regime SyntaxGym: the best-temperature aggregate and the T=1.0 per-suite
  accuracies of each long checkpoint
  (records/syntaxgym/long_models_per_suite.json); the no-reflexive column pools
  the per-suite values over the 19 non-reflexive suites, weighted by the
  per-suite item counts in SUITE_ITEMS.
- Short regime (3 seeds): per-seed end-of-training BLiMP and SyntaxGym accuracy from
  records/scaling/scaling_results.json (30M model, 100% data), and the no-reflexive
  column from the per-suite short-regime evaluation
  (records/syntaxgym/short_regime_per_suite.json).
Short-regime cells are mean +/- sample standard deviation over 3 seeds.
"""
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from records_io import TABLES_OUT, load_json  # noqa: E402

OUT = TABLES_OUT

SUITE_ITEMS = {"number": 19, "reflexive": 19, "npi": 38, "fgd": 24, "center": 28,
               "cleft": 40, "subordination": 23}


def n_items(suite):
    for prefix, n in SUITE_ITEMS.items():
        if suite.startswith(prefix):
            return n
    raise KeyError(suite)


def weighted_no_reflexive(per_suite):
    nr = {s: v for s, v in per_suite.items() if not s.startswith("reflexive")}
    return sum(v * n_items(s) for s, v in nr.items()) / sum(n_items(s) for s in nr)


def main():
    long_blimp = {
        arm: load_json(f"blimp/{arm}_long_full_blimp.json")
        for arm in ("baseline", "sambal")
    }
    long_sg = load_json("syntaxgym/long_models_per_suite.json")
    scaling = load_json("scaling/scaling_results.json")
    short_sg = load_json("syntaxgym/short_regime_per_suite.json")

    rows = []
    for arm, scaling_key in (("baseline", "Baseline"), ("sambal", "SAMBAL")):
        label = "baseline" if arm == "baseline" else "SAMBAL"
        lb = long_blimp[arm]["best_temp_avg_uid_accuracy"]
        sg_rec = long_sg["checkpoints"][arm]
        lsg = sg_rec["best_temp_avg_accuracy"]
        lsg_nr = weighted_no_reflexive(sg_rec["per_suite"])
        rows.append((f"{label} (long)", f"{lb:.1f}", f"{lsg:.2f}", f"{lsg_nr:.1f}"))

        cell = scaling[scaling_key]["30m"]["1.0"]
        bl = [r["blimp"] for r in cell]
        sg = [r["syntaxgym"] for r in cell]
        nr = [weighted_no_reflexive(short_sg["checkpoints"][f"{arm}_seed{i}"]["per_suite"])
              for i in range(3)]
        rows.append((f"{label} (short)",
                     f"{statistics.mean(bl):.1f} +/- {statistics.stdev(bl):.1f}",
                     f"{statistics.mean(sg):.1f} +/- {statistics.stdev(sg):.1f}",
                     f"{statistics.mean(nr):.1f} +/- {statistics.stdev(nr):.1f}"))

    order = [rows[0], rows[2], rows[1], rows[3]]
    print(f"{'Model':<18} {'BLiMP':>14} {'SG':>14} {'SG (no refl.)':>14}")
    for name, b, s, snr in order:
        print(f"{name:<18} {b:>14} {s:>14} {snr:>14}")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "table2_main_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "blimp", "syntaxgym", "syntaxgym_no_reflexives"])
        for r in order:
            w.writerow(r)
    print(f"\nwrote {OUT}/table2_main_results.csv")


if __name__ == "__main__":
    main()
