#!/usr/bin/env python3
"""Table 6 — SyntaxGym per-suite accuracy (%) for the two long-regime models.

Reads the committed per-suite SyntaxGym results
(records/syntaxgym/long_models_per_suite.json), prints the 25 core suites in
the paper's order, and writes a CSV.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from records_io import TABLES_OUT, load_json  # noqa: E402

RES = "syntaxgym/long_models_per_suite.json"
OUT = TABLES_OUT

ORDER = [
    "number_orc", "number_prep", "number_src",
    "reflexive_orc_fem", "reflexive_orc_masc", "reflexive_prep_fem",
    "reflexive_prep_masc", "reflexive_src_fem", "reflexive_src_masc",
    "npi_orc_any", "npi_orc_ever", "npi_src_any", "npi_src_ever",
    "fgd_hierarchy", "fgd_object", "fgd_pp", "fgd_subject",
    "center_embed", "center_embed_mod", "cleft", "cleft_modifier",
    "subordination", "subordination_orc-orc", "subordination_pp-pp",
    "subordination_src-src",
]


def main():
    data = load_json(RES)["checkpoints"]
    base = data["baseline"]["per_suite"]
    samb = data["sambal"]["per_suite"]
    assert set(base) == set(ORDER) and set(samb) == set(ORDER)

    print(f"{'SyntaxGym suite':<26} {'baseline':>9} {'SAMBAL':>8}")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "table6_syntaxgym_per_suite.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["suite", "baseline", "sambal"])
        for s in ORDER:
            print(f"{s:<26} {base[s]:>9.2f} {samb[s]:>8.2f}")
            w.writerow([s, f"{base[s]:.2f}", f"{samb[s]:.2f}"])
    print(f"\nAggregate (best-temp average): baseline "
          f"{data['baseline']['best_temp_avg_accuracy']:.2f}  "
          f"SAMBAL {data['sambal']['best_temp_avg_accuracy']:.2f}")
    print(f"wrote {OUT}/table6_syntaxgym_per_suite.csv")


if __name__ == "__main__":
    main()
