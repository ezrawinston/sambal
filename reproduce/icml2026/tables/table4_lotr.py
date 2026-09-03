#!/usr/bin/env python3
"""Table 4 — specialization on LoTR with LoRA fine-tuning.

Reads the committed LoRA run records (records/lora/lora_runs.json: pre/post
fine-tuning LoTR test perplexity and full-BLiMP best-temperature accuracy) and
the from-scratch grid (records/lora/from_scratch.json), prints the table, and
writes a CSV. The from-scratch row reports the grid configuration with the best
(lowest) held-out perplexity.
"""
import csv
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from records_io import TABLES_OUT, load_json  # noqa: E402

RES = "lora"
OUT = TABLES_OUT


def round1(v):
    # Display rounding goes through Decimal: some record values are themselves
    # decimal-rounded (e.g. a perplexity of 29.15), and binary-float ":.1f"
    # rounds such halfway values down.
    return Decimal(str(v)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)


def main():
    runs = load_json(f"{RES}/lora_runs.json")["runs"]
    fs = load_json(f"{RES}/from_scratch.json")["grids"]["lotr"]

    rows = []
    for arm, key in (("baseline (Long)", "lotr_baseline"), ("SAMBAL (Long)", "lotr_sambal")):
        r = runs[key]
        rows.append((arm, r["pre_ft_test_ppl"], r["pre_ft_blimp_test_best_temp_avg"]))
        rows.append((f"{arm} fine-tuned", r["post_ft_test_ppl"],
                     r["post_ft_blimp_test_best_temp_avg"]))
    rows = [rows[0], rows[1], rows[2], rows[3]]

    best_cfg = min(fs, key=lambda c: fs[c]["final_dev_ppl"])
    b = fs[best_cfg]
    rows.append((f"Best LoTR from-scratch model ({best_cfg})",
                 b["final_dev_ppl"], b["final_blimp_fast_best_temp_avg"]))

    print(f"{'Model':<44} {'LoTR PPL':>9} {'BLiMP Acc.':>11}")
    for name, ppl, acc in rows:
        print(f"{name:<44} {round1(ppl):>9} {round1(acc):>11}")
    print("\nNote: the from-scratch row reports the best grid model's held-out dev")
    print("perplexity and its inline BLiMP (fast-subset) accuracy; the LoRA rows")
    print("report test-set perplexity and full-BLiMP accuracy.")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "table4_lotr.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "lotr_ppl", "blimp_acc"])
        for name, ppl, acc in rows:
            w.writerow([name, f"{ppl:.2f}", f"{acc:.2f}"])
    print(f"wrote {OUT}/table4_lotr.csv")


if __name__ == "__main__":
    main()
