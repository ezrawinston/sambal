#!/usr/bin/env python3
"""Table 11 — specialization on PubMed with LoRA fine-tuning.

Reads the committed LoRA run records (records/lora/lora_runs.json) and the
PubMed from-scratch grid (records/lora/from_scratch.json), prints the table,
and writes a CSV. The from-scratch row is the grid configuration with the best
(lowest) test perplexity.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from records_io import TABLES_OUT, load_json  # noqa: E402

RES = "lora"
OUT = TABLES_OUT


def main():
    runs = load_json(f"{RES}/lora_runs.json")["runs"]
    fs = load_json(f"{RES}/from_scratch.json")["grids"]["pubmed"]

    rows = []
    for arm, key in (("baseline (Long)", "pubmed_baseline"), ("SAMBAL (Long)", "pubmed_sambal")):
        r = runs[key]
        rows.append((arm, r["pre_ft_test_ppl"], r["pre_ft_blimp_test_best_temp_avg"]))
        rows.append((f"{arm} fine-tuned", r["post_ft_test_ppl"],
                     r["post_ft_blimp_test_best_temp_avg"]))

    best_cfg = min(fs, key=lambda c: fs[c]["test_ppl"])
    b = fs[best_cfg]
    rows.append((f"Best PubMed from-scratch model ({best_cfg})",
                 b["test_ppl"], b["blimp_test_best_temp_avg"]))

    print(f"{'Model':<46} {'PubMed PPL':>11} {'BLiMP Acc.':>11}")
    for name, ppl, acc in rows:
        print(f"{name:<46} {ppl:>11.2f} {acc:>11.2f}")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "table11_pubmed.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "pubmed_ppl", "blimp_acc"])
        for name, ppl, acc in rows:
            w.writerow([name, f"{ppl:.2f}", f"{acc:.2f}"])
    print(f"wrote {OUT}/table11_pubmed.csv")


if __name__ == "__main__":
    main()
