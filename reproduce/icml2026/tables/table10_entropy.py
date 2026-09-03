#!/usr/bin/env python3
"""Table 10 — lexical entropy at the matched budget.

Reads the committed matched-budget entropy measurements
(records/entropy/entropy_matched_budget.json, computed with
evals/entropy/lexical_entropy.py over the three corpora), prints the
H1 levels, the Δh2 differences against the baseline, and the word-type
counts, and writes a CSV to reproduce/icml2026/out/tables/.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from records_io import TABLES_OUT, load_json  # noqa: E402

RESULTS = "entropy/entropy_matched_budget.json"
OUT = TABLES_OUT


def main():
    d = load_json(RESULTS)
    corpora = d["corpora"]
    base_h2 = corpora["baseline"]["h2_plugin"]

    print(f"budget: {d['budget_word_tokens']:,} word tokens ({d['estimator']})\n")
    print(f"{'Corpus':<16} {'H1':>7} {'dh2':>7} {'types':>8}")
    rows = []
    for name in ["baseline", "sambal", "vocab_filtered"]:
        c = corpora[name]
        dh2 = c["h2_plugin"] - base_h2
        dh2_str = "-" if name == "baseline" else f"{dh2:+.2f}"
        print(f"{name:<16} {c['H1_plugin']:>7.2f} {dh2_str:>7} {c['word_types']:>8,}")
        rows.append([name, f"{c['H1_plugin']:.4f}",
                     "" if name == "baseline" else f"{dh2:+.4f}", c["word_types"]])

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "table10_entropy.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["corpus", "H1_plugin_bits_per_word", "delta_h2_vs_baseline", "word_types"])
        w.writerows(rows)
    print(f"\nwrote {OUT}/table10_entropy.csv")


if __name__ == "__main__":
    main()
