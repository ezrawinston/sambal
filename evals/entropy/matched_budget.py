#!/usr/bin/env python3
"""Build the matched-budget entropy record consumed by Table 10 / App. H of the ICML 2026 paper.

Runs `lexical_entropy` on the three corpora at one common word-token budget —
the vocabulary-filtered control corpus's full length, measured here rather
than hardcoded — and writes the combined record in the exact schema of
`reproduce/icml2026/records/entropy/entropy_matched_budget.json`
(plug-in estimator; h(n) = H(n) - H(n-1)).

Default input paths follow the repository data layout (SAMBAL_DATA_DIR
honored); point the three path flags elsewhere to run on copies.

    python evals/entropy/matched_budget.py \
        --out reproduce/icml2026/records/entropy/entropy_matched_budget.json
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lexical_entropy import measure_corpus  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_ROOT = pathlib.Path(os.environ.get("SAMBAL_DATA_DIR") or (REPO_ROOT / "data"))

# Read far past any corpus length so the common budget is applied as a slice
# of the full token stream (not as a read cap).
READ_CAP = 100_000_000
MAX_N = 3


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", type=pathlib.Path,
                    default=DATA_ROOT / "babycosmofine/train.jsonl")
    ap.add_argument("--sambal", type=pathlib.Path,
                    default=DATA_ROOT / "babycosmofine_sambal/train_sambal.jsonl")
    ap.add_argument("--vocab-filtered", type=pathlib.Path,
                    default=DATA_ROOT / "babycosmofine_top25k/train_top25k.jsonl")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    return ap.parse_args()


def rung(measurement, budget):
    curve = measurement["results"]["budgets"][str(budget)]
    return {
        "word_types": curve[1]["types"],
        "H1_plugin": curve[1]["H_plugin"],
        "h2_plugin": curve[2]["plugin"],
        "h3_plugin": curve[3]["plugin"],
    }


def main():
    args = parse_args()

    # The budget anchor: the vocab-filtered corpus's full word-token length.
    anchor = measure_corpus("vocab_filtered", [str(args.vocab_filtered)],
                            READ_CAP, MAX_N, [READ_CAP], fmt="jsonl")
    budget = max(int(b) for b in anchor["results"]["budgets"])
    print(f"[matched-budget] anchor length: {budget:,} word tokens")

    record = {
        "budget_word_tokens": budget,
        "estimator": "plug-in",
        "note": "Word-token n-gram entropies at the matched budget (the "
                "vocab-filtered corpus's full length); h(n) values are "
                "conditional block entropies H(n)-H(n-1).",
        "corpora": {},
    }
    record["corpora"]["baseline"] = rung(
        measure_corpus("baseline", [str(args.baseline)], READ_CAP, MAX_N,
                       [budget], fmt="jsonl"), budget)
    record["corpora"]["sambal"] = rung(
        measure_corpus("sambal", [str(args.sambal)], READ_CAP, MAX_N,
                       [budget], fmt="jsonl"), budget)
    record["corpora"]["vocab_filtered"] = rung(anchor, budget)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(record, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
