#!/usr/bin/env python3
"""Fetch the pinned SyntaxGym test suites into data/syntaxgym/.

The standalone evaluator (syntaxgym_eval.py) downloads suites on demand;
the trainers' final test evaluation and the batch evaluator only READ
`data/syntaxgym/` — run this once (network required) before jobs that use
them. Idempotent: suites already present are kept.
"""
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from syntaxgym_eval import ALL_HU2020_SYNTAXGYM, download_suite_json


def main():
    out = _HERE / "data/syntaxgym"
    for name in ALL_HU2020_SYNTAXGYM:
        download_suite_json(name, cache_dir=out)
    print(f"{len(ALL_HU2020_SYNTAXGYM)} suites present under {out}")


if __name__ == "__main__":
    main()
