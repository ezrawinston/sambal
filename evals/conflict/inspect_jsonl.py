#!/usr/bin/env python3
"""Print a few examples from a JSONL suite."""
from __future__ import annotations

import argparse, json
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

from conflict_rr_suite.text_utils import is_swap_only

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    with args.jsonl.open("r", encoding="utf-8") as f:
        for i,line in enumerate(f):
            if i >= args.n:
                break
            ex = json.loads(line)
            g = ex["sentence_good"]
            b = ex["sentence_bad"]
            ok, idxs, toks = is_swap_only(g,b)
            print("="*80)
            print("UID:", ex.get("UID"))
            print("family:", ex.get("linguistics_term"))
            print("GOOD:", g)
            print("BAD :", b)
            print("swap_only:", ok, "idxs:", idxs, "toks:", toks)
            md = ex.get("metadata", {})
            if "baseline" in md:
                print("baseline delta_conf:", md["baseline"].get("delta_conf", md["baseline"].get("delta")))
            if "control_good" in md:
                print("control_good:", md["control_good"])
    print("Done.")

if __name__ == "__main__":
    main()
