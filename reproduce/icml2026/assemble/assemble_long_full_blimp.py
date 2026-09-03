#!/usr/bin/env python3
"""Assemble records/blimp/<arm>_long_full_blimp.json from a log's [blimp-test] block.

A full-BLiMP evaluation is printed as a block of consecutive unrounded
`[blimp-test]` lines:

    [blimp-test] blimp/best_temperature: 2.6000001430511475
    [blimp-test] blimp/best_temp_avg_uid_accuracy: 79.24029850746271
    [blimp-test] blimp/best_temp_field_morphology: 92.83888888888889
    [blimp-test] blimp/per_uid/adjunct_island: 80.80000000000001
    ...

A log may hold more than one such block. In particular, the committed
long-model records are the PRE-fine-tuning evaluation each LoRA fine-tuning
run performs on its base checkpoint: `lm/gpt-bert/pretraining/finetune_lora.py`
scores `--blimp_test_data_path` before training and again after, printing one
block each, so those logs hold two blocks and `--block 1` (the default)
selects the pre-fine-tuning one. The choice is explicit because selecting the
wrong block silently yields the post-fine-tuning numbers.

This script collects the selected block into the committed record's schema:
the 11 scalar metrics at the top level (`best_temp_*`, `temp_1_*`,
`best_temperature`) and the 67 per-paradigm accuracies nested under `per_uid`,
keys sorted, indent 1, no trailing newline.

The 2-decimal `Best temperature: X -> Avg UID: Y%` summary and the standalone
scorer's report files are NOT usable sources: at the full-BLiMP denominator
two decimals do not determine the underlying values.

Usage:

    python assemble_long_full_blimp.py --log <fine-tuning or evaluation log> \
        --out reproduce/icml2026/records/blimp/baseline_long_full_blimp.json
"""
import argparse
import json
import re
from pathlib import Path

METRIC_LINE = re.compile(r"^\s*\[blimp-test\]\s+blimp/([\w/]+):\s*([0-9.eE+-]+)\s*$")
N_PARADIGMS = 67
SCALARS = {
    "best_temperature",
    "best_temp_avg_uid_accuracy", "temp_1_avg_uid_accuracy",
    "best_temp_field_morphology", "best_temp_field_semantics",
    "best_temp_field_syntax", "best_temp_field_syntax_semantics",
    "temp_1_field_morphology", "temp_1_field_semantics",
    "temp_1_field_syntax", "temp_1_field_syntax_semantics",
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, required=True,
                    help="Log containing one or more [blimp-test] blocks.")
    ap.add_argument("--block", type=int, default=1,
                    help="Which [blimp-test] block to collect when the log holds "
                         "several, 1-based (default 1 - for the LoRA fine-tuning "
                         "logs that is the pre-fine-tuning evaluation of the base "
                         "checkpoint; block 2 is the post-fine-tuning one).")
    ap.add_argument("--out", type=Path, required=True)
    return ap.parse_args()


def metric_blocks(text):
    """Maximal runs of consecutive [blimp-test] lines, as lists of (key, raw)."""
    blocks, current = [], []
    for line in text.splitlines():
        m = METRIC_LINE.match(line)
        if m:
            current.append(m.groups())
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def main():
    args = parse_args()
    blocks = metric_blocks(args.log.read_text())
    if not blocks:
        raise SystemExit(f"{args.log}: no [blimp-test] block found")
    if args.block < 1 or args.block > len(blocks):
        raise SystemExit(f"{args.log}: --block {args.block} requested but the log "
                         f"holds {len(blocks)} [blimp-test] block(s)")

    record, per_uid = {}, {}
    seen = set()
    for key, raw in blocks[args.block - 1]:
        if key in seen:
            raise SystemExit(f"{args.log}: metric 'blimp/{key}' appears more than "
                             f"once within block {args.block}")
        seen.add(key)
        val = float(raw)
        if key.startswith("per_uid/"):
            per_uid[key[len("per_uid/"):]] = val
        elif key in SCALARS:
            record[key] = val

    missing = SCALARS - set(record)
    if missing:
        raise SystemExit(f"{args.log}: block {args.block} is missing metrics "
                         f"{sorted(missing)}")
    if len(per_uid) != N_PARADIGMS:
        raise SystemExit(f"{args.log}: block {args.block} has {len(per_uid)} "
                         f"per-paradigm entries, expected {N_PARADIGMS} "
                         f"(is this a fast-subset evaluation?)")

    record["per_uid"] = dict(sorted(per_uid.items()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(dict(sorted(record.items())), f, indent=1)
    print(f"wrote {args.out}  (block {args.block} of {len(blocks)}; "
          f"{len(per_uid)} paradigms)")


if __name__ == "__main__":
    main()
