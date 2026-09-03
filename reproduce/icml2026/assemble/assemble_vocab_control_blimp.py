#!/usr/bin/env python3
"""Assemble records/vocab_control/blimp.json from the control runs' training logs.

The vocab-filtered control was trained at three model sizes x three seeds in the
short regime at the 10% token budget. Each run's end-of-training full-BLiMP
evaluation prints its unrounded accuracy as

    [blimp-test] blimp/best_temp_avg_uid_accuracy: 58.41044776119401

which is what this script collects (not the 2-decimal `Best temperature:` line
above it, not the periodic in-training evaluations, and not the end-of-training
`[blimp-final]` line — those score the 200-pair fast subset).

Logs do not record which (size, seed) produced them, so the mapping is supplied
explicitly. For the committed record the per-log mode is REQUIRED: the
producing runs' job ids form three disjoint ranges and the slurm job name
changes with model size, so neither the contiguous job-id range mode nor a
single --log-pattern can address them. Identify each log's size and seed from
the log's own content — it names its wandb run directory, whose config records
the model config file and the seed, and the model dump printed in the log
(embedding width, parameter count) corroborates the size.

SIZE accepts the launcher-side size token (05m, as passed to
train_vocab_control.sh) or the record key (5M); either way the record is
keyed 5M/14M/30M, matching the committed file.

Usage:

    python assemble_vocab_control_blimp.py \
        --log 5M 0 logs/<5m-seed0>.out --log 5M 1 logs/<5m-seed1>.out \
        ... one --log SIZE SEED PATH per run, nine in all ... \
        --out reproduce/icml2026/records/vocab_control/blimp.json

The --slurm-id-start/--slurm-id-end range mode remains for submissions that
are one contiguous sweep in nested order (outer loop over sizes, inner loop
over seeds — the convention `reproduce/icml2026/assemble/parse_ablation_results.py`
uses).
"""
import argparse
import json
import re
from pathlib import Path

DEFAULT_SIZES = ["5M", "14M", "30M"]
DEFAULT_SEEDS = [0, 1, 2]

NOTE = ("Vocab-filtered control: short-regime training on the original text "
        "filtered to the top-25k replacement vocabulary at the 10% token budget "
        "(as-ran data fraction 0.202 of the filtered corpus), 3 seeds per model "
        "size. Values are each run's end-of-training full-BLiMP (1000 pairs per "
        "paradigm) best-temperature accuracy evaluated on the EMA weights (the "
        "trainer's final test-evaluation block).")

FINAL_BLIMP = re.compile(
    r"\[blimp-test\]\s+blimp/best_temp_avg_uid_accuracy:\s*([0-9.eE+-]+)")


def norm_size(tok: str) -> str:
    body, unit = tok.strip()[:-1], tok.strip()[-1]
    return (body.lstrip("0") or "0") + unit.upper()


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", nargs=3, action="append", metavar=("SIZE", "SEED", "PATH"),
                    help="Name one run's log explicitly; repeatable.")
    ap.add_argument("--log-dir", type=Path, default=Path("."))
    ap.add_argument("--log-pattern", default="train_vocab_control-{}.out")
    ap.add_argument("--slurm-id-start", type=int)
    ap.add_argument("--slurm-id-end", type=int)
    ap.add_argument("--sizes", nargs="+", default=DEFAULT_SIZES)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--out", type=Path, required=True)
    return ap.parse_args()


def final_blimp(path: Path) -> float:
    hits = FINAL_BLIMP.findall(Path(path).read_text())
    if len(hits) != 1:
        raise SystemExit(f"{path}: expected exactly one final [blimp-test] "
                         f"best_temp_avg_uid_accuracy line, found {len(hits)}")
    return float(hits[0])


def main():
    args = parse_args()
    finals = {}

    if args.log:
        for size, seed, path in args.log:
            finals.setdefault(norm_size(size), {})[f"seed{int(seed)}"] = final_blimp(Path(path))
    else:
        if args.slurm_id_start is None or args.slurm_id_end is None:
            raise SystemExit("give either --log SIZE SEED PATH (repeatable) or "
                             "--slurm-id-start/--slurm-id-end")
        ids = list(range(args.slurm_id_start, args.slurm_id_end + 1))
        expected = len(args.sizes) * len(args.seeds)
        if len(ids) != expected:
            raise SystemExit(f"job-id range covers {len(ids)} runs, expected {expected}")
        i = 0
        for size in map(norm_size, args.sizes):
            finals[size] = {}
            for seed in args.seeds:
                log = args.log_dir / args.log_pattern.format(ids[i])
                if not log.exists():
                    raise SystemExit(f"missing log for {size} seed{seed}: {log}")
                finals[size][f"seed{seed}"] = final_blimp(log)
                i += 1

    record = {
        "final_full_blimp_best_temp_avg":
            {k: dict(sorted(v.items())) for k, v in sorted(finals.items())},
        "note": NOTE,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(record, f, indent=1)
        f.write("\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
