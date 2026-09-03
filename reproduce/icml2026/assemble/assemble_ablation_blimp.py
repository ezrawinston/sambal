#!/usr/bin/env python3
"""Assemble records/ablations/blimp_finals.json from the ablation training logs.

Each pipeline-component ablation was trained for 3 seeds in the short regime;
at the end of training each run performs one full-BLiMP evaluation and prints
its unrounded metrics with a `[blimp-test]` prefix:

    BLiMP Results:
      Best temperature: 1.45 -> Avg UID: 70.41%
      [blimp-test] blimp/best_temp_avg_uid_accuracy: 70.41343283582091

This script reads that unrounded line -- not the 2-decimal `Best temperature:`
summary above it, which does not carry enough precision to recover the value,
and not the periodic in-training evaluations, which score the fast subset.

The logs do not record which variant/seed produced them, so the mapping is
supplied the same way `reproduce/icml2026/assemble/parse_ablation_results.py` supplies it:
one log per job id, assigned in nested submission order (outer loop over
variants, inner loop over seeds), i.e. the order produced by

    for v in <ablations>; do for s in <seeds>; do sbatch ... ; done; done

Usage (defaults read the pretrain launcher's logs/train_v1_sambal_lr_seed-<id>.out;
point --log-dir/--log-pattern at logs named otherwise):

    python assemble_ablation_blimp.py \
        --slurm-id-start 123450 --slurm-id-end 123464 \
        --out reproduce/icml2026/records/ablations/blimp_finals.json
"""
import argparse
import json
import re
from pathlib import Path

# Variant slugs in submission order (outer loop), paired with the key each one
# carries in the committed record.
DEFAULT_VARIANTS = [
    ("no_ctx_buckets", "no_context_buckets"),
    ("no_gender", "no_gender"),
    ("no_human", "no_humanness"),
    ("no_toinf", "no_toinf"),
    ("no_ud_roundtrip", "no_ud_roundtrip"),
]
DEFAULT_SEEDS = [0, 1, 2]

NOTE = ("Pipeline-component ablations: short-regime training (3 seeds each) "
        "on corpus variants with one component disabled; final BLiMP accuracy "
        "at best temperature over the full 67,000-item BLiMP set (not the "
        "inline fast subset used during training).")

FINAL_BLIMP = re.compile(
    r"\[blimp-test\]\s+blimp/best_temp_avg_uid_accuracy:\s*([0-9.eE+-]+)")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-dir", type=Path, default=Path("logs"))
    ap.add_argument("--log-pattern", default="train_v1_sambal_lr_seed-{}.out",
                    help="Log filename pattern relative to --log-dir; '{}' is the job id "
                         "(default: the launchers' logs/<job name>-<job id>.out naming).")
    ap.add_argument("--slurm-id-start", type=int, required=True)
    ap.add_argument("--slurm-id-end", type=int, required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--out", type=Path, required=True)
    return ap.parse_args()


def final_blimp(path: Path) -> float:
    """The run's end-of-training full-BLiMP best-temperature accuracy."""
    hits = FINAL_BLIMP.findall(path.read_text())
    if len(hits) != 1:
        raise SystemExit(
            f"{path}: expected exactly one final [blimp-test] "
            f"best_temp_avg_uid_accuracy line, found {len(hits)}")
    return float(hits[0])


def main():
    args = parse_args()
    ids = list(range(args.slurm_id_start, args.slurm_id_end + 1))
    expected = len(DEFAULT_VARIANTS) * len(args.seeds)
    if len(ids) != expected:
        raise SystemExit(f"job-id range covers {len(ids)} runs, "
                         f"expected {expected} ({len(DEFAULT_VARIANTS)} variants "
                         f"x {len(args.seeds)} seeds)")

    finals, i = {}, 0
    for _slug, key in DEFAULT_VARIANTS:
        finals[key] = {}
        for seed in args.seeds:
            log = args.log_dir / args.log_pattern.format(ids[i])
            if not log.exists():
                raise SystemExit(f"missing log for {key} seed{seed}: {log}")
            finals[key][f"seed{seed}"] = final_blimp(log)
            i += 1

    record = {"final_blimp_best_temp_avg": finals, "note": NOTE}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # No trailing newline: matches the committed record byte-for-byte.
    with args.out.open("w") as f:
        json.dump(record, f, indent=1)
    print(f"wrote {args.out}  ({len(ids)} runs)")


if __name__ == "__main__":
    main()
