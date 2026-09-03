#!/usr/bin/env python
"""Derive the evaluated conflict suite from the candidate pool.

The benchmark is built in two stages (see ``evals/conflict/README.md``):

1. ``build_suite.py`` generates a margin-gated *candidate pool* (shipped at
   ``evals/conflict/pool/conflict_pool.jsonl``);
2. this script applies the ICML 2026 paper's retention criterion (ii) -- keep only the
   conflict items on which the oracle model is actually lured by the
   ungrammatical-but-plausible sentence -- by scoring the whole pool with that
   model and dropping every conflict item it answers *correctly*.

The scoring path is the one the reported accuracies come from: sentence
log-probability sums from ``LanguageModel.sequence_logprobs`` over the same
61-temperature grid the reported sweeps use, with the operating temperature
the one that gets the most pairs right. Nothing here hardcodes an item list:
the surviving set is whatever the model's verdicts produce.

Example
-------
    python evals/conflict/derive_conflicts.py \
        --backend gptbert \
        --backend-arg checkpoint=lm/gpt-bert/trained_models/gptbert_babycosmofine_long_ema.bin \
        --out evals/conflict/conflicts.jsonl

Runs on CPU in about a minute for the shipped 213 + 213 pool.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from evals.backends import LanguageModel  # noqa: E402
from evals.common import TEMPERATURE_STEP, add_backend_arguments, backend_from_args, temperature_grid  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def parse_arguments():
    p = argparse.ArgumentParser(
        description="Derive the evaluated conflict suite from the candidate pool "
                    "by dropping the conflict items the oracle model gets right.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pool", type=pathlib.Path,
                   default=REPO_ROOT / "evals/conflict/pool/conflict_pool.jsonl",
                   help="Candidate conflict pool (jsonl).")
    p.add_argument("--controls", type=pathlib.Path,
                   default=REPO_ROOT / "evals/conflict/controls.jsonl",
                   help="Matched control pairs (jsonl). Scored alongside the pool "
                        "because the operating temperature is chosen over the whole "
                        "evaluation.")
    add_backend_arguments(p, role="plausibility oracle")
    p.add_argument("--out", type=pathlib.Path,
                   default=REPO_ROOT / "evals/conflict/conflicts.jsonl",
                   help="Where to write the derived suite.")
    p.add_argument("--compare", type=pathlib.Path, default=None,
                   help="Reference suite to diff the derivation against. "
                        "Defaults to --out when that file already exists.")
    p.add_argument("--dry-run", action="store_true",
                   help="Derive and compare, but do not write --out.")
    p.add_argument("--force", action="store_true",
                   help="Write --out even when the comparison reports a mismatch. "
                        "Without it a mismatching derivation is reported but the "
                        "committed suite is left in place.")
    p.add_argument("--verdicts-out", type=pathlib.Path, default=None,
                   help="Optional path for the per-item, per-temperature verdict dump.")
    return p.parse_args()


def read_rows(path):
    """Return [(uid, raw_line, parsed_row)], preserving the exact bytes of each line."""
    rows = []
    with open(path, "r") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows.append((json.loads(line)["UID"], line, json.loads(line)))
    return rows


def score_pair(model: LanguageModel, sentence_good: str, sentence_bad: str,
               temperatures: Sequence[float]) -> Tuple[List[List[float]], List[int]]:
    """Both sentences' sums per temperature, and whether the good one ranks first."""
    seqs = [model.encode(sentence_good), model.encode(sentence_bad)]
    scores = model.sequence_logprobs(seqs, temperatures)          # [2, n_temperatures]
    ranking = torch.argsort(scores.t(), dim=1, descending=True).tolist()
    return scores.tolist(), [1 if order[0] == 0 else 0 for order in ranking]


def derive(model: LanguageModel, pool_rows, control_rows, heartbeat: float = 15.0) -> Dict:
    """Score pool and controls on the grid and keep the conflict items the model gets wrong.

    Returns ``summary`` (counts and the operating temperature), ``rows`` (one
    record per scored pair) and ``protocol``, plus ``verdicts`` -- the payload
    of the ``--verdicts-out`` dump.
    """
    temperatures = temperature_grid().tolist()
    n_temps = len(temperatures)

    # correct[t] counts every scored pair (conflicts AND controls) at temperature t.
    # The sweep's own selection statistic is the mean of the per-UID accuracies, and
    # every UID here holds exactly one pair, so that mean is a monotone function of
    # this count over a fixed denominator -- the two selections coincide.
    correct = [0] * n_temps
    verdicts: Dict[str, List[int]] = {}
    scored: List[Tuple[str, str, dict, List[List[float]], List[int]]] = []

    t0 = time.time()
    n_done = 0
    last_beat = t0
    for tag, rows in (("conflicts", pool_rows), ("controls", control_rows)):
        for uid, _raw, row in rows:
            scores, hits = score_pair(model, row["sentence_good"], row["sentence_bad"], temperatures)
            verdicts[tag + "|" + uid] = hits
            scored.append((tag, uid, row, scores, hits))
            for i, hit in enumerate(hits):
                correct[i] += hit
            n_done += 1
            now = time.time()
            if now - last_beat >= heartbeat:
                el = now - t0
                print(f"heartbeat: {n_done} pairs, {el:.0f}s elapsed, "
                      f"{el / max(n_done, 1):.2f}s/pair", flush=True)
                last_beat = now
    print(f"scored {n_done} pairs in {time.time() - t0:.0f}s", flush=True)

    # Operating temperature = the one maximising the count above.
    #
    # TIE RULE: on ties take the FIRST (lowest) temperature index, the occurrence
    # torch.argmax returns. The committed pool's evaluation lands on a wide tie (the
    # same maximum is attained at several temperatures) and the *first* of those is
    # the temperature whose verdicts define the suite -- picking any other member of
    # the tie changes which items are dropped. max() over enumerate() below keeps the
    # first index for the same reason, because it only replaces the incumbent on a
    # strict improvement.
    best_i = max(range(n_temps), key=lambda i: (correct[i], -i))
    best_t = best_i * TEMPERATURE_STEP

    rows_out = []
    for tag, uid, row, scores, hits in scored:
        good, bad = scores[0][best_i], scores[1][best_i]
        rows_out.append({
            "uid": uid,
            "set": tag,
            "sentence_good": row["sentence_good"],
            "sentence_bad": row["sentence_bad"],
            "score_good": good,
            "score_bad": bad,
            "margin": good - bad,
            "prediction": "good" if hits[best_i] else "bad",
            "correct": bool(hits[best_i]),
            "kept": tag == "conflicts" and not hits[best_i],
        })

    n_kept = sum(1 for r in rows_out if r["kept"])
    summary = {
        "n_pool": len(pool_rows),
        "n_controls": len(control_rows),
        "n_pairs": n_done,
        "operating_temperature": best_t,
        "operating_temperature_index": best_i,
        "n_correct": correct[best_i],
        "accuracy": correct[best_i] / n_done if n_done else float("nan"),
        "n_dropped": len(pool_rows) - n_kept,
        "n_kept": n_kept,
    }
    protocol = {
        "model": model.name,
        "device": str(model.device),
        "n_temperatures": n_temps,
        "temperature_step": TEMPERATURE_STEP,
        "tie_rule": "first index",
    }
    return {"summary": summary, "rows": rows_out, "protocol": protocol,
            "verdicts": {"temperatures": temperatures,
                         "correct_per_temperature": correct,
                         "verdicts": verdicts}}


def main():
    args = parse_arguments()

    model = backend_from_args(args)
    print(f"model: {model.name}", flush=True)
    print(f"device: {model.device}", flush=True)

    pool_rows = read_rows(args.pool)
    control_rows = read_rows(args.controls)
    print(f"pool: {len(pool_rows)} conflict items", flush=True)
    print(f"controls: {len(control_rows)} control items", flush=True)
    print(f"temperatures: {len(temperature_grid())}", flush=True)

    result = derive(model, pool_rows, control_rows)
    result["protocol"].update(pool=str(args.pool), controls=str(args.controls),
                              backend=args.backend, backend_arg=list(args.backend_arg))
    summary = result["summary"]
    correct = result["verdicts"]["correct_per_temperature"]
    best_i = summary["operating_temperature_index"]
    best_t = summary["operating_temperature"]

    tied = [j * TEMPERATURE_STEP for j in range(len(correct)) if correct[j] == correct[best_i]]
    print(f"best temperature: {best_t:.2f} (index {best_i}); "
          f"{summary['n_correct']}/{summary['n_pairs']} pairs correct "
          f"= {summary['accuracy'] * 100:.4f}%", flush=True)
    print(f"tied temperatures at that maximum: "
          f"{', '.join(f'{t:.2f}' for t in tied)}", flush=True)

    raw_by_uid = {uid: raw for uid, raw, _p in pool_rows}
    kept_uids = [r["uid"] for r in result["rows"] if r["kept"]]
    kept_lines = [raw_by_uid[uid] for uid in kept_uids]
    print(f"dropped (model correct at t={best_t:.2f}): {summary['n_dropped']}", flush=True)
    print(f"kept: {len(kept_lines)}", flush=True)

    fam = Counter(json.loads(line)["linguistics_term"] for line in kept_lines)
    for key in sorted(fam):
        print(f"  {key}: {fam[key]}", flush=True)

    compare_path = args.compare
    if compare_path is None and args.out.exists():
        compare_path = args.out
    status = 0
    if compare_path is not None and pathlib.Path(compare_path).exists():
        ref_rows = read_rows(compare_path)
        ref_uids = [uid for uid, _r, _p in ref_rows]
        ref_by_uid = {uid: raw for uid, raw, _p in ref_rows}
        derived_by_uid = dict(zip(kept_uids, kept_lines))
        missing = [u for u in ref_uids if u not in derived_by_uid]
        extra = [u for u in kept_uids if u not in ref_by_uid]
        changed = [u for u in kept_uids
                   if u in ref_by_uid and ref_by_uid[u] != derived_by_uid[u]]
        print(f"\ncompare against {compare_path}: "
              f"{len(ref_rows)} reference rows vs {len(kept_lines)} derived rows",
              flush=True)
        if missing or extra or changed:
            status = 1
            print("MISMATCH -- the derivation does not reproduce the reference set.",
                  flush=True)
            for uid in missing:
                print(f"  only in reference (derivation dropped it): {uid}", flush=True)
            for uid in extra:
                print(f"  only in derivation (reference dropped it): {uid}", flush=True)
            for uid in changed:
                print(f"  row bytes differ: {uid}", flush=True)
            print("A handful of boundary flips usually means numeric drift "
                  "(different hardware, torch build, or dtype); a large or "
                  "systematic difference means a different model.",
                  flush=True)
        else:
            print("MATCH -- derived set is byte-identical to the reference.", flush=True)

    if args.verdicts_out is not None:
        with open(args.verdicts_out, "w") as fh:
            json.dump(result["verdicts"], fh)
        print(f"wrote verdicts: {args.verdicts_out}", flush=True)

    if args.dry_run:
        print("dry run: --out not written", flush=True)
    elif status and not args.force:
        print(f"not writing {args.out}: the derivation differs from the reference "
              f"above. Re-run with --force to overwrite it, or with a different "
              f"--out to keep the committed suite.", flush=True)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            fh.writelines(kept_lines)
        print(f"wrote {len(kept_lines)} rows: {args.out}", flush=True)

    return status


if __name__ == "__main__":
    sys.exit(main())
