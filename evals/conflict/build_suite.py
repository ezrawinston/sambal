#!/usr/bin/env python3
"""Build a role-reversal grammar-vs-plausibility conflict suite.

This script:
  1) samples many swap-only template candidates from curated semantic frames
  2) scores them with a candidate model (sentence log-probability sums at one
     temperature)
  3) keeps items where:
       - the model passes the *control* (grammar is easy): control_good >> bad
       - the model is *lured* on the conflict: bad >> conflict_good
  4) optionally scores a second model and can require it to pass the control
     and/or prefer the grammatical conflict_good

Outputs:
  - out_conflicts: minimal pairs (grammatical vs ungrammatical), swap-only
  - out_controls:  minimal pairs that isolate the grammar phenomenon (plausible)

Example
-------
    python evals/conflict/build_suite.py \
        --backend gptbert --backend-arg checkpoint=CANDIDATE.bin \
        --other-backend-arg checkpoint=CONTROL_PASSING.bin \
        --out_conflicts evals/conflict/pool/conflict_pool.jsonl \
        --out_controls evals/conflict/controls.jsonl \
        --n_sva 200 --n_det 200 --n_passive 200 --tau 0.1
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from evals.backends import LanguageModel  # noqa: E402
from evals.common import add_backend_arguments, backend_from_args  # noqa: E402
from conflict_rr_suite.builder import (  # noqa: E402
    ScoredItem,
    build_suite,
    write_jsonl_conflicts,
    write_jsonl_controls,
)


def parse_arguments() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # the model that judges plausibility
    add_backend_arguments(ap)
    # optional second model, present when at least one --other-backend-arg is given
    add_backend_arguments(ap, prefix="other-", role="control-passing model", with_device=False)

    # output
    ap.add_argument("--out_conflicts", type=Path, required=True)
    ap.add_argument("--out_controls", type=Path, required=True)
    ap.add_argument("--prefix", type=str, default="rr")

    # suite size
    ap.add_argument("--n_sva", type=int, default=200)
    ap.add_argument("--n_det", type=int, default=200)
    ap.add_argument("--n_passive", type=int, default=0, help="number of passive-role-reversal items")
    ap.add_argument("--seed", type=int, default=0)

    # sampling
    ap.add_argument("--pool_multiplier", type=int, default=200, help="candidates per desired item")
    ap.add_argument("--allow_animate_objects", action="store_true", help="include chase-style frames")

    # scoring
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--batch_items", type=int, default=64, help="candidates scored per selection round")

    # selection thresholds; --tau sets all three margins at once, explicit flags win
    ap.add_argument("--tau", type=float, default=None,
                    help="shorthand for --ctrl_margin, --conf_margin and --other_ctrl_margin")
    ap.add_argument("--ctrl_margin", type=float, default=None, help="default: 1.0")
    ap.add_argument("--conf_margin", type=float, default=None, help="default: 0.5")

    # secondary model constraints
    ap.add_argument("--require_other_ctrl", action="store_true")
    ap.add_argument("--other_ctrl_margin", type=float, default=None, help="default: 0.5")
    ap.add_argument("--require_other_prefers_good", action="store_true")

    # diversity
    ap.add_argument("--max_per_frame", type=int, default=30)
    ap.add_argument("--max_per_noun_pair", type=int, default=3)

    return ap.parse_args()


def _margin(explicit: Optional[float], tau: Optional[float], default: float) -> float:
    """The explicit flag when it was given, else ``--tau``, else the flag's default."""
    if explicit is not None:
        return explicit
    return default if tau is None else tau


def _row(sc: ScoredItem) -> Dict:
    """One selected item: the sentences scored, their scores and the margins."""
    it = sc.item
    row = {
        "family": it.family,
        "frame": it.meta.get("frame"),
        "sentence_good": it.sentence_good,
        "sentence_bad": it.sentence_bad,
        "control_good": it.control_good,
        "score_good": sc.base_good,
        "score_bad": sc.base_bad,
        "score_control_good": sc.base_ctrl_good,
        "delta_conf": sc.base_delta_conf,
        "delta_ctrl": sc.base_delta_ctrl,
        "prediction": "good" if sc.base_delta_conf > 0 else "bad",
    }
    if sc.other_good is not None:
        row.update(other_score_good=sc.other_good, other_score_bad=sc.other_bad,
                   other_score_control_good=sc.other_ctrl_good,
                   other_delta_conf=sc.other_delta_conf, other_delta_ctrl=sc.other_delta_ctrl)
    return row


def generate(base_model: LanguageModel, other_model: Optional[LanguageModel],
             out_conflicts: Path, out_controls: Path, prefix: str = "rr", **options) -> Dict:
    """Build the suite, write both jsonl files, and return summary / rows / protocol."""
    conflicts, controls = build_suite(base_model=base_model, other_model=other_model, **options)
    write_jsonl_conflicts(out_conflicts, conflicts, prefix=prefix)
    write_jsonl_controls(out_controls, controls, prefix=prefix + "_ctrl")
    return {
        "summary": {"n_conflicts": len(conflicts), "n_controls": len(controls)},
        "rows": [_row(sc) for sc in conflicts],
        "protocol": dict(options, prefix=prefix, model=base_model.name,
                         other_model=other_model.name if other_model is not None else None,
                         device=str(base_model.device),
                         out_conflicts=str(out_conflicts), out_controls=str(out_controls)),
    }


def main():
    args = parse_arguments()

    ctrl_margin = _margin(args.ctrl_margin, args.tau, 1.0)
    conf_margin = _margin(args.conf_margin, args.tau, 0.5)
    other_ctrl_margin = _margin(args.other_ctrl_margin, args.tau, 0.5)

    base_model = backend_from_args(args)
    print(f"candidate model: {base_model.name} on {base_model.device}")

    other_model = None
    if args.other_backend_arg:
        other_model = backend_from_args(args, prefix="other-")
        print(f"control-passing model: {other_model.name} on {other_model.device}")

    result = generate(
        base_model,
        other_model,
        out_conflicts=args.out_conflicts,
        out_controls=args.out_controls,
        prefix=args.prefix,
        n_sva=args.n_sva,
        n_det=args.n_det,
        n_passive=args.n_passive,
        seed=args.seed,
        pool_multiplier=args.pool_multiplier,
        only_inanimate_objects=not args.allow_animate_objects,
        temperature=args.temperature,
        batch_items=args.batch_items,
        ctrl_margin=ctrl_margin,
        conf_margin=conf_margin,
        require_other_ctrl=args.require_other_ctrl,
        other_ctrl_margin=other_ctrl_margin,
        require_other_prefers_good=args.require_other_prefers_good,
        max_per_frame=args.max_per_frame,
        max_per_noun_pair=args.max_per_noun_pair,
    )

    summary = result["summary"]
    print(f"Selected {summary['n_conflicts']} conflict items "
          f"and {summary['n_controls']} control items.")
    print(f"Wrote conflicts -> {args.out_conflicts}")
    print(f"Wrote controls  -> {args.out_controls}")


if __name__ == "__main__":
    main()
