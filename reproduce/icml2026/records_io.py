#!/usr/bin/env python3
"""Loader for the committed records, shared by the figure and table scripts.

Every figure/table script reads its inputs through this module so that a missing
record produces one clean line naming the file and the step that produces
it, instead of a bare traceback:

    $ python figures/fig1_swap_probe.py
    missing record: reproduce/icml2026/records/swap_probe/paired_probe.json
      -> produced by evals/swap_probe/paired_knowledge_probe.py (both arms'
         checkpoints); see reproduce/icml2026/RESULTS.md

Paths passed to the loaders are relative to `reproduce/icml2026/records/`.
Scripts import it __file__-relative, so the documented invocation
(`python figures/figX.py` from `reproduce/icml2026/`) works from any cwd:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from records_io import load_json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RECORDS_DIR = Path(__file__).resolve().parent / "records"
README = "reproduce/icml2026/RESULTS.md"

# Rendered outputs. The copies under reproduce/icml2026/out/ ship with the
# repository; the figure/table scripts overwrite them in place.
OUT_DIR = Path(__file__).resolve().parent / "out"
FIGURES_OUT = OUT_DIR / "figures"
TABLES_OUT = OUT_DIR / "tables"

# What produces each committed record, summarised from RESULTS.md and the
# producing guide sections. Keys are paths relative to RECORDS_DIR; a key
# ending in "/" is a directory-level fallback for every file beneath it.
PRODUCERS = {
    "swap_probe/paired_probe.json":
        "evals/swap_probe/paired_knowledge_probe.py (both arms' checkpoints)",
    "ewok/":
        "evals/ewok/ewok_eval.py (best-temperature report, filtered EWoK)",
    "infonce/":
        "evals/infonce/run_infonce.py (mean pooling, 12 layers, UD-EWT dev)",
    "scaling/scaling_results.json":
        "reproduce/icml2026/assemble/assemble_scaling_results.py parsing the scaling-grid "
        "training logs",
    "blimp/":
        "full-BLiMP best-temperature scoring of each long checkpoint; the "
        "scoring command is in reproduce/icml2026/EVALUATION.md (1.4)",
    "syntaxgym/long_models_per_suite.json":
        "evals/syntaxgym/syntaxgym_eval.py on the two long checkpoints",
    "syntaxgym/short_regime_per_suite.json":
        "evals/syntaxgym/syntaxgym_eval.py (launched by reproduce/icml2026/scoring/"
        "eval_syntaxgym.sh) on the six short-regime checkpoints",
    "reflexives/":
        "the targeted-augmentation retrains under reproduce/icml2026/reflexives/ "
        "(final test-split SyntaxGym evaluations)",
    "conflict/":
        "evals/blimp/blimp_eval.py on the conflict suite (evals/conflict; "
        "conflicts.jsonl is derived from the 213-item candidate pool by "
        "evals/conflict/derive_conflicts.py)",
    "snli/":
        "evals/snli/snli_probe.py (probe on frozen sentence representations)",
    "lora/lora_runs.json":
        "lm/gpt-bert/pretraining/finetune_lora.py runs (pre/post fine-tuning "
        "perplexity and full-BLiMP accuracy)",
    "lora/from_scratch.json":
        "the from-scratch training grids on the LoTR and PubMed corpora",
    "ablations/blimp_finals.json":
        "short-regime training on the component-ablated corpus variants "
        "(3 seeds each; final best-temperature BLiMP)",
    "vocab_control/blimp.json":
        "short-regime training on the vocab-filtered control corpus at the 10% "
        "token budget (3 seeds x 3 sizes; end-of-training full BLiMP)",
    "entropy/entropy_matched_budget.json":
        "evals/entropy/matched_budget.py over the three corpora at one "
        "common word-token budget",
    "blimpvocab/inline_finals.json":
        "last-epoch inline fast-subset evaluations of the benchmark-vocabulary "
        "variant short runs",
}


def _producer(rel: str) -> str:
    if rel in PRODUCERS:
        return PRODUCERS[rel]
    for key, hint in PRODUCERS.items():
        if key.endswith("/") and rel.startswith(key):
            return hint
    return "an evaluation run recorded in " + README


def record_path(rel) -> Path:
    """Absolute path to a committed record, checked for existence.

    Exits with a one-line diagnostic (not a traceback) if the file is absent.
    """
    rel = str(rel)
    path = RECORDS_DIR / rel
    if not path.exists():
        sys.exit(
            f"missing record: reproduce/icml2026/records/{rel}\n"
            f"  -> produced by {_producer(rel)}; see {README}"
        )
    return path


def load_json(rel):
    """Parse a committed JSON record."""
    return json.loads(record_path(rel).read_text())


def load_text(rel) -> str:
    """Read a committed text record (reports, id lists)."""
    return record_path(rel).read_text()


# The swap probe's expected protocol: the run settings the probe
# writes into its own output, and the flags that produce them. Every reader
# of swap_probe/paired_probe.json must call check_swap_probe_protocol —
# the margin fields exist under any flag combination but mean different
# things, so an off-protocol json is structurally valid and silently wrong.
SWAP_PROBE_PROTOCOL = {
    "grammar_mode": "nonsense_gram",
    "counterbalance_order": True,
    "score_all_tokens": False,
}
SWAP_PROBE_PROTOCOL_FLAGS = (
    "--grammar_mode nonsense_gram --counterbalance_order   "
    "(and no --score_all_tokens)"
)


def check_swap_probe_protocol(data, rel="swap_probe/paired_probe.json"):
    """Refuse a probe json that was not produced under the expected protocol.

    Checks the run settings the probe records at top level, and - for a record
    old or hand-assembled enough to lack them - the per-row fields that only
    this protocol produces (`grammar_mode` on graded rows, and the
    counterbalanced dK component `knowledge_margin_cb`).
    """
    problems = []
    for key, want in SWAP_PROBE_PROTOCOL.items():
        if key not in data:
            problems.append(f"top-level {key!r} is missing")
        elif data[key] != want:
            problems.append(f"top-level {key} is {data[key]!r}, expected {want!r}")

    for arm in ("normal", "sambal"):
        rows = data.get("dual_pair", {}).get(arm, {}).get("rows", [])
        if not rows:
            problems.append(f"dual_pair -> {arm} -> rows is empty")
            continue
        graded = [r for r in rows if r.get("has_grammar")]
        bad_mode = {r.get("grammar_mode") for r in graded} - {SWAP_PROBE_PROTOCOL["grammar_mode"]}
        if bad_mode:
            problems.append(
                f"{arm}: rows carry grammar_mode {sorted(map(str, bad_mode))}, "
                f"expected {SWAP_PROBE_PROTOCOL['grammar_mode']!r}"
            )
        n_no_cb = sum(1 for r in rows if r.get("knowledge_margin_cb") is None)
        if n_no_cb:
            problems.append(
                f"{arm}: {n_no_cb}/{len(rows)} rows have no knowledge_margin_cb "
                f"(the counterbalanced dK component)"
            )

    if problems:
        sys.exit(
            "records/%s was not produced under the swap probe's expected protocol:\n  - %s\n"
            "  -> re-run evals/swap_probe/paired_knowledge_probe.py with:\n"
            "       %s\n"
            "     The margin fields exist under any flags but mean different "
            "things; see evals/swap_probe/README.md."
            % (rel, "\n  - ".join(problems), SWAP_PROBE_PROTOCOL_FLAGS)
        )

