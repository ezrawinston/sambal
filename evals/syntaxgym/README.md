# SyntaxGym

Targeted syntactic evaluation (25 of the 31 SyntaxGym test suites) plus the
reflexive-binding diagnostics
([Guide 1.2](../../reproduce/icml2026/EVALUATION.md#12-benchmark-data-one-time-fetches),
[Guide 1.3](../../reproduce/icml2026/EVALUATION.md#13-syntaxgym--table-2-sg-columns-table-6-app-d),
[Guide 1.10](../../reproduce/icml2026/EVALUATION.md#110-reflexive-probes--app-d-diagnostics)).

## Layout

- [`syntaxgym_eval.py`](syntaxgym_eval.py) — the suite scorer, for any model
  behind [`evals.backends.LanguageModel`](../backends/base.py): the trainers
  call it for their inline and final evaluations, and its command line scores
  a list of checkpoints.
- [`fetch_data.py`](fetch_data.py) — populates the git-ignored
  `data/syntaxgym/` with the full suites from the SyntaxGym distribution,
  pinned to `cpllab/syntactic-generalization@2f42038477a9f03ab66f308a963ed49864cd8111`.
- [`syntaxgym_fast/`](syntaxgym_fast/) — the frozen fast subset every
  training run's inline validation scored.
  [`create_fast_subset.py`](create_fast_subset.py) regenerates it exactly
  (seed 42, 20% per suite, at least 2 items).
- [`reflexive_probes/`](reflexive_probes/) — the reflexive diagnostic
  probes: BLiMP-format minimal pairs, the ICML 2026 paper's App. D targeted
  diagnostics, scored
  with [`evals/blimp/blimp_eval.py`](../blimp/blimp_eval.py) pointed at that
  directory:
  - [`reflexive_attraction_pl_attractor.jsonl`](reflexive_probes/reflexive_attraction_pl_attractor.jsonl)
    — singular controller with an intervening plural noun (the interference
    diagnostic; 180 pairs)
  - [`reflexive_attraction_sg_attractor.jsonl`](reflexive_probes/reflexive_attraction_sg_attractor.jsonl)
    — singular-attractor control
  - [`reflexive_fixed_himself_head_number.jsonl`](reflexive_probes/reflexive_fixed_himself_head_number.jsonl) /
    [`reflexive_fixed_themselves_head_number.jsonl`](reflexive_probes/reflexive_fixed_themselves_head_number.jsonl)
    — fixed-form licensing controls (head-number tests)
  - `reflexive_no_attractor_neutralNP_*.jsonl` — no-attractor controls
    (regenerable by [`make_debug_suite.py`](make_debug_suite.py), seeded)
  - [`reflexive_no_attractor_pronoun_antecedent.jsonl`](reflexive_probes/reflexive_no_attractor_pronoun_antecedent.jsonl)
    — pronoun-antecedent no-attractor variant

- [`aug_sets/`](aug_sets/) — the reflexive / number augmentation sentence
  sets consumed by [`reproduce/icml2026/reflexives/`](../../reproduce/icml2026/reflexives/).
- [`count_items.py`](count_items.py) — item-count utility.
