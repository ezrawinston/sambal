# EWoK

World-knowledge minimal-pair probe. [`ewok_eval.py`](ewok_eval.py) scores any
directory of files in EWoK's format for any model behind
[`evals.backends.LanguageModel`](../backends/base.py); each item is scored as
`"Context1 Target1"` against `"Context1 Target2"`, with `Context1` as the
prefix ([Guide 1.5](../../reproduce/icml2026/EVALUATION.md#15-ewok--figure-2-43)). The items are
gated and not redistributable; they are fetched into the git-ignored
`downloads/` ([Guide 1.2](../../reproduce/icml2026/EVALUATION.md#12-benchmark-data-one-time-fetches)).
