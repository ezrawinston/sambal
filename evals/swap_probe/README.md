# Semantic-plausibility swap probe

Paired grammaticality/plausibility probe: for each item, role-swapped sentence
variants separate the grammatical margin from the world-knowledge margin
(ΔG vs ΔK). [`paired_knowledge_probe.py`](paired_knowledge_probe.py) holds
the items and the scoring; it compares two models, each built through the
[`evals.backends.LanguageModel`](../backends/base.py) interface
([Guide 1.6](../../reproduce/icml2026/EVALUATION.md#16-swap-probe--table-3-figure-1)).
