# evals

One directory per benchmark or probe suite: the runner that scores a
model on it, and — for the suites authored for the ICML 2026 paper — the
items and their generators. Externally published benchmarks are fetched into
a git-ignored directory beside their runner. Every runner takes its model
through [`evals.backends.LanguageModel`](backends/base.py), so it scores any model
with a backend; the released checkpoints use the `gptbert` one
([EVALUATION.md](../reproduce/icml2026/EVALUATION.md)).

| directory | evaluates |
|---|---|
| [`backends/`](backends/) | nothing — the `evals.backends.LanguageModel` interface the runners score through, its conformance checks, and the backend for the released models |
| [`blimp/`](blimp/) | BLiMP minimal-pair grammaticality |
| [`syntaxgym/`](syntaxgym/) | SyntaxGym targeted syntactic suites and the reflexive-binding diagnostics |
| [`ewok/`](ewok/) | EWoK world-knowledge minimal pairs |
| [`swap_probe/`](swap_probe/) | the semantic-plausibility swap probe — grammatical vs world-knowledge margins on role-swapped sentences |
| [`infonce/`](infonce/) | per-layer InfoNCE alignment of hidden states with UPOS tags vs lexical identity, on UD-EWT |
| [`snli/`](snli/) | SNLI and SNLI-hard probing on frozen representations |
| [`conflict/`](conflict/) | the grammar–plausibility conflict benchmark and its generator |
| [`entropy/`](entropy/) | word-level lexical entropy of a corpus |
