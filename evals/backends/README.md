# backends

The model side of every evaluator. [`base.py`](base.py) defines
`LanguageModel` — encode/decode, per-token log-probabilities at a set of
temperatures, hidden states — and the `--backend` / `--backend-arg` resolution
the command lines share; [`gptbert.py`](gptbert.py) implements it for the
released checkpoints (scoring variants `mlm_shift`, the ICML 2026 paper's, plus `mlm`,
`causal` and `prefix`); [`conformance.py`](conformance.py) checks any
implementation against the interface contract. A new model family is one
module with a `LanguageModel` subclass that passes `assert_conformant`; the
evaluators need no change.
