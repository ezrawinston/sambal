# SNLI

Natural-language-inference probing (SNLI and the SNLI-hard subset) on frozen
model representations. [`snli_probe.py`](snli_probe.py) trains a multinomial
logistic-regression probe on mean-pooled sentence representations from a
frozen checkpoint and evaluates 3-way NLI accuracy; probe C is selected on
dev from {0.1, 1.0, 10.0} ([Guide 1.8](../../reproduce/icml2026/EVALUATION.md#18-snli-probe--table-8-app-e)).
The model is any backend implementing
[`evals.backends.LanguageModel`](../backends/base.py).

## Data

[`fetch_data.py`](fetch_data.py) downloads SNLI 1.0 into
`evals/snli/downloads/snli_1.0/` — the three main splits from the
`uclnlp/inferbeddings` mirror and the hard subset from Stanford — and checks each file against the md5 of the copy the
ICML 2026 paper's runs used:

| file | md5 |
|---|---|
| `snli_1.0_train.jsonl` | `ff0cea1eb2dd6d4cec2d5698f6f66ee5` |
| `snli_1.0_dev.jsonl` | `b23798a0751d9a3dccac16a232496ca5` |
| `snli_1.0_test.jsonl` | `2392a35de49893c84b08fb92de6fc5e4` |
| `snli_1.0_test_hard.jsonl` | `75430b13d2a95c752935cb993d00d7fb` |
