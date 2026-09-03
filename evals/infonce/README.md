# InfoNCE representation alignment

Per-layer InfoNCE alignment of hidden states with (a) UPOS tags and (b)
lexical identity, in a supervised-contrastive setup where positives share a
label. The ICML 2026 paper's Figure 3 plots the per-layer difference
(UPOS alignment − Lex alignment) for the two arms.

[`run_infonce.py`](run_infonce.py) is the runner ([Guide 1.7](../../reproduce/icml2026/EVALUATION.md#17-infonce--figure-3))
and [`compare_infonce.py`](compare_infonce.py) the cross-arm plotter;
[`data_ud.py`](data_ud.py) is the CoNLL-U loader and [`common.py`](common.py)
holds the pooling and InfoNCE helpers.

## Data

The probe reads a Universal Dependencies treebank in CoNLL-U format. The
paper used the English EWT dev split from Universal Dependencies release
2.16 (`en_ewt-ud-dev.conllu`, 1,805,545 bytes, md5
`cb1da95ff28a449cb9bc286c922e15e1`) ([Guide 1.2](../../reproduce/icml2026/EVALUATION.md#12-benchmark-data-one-time-fetches)).
