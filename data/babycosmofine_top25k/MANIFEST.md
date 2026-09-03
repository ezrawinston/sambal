# babycosmofine_top25k — manifest

The vocab-filtered control corpus (Table 9 of the ICML 2026 paper): the source corpus with every
sentence containing a word outside wordfreq's top-25k English list removed —
built by [`build_top25k.py`](build_top25k.py)
([Guide 3.6](../../reproduce/icml2026/DATA.md#36-vocab-filtered-control-corpus)) and trained at three sizes
([Guide 2.2](../../reproduce/icml2026/TRAINING.md#22-vocab-filtered-control-runs--table-9)).

| file | fingerprint | from |
|---|---|---|
| `train_top25k.jsonl` | 20,084 records; 28,523,019 bytes; md5 `d262acbde1c797ae9cc060464f080125` | `build_top25k.py` over `babycosmofine/train.jsonl` |
| `train_top25k_10M_tokenized.bin` | 20,084 documents; 6,885,441 tokens | tokenization ([Guide 3.7](../../reproduce/icml2026/DATA.md#37-tokenization)) |
