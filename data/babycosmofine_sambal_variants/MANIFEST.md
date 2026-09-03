# babycosmofine_sambal_variants — manifest

The five component-ablation corpora (Table 7 of the ICML 2026 paper) and
the benchmark-vocabulary corpus (its §5): the `babycosmofine_sambal` generation with one override each
([Guide 3.5](../../reproduce/icml2026/DATA.md#35-corpus-variants)). Not distributed.
Each variant has its own subdirectory holding its 24 shards, their
concatenation `train_sambal_<variant>.jsonl`, and its tokenized
`train_sambal_<variant>_10M_tokenized.bin`:

| `<variant>/` | override | trained for (ICML 2026 paper) |
|---|---|---|
| `no_toinf` | `"skip_toinf_freeze": true` | Table 7 |
| `no_gender` | `"respect_gender": false` | Table 7 |
| `no_ud_roundtrip` | `"roundtrip_ud": false` | Table 7 |
| `no_humanness` | `"human_to_human": false` | Table 7 |
| `no_context_buckets` | `"ctx_lemma_gate": "off"` | Table 7, and Table 8's third row |
| `blimpvocab_flat` | `"ctx_lemma_gate": "flat"` (uniform sampling within the context bucket), plus `"allowed_vocab_path": "sambal/resources/blimpvocab.txt"` in the `paths` block | §5 |
