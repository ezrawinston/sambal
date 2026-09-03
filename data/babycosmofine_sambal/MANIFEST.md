# babycosmofine_sambal — manifest

The sambal ablation of `babycosmofine/train.jsonl` — the
ICML 2026 paper's ablated corpus, doc-aligned with the source (profile
[`sambal/profiles/icml2026.json`](../../sambal/profiles/icml2026.json)).

| file | fingerprint | from |
|---|---|---|
| `train_sambal.jsonl` | 21,933 records; 60,776,246 bytes; md5 `aa6bc21ca98efcef8c909b771073089f` | the [dataset release](https://huggingface.co/datasets/ezrawinston/babycosmofine-sambal), or regeneration ([Guide 3.4](../../reproduce/icml2026/DATA.md#34-ablated-corpus)): 24 shards, seed = shard id, `shard_00.jsonl` … `shard_23.jsonl` concatenated in shard order — a regeneration reproduces the distribution, not these bytes |
| `train_sambal_10M_tokenized.bin` | 21,933 documents; 15,640,481 tokens | tokenization ([Guide 3.7](../../reproduce/icml2026/DATA.md#37-tokenization)), or the dataset release |
| `train_sambal_10M_tokenized_with_reflexives_number.bin` | 22,617 documents; 15,647,365 tokens | the bin above with the 684-row augmentation set appended ([Guide 3.8](../../reproduce/icml2026/DATA.md#38-reflexive-augmented-corpora)) |
