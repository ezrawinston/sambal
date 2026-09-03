# lotr — manifest

The Lord of the Rings trilogy — the small-domain adaptation corpus (Table 4 of the ICML 2026 paper).
The text is under copyright: supply your own plain-text copy of the trilogy as
`lotr.txt` (editions can be found on GitHub), then chunk and split it
([Guide 3.9](../../reproduce/icml2026/DATA.md#39-domain-corpora-lotr-pubmed)). A copy
with a different checksum still runs, but its splits will not match the counts
below.

| file | fingerprint | from |
|---|---|---|
| `lotr.txt` | 3,262,595 bytes; md5 `0f803a1fd5a5e652d7324014a76743ec`; sha256 `18350dd7a8b3f2638aa47b88bb87f7c6f9e850fb0103f53d964dd8018ca2e208` | you |
| `lotr.jsonl` | 1,156 chunks; 3,057,790 bytes; md5 `29c6566f58e09037794f138a8dc68c6a` | [`chunk_lotr.py`](../../data/lotr/chunk_lotr.py): ~510-word chunks |
| `lotr_train.bin` | 924 chunks; 675,670 tokens | [`split_lotr.py`](../../data/lotr/split_lotr.py): a deterministic 80/10/10 prefix split by chunk, tokenized |
| `lotr_dev.bin` | 116 chunks; 84,942 tokens | " |
| `lotr_test.bin` | 116 chunks; 84,352 tokens | " |
