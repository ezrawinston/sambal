# pubmed — manifest

PubMed abstracts — the second small-domain adaptation corpus (Table 11 of the ICML 2026 paper), sized
to the LotR corpus ([Guide 3.9](../../reproduce/icml2026/DATA.md#39-domain-corpora-lotr-pubmed)).

| file | fingerprint | from |
|---|---|---|
| [`split_indices.json`](split_indices.json) | committed | the recorded document permutation of the ~80/10/10 split (2,603 / 325 / 326 documents) |
| `pubmed_abstracts.jsonl` | 3,254 abstracts; 575,129 words; 3,949,312 bytes; md5 `20a81c7760d2327b257554b2f6daae67` | [`get_pubmed.py`](../../data/pubmed/get_pubmed.py): abstracts streamed from [`casinca/PUBMED_title_abstracts_2019_baseline`](https://huggingface.co/datasets/casinca/PUBMED_title_abstracts_2019_baseline) @ `5b8dcdcf6657dc120049448261445a5a9848fc8b` up to 575,230 whitespace words (the LotR corpus's count), one `{"text": ...}` per line |
| `pubmed_abstracts_train.bin` | 2,603 documents; 871,736 tokens | [`split_pubmed.py`](../../data/pubmed/split_pubmed.py): the recorded split, tokenized |
| `pubmed_abstracts_dev.bin` | 325 documents; 104,584 tokens | " |
| `pubmed_abstracts_test.bin` | 326 documents; 110,583 tokens | " |
