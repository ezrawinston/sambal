# babycosmofine — manifest

The source pretraining corpus (Charpentier & Samuel, 2024).

| file | fingerprint | from |
|---|---|---|
| `train.jsonl` | 21,933 records; 60,640,109 bytes; md5 `72d263b81a7e5e81c48fcaee483fb5dd` | [`ltg/babylm-2024-baby-cosmo-fine-10m`](https://huggingface.co/datasets/ltg/babylm-2024-baby-cosmo-fine-10m) @ `5179e7ac0b6be2083ed03444a3a8c3d2c96061a2`, via [`fetch.sh`](fetch.sh) ([Guide 3.2](../../reproduce/icml2026/DATA.md#32-source-corpus)) |
| `train_10M_tokenized.bin` | 21,933 documents; 15,759,951 tokens | tokenization ([Guide 3.7](../../reproduce/icml2026/DATA.md#37-tokenization)), or the [dataset release](https://huggingface.co/datasets/ezrawinston/babycosmofine-sambal) |
| `train_10M_tokenized_with_reflexives_number.bin` | 22,617 documents; 15,766,835 tokens | the bin above with the 684-row augmentation set appended ([Guide 3.8](../../reproduce/icml2026/DATA.md#38-reflexive-augmented-corpora)) |
