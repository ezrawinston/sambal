# Main pretraining

The launchers train the paper's main models
([Guide 2.1](../TRAINING.md#21-main-pretraining--the-table-26--figure-13-models)).

| launcher | what it trains |
|---|---|
| [`train_long_gptbert.sh`](train_long_gptbert.sh) | baseline arm, long regime (one 8-GPU node) |
| [`train_long_sambal.sh`](train_long_sambal.sh) | SAMBAL arm, long regime (one 8-GPU node) |
| `train_v1_gptbert_lr_seed.sh <lr> <seed> [train_bin]` | baseline arm, short regime (single GPU) |
| `train_v1_sambal_lr_seed.sh <lr> <seed> [train_bin]` | SAMBAL arm, short regime (single GPU) |
| `train_vocab_control.sh <05m\|14m\|30m> <seed>` | vocab-filtered control corpus, short regime at the Table 9 sizes (single GPU; [Guide 2.2](../TRAINING.md#22-vocab-filtered-control-runs--table-9)) |

The optional `[train_bin]` swaps in another tokenized corpus — the corpus
variants and the reflexive-augmentation retrains
([Guide 2.3](../TRAINING.md#23-component-ablation-retrains--table-7-table-8-row-3),
[Guide 2.6](../TRAINING.md#26-reflexive-augmentation-retrains--app-d)); a
relative path resolves from the repository root.
