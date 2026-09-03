# gpt-bert (upstream snapshot)

The GPT-BERT training/evaluation stack, copied from
https://github.com/ltgoslo/gpt-bert at pinned commit
`9fc15074d5e2620ea4ad0871d57454c0f35e3ff0`, followed by this project's
modifications.

Modified upstream files:

- [`corpus_tokenization/tokenize_corpus.py`](corpus_tokenization/tokenize_corpus.py)
  — reads `document["text"]` from
  jsonl corpora (upstream read plain-text lines); tokenizer defaults point
  at the committed [`gpt-bert-babylm-small/tokenizer.json`](gpt-bert-babylm-small/tokenizer.json)
  (upstream
  defaulted to `tokenizers/tokenizer_100M.json`)
- [`corpus_tokenization/README.md`](corpus_tokenization/README.md) — flag
  documentation aligned with those
  defaults
- [`pretraining/dataset.py`](pretraining/dataset.py) — dataset handling used by the trainers
- [`pretraining/model_logging.py`](pretraining/model_logging.py) — rank
  detection via `RANK` (upstream read
  `SLURM_PROCID`)
- [`pretraining/train_10m.py`](pretraining/train_10m.py) — wandb project/entity
  from the environment;
  creates the checkpoint output directory at save time 

Added files:

- `VENDORED.md` (this file)
- [`gpt-bert-babylm-small/tokenizer.json`](gpt-bert-babylm-small/tokenizer.json) — the upstream
  `ltg/gpt-bert-babylm-small` tokenizer, from that repo at
  revision `4f21977cd05d9ed6bba16aa13e6c94d551c2a456`
  (md5 `9b49561a31e76c1726fc2dfcb3c004bd`)
- [`configs/tiny.json`](configs/tiny.json),
  [`configs/mini.json`](configs/mini.json) — small-model configs used by the
  from-scratch grids
- `configs/scaling_configs/{05m,10m,14m,30m}.json` — the scaling-grid model
  sizes (generator:
  [`reproduce/icml2026/scaling/gen_configs.py`](../../reproduce/icml2026/scaling/gen_configs.py))
- [`pretraining/train_v1.py`](pretraining/train_v1.py) — the short-regime trainer (inline BLiMP/
  SyntaxGym evaluation, final test evaluation)
- [`pretraining/train_long.py`](pretraining/train_long.py) — the long-regime 8-GPU trainer
- [`pretraining/train_lotr.py`](pretraining/train_lotr.py) — small-domain from-scratch trainer
- [`pretraining/lora.py`](pretraining/lora.py),
  [`pretraining/finetune_lora.py`](pretraining/finetune_lora.py) — LoRA adapter
  fine-tuning
- [`pretraining/gen_text.py`](pretraining/gen_text.py),
  [`pretraining/gen_text_beam.py`](pretraining/gen_text_beam.py) — generation
  sampling scripts
