# LoRA fine-tuning and small-domain from-scratch training

LotR (Table 4) and PubMed (Table 11)
([Guide 2.7](../TRAINING.md#27-lora-fine-tuning-and-small-domain-from-scratch-training--tables-4-11)).

- `finetune_{gptbert,sambal}[_pubmed]_slurm.sh <rank> <alpha>` — LoRA fine-tuning of the arm's long-regime EMA checkpoint.
- [`sweep_lotr.sh`](sweep_lotr.sh) / [`sweep_pubmed.sh`](sweep_pubmed.sh) — the from-scratch grids, one [`run_lotr_experiment.sh`](run_lotr_experiment.sh) / [`run_pubmed_experiment.sh`](run_pubmed_experiment.sh) job per cell.
- [`train_lotr_slurm.sh`](train_lotr_slurm.sh) — the `lotr_tiny_from_scratch` cell, launched on its own.
- [`gen_lotr_sents.sh`](gen_lotr_sents.sh) / [`gen_lotr_sents_beam.sh`](gen_lotr_sents_beam.sh) — the appendix's generation samples.
