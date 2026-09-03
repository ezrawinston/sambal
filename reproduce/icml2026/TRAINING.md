# 2. Training the models

Install the sambal package and, for slurm submissions, name the partition, QOS and
CUDA module:

```bash
pip install -e .
export SBATCH_PARTITION=<your partition> SBATCH_QOS=<qos>   # if submitting with sbatch
export SAMBAL_CUDA_MODULE=cuda-12.9   # if the cluster uses environment modules (default shown)
```

Every `sbatch` command in this guide also runs with `bash` instead of `sbatch`.

The trainers read tokenized `.bin` corpora: regenerate them
([DATA.md](DATA.md)) or download them from the
[dataset release](https://huggingface.co/datasets/ezrawinston/babycosmofine-sambal):

```bash
hf download ezrawinston/babycosmofine-sambal --repo-type dataset \
    train_10M_tokenized.bin --local-dir ${SAMBAL_DATA_DIR:-data}/babycosmofine
hf download ezrawinston/babycosmofine-sambal --repo-type dataset \
    train_sambal.jsonl train_sambal_10M_tokenized.bin --local-dir ${SAMBAL_DATA_DIR:-data}/babycosmofine_sambal
```

They also read the benchmark data:

```bash
python evals/syntaxgym/fetch_data.py            # full SyntaxGym suites (pinned)
python evals/blimp/fetch_data.py                # full BLiMP (pinned) + the trainers' fast subsets
```

Checkpoints land in `lm/gpt-bert/trained_models/`, each job's log in `logs/`.

## 2.1 Main pretraining — the Table 2–6 / Figure 1–3 models

Long regime (one 8-GPU node each) and the short-regime seed trios (single
GPU each):

```bash
sbatch reproduce/icml2026/pretrain/train_long_gptbert.sh
sbatch reproduce/icml2026/pretrain/train_long_sambal.sh
for SEED in 0 1 2; do
  sbatch reproduce/icml2026/pretrain/train_v1_gptbert_lr_seed.sh 0.007 $SEED
  sbatch reproduce/icml2026/pretrain/train_v1_sambal_lr_seed.sh 0.007 $SEED
done
```

The long training regime takes around 12 hrs on 8 A100s, the short regime takes around 2 hours on 1 A100.

## 2.2 Vocab-filtered control runs — Table 9

3 sizes × 3 seeds on the control corpus
([3.6](DATA.md#36-vocab-filtered-control-corpus), tokenized first:
[3.7](DATA.md#37-tokenization)):

```bash
for SEED in 0 1 2; do
  sbatch reproduce/icml2026/pretrain/train_vocab_control.sh 05m $SEED
  sbatch reproduce/icml2026/pretrain/train_vocab_control.sh 14m $SEED
  sbatch reproduce/icml2026/pretrain/train_vocab_control.sh 30m $SEED
done
```

## 2.3 Component-ablation retrains — Table 7, Table 8 row 3

Per corpus variant ([3.5](DATA.md#35-corpus-variants)), 3 seeds on the variant's
tokenized bin:

```bash
for SEED in 0 1 2; do
  sbatch reproduce/icml2026/pretrain/train_v1_sambal_lr_seed.sh 0.007 $SEED \
      ${SAMBAL_DATA_DIR:-data}/babycosmofine_sambal_variants/no_toinf/train_sambal_no_toinf_10M_tokenized.bin
done
# repeat for no_gender, no_ud_roundtrip, no_humanness, no_context_buckets
```

The `no_context_buckets` seed-2 run is also Table 8's third-row model
([1.8](EVALUATION.md#18-snli-probe--table-8-app-e)).

## 2.4 Benchmark-vocabulary variant runs — §5 discussion

Short runs on the restricted-vocabulary corpus
([3.5](DATA.md#35-corpus-variants)) beside matched runs on the plain ablated
corpus:

```bash
for LR in 0.0035 0.007; do for SEED in 13 42; do
  sbatch reproduce/icml2026/pretrain/train_v1_sambal_lr_seed.sh $LR $SEED \
      ${SAMBAL_DATA_DIR:-data}/babycosmofine_sambal_variants/blimpvocab_flat/train_sambal_blimpvocab_flat_10M_tokenized.bin
  sbatch reproduce/icml2026/pretrain/train_v1_sambal_lr_seed.sh $LR $SEED
done; done
```

The §5 adaptation row: a long-regime model on the variant, then its LoRA
run (needs the LotR corpus, [3.9](DATA.md#39-domain-corpora-lotr-pubmed)):

```bash
sbatch reproduce/icml2026/pretrain/train_long_sambal.sh \
    ${SAMBAL_DATA_DIR:-data}/babycosmofine_sambal_variants/blimpvocab_flat/train_sambal_blimpvocab_flat_10M_tokenized.bin
sbatch reproduce/icml2026/lora/finetune_sambal_slurm.sh 32 16 \
    lm/gpt-bert/trained_models/gptbert_sambal_long_train_sambal_blimpvocab_flat_10M_tokenized_ema.bin
```

## 2.5 Scaling grid — Figures 4–6

One launcher submits all 150 cells (`START_CELL=<n>` resumes an
interrupted wave); without slurm it runs them one after another.
`--dry-run` prints the 150 per-cell commands instead:

```bash
./reproduce/icml2026/scaling/launch_scaling_ablation.sh
./reproduce/icml2026/scaling/launch_scaling_ablation.sh --dry-run
```

## 2.6 Reflexive-augmentation retrains — App. D

3 seeds per arm on the augmented corpora ([3.8](DATA.md#38-reflexive-augmented-corpora)):

```bash
for SEED in 0 1 2; do
  sbatch reproduce/icml2026/pretrain/train_v1_gptbert_lr_seed.sh 0.007 $SEED \
      ${SAMBAL_DATA_DIR:-data}/babycosmofine/train_10M_tokenized_with_reflexives_number.bin
  sbatch reproduce/icml2026/pretrain/train_v1_sambal_lr_seed.sh 0.007 $SEED \
      ${SAMBAL_DATA_DIR:-data}/babycosmofine_sambal/train_sambal_10M_tokenized_with_reflexives_number.bin
done
```

## 2.7 LoRA fine-tuning and small-domain from-scratch training — Tables 4, 11

Needs the domain corpora ([3.9](DATA.md#39-domain-corpora-lotr-pubmed)) and the
long checkpoints — from [2.1](#21-main-pretraining--the-table-26--figure-13-models), or the released ones:

```bash
hf download ezrawinston/gptbert-babycosmofine --include "*.bin" --local-dir lm/gpt-bert/trained_models
hf download ezrawinston/gptbert-sambal       --include "*.bin" --local-dir lm/gpt-bert/trained_models
```

```bash
sbatch reproduce/icml2026/lora/finetune_gptbert_slurm.sh 32 16   # <rank> <alpha> [base checkpoint]
sbatch reproduce/icml2026/lora/finetune_sambal_slurm.sh 32 16
sbatch reproduce/icml2026/lora/finetune_gptbert_pubmed_slurm.sh 32 16
sbatch reproduce/icml2026/lora/finetune_sambal_pubmed_slurm.sh 32 16
bash reproduce/icml2026/lora/sweep_lotr.sh      # from-scratch grids (without slurm the cells run in sequence)
bash reproduce/icml2026/lora/sweep_pubmed.sh
sbatch reproduce/icml2026/lora/train_lotr_slurm.sh   # one more from-scratch LotR cell (lotr_tiny_from_scratch)
```

One sweep cell on its own (`EXP_CONFIG` must be an absolute path):

```bash
sbatch --export=ALL,EXP_NAME=tiny_bs32_lr1e-3,EXP_CONFIG=$PWD/lm/gpt-bert/configs/tiny.json,EXP_BATCH_SIZE=32,EXP_LR=1e-3,EXP_EPOCHS=200 \
    reproduce/icml2026/lora/run_lotr_experiment.sh
# without slurm, the same variables in the environment:
EXP_NAME=tiny_bs32_lr1e-3 EXP_CONFIG=$PWD/lm/gpt-bert/configs/tiny.json EXP_BATCH_SIZE=32 EXP_LR=1e-3 EXP_EPOCHS=200 \
    bash reproduce/icml2026/lora/run_lotr_experiment.sh
```

The App. I generation samples read the adapters under
`lm/gpt-bert/pretraining/ft_out/` — written by the LotR launchers above, or
the released copies:

```bash
mkdir -p lm/gpt-bert/pretraining/ft_out/lotr_lora_r_32_a_16_sambal \
         lm/gpt-bert/pretraining/ft_out/lotr_lora_r_32_a_16_gptbert
cp "$(hf download ezrawinston/gptbert-sambal lora/lotr_sambal_best_ppl_trainable_params.pt)" \
   lm/gpt-bert/pretraining/ft_out/lotr_lora_r_32_a_16_sambal/best_ppl_trainable_params.pt
cp "$(hf download ezrawinston/gptbert-babycosmofine lora/lotr_baseline_best_ppl_trainable_params.pt)" \
   lm/gpt-bert/pretraining/ft_out/lotr_lora_r_32_a_16_gptbert/best_ppl_trainable_params.pt
```

```bash
(cd lm/gpt-bert/pretraining && bash ../../../reproduce/icml2026/lora/gen_lotr_sents.sh)   # or gen_lotr_sents_beam.sh
```

## 2.8 Records from training logs

These records come from the training jobs' logs; each command writes its
committed record in place:

```bash
# records/scaling/scaling_results.json
python reproduce/icml2026/assemble/assemble_scaling_results.py --log-dir logs \
    --out reproduce/icml2026/records/scaling/scaling_results.json

# records/blimp/*_long_full_blimp.json — from a LoRA run's pre-fine-tuning block
python reproduce/icml2026/assemble/assemble_long_full_blimp.py --log logs/finetune_gptbert_lotr-<id>.out \
    --out reproduce/icml2026/records/blimp/baseline_long_full_blimp.json
# ... and logs/finetune_sambal-<id>.out → sambal_long_full_blimp.json

# records/ablations/blimp_finals.json — from the Table 7 retrains' logs (resolved as logs/train_v1_sambal_lr_seed-<id>.out)
python reproduce/icml2026/assemble/assemble_ablation_blimp.py --slurm-id-start <id> --slurm-id-end <id> \
    --out reproduce/icml2026/records/ablations/blimp_finals.json
# (mean ± sd summary: reproduce/icml2026/assemble/parse_ablation_results.py, same flags)

# records/vocab_control/blimp.json — from the nine control runs' logs
python reproduce/icml2026/assemble/assemble_vocab_control_blimp.py --log 05m 0 logs/train_vocab_control-<id>.out ... \
    --out reproduce/icml2026/records/vocab_control/blimp.json
```

The scaling assembler merges into the existing record (seven entries come
from no log: [RESULTS.md](RESULTS.md#bases-and-weights)).
[`records/reflexives/syntaxgym_finals.json`](records/reflexives/syntaxgym_finals.json),
[`records/lora/lora_runs.json`](records/lora/lora_runs.json),
[`records/lora/from_scratch.json`](records/lora/from_scratch.json) and
[`records/blimpvocab/inline_finals.json`](records/blimpvocab/inline_finals.json)
have no assembler: their values are the final evaluation lines of the
runs' logs (LoRA runs also write `ft_out/<run>/metrics.json`, in the
record's shape).
