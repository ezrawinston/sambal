# 1. Evaluation and analysis

Every evaluation on the released checkpoints, then the figures and tables.
Evaluations reproduce the committed [`records/`](records/) values.
Regenerated files can differ from the committed copies in layout and key
names — compare values rather than bytes. [RESULTS.md](RESULTS.md) maps
every paper result to the script that
renders it and the records it reads; those scripts read only the committed
[`records/`](records/), so they should be runnable from a fresh repo checkout.

## 1.1 Model checkpoints

Install the package, then download the checkpoints from Hugging Face into
`lm/gpt-bert/trained_models/`:

```bash
pip install -e .
hf download ezrawinston/gptbert-babycosmofine --include "*.bin" --local-dir lm/gpt-bert/trained_models
hf download ezrawinston/gptbert-sambal       --include "*.bin" --local-dir lm/gpt-bert/trained_models
```

This downloads the two long-regime models (`gptbert_babycosmofine_long.bin`,
`gptbert_babycosmofine_long_ema.bin`, `gptbert_sambal_long_ema.bin`) and the
six short-regime seed checkpoints
(`gptbert_{babycosmofine,sambal}_short_lr_0.007_seed_{0,1,2}_ema.bin`).

## 1.2 Benchmark data (one-time fetches)

```bash
python evals/syntaxgym/fetch_data.py            # full SyntaxGym suites (pinned)
python evals/blimp/fetch_data.py                # full BLiMP (pinned) + the trainers' fast subsets
```

```bash
# UD-EWT dev split (InfoNCE), pinned revision
mkdir -p evals/infonce/downloads
curl -L -o evals/infonce/downloads/en_ewt-ud-dev.conllu \
  https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/4c89b5833a70aa5ed3a00bad2f23f57992cc7df8/en_ewt-ud-dev.conllu
```

```bash
python evals/snli/fetch_data.py                 # SNLI 1.0 + the hard subset (md5-gated)
```

EWoK is gated and not redistributable: accept the usage agreement for
[`ewok-core/ewok-core-1.0`](https://huggingface.co/datasets/ewok-core/ewok-core-1.0)
on Hugging Face, then run the BabyLM evaluation pipeline's
download-and-filter step:

```bash
git clone https://github.com/babylm/evaluation-pipeline-2024
git -C evaluation-pipeline-2024 checkout d680c2875e2141d0db14f68c305caca9e6d86b85
pip install -e evaluation-pipeline-2024
hf auth login
(cd evaluation-pipeline-2024 && python ewok/dl_and_filter.py)
mkdir -p evals/ewok/downloads
mv evaluation-pipeline-2024/evaluation_data/ewok_filtered evals/ewok/downloads/ewok_filtered
```

## 1.3 SyntaxGym — Table 2 (SG columns), Table 6, App. D

Long models (baseline from its raw weights, SAMBAL from EMA —
[RESULTS.md](RESULTS.md#bases-and-weights)):

```bash
python evals/syntaxgym/syntaxgym_eval.py \
    --input_path evals/syntaxgym/data/syntaxgym --suite_set core_only \
    --output_json reproduce/icml2026/records/syntaxgym/long_models_per_suite.json \
    --models baseline:checkpoint=lm/gpt-bert/trained_models/gptbert_babycosmofine_long.bin \
    --models sambal:checkpoint=lm/gpt-bert/trained_models/gptbert_sambal_long_ema.bin
```

Short-regime seeds (the launcher writes
[`records/syntaxgym/short_regime_per_suite.json`](records/syntaxgym/short_regime_per_suite.json)):

```bash
export SBATCH_PARTITION=<your partition> SBATCH_QOS=<qos>   # if submitting with sbatch
export SAMBAL_CUDA_MODULE=cuda-12.9   # if the cluster uses environment modules (default shown)
sbatch reproduce/icml2026/scoring/eval_syntaxgym.sh   # or run it with bash instead of sbatch
```

## 1.4 BLiMP — Table 2 (BLiMP column), Table 5

```bash
python evals/blimp/blimp_eval.py \
    --input_path evals/blimp/data \
    --record reproduce/icml2026/records/blimp/baseline_long_full_blimp.json \
    --backend-arg checkpoint=lm/gpt-bert/trained_models/gptbert_babycosmofine_long_ema.bin
python evals/blimp/blimp_eval.py \
    --input_path evals/blimp/data \
    --record reproduce/icml2026/records/blimp/sambal_long_full_blimp.json \
    --backend-arg checkpoint=lm/gpt-bert/trained_models/gptbert_sambal_long_ema.bin
```

## 1.5 EWoK — Figure 2, §4.3

```bash
python evals/ewok/ewok_eval.py \
    --input_path evals/ewok/downloads/ewok_filtered \
    --record_prefix reproduce/icml2026/records/ewok/baseline \
    --backend-arg checkpoint=lm/gpt-bert/trained_models/gptbert_babycosmofine_long_ema.bin
python evals/ewok/ewok_eval.py \
    --input_path evals/ewok/downloads/ewok_filtered \
    --record_prefix reproduce/icml2026/records/ewok/sambal \
    --backend-arg checkpoint=lm/gpt-bert/trained_models/gptbert_sambal_long_ema.bin
```

## 1.6 Swap probe — Table 3, Figure 1

```bash
python evals/swap_probe/paired_knowledge_probe.py \
    --sambal-backend-arg checkpoint=lm/gpt-bert/trained_models/gptbert_sambal_long_ema.bin \
    --normal-backend-arg checkpoint=lm/gpt-bert/trained_models/gptbert_babycosmofine_long_ema.bin \
    --grammar_mode nonsense_gram --counterbalance_order \
    --out_json reproduce/icml2026/records/swap_probe/paired_probe.json
```

The two protocol flags are not the defaults: `--counterbalance_order` makes
ΔK the mean over both clause orders rather than one, and
`--grammar_mode nonsense_gram` makes ΔG compare the two
grammatical-but-implausible arrangements against the four ungrammatical
ones rather than the default `primary` pair; `--score_all_tokens` stays off,
so only the swapped spans are scored.

## 1.7 InfoNCE — Figure 3

```bash
python evals/infonce/run_infonce.py \
    --ud_conllu evals/infonce/downloads/en_ewt-ud-dev.conllu \
    --backend-arg checkpoint=lm/gpt-bert/trained_models/gptbert_sambal_long_ema.bin \
    --pool mean --tau 0.1 --lex_exclude_func \
    --layers -1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 \
    --out_json reproduce/icml2026/records/infonce/sambal_infonce_hidden.json
# repeat with gptbert_babycosmofine_long_ema.bin → gptbert_infonce_hidden.json
```

## 1.8 SNLI probe — Table 8, App. E

Rows 1–2 are the long models; row 3 is the `no_context_buckets` seed-2
retrain of [2.3](TRAINING.md#23-component-ablation-retrains--table-7-table-8-row-3):

```bash
python evals/snli/snli_probe.py \
    --snli_dir evals/snli/downloads/snli_1.0 --snli_train_limit 100000 \
    --record_prefix reproduce/icml2026/records/snli/baseline \
    --backend-arg checkpoint=lm/gpt-bert/trained_models/gptbert_babycosmofine_long_ema.bin
python evals/snli/snli_probe.py \
    --snli_dir evals/snli/downloads/snli_1.0 --snli_train_limit 100000 \
    --record_prefix reproduce/icml2026/records/snli/sambal \
    --backend-arg checkpoint=lm/gpt-bert/trained_models/gptbert_sambal_long_ema.bin
python evals/snli/snli_probe.py \
    --snli_dir evals/snli/downloads/snli_1.0 --snli_train_limit 100000 \
    --record_prefix reproduce/icml2026/records/snli/sambal_no_ctx \
    --backend-arg checkpoint=lm/gpt-bert/trained_models/gptbert_sambal_short_train_sambal_no_context_buckets_10M_tokenized_lr_0.007_seed_2_<stamp>_ema.bin
```

## 1.9 Conflict benchmark — §4.4, App. F

```bash
python evals/blimp/blimp_eval.py \
    --input_path evals/conflict \
    --record_prefix reproduce/icml2026/records/conflict/baseline \
    --backend-arg checkpoint=lm/gpt-bert/trained_models/gptbert_babycosmofine_long_ema.bin
python evals/blimp/blimp_eval.py \
    --input_path evals/conflict \
    --record_prefix reproduce/icml2026/records/conflict/sambal \
    --backend-arg checkpoint=lm/gpt-bert/trained_models/gptbert_sambal_long_ema.bin
```

Regenerating the suite itself: stage 1 builds the candidate pool and the
controls from the two checkpoints (GPU); stage 2 keeps the pool items the
baseline gets wrong at its best temperature, the committed
`evals/conflict/conflicts.jsonl`:

```bash
export SBATCH_PARTITION=<your partition> SBATCH_QOS=<qos>   # if submitting with sbatch
export SAMBAL_CUDA_MODULE=cuda-12.9   # if the cluster uses environment modules (default shown)
sbatch reproduce/icml2026/scoring/build_suite_slurm.sh   # or run it with bash instead of sbatch
python evals/conflict/derive_conflicts.py \
    --backend-arg checkpoint=lm/gpt-bert/trained_models/gptbert_babycosmofine_long_ema.bin \
    --out evals/conflict/conflicts.jsonl --force   # without --force an existing file is only compared
```

## 1.10 Reflexive probes — App. D diagnostics

```bash
python evals/blimp/blimp_eval.py \
    --input_path evals/syntaxgym/reflexive_probes \
    --output_dir reproduce/icml2026/records/syntaxgym/probe_reports \
    --backend-arg checkpoint=lm/gpt-bert/trained_models/gptbert_sambal_long_ema.bin
# repeat with gptbert_babycosmofine_long_ema.bin
```

The probe files are scored together: the reported best temperature is
selected jointly across them ([RESULTS.md](RESULTS.md#bases-and-weights)).

## 1.11 Figures and tables

One script per paper item ([RESULTS.md](RESULTS.md) says which reads what),
each overwriting its committed rendering under [`out/`](out/):

```bash
for s in reproduce/icml2026/figures/fig*.py reproduce/icml2026/tables/table*.py; do python "$s"; done
```
