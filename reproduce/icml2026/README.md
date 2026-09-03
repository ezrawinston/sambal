# icml2026

Replicating the ICML 2026 paper:

1. **[Evaluation and analysis](EVALUATION.md)** — run
   evaluations on the released checkpoints and rebuild figures and
   tables. (GPU or CPU)
2. **[Model training](TRAINING.md)**
3. **[Regenerating the sambal data](DATA.md)** — rebuild the
   ablated corpora and their variants, the control and domain corpora, and
   the engine's resource lexicons.

[RESULTS.md](RESULTS.md) maps every paper result.

## Compute details

**Running the jobs.** Every training and generation job is a shell launcher
around one `python`/`torchrun` command: run it with `bash` on a GPU machine,
or submit it with `sbatch` on a slurm cluster, from the repository root
either way (`export REPO_ROOT=/path/to/checkout` to run from elsewhere). The
corpus stage also has a Modal driver,
[`corpus/modal_sambal_corpus.py`](corpus/modal_sambal_corpus.py).

**Hardware.** One GPU for everything except the two long-regime pretraining
launchers, which need one 8-GPU node ([`train_long.py`](../../lm/gpt-bert/pretraining/train_long.py) partitions
the 7:8 masked/causal hybrid across ranks and asserts a world size divisible
by 8). The released checkpoints evaluate on one GPU or on CPU (slowly); the
trainers need CUDA. On clusters with environment modules the launchers load
`SAMBAL_CUDA_MODULE` (default `cuda-12.9`); elsewhere they skip the step.

**Paths and environment.**

- Corpora live under `data/`, or under `SAMBAL_DATA_DIR` if it is set.
- Launchers source `$SAMBAL_ENV` if set, else a repo-root `.venv/`.
- The pretraining and from-scratch trainers log each run to Weights &
  Biases. `WANDB_MODE=offline` (the launchers' default) keeps the run under
  `lm/gpt-bert/pretraining/wandb/` and needs no account; `disabled` turns
  logging off; `online` uploads to project `WANDB_PROJECT` (default
  `sambal`) under `WANDB_ENTITY`, authenticated by `wandb login` or
  `WANDB_API_KEY` (`WANDB_BASE_URL` for a self-hosted server).

Directory map:

| dir | contents |
|---|---|
| [`corpus/`](corpus/) | corpus ablation (slurm + Modal drivers) |
| [`stats/`](stats/) | context-statistics collection over the source corpus |
| [`pretrain/`](pretrain/) | pretraining launchers: long regime, short regime, vocab control |
| [`scaling/`](scaling/) | model-size × data-fraction grid |
| [`reflexives/`](reflexives/) | SyntaxGym reflexive/number corpus augmentation + retrains |
| [`lora/`](lora/) | LoRA fine-tuning and the from-scratch grids on the domain corpora |
| [`scoring/`](scoring/) | slurm launchers for the SyntaxGym batch scoring and the conflict-suite generation |
| [`assemble/`](assemble/) | the scripts that turn training logs into the committed scaling, ablation and vocab-control records |
| [`records/`](records/) | the committed records — what the figure and table scripts read |
| [`figures/`](figures/) | one script per paper figure → [`out/figures/`](out/figures/) |
| [`tables/`](tables/) | one script per derived table → CSV in [`out/tables/`](out/tables/) |
| [`out/`](out/) | the rendered figures and tables (outputs already committed; the scripts overwrite them in place) |
