# Scaling ablation grid (Figs. 4–6; Tables 2/7/9 reference rows)

The model-size × data-fraction grid
([Guide 2.5](../TRAINING.md#25-scaling-grid--figures-46)).

- [`launch_scaling_ablation.sh`](launch_scaling_ablation.sh) — submits all
  150 cells, or runs them in sequence without slurm (`--dry-run` prints the
  per-cell commands).
- [`train_scaling_ablation.sh`](train_scaling_ablation.sh) — the per-cell
  worker, invoked by the launcher with the cell's corpus/size/fraction/seed.
- [`gen_configs.py`](gen_configs.py) — generator of the four model-size
  configs the grid trains (`lm/gpt-bert/configs/scaling_configs/{05m,10m,14m,30m}.json`,
  committed); running it writes all 30 candidate sizes (01m–30m) into that
  directory, over the committed four.
