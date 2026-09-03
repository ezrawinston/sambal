# Results

Every figure, table and quoted number of the paper: the script that renders
it, and the committed [`records/`](records/) it reads, each with the guide
step that produces that record. The scripts read only the committed `records/`; each runs from the repository
root as `python reproduce/icml2026/<script>` and overwrites its committed
rendering under [`out/`](out/). Model keys are `baseline` (standard
pretraining) and `sambal` (ablated corpus); unless stated otherwise, records
are for the long-regime checkpoints.

| paper item | script → output under `out/` | records read (producing step) |
|---|---|---|
| Figure 1 (swap-probe scatter) | [`figures/fig1_swap_probe.py`](figures/fig1_swap_probe.py) → `figures/paired_probe_scatter.{pdf,png}` | [`swap_probe/paired_probe.json`](records/swap_probe/paired_probe.json) ([Guide 1.6](EVALUATION.md#16-swap-probe--table-3-figure-1)) |
| Figure 2 (EWoK by suite) | [`figures/fig2_ewok.py`](figures/fig2_ewok.py) → `figures/ewok.{pdf,png}` | `ewok/{baseline,sambal}_report.txt` ([Guide 1.5](EVALUATION.md#15-ewok--figure-2-43)) |
| Figure 3 (per-layer InfoNCE) | [`figures/fig3_infonce.py`](figures/fig3_infonce.py) → `figures/infonce_compare_hidden_diff.png` | `infonce/{gptbert,sambal}_infonce_hidden.json` ([Guide 1.7](EVALUATION.md#17-infonce--figure-3)) |
| Figure 4 (scaling curves + bars) | [`figures/fig4_scaling.py`](figures/fig4_scaling.py) → `figures/blimp_lines.png`, `figures/blimp_grouped_bars.pdf` | [`scaling/scaling_results.json`](records/scaling/scaling_results.json) ([Guide 2.8](TRAINING.md#28-records-from-training-logs)) |
| Figure 5 (small-budget scaling) | [`figures/fig5_scaling_small.py`](figures/fig5_scaling_small.py) → `figures/blimp_grouped_bars_small.pdf` | [`scaling/scaling_results.json`](records/scaling/scaling_results.json) ([Guide 2.8](TRAINING.md#28-records-from-training-logs)) |
| Figure 6 (efficiency scatter) | [`figures/fig6_efficiency.py`](figures/fig6_efficiency.py) → `figures/efficiency_scatter.{pdf,png}` | [`scaling/scaling_results.json`](records/scaling/scaling_results.json) ([Guide 2.8](TRAINING.md#28-records-from-training-logs)) |
| Table 2 (main benchmarks) | [`tables/table2_main_results.py`](tables/table2_main_results.py) → `tables/table2_main_results.csv` | `blimp/{baseline,sambal}_long_full_blimp.json` ([Guide 1.4](EVALUATION.md#14-blimp--table-2-blimp-column-table-5), [Guide 2.8](TRAINING.md#28-records-from-training-logs)); [`syntaxgym/long_models_per_suite.json`](records/syntaxgym/long_models_per_suite.json), [`syntaxgym/short_regime_per_suite.json`](records/syntaxgym/short_regime_per_suite.json) ([Guide 1.3](EVALUATION.md#13-syntaxgym--table-2-sg-columns-table-6-app-d)); [`scaling/scaling_results.json`](records/scaling/scaling_results.json) ([Guide 2.8](TRAINING.md#28-records-from-training-logs)) |
| Table 3 (swap probe) | [`tables/table3_swap_probe.py`](tables/table3_swap_probe.py) → `tables/table3_swap_probe.csv` | [`swap_probe/paired_probe.json`](records/swap_probe/paired_probe.json) ([Guide 1.6](EVALUATION.md#16-swap-probe--table-3-figure-1)) |
| Table 4 (LotR specialization) | [`tables/table4_lotr.py`](tables/table4_lotr.py) → `tables/table4_lotr.csv` | [`lora/lora_runs.json`](records/lora/lora_runs.json), [`lora/from_scratch.json`](records/lora/from_scratch.json) ([Guide 2.7](TRAINING.md#27-lora-fine-tuning-and-small-domain-from-scratch-training--tables-4-11)) |
| Table 5 (BLiMP per paradigm) | [`tables/table5_blimp_per_suite.py`](tables/table5_blimp_per_suite.py) → `tables/table5_blimp_per_suite.csv` | `blimp/{baseline,sambal}_long_full_blimp.json` ([Guide 1.4](EVALUATION.md#14-blimp--table-2-blimp-column-table-5), [Guide 2.8](TRAINING.md#28-records-from-training-logs)) |
| Table 6 (SyntaxGym per suite) | [`tables/table6_syntaxgym_per_suite.py`](tables/table6_syntaxgym_per_suite.py) → `tables/table6_syntaxgym_per_suite.csv` | [`syntaxgym/long_models_per_suite.json`](records/syntaxgym/long_models_per_suite.json) ([Guide 1.3](EVALUATION.md#13-syntaxgym--table-2-sg-columns-table-6-app-d)) |
| Table 7 (component ablations) | [`tables/table7_ablations.py`](tables/table7_ablations.py) → `tables/table7_ablations.csv` | [`ablations/blimp_finals.json`](records/ablations/blimp_finals.json), [`scaling/scaling_results.json`](records/scaling/scaling_results.json) ([Guide 2.8](TRAINING.md#28-records-from-training-logs)) |
| Table 8 (SNLI) | [`tables/table8_snli.py`](tables/table8_snli.py) → `tables/table8_snli.csv` | `snli/{baseline,sambal,sambal_no_ctx}_metrics.json`, `snli/*_predictions_test_labels.tsv`, [`snli/test_hard_pair_ids.txt`](records/snli/test_hard_pair_ids.txt) ([Guide 1.8](EVALUATION.md#18-snli-probe--table-8-app-e)) |
| Table 9 (vocab-filtered control) | [`tables/table9_vocab_control.py`](tables/table9_vocab_control.py) → `tables/table9_vocab_control.csv` | [`scaling/scaling_results.json`](records/scaling/scaling_results.json), [`vocab_control/blimp.json`](records/vocab_control/blimp.json) ([Guide 2.8](TRAINING.md#28-records-from-training-logs)) |
| Table 10 (lexical entropy) | [`tables/table10_entropy.py`](tables/table10_entropy.py) → `tables/table10_entropy.csv` | [`entropy/entropy_matched_budget.json`](records/entropy/entropy_matched_budget.json) ([Guide 3.10](DATA.md#310-entropy-record--table-10)) |
| Table 11 (PubMed specialization) | [`tables/table11_pubmed.py`](tables/table11_pubmed.py) → `tables/table11_pubmed.csv` | [`lora/lora_runs.json`](records/lora/lora_runs.json), [`lora/from_scratch.json`](records/lora/from_scratch.json) ([Guide 2.7](TRAINING.md#27-lora-fine-tuning-and-small-domain-from-scratch-training--tables-4-11)) |
| §4.2, App. D (reflexive diagnostics, augmentation retrains) | prose | per-suite values of the SyntaxGym records ([Guide 1.3](EVALUATION.md#13-syntaxgym--table-2-sg-columns-table-6-app-d)); the probe reports of [Guide 1.10](EVALUATION.md#110-reflexive-probes--app-d-diagnostics); [`reflexives/syntaxgym_finals.json`](records/reflexives/syntaxgym_finals.json) ([Guide 2.8](TRAINING.md#28-records-from-training-logs)) |
| §4.3 EWoK averages and per-domain scores | prose | `ewok/{baseline,sambal}_report.txt` ([Guide 1.5](EVALUATION.md#15-ewok--figure-2-43)) |
| §4.3 swap-probe accuracies and margins | Table 3's script | [`swap_probe/paired_probe.json`](records/swap_probe/paired_probe.json) ([Guide 1.6](EVALUATION.md#16-swap-probe--table-3-figure-1)) |
| §4.4 conflict and control accuracies | prose | `conflict/{baseline,sambal}_{best_temperature,temperature_1}_report.txt` ([Guide 1.9](EVALUATION.md#19-conflict-benchmark--44-app-f)) |
| §4.5 scaling examples; efficiency-fit statistics (R², r, slope) | the Figure 4–6 scripts | [`scaling/scaling_results.json`](records/scaling/scaling_results.json) ([Guide 2.8](TRAINING.md#28-records-from-training-logs)) |
| §4.6, App. J (LoRA adaptation perplexities and BLiMP costs) | prose | [`lora/lora_runs.json`](records/lora/lora_runs.json) ([Guide 2.7](TRAINING.md#27-lora-fine-tuning-and-small-domain-from-scratch-training--tables-4-11)) |
| §4.6, App. I (LotR generations) | [`lora/gen_lotr_sents.sh`](lora/gen_lotr_sents.sh) / [`lora/gen_lotr_sents_beam.sh`](lora/gen_lotr_sents_beam.sh) → stdout (stochastic sampling; the paper prints a selection) | the fine-tuned LotR checkpoints ([Guide 2.7](TRAINING.md#27-lora-fine-tuning-and-small-domain-from-scratch-training--tables-4-11)) |
| §5 (benchmark-vocabulary variant) | prose | [`blimpvocab/inline_finals.json`](records/blimpvocab/inline_finals.json) and the `lotr_sambal_blimpvocab_flat` entry of [`lora/lora_runs.json`](records/lora/lora_runs.json) ([Guide 2.4](TRAINING.md#24-benchmark-vocabulary-variant-runs--5-discussion)) |
| App. H (corpus entropies, word-type counts) | prose | [`entropy/entropy_matched_budget.json`](records/entropy/entropy_matched_budget.json) ([Guide 3.10](DATA.md#310-entropy-record--table-10)) and the corpus builders under [`data/`](../../data/) |
| App. A (pipeline constants: thresholds, retry counts, vocabulary sizes) | prose | [`sambal/profiles/icml2026.json`](../../sambal/profiles/icml2026.json) and the [`sambal/config.py`](../../sambal/config.py) defaults |

Known divergences from the printed paper are listed in
[`ERRATA.md`](../../ERRATA.md).

## Bases and weights

Bases differ where the underlying evaluations differed: inline training-time
BLiMP evaluations use the fast subsets
([`evals/blimp/README.md`](../../evals/blimp/README.md)), standalone and
final evaluations use full BLiMP; short-regime rows aggregate 3 seeds
(Table 2 with sample, Table 7 with population standard deviation).

Temperature bases (best temperature for the aggregates, T=1.0 for every
per-suite SyntaxGym value): [`ERRATA.md`](../../ERRATA.md#clarifications).

Reported numbers come from EMA checkpoints, with one exception: the
baseline long model's SyntaxGym results (the SyntaxGym column and the
per-suite table) are from its raw, non-EMA weights, so
[`syntaxgym/long_models_per_suite.json`](records/syntaxgym/long_models_per_suite.json)
scores the baseline from its raw checkpoint and SAMBAL from its EMA
checkpoint. Both baseline weight files are released with the models
(`gptbert_babycosmofine_long.bin`, raw, and
`gptbert_babycosmofine_long_ema.bin`).

[`lora/lora_runs.json`](records/lora/lora_runs.json) carries two
post-fine-tuning bases: `post_ft_test_ppl` is the final training state,
while the BLiMP numbers — like the released adapters — are the
best-validation-perplexity restore.

Seven entries of [`scaling/scaling_results.json`](records/scaling/scaling_results.json)
come from no training log: the six 30M@100% entries are best-temperature
evaluations of those runs' EMA checkpoints, and the SAMBAL 14M@0.15
seed-123 entry is a backfill retrain ([`ERRATA.md`](../../ERRATA.md) entry 1).

EWoK's best temperature for the SAMBAL arm is a near-tie between the
adjacent grid points 2.65 and 2.70. The committed report is the run in the
released environment (2.65); a rerun on other hardware can select 2.70,
which moves the per-domain values while the mean still prints as 50.8.

The App. D probe accuracies ([Guide 1.10](EVALUATION.md#110-reflexive-probes--app-d-diagnostics))
are each probe file's per-UID accuracy at T=1.0 and at the best temperature
selected jointly across the probe files, which are scored together in one
run per checkpoint — the argmax of the cross-file average. The per-probe
numbers are the per-UID lines of `best_temperature_report.txt` and
`temperature_1_report.txt`. A probe file scored alone reproduces its T=1.0
value, but its best-temperature number then reflects that file's own
argmax, which is not the reported basis.
