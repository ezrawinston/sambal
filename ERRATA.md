# Errata

Corrections to the ICML 2026 paper as published. The committed files in this repository
carry the corrected values. Paths below are relative to [`reproduce/icml2026/`](reproduce/icml2026/) unless they
start with [`evals/`](evals/).

| # | paper | published → corrected | why | record |
|---|---|---|---|---|
| 1 | Figs. 4–5, App. Fig. 5, §4.5 (scaling grid, SAMBAL 14M @ 15%) | SG 44.48 (3.52), BLiMP 64.65 (0.10), over 2 seeds → SG 42.53 (3.98), BLiMP 64.45 (0.30), over 3 | one of the stated 3 seed runs was skipped at paper time; since backfilled | [`records/scaling/scaling_results.json`](reproduce/icml2026/records/scaling/scaling_results.json) (`backfill` key) |
| 2 | §4.1 (long-regime protocol) | "5000 steps, average batch 1.1M tokens, ~350 epochs" → 5400 steps; 1.1M is the *starting* batch — the ramp averages 2.13M/step (≈730 corpus passes) | description only | [`pretrain/train_long_gptbert.sh`](reproduce/icml2026/pretrain/train_long_gptbert.sh), [`pretrain/train_long_sambal.sh`](reproduce/icml2026/pretrain/train_long_sambal.sh) |
| 3 | Table 2 (baseline short, SyntaxGym) | 54.7 (2.0) → 59.2 (8.8) | one per-seed value (55.59) matches no surviving checkpoint; the released one scores 69.15. The short-regime SG gap widens 8.1 → 12.6 | [`records/scaling/scaling_results.json`](reproduce/icml2026/records/scaling/scaling_results.json) (`corrected` key) |
| 4 | Table 3 / Fig. 1, Fig. 3, Fig. 2 + §4.3 EWoK (baseline arm) | Table 3 baseline row 55.0 / 0.65 / 100.0 / 7.61 → 52.0 / 0.97 / 100.0 / 7.38; EWoK average 54.3 → 54.0 | the baseline records came from an earlier checkpoint of the same recipe; re-scored on the released one. Readings unchanged | [`records/swap_probe/paired_probe.json`](reproduce/icml2026/records/swap_probe/paired_probe.json), [`records/infonce/gptbert_infonce_hidden.json`](reproduce/icml2026/records/infonce/gptbert_infonce_hidden.json), [`records/ewok/baseline_report.txt`](reproduce/icml2026/records/ewok/baseline_report.txt) |
| 5 | §4.4 and App. F (conflict benchmark) | N = 186, baseline conflict 10.2% (19/186) → N = 187, baseline 10.70% (20/187); SAMBAL and controls stay 100% | one item of the paper's set has no recoverable removal procedure; the repository ships the fully mechanical 187-item set, both models re-scored | [`evals/conflict/`](evals/conflict/) (suite, pool, [`derive_conflicts.py`](evals/conflict/derive_conflicts.py)); `records/conflict/*_report.txt` |
| 6 | App. F.2 (item selection; a generation constraint) | "a fixed number per family with simple de-duplication" → the selection was the baseline-verdict prune (§4.4's criterion ii); the stated Δ<sub>ctrl</sub> ≥ 0.1 constraint was never armed (28/213 pool items violate it) | description only | [`evals/conflict/README.md`](evals/conflict/README.md); [`reproduce/icml2026/scoring/build_suite_slurm.sh`](reproduce/icml2026/scoring/build_suite_slurm.sh) |
| 7 | Table 8 and App. E (SAMBAL, SNLI-hard) | 37.5 → 36.9 | the published 37.5 matches a probe trained on the full SNLI training set, not the 100,000-pair probe that gives the 60.1 beside it; recomputed from that probe's predictions over the official hard-subset ids | [`records/snli/sambal_predictions_test_labels.tsv`](reproduce/icml2026/records/snli/sambal_predictions_test_labels.tsv), [`records/snli/test_hard_pair_ids.txt`](reproduce/icml2026/records/snli/test_hard_pair_ids.txt) |
| 8 | Table 8 third row and App. E (SAMBAL, no ctx. samp.) | SNLI 59.7 → 60.2; SNLI-hard (not printed) → 37.3 | the published variant's probe was trained on 50,000 pairs against the main rows' 100,000; re-probed at 100,000 with its per-item predictions kept. Also note: the variant is short-regime trained vs the long-regime main rows ([Guide 2.3](reproduce/icml2026/TRAINING.md#23-component-ablation-retrains--table-7-table-8-row-3)). App. E's reading unchanged | [`records/snli/sambal_no_ctx_metrics.json`](reproduce/icml2026/records/snli/sambal_no_ctx_metrics.json), [`records/snli/sambal_no_ctx_predictions_test_labels.tsv`](reproduce/icml2026/records/snli/sambal_no_ctx_predictions_test_labels.tsv) |

Smaller differences, corrected in the committed records and figure/table scripts: the Table 2 SAMBAL-short SG-without-reflexives
sd 4.9 → 4.5; long-baseline SyntaxGym 70.47 → 70.49 (Table 2, §4.2); baseline
ΔK 7.62 → 7.61 (Table 3); fine-tuned SAMBAL BLiMP 71.4 → 71.5 (Table 4); the
Fig. 6 efficiency fit R² = 0.76, r = −0.87 → R² = 0.74, r = −0.86; the 30M
vocab-filtered sd 0.76 → 0.75 (Table 9); the unigram entropies H1 =
10.80 / 10.45 / 10.16 → 10.83 / 10.48 / 10.19 (Table 10, §4.5); and the
SAMBAL EWoK report, re-scored in the released environment where the best
temperature is 2.65 rather than 2.70 — the printed 50.8 average is unchanged
(50.81 → 50.77) but Fig. 2's SAMBAL per-domain bars move.

## Clarifications

Published numbers that are correct as printed but whose bases are not
like-for-like:

- **Table 2's SG-without-reflexives column** — an item-count-weighted mean of
  T = 1.0 per-suite accuracies; alternative aggregations move the column by
  up to 3 points without changing the SAMBAL-above-baseline comparison.
- **Best temperature vs. T = 1.0** — aggregate BLiMP/SyntaxGym accuracies are
  at each model's best temperature, but all per-suite SyntaxGym values (and
  everything derived from them: Table 6, the SG-without-reflexives columns,
  App. D's reflexive-suite averages) are at T = 1.0. Basis notes:
  [`reproduce/icml2026/RESULTS.md`](reproduce/icml2026/RESULTS.md).
