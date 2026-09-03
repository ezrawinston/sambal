# reflexives

Reflexive/number-agreement corpus augmentation and the retrains on the
augmented corpora, the targeted-augmentation experiment of the paper's
reflexive-diagnosis appendix (corpora
[Guide 3.8](../DATA.md#38-reflexive-augmented-corpora), retrains
[Guide 2.6](../TRAINING.md#26-reflexive-augmentation-retrains--app-d)).

- [`run_reflexive_number_pipeline.sh`](run_reflexive_number_pipeline.sh) —
  builds both augmented corpora end to end (extract → ablate → tokenize → append).
- [`extract_reflexive_sentences.py`](extract_reflexive_sentences.py) — the
  extraction step: match sentences of the SyntaxGym reflexive and
  number-agreement suites.
- [`merge_tokenized.py`](merge_tokenized.py) — appends any tokenized
  augmentation set to any tokenized base, so other base × set combinations
  build the same way.
