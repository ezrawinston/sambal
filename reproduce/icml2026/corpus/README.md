# Corpus generation

The sambal ablation of the source corpus with the paper profile
([`sambal/profiles/icml2026.json`](../../../sambal/profiles/icml2026.json)),
one shard per job ([Guide 3.4](../DATA.md#34-ablated-corpus)); the corpus
variants are the same run with one override each
([Guide 3.5](../DATA.md#35-corpus-variants)).

- [`run_sambal_corpus.sbatch`](run_sambal_corpus.sbatch) — slurm; `CONFIG_OVERRIDES`,
  `PATHS_OVERRIDES` and `OUT_DIR` select a variant.
- [`modal_sambal_corpus.py`](modal_sambal_corpus.py) — Modal variant, one container per shard.
