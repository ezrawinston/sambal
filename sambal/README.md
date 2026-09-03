# sambal

The ablation engine: a spaCy-parsed corpus is rewritten word by word,
replacing content words with others of the same syntactic class — verbs by
VerbNet frame, nouns by countability, adjectives and adverbs by degree — and
re-inflecting them.
Every driver runs as `python -m sambal.<module> --config <json>`.
[`profiles/icml2026.json`](profiles/icml2026.json) holds the ICML 2026
paper's settings ([Guide 3.4](../reproduce/icml2026/DATA.md#34-ablated-corpus)).

| module or directory | what |
|---|---|
| [`augment.py`](augment.py) | one-shot driver: parse and ablate a corpus (`--shard i --num-shards N` slices the input) |
| [`parse.py`](parse.py), [`augment_from_docbin.py`](augment_from_docbin.py) | the same in two steps: parse to DocBins, then ablate them |
| [`stats.py`](stats.py), [`stats_merge.py`](stats_merge.py) | the context statistics the profile reads, and the merge of sharded runs ([Guide 3.3](../reproduce/icml2026/DATA.md#33-context-statistics)) |
| [`engine.py`](engine.py), [`config.py`](config.py) | the `Augmenter` class and its `Config` / `ResourcePaths` dataclasses — the keys a config may override |
| [`profiles/`](profiles/) | named engine configurations |
| [`resources/`](resources/) | the lexicons and the context-statistics pickle ([`MANIFEST.md`](resources/MANIFEST.md)); their generators in [`resources/generation/`](resources/generation/) |
