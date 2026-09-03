# data

One directory per corpus: its fetch/build script where there is one, and a
`MANIFEST.md` listing the files that land there with their fingerprints. The
corpora themselves are gitignored and live here by default; export
`SAMBAL_DATA_DIR` (an absolute path) to keep them elsewhere — every launcher
and builder reads it ([DATA.md](../reproduce/icml2026/DATA.md)).

| directory | corpus (paper items: ICML 2026) |
|---|---|
| [`babycosmofine/`](babycosmofine/) | the source pretraining corpus — the 10M-word "baby-cosmo-fine" blend of BabyLM, Cosmopedia and FineWeb text (Charpentier & Samuel, 2024) |
| [`babycosmofine_sambal/`](babycosmofine_sambal/) | its sambal ablation — the ICML 2026 paper's ablated corpus |
| [`babycosmofine_sambal_variants/`](babycosmofine_sambal_variants/) | the five component-ablation corpora (Table 7) and the benchmark-vocabulary corpus (§5), one subdirectory each |
| [`babycosmofine_top25k/`](babycosmofine_top25k/) | the vocab-filtered control corpus (Table 9) |
| [`lotr/`](lotr/) | the Lord of the Rings trilogy — small-domain adaptation (Table 4) |
| [`pubmed/`](pubmed/) | PubMed abstracts, sized to the LotR corpus (Table 11) |

The trainers read only the tokenized `.bin` files. The dataset release
carries the two main pretraining bins, for the main models and the
scaling grid ([TRAINING.md](../reproduce/icml2026/TRAINING.md)); every other
corpus is built and tokenized locally.
