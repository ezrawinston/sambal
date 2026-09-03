# Licenses

The code in this repository is released under the MIT License (see
[`LICENSE`](LICENSE)). The tree under [`lm/gpt-bert/`](lm/gpt-bert/) retains its
upstream MIT license.
Committed data resources derived from third-party sources carry their own
licenses, listed below.

| material | license | notes |
|---|---|---|
| [`sambal/resources/`](sambal/resources/) lexicons derived from English Wiktionary (the countability lists and the NPI list) | CC BY-SA 3.0 (Wiktionary dual-licenses CC BY-SA 3.0 / GFDL; used here under CC BY-SA 3.0) | extracted via Kaikki/Wiktextract — cite Ylonen (2022) |
| [`licensor_patterns_full.jsonl`](sambal/resources/licensor_patterns_full.jsonl) | MIT (this repository's license) | hand-authored NPI licensor pattern inventory written for this project; not derived from external resources |
| function words, fixed multiword expressions, phrasal verbs | CC BY-SA 4.0 | derived from UD_English-EWT and STREUSLE |
| `wordnet_*.jsonl`, [`human_nouns.jsonl`](sambal/resources/human_nouns.jsonl) | WordNet License | via NLTK |
| [`gendered_words_filtered.json`](sambal/resources/gendered_words_filtered.json) | CC BY 3.0 | from ecmonsen/gendered_words (WordNet content under the WordNet License) |
| given-name lists | US public domain | SSA baby-names national data |
| to-infinitive lexicons ([`toinf_verbs.jsonl`](sambal/resources/toinf_verbs.jsonl), [`toinf_adjs.jsonl`](sambal/resources/toinf_adjs.jsonl)) | see the ERG/ACE distributions (open source) | derived from the English Resource Grammar via the ACE parser |
| [`evals/syntaxgym/syntaxgym_fast/`](evals/syntaxgym/syntaxgym_fast/) and the reflexive probe files | MIT | items from cpllab/syntactic-generalization |
| EWoK | not redistributed | fetched at eval time; its terms do not permit redistribution |
| tokenizer `gpt-bert-babylm-small` | MIT | committed at [`lm/gpt-bert/gpt-bert-babylm-small/tokenizer.json`](lm/gpt-bert/gpt-bert-babylm-small/tokenizer.json), from its upstream Hugging Face distribution |
| source pretraining corpus ([`data/babycosmofine/`](data/babycosmofine/)) | no license declared upstream | fetched; see [`data/babycosmofine/MANIFEST.md`](data/babycosmofine/MANIFEST.md) |

Full per-resource provenance: [`sambal/resources/MANIFEST.md`](sambal/resources/MANIFEST.md) and
[`sambal/resources/generation/README.md`](sambal/resources/generation/README.md).
