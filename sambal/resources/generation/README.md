# Resource generation

The scripts regenerate the files in [`sambal/resources/`](../) from their
upstream sources, in place; the inputs table below says which are pinned.
Run `bash fetch_inputs.sh` first (the Kaikki dump, `gendered_words.json`,
ACE+ERG); the generators fetch everything else
themselves, except the BLiMP files of [`gen_blimpvocab.py`](gen_blimpvocab.py),
which [`evals/blimp/fetch_data.py`](../../../evals/blimp/fetch_data.py) fetches.

[`gen_proper_nouns.py`](gen_proper_nouns.py) imports the `sambal` package,
so run it from the repository root (or with `PYTHONPATH` set to it).

## Script → resources map

| script | generates |
|---|---|
| [`gen_lexicons.py`](gen_lexicons.py) | `wordnet_{nouns,adjectives,adverbs}.jsonl`, [`human_nouns.jsonl`](../human_nouns.jsonl), [`given_names_ssa.txt`](../given_names_ssa.txt), [`function_words_ud_ewt.txt`](../function_words_ud_ewt.txt), [`fixed_mwes_all.txt`](../fixed_mwes_all.txt) (+ its three parts), [`streusle_vmwe_patterns.jsonl`](../streusle_vmwe_patterns.jsonl), [`english_npis_wiktionary.tsv`](../english_npis_wiktionary.tsv) |
| [`gen_countability.py`](gen_countability.py) | [`count.txt`](../count.txt), [`mass.txt`](../mass.txt), [`both.txt`](../both.txt), [`pluralia_tantum.txt`](../pluralia_tantum.txt) |
| [`gen_gender_names.py`](gen_gender_names.py) | [`male_given_names.txt`](../male_given_names.txt), [`female_given_names.txt`](../female_given_names.txt), [`neutral_given_names.txt`](../neutral_given_names.txt) |
| [`gen_toinf_lexicons.py`](gen_toinf_lexicons.py) | [`toinf_verbs.jsonl`](../toinf_verbs.jsonl), [`toinf_adjs.jsonl`](../toinf_adjs.jsonl) |
| [`gen_allowed_vocab.py`](gen_allowed_vocab.py) | [`allowed_vocab.txt`](../allowed_vocab.txt) |
| [`gen_blimpvocab.py`](gen_blimpvocab.py) | [`blimpvocab.txt`](../blimpvocab.txt) |
| [`gen_proper_nouns.py`](gen_proper_nouns.py) | [`allowed_proper_nouns.txt`](../allowed_proper_nouns.txt) |
| [`prefilter_gendered_words.py`](prefilter_gendered_words.py) | [`gendered_words_filtered.json`](../gendered_words_filtered.json) |
| — | [`licensor_patterns_full.jsonl`](../licensor_patterns_full.jsonl), hand-authored: 81 NPI licensor patterns in spaCy matcher form |

## Upstream inputs

| input | used by | fetched from | revision as built |
|---|---|---|---|
| BabyLM 2024 10M train split (`train_10M.zip`) | [`gen_allowed_vocab.py`](gen_allowed_vocab.py) | `https://osf.io/ad7qg/` → `text_data/train_10M.zip` (direct: `https://osf.io/download/5mk3x/`) | a static archive — md5 `dc67e5d2db96ed4f76cb072e2504a18c`, 18139949 bytes; per-file checksums below |
| BLiMP paradigm files | [`gen_blimpvocab.py`](gen_blimpvocab.py) | [`evals/blimp/fetch_data.py`](../../../evals/blimp/fetch_data.py) | commit `3e56b06fcabca9b30822fc66435fca6b1aa40bb1` of the BLiMP repository |
| UD_English-EWT | [`gen_lexicons.py`](gen_lexicons.py) | `https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/<ref>/en_ewt-ud-{train,dev,test}.conllu` | commit `4c89b5833a70aa5ed3a00bad2f23f57992cc7df8` (UD 2.16, tag `r2.16`), the `--ud-ewt-ref` default; current `master` no longer reproduces the committed files |
| STREUSLE `streusle.conllulex` | [`gen_lexicons.py`](gen_lexicons.py) | `https://raw.githubusercontent.com/nert-nlp/streusle/<ref>/streusle.conllulex` | tag `v4.7.1` = commit `9eb4fa91fce7edc28dcfb3761a81dbbd473de9a9`, the `--streusle-ref` default; md5 `45bdbc23800913cdc3d5a12488f05f6c`, 6261661 bytes |
| WordNet | [`gen_lexicons.py`](gen_lexicons.py) | the NLTK `wordnet` corpus | version not recorded by the run |
| SSA baby names `names.zip` | [`gen_lexicons.py`](gen_lexicons.py), [`gen_gender_names.py`](gen_gender_names.py) | `https://www.ssa.gov/oact/babynames/names.zip` | shipped in-repo as [`generation/ssa_names.zip`](ssa_names.zip) (md5 `6c4c0cb728803f2a589ad311b0b098fc`, 7726516 bytes); the endpoint is unversioned and rejects scripted requests |
| English Wiktionary NPI category | [`gen_lexicons.py`](gen_lexicons.py) | `https://en.wiktionary.org/w/api.php` | not pinned — a live category listing that has changed since the committed run |
| ecmonsen/gendered_words `gendered_words.json` | [`prefilter_gendered_words.py`](prefilter_gendered_words.py) | [`fetch_inputs.sh`](fetch_inputs.sh) | commit `5a75616c917bed7f32415526bce7c781c2be5296` (unchanged upstream since 2021-09-05), md5 `9e7709dbbc931e1f4e1d1b67829b69bb`, 554118 bytes |
| Kaikki/Wiktextract dump | [`gen_countability.py`](gen_countability.py) | [`fetch_inputs.sh`](fetch_inputs.sh) (`https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz`) | not pinned — a rolling "latest" URL with no version-addressable archive |
| ERG grammar image + ACE | [`gen_toinf_lexicons.py`](gen_toinf_lexicons.py) | [`fetch_inputs.sh`](fetch_inputs.sh) | partly — ACE 0.9.34 with the ERG 2025 release compiled from its source tree (`ace -G erg-2025.dat -g ace/config.tdl`; ACE ships no prebuilt 2025 image, and the prebuilt 2018 image drops rows — see below), PyDelphin 1.10.0, lemminflect 0.2.3, NLTK VerbNet 2.1 |

### Reproducing [`allowed_vocab.txt`](../allowed_vocab.txt)

A spaCy token-frequency count over the six plain-text files of the
**BabyLM 2024 10M-word train split** at `--min-freq 1` — the raw BabyLM
split, not `data/babycosmofine/train.jsonl` (the BabyLM + Cosmopedia +
FineWeb blend). From the repository root:

    mkdir -p sambal/resources/generation/downloads
    curl -L -o sambal/resources/generation/downloads/train_10M.zip https://osf.io/download/5mk3x/
    unzip -d sambal/resources/generation/downloads sambal/resources/generation/downloads/train_10M.zip

The six files it unpacks into `train_10M/`:

| file | bytes | md5 |
|---|---:|---|
| `bnc_spoken.train` | 4884146 | `f2cb4f997bd672f2249891d9b73884e2` |
| `childes.train` | 15485295 | `b3248fd1869ae7d6f09b368d75b85f0f` |
| `gutenberg.train` | 14001510 | `e4bbf11bb0507d4dd68f7201d3805ec5` |
| `open_subtitles.train` | 10828244 | `67224789d3d7d50e662f7475168a26ba` |
| `simple_wiki.train` | 8432882 | `6af2fc2f38d18beb471eae04c5d41f22` |
| `switchboard.train` | 719322 | `92696a448b0ca12991837ab007eb8f13` |

All six files, in this order:

    python sambal/resources/generation/gen_allowed_vocab.py \
      --out sambal/resources/allowed_vocab.txt \
      --spacy-model en_core_web_sm \
      --min-freq 1 \
      sambal/resources/generation/downloads/train_10M/bnc_spoken.train \
      sambal/resources/generation/downloads/train_10M/childes.train \
      sambal/resources/generation/downloads/train_10M/gutenberg.train \
      sambal/resources/generation/downloads/train_10M/open_subtitles.train \
      sambal/resources/generation/downloads/train_10M/simple_wiki.train \
      sambal/resources/generation/downloads/train_10M/switchboard.train

The committed file (147,938 lines) was built with spaCy 3.8.7 /
en_core_web_sm 3.8.0; other spaCy versions tokenize differently.

Two properties of the committed file are easy to lose when regenerating it:
**case is preserved** (the engine's proper-noun and given-name gates match
capitalized candidates as written, so a lowercased vocabulary would silently
disable them), and CHILDES speaker codes such as `MOT` survive as ordinary
tokens.

### Reproducing [`blimpvocab.txt`](../blimpvocab.txt)

The same count over the sentences of the 67 BLiMP paradigm files, both
members of every pair, at `--min-freq 1`. From the repository root, after
`python evals/blimp/fetch_data.py`:

    python sambal/resources/generation/gen_blimpvocab.py \
      --out sambal/resources/blimpvocab.txt \
      --spacy-model en_core_web_sm \
      --min-freq 1 \
      evals/blimp/data

The committed file (2,411 lines) was built with spaCy 3.8.7 /
en_core_web_sm 3.8.0.

### Reproducing [`gendered_words_filtered.json`](../gendered_words_filtered.json)

Its defaults match the committed file:

    python sambal/resources/generation/prefilter_gendered_words.py \
      --gendered-words-in sambal/resources/generation/downloads/gendered_words.json \
      --allowed-vocab sambal/resources/allowed_vocab.txt \
      --out sambal/resources/gendered_words_filtered.json

### Reproducing the countability lists

The Kaikki dump from [`fetch_inputs.sh`](fetch_inputs.sh); the committed
lists are single-token (hyphens and apostrophes allowed, no spaces):

    python sambal/resources/generation/gen_countability.py --single-token \
      sambal/resources/generation/downloads/raw-wiktextract-data.jsonl.gz

### Reproducing the to-infinitive lexicons

[`gen_toinf_lexicons.py`](gen_toinf_lexicons.py) parses every candidate lemma with ACE under the ERG,
classifies it from the lexical types and applies the curated tables at the
top of the script:

    python sambal/resources/generation/gen_toinf_lexicons.py --erg <ERG .dat> --ace <ace binary> \
      --adj_in sambal/resources/wordnet_adjectives.jsonl --verb_in verbnet

`verbnet` is the NLTK VerbNet corpus. The prebuilt ERG 2018 image drops
rows; compile the 2025 grammar (the inputs table has the command).

### Reproducing [`allowed_proper_nouns.txt`](../allowed_proper_nouns.txt)

The gate keeps the vocabulary entries attested purely as singular proper
nouns (NNP), judged from a per-lemma tag profile the script builds one of
two ways.

**From the statistics pickle** (the committed file's path).

    PYTHONPATH=. python sambal/resources/generation/gen_proper_nouns.py \
      --allowed-vocab sambal/resources/allowed_vocab.txt \
      --stats-pkl <proper-noun-retaining lemma statistics> \
      --min-count 80 \
      --out sambal/resources/allowed_proper_nouns.txt

With the committed vocabulary and that pickle — a lemma-statistics run with
proper nouns retained, which the shipped `sambal.stats` skips; not shipped —
this reproduces the committed 92-entry file.

**From the corpus.** `--corpus` builds the profile by tagging a JSONL corpus
with spaCy — here the pretraining blend, not the raw BabyLM split:

    PYTHONPATH=. python sambal/resources/generation/gen_proper_nouns.py \
      --allowed-vocab sambal/resources/allowed_vocab.txt \
      --corpus data/babycosmofine/train.jsonl \
      --jsonl-field text \
      --spacy-model en_core_web_trf \
      --min-count 80 \
      --out sambal/resources/allowed_proper_nouns.corpus.txt

The profile differs (every lemma is tagged, and the purity test turns on
single tagging decisions), so the output is tagger-dependent and a subset of
the committed entries. `en_core_web_trf` wants a GPU.

### Reproducing [`gen_lexicons.py`](gen_lexicons.py)'s outputs

    python sambal/resources/generation/gen_lexicons.py

The generator appends two entries the extractors cannot produce: `'s` and
`'t` (ASCII and typographic apostrophe)
at the end of [`function_words_ud_ewt.txt`](../function_words_ud_ewt.txt) — the
UD token regex requires a
word character first — and `person` and `people` in [`human_nouns.jsonl`](../human_nouns.jsonl) —
WordNet files them under `noun.Tops`/`noun.group`, not `noun.person`.

Licenses of the derived resources: [`LICENSES.md`](../../../LICENSES.md).
