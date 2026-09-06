# 3. Regenerating the data

Install the package with the parsers and lexicons the engine imports and,
for slurm submissions, name the partition, QOS and CUDA module:

```bash
pip install -e .
python -m spacy download en_core_web_trf
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('verbnet'); nltk.download('wordnet')"
pip install cupy-cuda12x   # GPU parsing; pick the cupy build matching your CUDA
export SBATCH_PARTITION=<your partition> SBATCH_QOS=<qos>   # if submitting with sbatch
export SAMBAL_CUDA_MODULE=cuda-12.9   # if the cluster uses environment modules (default shown)
```

Every `sbatch` command in this guide also runs with `bash` instead of `sbatch`.

Everything below lands under `data/`, or under `SAMBAL_DATA_DIR` if it is
set ([Compute details](README.md#compute-details)).

## 3.1 Engine resources

Optional: the resource files are committed
([`sambal/resources/MANIFEST.md`](../../sambal/resources/MANIFEST.md));
rebuilding them from their upstream sources is
[`sambal/resources/generation/README.md`](../../sambal/resources/generation/README.md),
and the context-statistics pickle [3.3](#33-context-statistics).

## 3.2 Source corpus

```bash
bash data/babycosmofine/fetch.sh        # pinned HF revision, checksum-gated
```

## 3.3 Context statistics

Optional — the pickle the ablation profile reads
([`sambal/resources/lemma_stats_top_25k.pkl`](../../sambal/resources/lemma_stats_top_25k.pkl))
is committed:

```bash
sbatch reproduce/icml2026/stats/run_stats.sbatch
```

Sharded (four pieces shown): split the input, one job per piece, then merge
the pickles into the committed path:

```bash
split -n l/4 -d --additional-suffix=.jsonl \
    ${SAMBAL_DATA_DIR:-data}/babycosmofine/train.jsonl ${SAMBAL_DATA_DIR:-data}/babycosmofine/stats_shard_
for i in 00 01 02 03; do
  sbatch reproduce/icml2026/stats/run_stats.sbatch \
      ${SAMBAL_DATA_DIR:-data}/babycosmofine/stats_shard_$i.jsonl \
      ${SAMBAL_DATA_DIR:-data}/babycosmofine/stats_shard_$i.pkl
done
# after the jobs complete:
python -m sambal.stats_merge ${SAMBAL_DATA_DIR:-data}/babycosmofine 'stats_shard_*.pkl' \
    sambal/resources/lemma_stats_top_25k.pkl
```

## 3.4 Ablated corpus

24 shards, seed = shard id, GPU parse per shard. The engine's first start-up on a
machine takes several minutes to build its caches
(`sambal/resources/_augmenter_cache/`).

```bash
for i in $(seq 0 23); do
  sbatch reproduce/icml2026/corpus/run_sambal_corpus.sbatch $i 24
done
# after all 24 complete:
cat ${SAMBAL_DATA_DIR:-data}/babycosmofine_sambal/shard_*.jsonl \
    > ${SAMBAL_DATA_DIR:-data}/babycosmofine_sambal/train_sambal.jsonl
```

Or download the released corpus instead
([`data/babycosmofine_sambal/MANIFEST.md`](../../data/babycosmofine_sambal/MANIFEST.md)):

```bash
hf download ezrawinston/babycosmofine-sambal --repo-type dataset \
    train_sambal.jsonl --local-dir ${SAMBAL_DATA_DIR:-data}/babycosmofine_sambal
```

## 3.5 Corpus variants

The Table 7 and §5 variants are the same run with one override each
([manifest](../../data/babycosmofine_sambal_variants/MANIFEST.md)):

```bash
for i in $(seq 0 23); do
  CONFIG_OVERRIDES='{"skip_toinf_freeze": true}' \
  OUT_DIR=${SAMBAL_DATA_DIR:-data}/babycosmofine_sambal_variants/no_toinf \
    sbatch --export=ALL reproduce/icml2026/corpus/run_sambal_corpus.sbatch $i 24
done

for i in $(seq 0 23); do
  CONFIG_OVERRIDES='{"respect_gender": false}' \
  OUT_DIR=${SAMBAL_DATA_DIR:-data}/babycosmofine_sambal_variants/no_gender \
    sbatch --export=ALL reproduce/icml2026/corpus/run_sambal_corpus.sbatch $i 24
done

for i in $(seq 0 23); do
  CONFIG_OVERRIDES='{"roundtrip_ud": false}' \
  OUT_DIR=${SAMBAL_DATA_DIR:-data}/babycosmofine_sambal_variants/no_ud_roundtrip \
    sbatch --export=ALL reproduce/icml2026/corpus/run_sambal_corpus.sbatch $i 24
done

for i in $(seq 0 23); do
  CONFIG_OVERRIDES='{"human_to_human": false}' \
  OUT_DIR=${SAMBAL_DATA_DIR:-data}/babycosmofine_sambal_variants/no_humanness \
    sbatch --export=ALL reproduce/icml2026/corpus/run_sambal_corpus.sbatch $i 24
done

for i in $(seq 0 23); do
  CONFIG_OVERRIDES='{"ctx_lemma_gate": "off"}' \
  OUT_DIR=${SAMBAL_DATA_DIR:-data}/babycosmofine_sambal_variants/no_context_buckets \
    sbatch --export=ALL reproduce/icml2026/corpus/run_sambal_corpus.sbatch $i 24
done

# benchmark-vocabulary replacements: uniform sampling within the context bucket,
# replacement vocabulary sambal/resources/blimpvocab.txt
for i in $(seq 0 23); do
  CONFIG_OVERRIDES='{"ctx_lemma_gate": "flat"}' \
  PATHS_OVERRIDES='{"allowed_vocab_path": "sambal/resources/blimpvocab.txt"}' \
  OUT_DIR=${SAMBAL_DATA_DIR:-data}/babycosmofine_sambal_variants/blimpvocab_flat \
    sbatch --export=ALL reproduce/icml2026/corpus/run_sambal_corpus.sbatch $i 24
done
# after a variant's 24 shards complete:
cat ${SAMBAL_DATA_DIR:-data}/babycosmofine_sambal_variants/no_toinf/shard_*.jsonl \
    > ${SAMBAL_DATA_DIR:-data}/babycosmofine_sambal_variants/no_toinf/train_sambal_no_toinf.jsonl
```

## 3.6 Vocab-filtered control corpus

The App. H lexical-regularization control (Table 9;
[`data/babycosmofine_top25k/MANIFEST.md`](../../data/babycosmofine_top25k/MANIFEST.md)):

```bash
python data/babycosmofine_top25k/build_top25k.py
```

## 3.7 Tokenization

Every corpus a trainer reads (the bin lands beside its jsonl):

```bash
python lm/gpt-bert/corpus_tokenization/tokenize_corpus.py \
    --data_folder=${SAMBAL_DATA_DIR:-data}/babycosmofine \
    --train_file=train.jsonl --name=10M

python lm/gpt-bert/corpus_tokenization/tokenize_corpus.py \
    --data_folder=${SAMBAL_DATA_DIR:-data}/babycosmofine_sambal \
    --train_file=train_sambal.jsonl --name=10M

python lm/gpt-bert/corpus_tokenization/tokenize_corpus.py \
    --data_folder=${SAMBAL_DATA_DIR:-data}/babycosmofine_top25k \
    --train_file=train_top25k.jsonl --name=10M

python lm/gpt-bert/corpus_tokenization/tokenize_corpus.py \
    --data_folder=${SAMBAL_DATA_DIR:-data}/babycosmofine_sambal_variants/no_toinf \
    --train_file=train_sambal_no_toinf.jsonl --name=10M
# ... and the other variants of 3.5 the same way
```

## 3.8 Reflexive-augmented corpora

Builds the App. D augmented corpora (GPU): the augmentation set
([`aug_reflexive_number_match.jsonl`](../../evals/syntaxgym/aug_sets/aug_reflexive_number_match.jsonl))
and `<base>_with_reflexives_number.bin` beside each tokenized corpus:

```bash
bash reproduce/icml2026/reflexives/run_reflexive_number_pipeline.sh
```

## 3.9 Domain corpora (LotR, PubMed)

Place the LotR text at `${SAMBAL_DATA_DIR:-data}/lotr/lotr.txt` (not
distributed: [`data/lotr/MANIFEST.md`](../../data/lotr/MANIFEST.md)), then:

```bash
python data/lotr/chunk_lotr.py
python data/lotr/split_lotr.py
python data/pubmed/get_pubmed.py    # sized to lotr.txt if present, else to its reference word count
python data/pubmed/split_pubmed.py
```

## 3.10 Entropy record — Table 10

Needs the three corpora ([3.2](#32-source-corpus), [3.4](#34-ablated-corpus) or the
released corpus, [3.6](#36-vocab-filtered-control-corpus)); no GPU:

```bash
python evals/entropy/matched_budget.py \
    --out reproduce/icml2026/records/entropy/entropy_matched_budget.json
```
