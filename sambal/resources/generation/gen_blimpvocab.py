#!/usr/bin/env python3
"""
gen_blimpvocab.py

Build `blimpvocab.txt` — the benchmark-vocabulary gate of the `blimpvocab_flat`
corpus variant, committed in `sambal/resources/` — by counting spaCy tokens
over the sentences of the full BLiMP benchmark.

The committed file was built from the 67 BLiMP paradigm files (both
`sentence_good` and `sentence_bad` of every pair) at `--min-freq 1`:
2,411 entries, md5 `5cce7bb3d5e2acdece3059d57bf623ed`.

What it counts
--------------
- spaCy tokenization, so the gate agrees with the engine's token boundaries.
- Alphabetic tokens and tokens with internal apostrophes or hyphens
  (`don't`, `state-of-the-art`). Whitespace and punctuation tokens are skipped.
- Case is preserved: counts are kept per surface form, as in
  `allowed_vocab.txt`, because the engine's gates look candidates up as
  written.

Output: one `token<TAB>count` line per entry, most frequent first and ties in
codepoint order, entries below `--min-freq` dropped.

Usage
-----
  python evals/blimp/fetch_data.py          # once; pins the BLiMP commit
  python sambal/resources/generation/gen_blimpvocab.py \
      --out sambal/resources/blimpvocab.txt \
      --spacy-model en_core_web_sm \
      --min-freq 1 \
      evals/blimp/data

Inputs may be files, directories (walked recursively) or globs.

Reproducibility
---------------
The committed file is reproduced byte-for-byte under Python 3.9 with spaCy
3.8.7 and en_core_web_sm 3.8.0 from the BLiMP files `evals/blimp/fetch_data.py`
fetches. Tokenization is version-sensitive: other spaCy or model versions are
untested and are not expected to give identical bytes.
"""

import argparse, os, sys, re, json
from collections import Counter
from tqdm import tqdm

ALNUM_APOS_HYPHEN = re.compile(r"^(?:[A-Za-z]+(?:['-][A-Za-z]+)*)|(?:[A-Za-z]+(?:-[A-Za-z]+)+)$")


def iter_paths(paths):
    import glob
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for fn in sorted(files):
                    if fn.endswith(".jsonl"):
                        yield os.path.join(root, fn)
        else:
            for q in sorted(glob.glob(p)):
                if os.path.isfile(q) and q.endswith(".jsonl"):
                    yield q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--spacy-model", default="en_core_web_sm")
    ap.add_argument("--min-freq", type=int, default=1)
    ap.add_argument("inputs", nargs="+", help="BLiMP jsonl files, directories or globs")
    args = ap.parse_args()

    try:
        import spacy
    except Exception:
        print("Please `pip install spacy` and the model you name with --spacy-model", file=sys.stderr)
        sys.exit(2)

    nlp = spacy.load(args.spacy_model, disable=["parser", "senter", "ner", "lemmatizer"])
    vocab = Counter()

    def sentence_stream():
        for path in iter_paths(args.inputs):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "sentence_good" in data:
                        yield data["sentence_good"]
                    if "sentence_bad" in data:
                        yield data["sentence_bad"]

    for doc in tqdm(nlp.pipe(sentence_stream(), batch_size=2000), disable=not sys.stderr.isatty()):
        for tok in doc:
            if tok.is_space or tok.is_punct:
                continue
            if ALNUM_APOS_HYPHEN.match(tok.text):
                vocab[tok.text] += 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    kept = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for w, c in sorted(vocab.items(), key=lambda kv: (-kv[1], kv[0])):
            if c >= args.min_freq:
                out.write(f"{w}\t{c}\n")
                kept += 1
    print(f"Wrote {kept} tokens -> {args.out}")


if __name__ == "__main__":
    main()
