#!/usr/bin/env python3
"""
gen_allowed_vocab.py

Build `allowed_vocab.txt` — the surface-form vocabulary gate committed in
`sambal/resources/` — by counting spaCy tokens over one or more plain-text
corpus files.

The committed file was built from the six plain-text files of the BabyLM 2024
10M-word train split (`bnc_spoken`, `childes`, `gutenberg`, `open_subtitles`,
`simple_wiki`, `switchboard`) at `--min-freq 1`: 147,938 entries, md5
`46672aec4b1519b521123097b59e0f35`.

What it counts
--------------
- spaCy tokenization, so the gate agrees with the engine's token boundaries.
- Alphabetic tokens and tokens with internal apostrophes or hyphens
  (`don't`, `state-of-the-art`). Whitespace and punctuation tokens are skipped.
- **Case is preserved.** The engine's proper-noun and given-name gates are
  case-sensitive — a capitalized candidate is looked up as written — so counts
  are kept per surface form and never folded to lowercase. 77,652 of the
  committed file's 147,938 entries contain an uppercase character.
- CHILDES tier markers such as `*MOT:` lose their punctuation to the skip
  above, but the speaker code itself survives as an ordinary token: `MOT` and
  `CHI` are in the committed file (`MOT` is its second-most-frequent entry at
  251,187; `CHI` is eighth at 187,543).

Output: one `token<TAB>count` line per entry, most frequent first, entries
below `--min-freq` dropped.

Usage
-----
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

Inputs may be files, directories (walked recursively) or globs.

Reproducibility
---------------
The committed file is reproduced byte-for-byte under Python 3.9 with spaCy
3.8.7 and en_core_web_sm 3.8.0, in about 14 minutes on one CPU core for the
10M-word split. Tokenization is version-sensitive: other spaCy or model
versions are untested and are not expected to give identical bytes. Input
checksums and fetch instructions are in `generation/README.md`.
"""

import argparse, os, sys, re
from collections import Counter
from tqdm import tqdm

ALNUM_APOS_HYPHEN = re.compile(r"^(?:[A-Za-z]+(?:['’-][A-Za-z]+)*)|(?:[A-Za-z]+(?:-[A-Za-z]+)+)$")


def iter_paths(paths):
    import glob
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for fn in files:
                    yield os.path.join(root, fn)
        else:
            # allow globs
            for q in glob.glob(p):
                if os.path.isfile(q):
                    yield q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--spacy-model", default="en_core_web_sm")
    ap.add_argument("--min-freq", type=int, default=1)
    ap.add_argument("inputs", nargs="+", help="Files/dirs/globs")
    args = ap.parse_args()

    try:
        import spacy
    except Exception:
        print("Please `pip install spacy` and the model you name with --spacy-model", file=sys.stderr)
        sys.exit(2)

    nlp = spacy.load(args.spacy_model, disable=["parser","senter","ner","lemmatizer"])
    vocab = Counter()

    # stream files line-by-line; batch with nlp.pipe for speed
    def line_stream():
        for path in iter_paths(args.inputs):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    # skip empty-ish lines
                    if not line.strip():
                        continue
                    yield line.rstrip("\n")

    for doc in tqdm(nlp.pipe(line_stream(), batch_size=2000)):
        for tok in doc:
            if tok.is_space or tok.is_punct:
                continue
            s = tok.text
            # keep alphabetic, apostrophes, hyphens (e.g., don't, mother-in-law)
            if ALNUM_APOS_HYPHEN.match(s):
                vocab[s] += 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    kept = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for w, c in vocab.most_common():
            if c >= args.min_freq:
                out.write(f"{w}\t{c}\n")
                kept += 1

    print(f"Wrote {kept} tokens → {args.out}")

if __name__ == "__main__":
    main()
