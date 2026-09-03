#!/usr/bin/env python3
"""Build the lexical-regularization control corpus.

Filters the source corpus to sentences whose every word is in the top-25k
frequency vocabulary (wordfreq's English list): documents are split with NLTK
`sent_tokenize`; a word is a whitespace token with non-alphanumeric characters
stripped from both edges, lowercased; a sentence is kept iff it contains at
least one word and every word is in the vocabulary; kept sentences are joined
with a single space and empty documents are dropped.

Verified: this reproduces the released `train_top25k.jsonl` byte-for-byte
(20,084 documents, md5 d262acbde1c797ae9cc060464f080125) from the pinned
source corpus.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import nltk
from wordfreq import top_n_list

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("SAMBAL_DATA_DIR") or (REPO_ROOT / "data"))
EXPECT_MD5 = "d262acbde1c797ae9cc060464f080125"


def words(sentence):
    out = []
    for raw in sentence.split():
        i, j = 0, len(raw)
        while i < j and not raw[i].isalnum():
            i += 1
        while j > i and not raw[j - 1].isalnum():
            j -= 1
        w = raw[i:j].lower()
        if w:
            out.append(w)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path,
                    default=DATA_ROOT / "babycosmofine/train.jsonl")
    ap.add_argument("--out", type=Path,
                    default=DATA_ROOT / "babycosmofine_top25k/train_top25k.jsonl")
    ap.add_argument("--top-n", type=int, default=25000)
    args = ap.parse_args()

    for pkg in ["punkt", "punkt_tab"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    from nltk.tokenize import sent_tokenize

    vocab = set(top_n_list("en", args.top_n))

    lines = []
    with args.input.open(encoding="utf-8") as fin:
        for line in fin:
            text = json.loads(line)["text"]
            kept = []
            for s in sent_tokenize(text):
                ws = words(s)
                if ws and all(w in vocab for w in ws):
                    kept.append(s)
            out = " ".join(kept)
            if out.strip():
                lines.append(json.dumps({"text": out}))

    blob = ("\n".join(lines) + "\n").encode("utf-8")
    args.out.write_bytes(blob)
    digest = hashlib.md5(blob).hexdigest()
    print(f"wrote {args.out}: {len(lines)} documents, md5 {digest}")
    if args.top_n == 25000 and digest != EXPECT_MD5:
        print(f"WARNING: md5 differs from the released control corpus ({EXPECT_MD5}) — "
              "check the source corpus revision and nltk/wordfreq versions")


if __name__ == "__main__":
    main()
