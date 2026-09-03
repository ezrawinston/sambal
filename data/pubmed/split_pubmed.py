#!/usr/bin/env python3
"""Build the pubmed train/dev/test tokenized bins.

Tokenizes `data/pubmed/pubmed_abstracts.jsonl` (from `get_pubmed.py`) with the
gpt-bert tokenizer and slices it by the recorded document permutation in
`data/pubmed/split_indices.json` (an ~80/10/10 shuffled split), writing
`pubmed_abstracts_{train,dev,test}.bin`. With the recorded indices the output
is byte-identical to the corpora the experiments consumed.
"""
import argparse
import json
import os
from pathlib import Path

import torch
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("SAMBAL_DATA_DIR") or (REPO_ROOT / "data"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path,
                    default=DATA_ROOT / "pubmed/pubmed_abstracts.jsonl")
    ap.add_argument("--indices", type=Path,
                    default=REPO_ROOT / "data/pubmed/split_indices.json")
    ap.add_argument("--tokenizer", type=Path,
                    default=REPO_ROOT / "lm/gpt-bert/gpt-bert-babylm-small/tokenizer.json")
    ap.add_argument("--out-dir", type=Path, default=DATA_ROOT / "pubmed")
    args = ap.parse_args()

    tok = Tokenizer.from_file(str(args.tokenizer))
    docs = [json.loads(l)["text"] for l in args.jsonl.open(encoding="utf-8")]
    splits = json.loads(args.indices.read_text())

    n_idx = sum(len(v) for v in splits.values())
    if n_idx != len(docs):
        raise SystemExit(f"index count {n_idx} != document count {len(docs)} — "
                         "the jsonl does not match the recorded split")

    toks = [torch.tensor(tok.encode(t.strip(), add_special_tokens=False).ids,
                         dtype=torch.int16) for t in docs]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, idx in splits.items():
        out = args.out_dir / f"pubmed_abstracts_{split}.bin"
        torch.save([toks[i] for i in idx], out)
        print(f"wrote {out} ({len(idx)} documents)")


if __name__ == "__main__":
    main()
