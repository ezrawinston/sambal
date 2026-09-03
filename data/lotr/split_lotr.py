#!/usr/bin/env python3
"""Build the LotR train/dev/test tokenized bins.

Tokenizes `data/lotr/lotr.jsonl` (from `chunk_lotr.py`) with the gpt-bert
tokenizer and applies the deterministic 80/10/10 prefix split by document:
the first int(0.8*n) chunks are train and the remainder is halved into dev
then test (1156 chunks -> 924/116/116), writing `lotr_{train,dev,test}.bin`.
With the reference `lotr.txt` (see `data/lotr/MANIFEST.md`) the split
reproduces the corpora the experiments consumed.
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
    ap.add_argument("--jsonl", type=Path, default=DATA_ROOT / "lotr/lotr.jsonl")
    ap.add_argument("--tokenizer", type=Path,
                    default=REPO_ROOT / "lm/gpt-bert/gpt-bert-babylm-small/tokenizer.json")
    ap.add_argument("--out-dir", type=Path, default=DATA_ROOT / "lotr")
    args = ap.parse_args()

    tok = Tokenizer.from_file(str(args.tokenizer))
    docs = [json.loads(l)["text"] for l in args.jsonl.open(encoding="utf-8")]

    toks = [torch.tensor(tok.encode(t.strip(), add_special_tokens=False).ids,
                         dtype=torch.int16) for t in docs]

    n = len(toks)
    n_train = int(0.8 * n)
    n_dev = (n - n_train) // 2
    splits = {
        "train": toks[:n_train],
        "dev": toks[n_train:n_train + n_dev],
        "test": toks[n_train + n_dev:],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, tensors in splits.items():
        out = args.out_dir / f"lotr_{split}.bin"
        torch.save(tensors, out)
        print(f"wrote {out} ({len(tensors)} documents, "
              f"{sum(len(t) for t in tensors)} tokens)")


if __name__ == "__main__":
    main()
