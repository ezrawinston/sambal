#!/usr/bin/env python3
"""Chunk the LotR trilogy text into ~510-word documents.

Reads `data/lotr/lotr.txt` (user-supplied; checksum in `data/lotr/MANIFEST.md`)
and writes `data/lotr/lotr.jsonl`, one {"text": ...} chunk per line, for
`split_lotr.py` to tokenize and split.
"""
import argparse
import json
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("SAMBAL_DATA_DIR") or (REPO_ROOT / "data"))


def clean_and_chunk_text(input_file, output_file, words_per_chunk=510):
    # Read the file
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Remove newlines and join text together
    text = text.replace('\n', ' ')

    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)

    # Split into sentences (basic sentence splitting)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Group sentences into chunks of ~500 words
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        word_count = len(sentence.split())

        # If adding this sentence keeps us under target, add it
        if current_word_count + word_count <= words_per_chunk:
            current_chunk.append(sentence)
            current_word_count += word_count
        else:
            # Save current chunk if it has content
            if current_chunk:
                chunks.append(' '.join(current_chunk))

            # Start new chunk with current sentence
            current_chunk = [sentence]
            current_word_count = word_count

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    # Write chunks to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks, 1):
            record = {"text": chunk}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Processed {len(sentences)} sentences into {len(chunks)} chunks")
    print(f"Average chunk size: {sum(len(c.split()) for c in chunks) / len(chunks):.1f} words")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DATA_ROOT / "lotr/lotr.txt")
    ap.add_argument("--output", type=Path, default=DATA_ROOT / "lotr/lotr.jsonl")
    ap.add_argument("--words-per-chunk", type=int, default=510)
    args = ap.parse_args()
    clean_and_chunk_text(args.input, args.output, args.words_per_chunk)
