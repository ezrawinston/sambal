"""Fetch a LotR-sized sample of PubMed abstracts.

Streams the PubMed title+abstracts baseline dump (pinned upstream revision)
and keeps abstracts until their total word count matches the LotR corpus:
the whitespace word count of `data/lotr/lotr.txt` when that file is present,
otherwise REFERENCE_TARGET_WORDS — the count of the fingerprinted `lotr.txt`
in `data/lotr/MANIFEST.md`, i.e. the target the reference corpus was built
to — so the PubMed corpus builds without the LotR text. Writes
`data/pubmed/pubmed_abstracts.jsonl`; the reference checksum in
`data/pubmed/MANIFEST.md` is the gate that the fetched corpus matches the
one the experiments consumed.

Requires: pip install datasets
(and transformers, only if COUNT_MODE="hf_tokens").
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("SAMBAL_DATA_DIR") or (REPO_ROOT / "data"))

# ----------------------------
# Config
# ----------------------------

# The LotR corpus that sets the sample-size target: either
#   1) a plain .txt file, or
#   2) a .jsonl file with {"text": "..."} per line
LOTR_PATH = str(DATA_ROOT / "lotr/lotr.txt")

# Whitespace word count of the fingerprinted lotr.txt (data/lotr/MANIFEST.md):
# the sample-size target when LOTR_PATH is absent (COUNT_MODE "words" only).
REFERENCE_TARGET_WORDS = 575_230

# Output JSONL
OUT_PATH = str(DATA_ROOT / "pubmed/pubmed_abstracts.jsonl")

# Use "words" for whitespace-word count, or "hf_tokens" to match a tokenizer.
COUNT_MODE = "words"

# If COUNT_MODE == "hf_tokens", set this to your tokenizer name/path.
# Example: TOKENIZER_NAME = "gpt2"
TOKENIZER_NAME = None

# If False, keep full abstracts and stop at the closest size.
# If True, truncate the final abstract to hit the target exactly.
STRICT_EXACT = False

SEED = 42
SHUFFLE_BUFFER_SIZE = 100_000

# Raw file URL on Hugging Face, pinned to a fixed upstream revision
PUBMED_URL = (
    "https://huggingface.co/datasets/casinca/"
    "PUBMED_title_abstracts_2019_baseline/resolve/"
    "5b8dcdcf6657dc120049448261445a5a9848fc8b/"
    "PUBMED_title_abstracts_2019_baseline.jsonl.zst"
)

# ----------------------------
# Counting helpers
# ----------------------------

tokenizer = None
if COUNT_MODE == "hf_tokens":
    if not TOKENIZER_NAME:
        raise ValueError("Set TOKENIZER_NAME when COUNT_MODE='hf_tokens'.")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)


def count_units(text: str) -> int:
    if tokenizer is None:
        return len(text.split())
    return len(tokenizer.encode(text, add_special_tokens=False))


def truncate_to_units(text: str, n_units: int) -> str:
    if n_units <= 0:
        return ""

    if tokenizer is None:
        words = text.split()
        return " ".join(words[:n_units]).strip()

    ids = tokenizer.encode(text, add_special_tokens=False)
    truncated_ids = ids[:n_units]
    return tokenizer.decode(truncated_ids, skip_special_tokens=True).strip()


# ----------------------------
# Input readers
# ----------------------------

def iter_lotr_texts(path: str | Path) -> Iterator[str]:
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "text" not in obj:
                    raise KeyError(f"{path} is JSONL but a line has no 'text' field.")
                yield obj["text"]
    else:
        yield path.read_text(encoding="utf-8")


def extract_abstract(pubmed_text: str) -> str | None:
    """
    PubMed dataset rows are typically:
        TITLE + "\\n" + ABSTRACT

    We keep only the abstract. If a row has no newline or no abstract content,
    skip it by returning None.
    """
    parts = pubmed_text.split("\n", 1)
    if len(parts) != 2:
        return None

    abstract = parts[1].strip()
    return abstract if abstract else None


# ----------------------------
# Main
# ----------------------------

def resolve_target(lotr_path: str | Path) -> int:
    """Sample-size target: counted from the LotR corpus if present, else the
    reference count (words mode only)."""
    lotr_path = Path(lotr_path)
    if lotr_path.exists():
        target = sum(count_units(text) for text in iter_lotr_texts(lotr_path))
        print(f"Target size from {lotr_path} ({COUNT_MODE}): {target:,}")
        if COUNT_MODE == "words" and target != REFERENCE_TARGET_WORDS:
            print(f"Note: differs from the reference target {REFERENCE_TARGET_WORDS:,} "
                  "(a lotr.txt other than the fingerprinted one)")
        return target
    if COUNT_MODE != "words":
        raise SystemExit(f"{lotr_path} not found and there is no reference target for "
                         f"COUNT_MODE={COUNT_MODE!r}; supply the LotR corpus")
    print(f"{lotr_path} not found; using the reference target (words): "
          f"{REFERENCE_TARGET_WORDS:,}")
    return REFERENCE_TARGET_WORDS


def main() -> None:
    target_units = resolve_target(LOTR_PATH)

    # Stream the PubMed dataset and approximately shuffle it
    ds = load_dataset(
        "json",
        data_files=PUBMED_URL,
        split="train",
        streaming=True,
    ).shuffle(seed=SEED, buffer_size=SHUFFLE_BUFFER_SIZE)

    selected = []
    total_units = 0

    for row in ds:
        abstract = extract_abstract(row["text"])
        if not abstract:
            continue

        n = count_units(abstract)
        if n == 0:
            continue

        new_total = total_units + n

        # Still below target: keep whole abstract
        if new_total < target_units:
            selected.append({"text": abstract})
            total_units = new_total
            continue

        # Exact hit
        if new_total == target_units:
            selected.append({"text": abstract})
            total_units = new_total
            break

        # Overshoot: either keep the full abstract if it's closer,
        # or optionally truncate the final abstract to match exactly.
        remaining = target_units - total_units

        if STRICT_EXACT:
            truncated = truncate_to_units(abstract, remaining)
            if truncated:
                selected.append({"text": truncated})
                total_units += count_units(truncated)
        else:
            # Keep whichever is closer: stopping here vs. adding the whole abstract
            undershoot = target_units - total_units
            overshoot = new_total - target_units
            if overshoot < undershoot:
                selected.append({"text": abstract})
                total_units = new_total

        break

    # Write JSONL
    out_path = Path(OUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in selected:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote {len(selected):,} abstracts to {out_path}")
    print(f"Final size ({COUNT_MODE}): {total_units:,}")
    print(f"Target size ({COUNT_MODE}): {target_units:,}")
    print(f"Difference: {total_units - target_units:+,}")


if __name__ == "__main__":
    main()