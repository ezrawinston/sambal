#!/usr/bin/env python3
"""
gen_countability.py

Build the mass/count/both/plural-only countability lexicons from a
Kaikki/Wiktextract English Wiktionary JSONL dump (see fetch_inputs.sh for the
dump source). Shows progress bars with tqdm, writes debug anomalies.

Outputs:
  - mass.txt
  - count.txt
  - both.txt
  - pluralia_tantum.txt
  - lemma_meta.tsv (lemma\tbucket\tplural_forms\ttags_seen)
  - anomalies.jsonl (full JSON for entries dropped/flagged)
"""

import argparse
import gzip
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Set, List, Tuple
from tqdm import tqdm

# -------------------------
# Constants / heuristics
# -------------------------

# Kaikki/Wiktextract tags
KAIKKI_COUNT_TAGS = {"countable"}
KAIKKI_MASS_TAGS = {"uncountable"}
KAIKKI_PLURAL_ONLY_TAGS = {"plural-only", "pluralia-tantum"}
KAIKKI_SINGULAR_ONLY_TAGS = {"singular-only"}

# Usage labels to ignore for bucket decisions
KAIKKI_USAGE_IGNORED = {"in-plural", "usually-plural", "often-plural"}

# Misspelling filters
SPELLING_NOISE_TAGS = {
    "misspelling", "misspelling-of", "nonstandard-spelling",
    "eye-dialect", "obsolete-spelling-of", "obsolete-spelling",
}
MISSPELLING_CATEGORY_SUBSTR = {"english misspellings"}

# Regional/dialect auto-detection (no curated list)
REGION_ACRONYMS = {"US","UK","NZ","AU","CA","IE","IN","PH","SG","MY","ZA","HK"}
REGION_SEED = {
    "Philippines","Canada","Australia","New Zealand","Ireland","Scotland",
    "India","Singapore","Malaysia","South Africa","Hong Kong","Jamaica",
    "Caribbean",
}

def is_region_like_tag(tag: str) -> bool:
    if not tag:
        return False
    t = tag.strip()
    if t in REGION_ACRONYMS:
        return True
    if t.endswith(" English") or "-English" in t:
        return True
    if t in REGION_SEED:
        return True
    return False

# -------------------------
# Common helpers
# -------------------------

WORD_RE = re.compile(r"^[^\W_]+(?:[-'][^\W_]+)*$", re.UNICODE)  # single token: letters + optional - / '

def is_single_token(word: str) -> bool:
    return bool(WORD_RE.match(word))

def stream_jsonl(path: str):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield json.loads(s)

def add_row(meta, lemma, bucket, tags_seen, plural_forms):
    # Merge if lemma already present
    prev = meta.get(lemma)
    if prev:
        prev_bucket, prev_tags, prev_plurals = prev
        if prev_bucket != bucket:
            rank = {"BOTH": 3, "COUNT": 2, "PLURAL_ONLY": 2, "MASS": 1, "UNKNOWN": 0}
            bucket = max((prev_bucket, bucket), key=lambda b: rank.get(b, 0))
        tags_seen = set(prev_tags) | set(tags_seen)
        plural_forms = set(prev_plurals) | set(plural_forms)
    meta[lemma] = (bucket, set(tags_seen), set(plural_forms))

def write_lines(path, items):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for x in sorted(set(items)):
            f.write(x + "\n")

def write_outputs(meta, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    mass, count, both, plural_only = [], [], [], []
    with open(os.path.join(outdir, "lemma_meta.tsv"), "w", encoding="utf-8") as f:
        f.write("lemma\tbucket\tplural_forms\ttags_seen\n")
        for lemma, (bucket, tags, plurals) in sorted(meta.items()):
            if bucket == "MASS":
                mass.append(lemma)
            elif bucket == "COUNT":
                count.append(lemma)
            elif bucket == "BOTH":
                both.append(lemma)
            elif bucket == "PLURAL_ONLY":
                plural_only.append(lemma)
            f.write(f"{lemma}\t{bucket}\t{','.join(sorted(plurals))}\t{','.join(sorted(tags))}\n")
    write_lines(os.path.join(outdir, "mass.txt"), mass)
    write_lines(os.path.join(outdir, "count.txt"), count)
    write_lines(os.path.join(outdir, "both.txt"), both)
    write_lines(os.path.join(outdir, "pluralia_tantum.txt"), plural_only)

# -------------------------
# Anomaly logging
# -------------------------

_ANOMALIES: List[dict] = []

def _record_anomaly(entry: dict, reason: str):
    _ANOMALIES.append({"reason": reason, "entry": entry})

def write_anomalies(outdir: str, fname: str = "anomalies.jsonl"):
    if not _ANOMALIES:
        return
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, fname)
    with open(path, "w", encoding="utf-8") as f:
        for row in _ANOMALIES:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(_ANOMALIES)} anomalous entries to {path}")

# -------------------------
# Lemma / senses processing
# -------------------------

def _lemma_head_status(entry: dict) -> Tuple[bool, str]:
    """
    Inspect head_templates and return (is_lemma, status).
    status ∈ {"lemma","noun_form","ambiguous","no_head","pos_only_lemma","pos_only_nonlemma"}
    """
    heads = entry.get("head_templates") or []
    if not heads:
        if entry.get("pos") == "noun":
            return True, "no_head"
        return False, "pos_only_nonlemma"

    saw_lemma = False
    saw_form = False
    for ht in heads:
        name = (ht.get("name") or "").strip().lower()
        args = ht.get("args") or {}
        if isinstance(args, dict):
            arg2 = str(args.get("2", "")).strip().lower()
        elif isinstance(args, list) and len(args) > 1:
            arg2 = str(args[1]).strip().lower()
        else:
            arg2 = ""
        if name == "en-noun" or arg2 == "noun":
            saw_lemma = True
        if arg2 == "noun form" or name.endswith("noun form"):
            saw_form = True

    if saw_lemma and saw_form:
        return True, "ambiguous"
    if saw_lemma:
        return True, "lemma"
    if saw_form:
        return False, "noun_form"
    if entry.get("pos") == "noun":
        return True, "pos_only_lemma"
    return False, "pos_only_nonlemma"

def _filter_non_form_senses(senses: list) -> list:
    """
    Remove 'form-of' senses, misspellings, and regional/dialect senses.
    """
    out = []
    for s in (senses or []):
        stags = set(s.get("tags", []) or [])
        # drop 'form-of' senses entirely
        if "form-of" in stags:
            continue
        # drop misspellings
        if stags & SPELLING_NOISE_TAGS:
            continue
        # drop regional/dialect
        if any(is_region_like_tag(t) for t in stags):
            continue
        out.append(s)
    return out

def _lemma_plural_only_from_senses(entry_tags, senses) -> bool:
    """
    Lemma-level plural-only policy after filtering:
      - consider only non-form senses (we pass filtered senses already)
      - return True only if *every* such sense is plural-only
      - if none remain, fall back to entry-level plural-only
    """
    non_form_tags = [set(s.get("tags", []) or []) for s in (senses or [])]
    if non_form_tags:
        return all((stags & KAIKKI_PLURAL_ONLY_TAGS) for stags in non_form_tags)
    return bool(set(entry_tags or []) & KAIKKI_PLURAL_ONLY_TAGS)

def classify_from_kaikki_entry(e):
    """
    Returns: lemma, bucket, tags_seen(set), plural_forms(set) or None
    """
    lemma = e.get("word")
    if not lemma or e.get("lang_code") != "en" or e.get("pos") != "noun":
        return None

    # Drop obvious misspelling pages via categories
    for c in (e.get("categories") or []):
        if isinstance(c, str) and any(sub in c.lower() for sub in MISSPELLING_CATEGORY_SUBSTR):
            _record_anomaly(e, "page_misspelling_category")
            return None

    # Lemma vs noun form
    is_lemma, head_status = _lemma_head_status(e)
    if head_status in {"no_head", "ambiguous"}:
        _record_anomaly(e, head_status)
    if not is_lemma:
        if head_status == "noun_form":
            return None
        return None

    entry_tags = set(e.get("tags", []) or [])
    senses_all = e.get("senses", []) or []
    senses = _filter_non_form_senses(senses_all)

    # If nothing remains after filtering, skip lemma
    if not senses:
        _record_anomaly(e, "sense_filtered_empty")
        return None

    # Union of tags across filtered senses
    sense_tags_union = set()
    for s in senses:
        stags = set(s.get("tags", []) or [])
        stags -= KAIKKI_USAGE_IGNORED
        sense_tags_union |= stags
    tags_seen = entry_tags | sense_tags_union

    # Plural-only decision based on filtered senses
    is_plural_only = _lemma_plural_only_from_senses(entry_tags, senses)
    is_count = bool(tags_seen & KAIKKI_COUNT_TAGS)
    is_uncount = bool(tags_seen & KAIKKI_MASS_TAGS)

    # Plural forms (for metadata)
    plurals = set()
    for frm in e.get("forms", []) or []:
        ftags = set(frm.get("tags", []) or [])
        if "plural" in ftags:
            val = frm.get("form")
            if val:
                plurals.add(val)

    # Decide bucket
    if is_plural_only:
        bucket = "PLURAL_ONLY"
    elif is_count and is_uncount:
        bucket = "BOTH"
    elif is_uncount:
        bucket = "MASS"
    elif is_count:
        bucket = "COUNT"
    else:
        bucket = "COUNT"  # default

    return lemma, bucket, tags_seen, plurals

def build_from_kaikki(path: str, single_token: bool) -> dict:
    meta = {}
    for e in tqdm(stream_jsonl(path), desc="Kaikki JSON entries", unit="entry"):
        out = classify_from_kaikki_entry(e)
        if not out:
            continue
        lemma, bucket, tags, plurals = out
        if single_token and not is_single_token(lemma):
            continue
        add_row(meta, lemma, bucket, tags, plurals)
    return meta

# -------------------------
# CLI
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Build noun pools (mass/count/both/plural-only) from Kaikki/Wiktextract JSONL.")
    ap.add_argument("dump_path", help="Path to raw-wiktextract-data.jsonl[.gz]")
    ap.add_argument("--single-token", action="store_true", help="Keep only single-token lemmas (recommended).")
    ap.add_argument("--outdir", default="sambal/resources", help="Output directory (default: sambal/resources -- the committed lists, overwritten in place)")
    args = ap.parse_args()

    meta = build_from_kaikki(args.dump_path, args.single_token)
    write_outputs(meta, args.outdir)
    write_anomalies(args.outdir)

    total = len(meta)
    buckets = defaultdict(int)
    for _, (b, _, _) in meta.items():
        buckets[b] += 1
    print(f"Processed {len(meta)} lemmas.")
    for k in ("MASS", "COUNT", "BOTH", "PLURAL_ONLY"):
        print(f"  {k:12s} {buckets.get(k,0)}")
    print(f"Wrote outputs to: {os.path.abspath(args.outdir)}")

if __name__ == "__main__":
    main()
