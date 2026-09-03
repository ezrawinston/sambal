#!/usr/bin/env python3
"""prefilter_gendered_words.py

Prefilter `gendered_words.json` (e.g. from https://github.com/ecmonsen/gendered_words)
using an allowed-vocab list, to reduce replacement sampling failures.

Rationale
---------
The augmenter normally checks `allowed_vocab` *after* realizing a surface form
(e.g., lemma -> plural), because allowed-vocab is defined over surface forms.

This script does a conservative prefilter that still works on the JSON lexicon:
  1) Read allowed-vocab with the same "token\tcount" logic as the augmenter,
     applying a `min_vocab_freq` threshold.
  2) Lowercase all allowed-vocab tokens.
  3) For each gendered-words record:
       - If it is NOT a noun (based on `wordnet_senseno`), keep it.
       - If it is a noun:
           * lowercase the record's `word`
           * generate a noun plural form using the same pluralization logic as the augmenter
           * generate a noun singular form (because the source `word` might already be plural)
           * keep the record iff either the singular OR plural is present in allowed-vocab (lowercased)

Notes
-----
- This is intentionally conservative; it may still keep words that will later fail
  due to other constraints (NPIs, POS-tag mismatches, etc.).
- It only tries noun singularization/pluralization; non-nouns are passed through.

Usage
-----
python sambal/resources/generation/prefilter_gendered_words.py \
  --gendered-words-in sambal/resources/generation/downloads/gendered_words.json \
  --allowed-vocab sambal/resources/allowed_vocab.txt \
  --out sambal/resources/gendered_words_filtered.json

The committed `gendered_words_filtered.json` is this run against the pinned
`gendered_words.json` and the committed `allowed_vocab.txt`, at the default
`--min-vocab-freq 10`. See `generation/README.md`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import inflect
from lemminflect import getInflection


def load_allowed_vocab_lower(path: str, min_vocab_freq: int) -> Set[str]:
    """Load allowed vocab in the same way as the augmenter, then lowercase it.

    Expected file format (per line):
        token\tcount

    Lines that are blank or start with '#' are ignored.

    Returns:
        Set of allowed surface tokens lowercased.
    """
    av_lower: Set[str] = set()
    with open(path, "r", encoding="utf-8") as vf:
        for raw in vf:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            # take the first column (token) and ignore any count or extra columns
            entries = raw.split("\t", 1)
            if len(entries) < 2:
                # mirror augmenter behavior but avoid crashing on malformed lines
                continue
            tok = entries[0].strip()
            try:
                # augmenter does: int(entries[1].strip())
                # we do the same, but tolerate trailing columns by taking the first whitespace chunk
                count_str = entries[1].strip().split()[0]
                count = int(count_str)
            except Exception:
                continue
            if tok and count >= min_vocab_freq:
                av_lower.add(tok.lower())
    return av_lower


def pluralize_noun_like_augmenter(word_lc: str, infl: inflect.engine) -> str:
    """Pluralize noun using the same approach as Augmenter._realize for ptb_tag=='NNS'.

    Order:
      1) inflect_engine.plural_noun(word) if available and returns a truthy value
      2) lemminflect getInflection(word, tag='NNS')
      3) fallback to word itself

    `word_lc` must be lowercase already.
    """
    form: Optional[str] = None

    # Prefer noun-only pluralization when available.
    try:
        plural_noun_fn = getattr(infl, "plural_noun", None)
        if callable(plural_noun_fn):
            pn = plural_noun_fn(word_lc)
            # plural_noun() typically returns False when it can't pluralize
            # or thinks the input is already plural.
            if pn:
                form = pn
    except Exception:
        form = None

    # Fall back to lemminflect
    if not form:
        try:
            infls = getInflection(word_lc, tag="NNS")
            if infls:
                form = infls[0]
        except Exception:
            form = None

    # Last resort
    return form if form else word_lc


def singularize_noun(word_lc: str, infl: inflect.engine) -> str:
    """Best-effort noun singularization.

    We use inflect_engine.singular_noun(), which is noun-specific (unlike plural()),
    and fall back to the input if singular_noun() returns False / None.

    `word_lc` must be lowercase already.
    """
    try:
        sn = infl.singular_noun(word_lc)
        if sn:
            return str(sn)
    except Exception:
        pass
    return word_lc


def is_noun_record(rec: Dict[str, Any]) -> bool:
    """Return True iff the record looks like a WordNet noun sense.

    We treat "unknown" as non-noun (so it will be kept), matching:
      "almost all entries are nouns; if not, just include it".
    """
    ws = rec.get("wordnet_senseno")
    if isinstance(ws, str) and ws:
        parts = ws.split(".")
        # Typical: word.pos.sense -> abbess.n.01
        if len(parts) >= 3:
            pos = parts[1].lower()
            return pos == "n"
        # Fallback heuristic
        if ".n." in ws:
            return True
        if any(f".{p}." in ws for p in ("v", "a", "s", "r")):
            return False

    # If the dataset changes and introduces explicit POS fields, handle lightly.
    pos_field = rec.get("pos") or rec.get("part_of_speech") or rec.get("parts_of_speech")
    if isinstance(pos_field, str):
        pf = pos_field.lower()
        if pf in {"n", "noun"} or pf.startswith("n"):
            return True
        if pf in {"v", "verb", "a", "adj", "adjective", "r", "adv", "adverb", "s"}:
            return False
    if isinstance(pos_field, list):
        low = [str(x).lower() for x in pos_field]
        if any(x in {"n", "noun"} or x.startswith("n") for x in low):
            return True
        if any(x in {"v", "verb", "a", "adj", "adjective", "r", "adv", "adverb", "s"} for x in low):
            return False

    # Unknown -> non-noun
    return False


def should_keep_record(rec: Dict[str, Any], allowed_vocab_lower: Set[str], infl: inflect.engine) -> Tuple[bool, str]:
    """Return (keep?, reason) for a record."""
    if not is_noun_record(rec):
        return True, "non_noun"

    w = rec.get("word")
    if not isinstance(w, str) or not w.strip():
        return True, "missing_word"

    w_lc = w.strip().lower()

    if w_lc in allowed_vocab_lower:
        return True, "direct"

    plural = pluralize_noun_like_augmenter(w_lc, infl)
    if plural.lower() in allowed_vocab_lower:
        return True, "plural"

    singular = singularize_noun(w_lc, infl)
    if singular.lower() in allowed_vocab_lower:
        return True, "singular"

    return False, "drop"


def main() -> None:
    ap = argparse.ArgumentParser(description="Prefilter gendered_words.json by allowed vocab")
    ap.add_argument("--gendered-words-in", required=True, help="Path to input gendered_words.json")
    ap.add_argument("--allowed-vocab", required=True, help="Path to allowed vocab TSV (token\\tcount)")
    ap.add_argument("--min-vocab-freq", type=int, default=10,
                    help="Minimum vocab frequency (default: 10, the value the "
                         "committed gendered_words_filtered.json was built with)")
    ap.add_argument("--out", required=True, help="Path to write filtered JSON")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON output (indent=2)")
    args = ap.parse_args()

    allowed_lower = load_allowed_vocab_lower(args.allowed_vocab, args.min_vocab_freq)
    infl = inflect.engine()

    in_path = Path(args.gendered_words_in)
    data = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"Expected a JSON list in {in_path}, got {type(data)}")

    kept: List[Dict[str, Any]] = []
    stats = {
        "total": 0,
        "kept": 0,
        "dropped": 0,
        "kept_non_noun": 0,
        "kept_direct": 0,
        "kept_plural": 0,
        "kept_singular": 0,
        "kept_missing_word": 0,
    }

    for rec in data:
        if not isinstance(rec, dict):
            # keep unknown shapes
            kept.append(rec)
            stats["total"] += 1
            stats["kept"] += 1
            continue

        stats["total"] += 1
        keep, reason = should_keep_record(rec, allowed_lower, infl)
        if keep:
            kept.append(rec)
            stats["kept"] += 1
            if reason == "non_noun":
                stats["kept_non_noun"] += 1
            elif reason == "direct":
                stats["kept_direct"] += 1
            elif reason == "plural":
                stats["kept_plural"] += 1
            elif reason == "singular":
                stats["kept_singular"] += 1
            elif reason == "missing_word":
                stats["kept_missing_word"] += 1
        else:
            stats["dropped"] += 1

    out_path = Path(args.out)
    if args.pretty:
        out_path.write_text(json.dumps(kept, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write("[\n")
            for i, item in enumerate(kept):
                json_str = json.dumps(item, ensure_ascii=False)
                if i < len(kept) - 1:
                    f.write(json_str + ",\n")
                else:
                    f.write(json_str + "\n")
            f.write("]\n")

    # Print a small report
    print("=== prefilter_gendered_words report ===")
    print(f"allowed_vocab_lower: {len(allowed_lower):,} entries (min_freq={args.min_vocab_freq})")
    print(f"gendered_words_in:   {stats['total']:,} records")
    print(f"kept:               {stats['kept']:,}")
    print(f"dropped:            {stats['dropped']:,}")
    print("kept breakdown:")
    print(f"  non-noun:         {stats['kept_non_noun']:,}")
    print(f"  direct match:     {stats['kept_direct']:,}")
    print(f"  plural match:     {stats['kept_plural']:,}")
    print(f"  singular match:   {stats['kept_singular']:,}")
    print(f"  missing word:     {stats['kept_missing_word']:,}")
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
