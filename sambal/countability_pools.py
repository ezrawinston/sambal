"""
Countability pools from Wiktionary (Kaikki) for noun replacement.

Provides grammatical constraints for singular/plural noun substitution
based on countability categories (mass, count, both, pluralia tantum).
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Set


_WORD_RE = re.compile(r"^[^\W_]+(?:[-'][^\W_]+)*$", re.UNICODE)  # single token: letters + optional hyphen/' (no spaces)


def _is_single_token(s: str) -> bool:
    return bool(_WORD_RE.match(s))


class CountabilityPools:
    """
    Pools built from files:
      - mass.txt              # uncountable only
      - count.txt             # countable only
      - both.txt              # has both mass and count senses
      - pluralia_tantum.txt   # plural-only headwords (e.g., pants, scissors)

     Return a *lemma pool* that guarantees grammaticality without parsing determiners:
       - If source is BOTH -> restrict to BOTH in all contexts.
       - NNS (plural):
           COUNT -> count | both | plural-only
           MASS  -> count-capable | plural-only   (defensive)
           PLURAL_ONLY -> plural-only
           UNKNOWN -> count-capable | plural-only
       - NN bare (singular, no determiner):
           MASS -> mass-capable
           COUNT -> freeze (empty set)
       - NN non-bare (singular with determiner/modifiers):
           MASS -> MASS-only
           COUNT -> COUNT-only
     Notes:
       - PLURAL_ONLY never appears in NN output pools.
       - The source lemma is excluded from the returned set.
    """
    def __init__(self, mass, count, both, plural_only):
        self.mass = set(mass)
        self.count = set(count)
        self.both = set(both)
        self.plural_only = set(plural_only)

        # Build "capability" sets
        self.mass_capable = self.mass | self.both
        self.count_capable = self.count | self.both

    def filter(self, filter_fn):
        self.mass = filter_fn(self.mass)
        self.count = filter_fn(self.count)
        self.both = filter_fn(self.both)
        self.plural_only = filter_fn(self.plural_only)
        self.mass_capable = filter_fn(self.mass_capable)
        self.count_capable = filter_fn(self.count_capable)

    @classmethod
    def from_dir(cls, d: str, single_token_only: bool = True, drop_capitalized: bool = True):
        dpath = Path(d)

        def _has_capital(s: str) -> bool:
            # Unicode-aware: Lu/Lt categories count as "capital"
            return any(ch.isupper() for ch in s)

        def _clean_split(text: str):
            """
            Yield normalized items from a raw file:
              - strip BOM and inline '#...' comments
              - split on commas and whitespace (keep hyphens/apostrophes intact)
              - drop any token containing a capital letter if drop_capitalized is True
              - NFKC normalize; do NOT casefold yet (we want to inspect capitals first)
            """
            if text and text[0] == "\ufeff":  # BOM
                text = text[1:]

            for raw in text.splitlines():
                line = raw.split("#", 1)[0].strip()
                if not line:
                    continue
                for tok in re.split(r"[,\s]+", line):
                    tok = unicodedata.normalize("NFKC", tok).strip("\"'")
                    if not tok:
                        continue
                    if drop_capitalized and _has_capital(tok):
                        continue
                    yield tok

        def _load(name):
            p = dpath / name
            if not p.exists():
                return set()
            text = p.read_text(encoding="utf-8", errors="replace")
            items = list(_clean_split(text))
            if single_token_only:
                items = [w for w in items if _is_single_token(w)]
            # case-insensitive keys downstream
            return set(w.casefold() for w in items)

        mass = _load("mass.txt")
        count = _load("count.txt")
        both = _load("both.txt")
        plural_only = _load("pluralia_tantum.txt")
        return cls(mass, count, both, plural_only)

    def bucket_of(self, lemma: str) -> str:
        """Return 'MASS' | 'COUNT' | 'BOTH' | 'PLURAL_ONLY' | 'UNKNOWN' for a lemma."""
        l = lemma.lower()
        if l in self.plural_only:
            return "PLURAL_ONLY"
        if l in self.both:
            return "BOTH"
        if l in self.mass:
            return "MASS"
        if l in self.count:
            return "COUNT"
        return "UNKNOWN"

    def pool_for_slot(self, *, source_lemma: str, tag: str, is_bare_singular: bool) -> Set[str]:
        """
        Decide the *candidate pool* based on:
          - source lemma's bucket,
          - slot tag (NN vs NNS),
          - bare singular status.
        """
        src = source_lemma.lower()
        src_bucket = self.bucket_of(src)

        if tag == "NNS":
            if src_bucket == "BOTH":
                return self.both - {src}
            if src_bucket == "COUNT":
                return (self.count | self.both | self.plural_only) - {src}
            if src_bucket == "MASS":
                return (self.count_capable | self.plural_only) - {src}
            if src_bucket == "PLURAL_ONLY":
                return self.plural_only - {src}
            return (self.count_capable | self.plural_only) - {src}

        # NN (singular) slot
        if is_bare_singular:
            # only nouns that can surface bare as singular without an article:
            # BOTH nouns are mass-capable; MASS is mass-capable; COUNT-only should *not* freeze:
            # fall back to mass-capable so we keep grammaticality.
            if src_bucket == "BOTH":
                return self.both - {src}
            if src_bucket == "MASS":
                return self.mass_capable - {src}
            # Previously: return set()  # freeze COUNT-only in bare NN
            # New: allow replacement, but restrict to mass-capable (mass | both)
            return self.mass_capable - {src}
        else:
            # not bare -> stay in the same bucket
            if src_bucket == "BOTH":
                return self.both - {src}
            if src_bucket == "MASS":
                return self.mass - {src}
            if src_bucket == "COUNT":
                return self.count - {src}
            if src_bucket == "PLURAL_ONLY":
                # plural-only lemma appearing as NN (rare): safest is to skip
                return set()
            # UNKNOWN: conservative default -> treat as count-capable
            return (self.count_capable - {src})
