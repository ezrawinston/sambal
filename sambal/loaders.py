"""
File loading utilities for sambal.

Contains functions for loading various resource files (lexicons, patterns, etc.)
used by the Augmenter.
"""
from __future__ import annotations

import csv
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set

from spacy.matcher import Matcher, PhraseMatcher

from .pronoun_utils import GENDER_TAGS

if TYPE_CHECKING:
    from .config import ResourcePaths


def load_lines(path: Optional[str]) -> List[str]:
    """Load all non-empty lines from a text file.

    Args:
        path: Path to the text file. If None, returns empty list.

    Returns:
        List of stripped non-empty lines.

    Raises:
        FileNotFoundError: If path is provided but file doesn't exist.
    """
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return [ln.strip() for ln in p.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]


def load_first_col(path: Optional[str]) -> List[str]:
    """Load the first column from a delimited file.

    Automatically detects delimiter (comma, semicolon, or tab) from the first few lines.
    If no delimiter is detected, treats each line as a single value.

    Args:
        path: Path to the file. If None, returns empty list.

    Returns:
        List of first-column values.
    """
    if not path:
        return []
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return []
    head = lines[0]
    if any(d in head for d in [",", ";", "\t"]):
        scores = {d: sum(ln.count(d) for ln in lines[:5]) for d in (",", ";", "\t")}
        delim = max(scores, key=scores.get) or "\t"
        rdr = csv.reader(io.StringIO(raw), delimiter=delim)
        out = []
        for row in rdr:
            if row and row[0]:
                out.append(row[0].strip())
        return out
    return [ln.strip() for ln in lines]


def load_gendered_words_lexicon(path: Optional[str]) -> Dict[str, str]:
    """Load a word->gender map from the `ecmonsen/gendered_words` JSON file.

    Expected JSON schema: list of objects with at least:
      - word: str
      - gender: one of {"m","f","n","o"}

    Returns:
        dict[word_lower] = gender in {"m","f","n"} for *unambiguous* words only.
        Words with multiple genders across entries, or tagged "o", are dropped.
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, list):
        return {}

    genders_by_word: Dict[str, Set[str]] = defaultdict(set)
    for rec in data:
        if not isinstance(rec, dict):
            continue
        w = (rec.get("word") or "").strip().lower()
        g = (rec.get("gender") or "").strip().lower()
        if not w or g not in GENDER_TAGS:
            continue
        genders_by_word[w].add(g)

    out: Dict[str, str] = {}
    for w, gs in genders_by_word.items():
        if len(gs) == 1:
            out[w] = next(iter(gs))
    return out


def load_fixed_mwes(paths: "ResourcePaths", nlp) -> Optional[PhraseMatcher]:
    """Load fixed multi-word expressions as a PhraseMatcher.

    Args:
        paths: ResourcePaths with fixed_mwes_path.
        nlp: spaCy language model.

    Returns:
        PhraseMatcher with FIXED_MWE patterns, or None if no path/file.
    """
    if not paths.fixed_mwes_path:
        return None
    phrases = load_lines(paths.fixed_mwes_path)
    if not phrases:
        return None
    pm = PhraseMatcher(nlp.vocab, attr="LOWER")
    pm.add("FIXED_MWE", [nlp.make_doc(p) for p in phrases])
    return pm


def load_mwe_patterns(paths: "ResourcePaths", nlp) -> Optional[Matcher]:
    """Load spaCy Matcher patterns from a JSONL file.

    Each line is a JSON object with a 'patterns' dict containing optional
    'tight' and 'wildcard' lists of spaCy patterns.

    Args:
        paths: ResourcePaths with mwe_patterns_jsonl_path.
        nlp: spaCy language model.

    Returns:
        Matcher with VMWE_TIGHT and VMWE_WILDCARD patterns, or None if no path/file.
    """
    path = paths.mwe_patterns_jsonl_path
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    m = Matcher(nlp.vocab)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            pats = obj.get("patterns", {})
            for pat in pats.get("tight", []):
                m.add("VMWE_TIGHT", [pat])
            for pat in pats.get("wildcard", []):
                m.add("VMWE_WILDCARD", [pat])
    return m


