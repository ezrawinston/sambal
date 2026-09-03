"""
Static utility functions for token and dependency inspection.

These functions are pure utilities with no state dependencies, extracted from
the Augmenter class to reduce its complexity and improve reusability.
"""
from __future__ import annotations
from typing import Iterator, Tuple, Optional, Set
from spacy.tokens import Token


# =============================================================================
# Dependency Label Compatibility (spaCy vs UD)
# =============================================================================

def is_particle_dep(dep: str) -> bool:
    """Check if dep label indicates a verbal particle (spaCy: 'prt'; UD: 'compound:prt')."""
    return dep == "prt" or dep == "compound:prt"


def is_auxpass_dep(dep: str) -> bool:
    """Check if dep label indicates passive auxiliary (spaCy: 'auxpass'; UD: 'aux:pass')."""
    return dep == "auxpass" or dep == "aux:pass"


def is_nsubjpass_dep(dep: str) -> bool:
    """Check if dep label indicates passive subject (spaCy: 'nsubjpass'; UD: 'nsubj:pass')."""
    return dep == "nsubjpass" or dep == "nsubj:pass"


def is_obj_dep(dep: str) -> bool:
    """Check if dep label indicates an object (direct, indirect, or prepositional)."""
    return dep in {"obj", "dobj", "pobj", "iobj"}


# =============================================================================
# Token Type Checks
# =============================================================================

VERB_TAGS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
NOUN_TAGS = {"NN", "NNS", "NNP", "NNPS"}
PROPN_TAGS = {"NNP", "NNPS"}


def is_verb_like(tok: Token) -> bool:
    """Check if token is verb-like (has verb tag or VerbForm morphology)."""
    return tok.tag_ in VERB_TAGS or bool(tok.morph.get("VerbForm"))


def is_noun_like(tok: Token) -> bool:
    """Check if token is a common noun (not proper noun)."""
    return tok.tag_ in {"NN", "NNS"} or (tok.pos_ == "NOUN" and tok.tag_ not in PROPN_TAGS)


def is_infinitival_to_as_prep(tok: Token) -> bool:
    """
    Check if `tok` is an ADP token 'to' functioning as an infinitival marker
    (misparsed as a preposition).

    Returns True iff:
      - tok lemma is 'to'
      - it does NOT have a nominal object (pobj/obj)
      - it governs a bare-infinitive verb child (VB / VerbForm=Inf) via pcomp/xcomp
    """
    try:
        if tok.pos_ != "ADP":
            return False
        if tok.lemma_.lower() != "to":
            return False

        # Real PP "to NP" -> keep it
        if any(ch.dep_ in {"pobj", "obj"} for ch in tok.children):
            return False

        # Infinitival marker usually introduces a bare verb
        for ch in tok.children:
            if ch.dep_ in {"pcomp", "xcomp"} and ch.pos_ in {"VERB", "AUX"}:
                if ch.tag_ == "VB":
                    return True
                if "Inf" in set(ch.morph.get("VerbForm")):
                    return True
        return False
    except Exception:
        return False


def is_governed_prep_child(tok: Token) -> bool:
    """
    Check if `tok` is a governed preposition child of a verb.

    Governed preposition = headed by a verb (directly via 'prep' or indirectly
    via 'obl' + 'case'), excluding adjuncts and infinitival 'to'.
    """
    try:
        head = tok.head
        if head is None or tok.i == head.i:
            return False
        if tok.pos_ != "ADP":
            return False
        if tok.dep_ == "prep" and head.pos_ in {"VERB", "AUX"}:
            # Exclude infinitival 'to'
            if is_infinitival_to_as_prep(tok):
                return False
            return True
        if tok.dep_ == "case":
            grandhead = head.head
            if grandhead is not None and grandhead.pos_ in {"VERB", "AUX"}:
                if head.dep_.startswith("obl"):
                    return True
        return False
    except Exception:
        return False


# =============================================================================
# Preposition Extraction
# =============================================================================

# Temporal subordinators to ignore when they head gerund clauses
_TEMPORAL_SUBORDINATORS = {"before", "after", "while", "when", "since", "until", "upon"}


def iter_governed_preps(head: Token) -> Iterator[str]:
    """
    Yield governed preposition lemmas for 'head' in both schemes:
      - spaCy: head -> 'prep'(ADP)
      - UD:    head -> 'obl*' -> 'case'(ADP)

    Filters out infinitival 'to' and temporal subordinators with gerund clauses.
    """
    # spaCy style
    for ch in head.children:
        if ch.dep_ == "prep" and ch.pos_ == "ADP":
            # Avoid treating infinitival 'to' as governed PP
            if is_infinitival_to_as_prep(ch):
                continue
            # Filter temporal subordinators with pcomp (gerund clause)
            if any(gc.dep_ == "pcomp" for gc in ch.children):
                if ch.lemma_.lower() in _TEMPORAL_SUBORDINATORS:
                    continue
                yield ch.lemma_.lower()
                continue
            yield ch.lemma_.lower()

    # UD style
    for ch in head.children:
        if ch.dep_.startswith("obl"):
            for gc in ch.children:
                if gc.dep_ == "case" and gc.pos_ == "ADP":
                    # If infinitival VB got misattached as 'obl' with case 'to', ignore
                    try:
                        if (gc.lemma_.lower() == "to"
                                and ch.pos_ in {"VERB", "AUX"}
                                and (ch.tag_ == "VB" or "Inf" in set(ch.morph.get("VerbForm")))):
                            continue
                    except Exception:
                        pass
                    yield gc.lemma_.lower()


# =============================================================================
# String Utilities
# =============================================================================

def match_case_like(src: str, cand: str) -> str:
    """Match the casing style of `src` onto `cand` (best-effort)."""
    if not src or not cand:
        return cand
    if src.isupper():
        return cand.upper()
    if src.islower():
        return cand.lower()
    if src.istitle():
        return cand[:1].upper() + cand[1:].lower()
    # mixed: preserve first character case only
    if src[:1].isupper():
        return cand[:1].upper() + cand[1:]
    return cand


# =============================================================================
# Morphology Utilities
# =============================================================================

def morph_signature(tok: Token) -> Tuple[Tuple[str, str], ...]:
    """
    Create a hashable morphology signature for a token.
    Returns a tuple of (key, value) pairs sorted alphabetically.
    """
    try:
        d = tok.morph.to_dict()
    except Exception:
        return (("MORPH", str(tok.morph)),)

    items = []
    for k, v in d.items():
        if isinstance(v, (list, tuple)):
            vv = ",".join(map(str, v))
        else:
            vv = str(v)
        items.append((str(k), vv))
    return tuple(sorted(items))


def child_dep_counts(tok: Token) -> Tuple[Tuple[str, int], ...]:
    """Return sorted tuple of (dep_label, count) for token's children."""
    from collections import Counter
    c = Counter()
    try:
        for ch in tok.children:
            c[str(ch.dep_)] += 1
    except Exception:
        pass
    return tuple(sorted(c.items()))


def safe_lemma(tok: Token) -> str:
    """Get lowercase lemma, falling back to lowercase surface form."""
    lem = (tok.lemma_ or tok.text or "").strip().lower()
    return lem if lem else (tok.text or "").strip().lower()


# =============================================================================
# Expletive Detection
# =============================================================================

def has_expl_there_child(v: Token) -> bool:
    """True iff v has a direct 'expl' child 'there' (existential/copular)."""
    return any(ch.lemma_.lower() == "there" and ch.dep_ in {"expl", "nsubj"} for ch in v.children)
