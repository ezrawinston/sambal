"""
Token features dataclass and schema for context-based lemma statistics.

TokenFeatures consolidates computed linguistic features for a token, used by
both stats collection and runtime augmentation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class TokenFeatures:
    """Computed linguistic features for a token, used by both stats collection and augmentation.

    This dataclass consolidates features computed by multiple methods (_ptb_tag_for_*,
    _extract_frame, _parse_comp_kind, etc.) to avoid redundant computation across
    tokeninfo_key, _ctx_bucket_for_token, and augment_doc.
    """
    # Core spacy attrs
    pos: str
    tag: str
    dep: str
    morph_sig: Tuple[Tuple[str, str], ...]  # Full morph signature for tokeninfo_key

    # Head info
    head_pos: str
    head_tag: str
    head_dep: str
    head_is_self: bool

    # PTB tags (computed per POS type)
    ptb_noun: str
    ptb_noun_forced: str
    ptb_adj: str
    ptb_adv: str
    ptb_verb: str

    # Noun features
    is_bare_singular: bool

    # Verb frame/valency features
    is_verb_like: bool
    frame_key: str
    frame_key_raw: str  # before valency fixes
    frame_has_prt: bool
    frame_req_prep: str
    preps_set: Tuple[str, ...]

    # Complement features
    comp_kind: str
    comp_kind_simple: str

    # Clause features
    is_passive: bool
    has_aux: bool
    has_heavy_aux: bool
    has_expl_there: bool
    is_to_be_xcomp: bool

    # Clausal complement markers (for verbs)
    ccomp_marks: Tuple[str, ...]
    xcomp_has_to: bool
    xcomp_has_for: bool
    xcomp_is_ger: bool


# Schema mapping TokenFeatures field names to tuple indices for stats collection.
# Used by collect_lemma_token_info_stats and bucket_builders.
TOKENINFO_SCHEMA = {
    "is_bare_singular": 2,
    "pos": 3,
    "tag": 4,
    "dep": 5,
    "head_pos": 7,
    "head_tag": 8,
    "ptb_noun_forced": 14,
    "ptb_adj": 15,
    "ptb_adv": 16,
    "is_verb_like": 17,
    "frame_key": 18,
    "frame_has_prt": 19,
    "preps_set": 21,
    "comp_kind": 22,
    "is_passive": 24,
    "has_expl_there": 27,
    "to_be_xcomp": 28,
    "xcomp_has_to": 30,
    "xcomp_has_for": 31,
    "xcomp_is_ger": 32,
    "ptb_verb_forced": 33,
}
