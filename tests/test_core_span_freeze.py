"""Unit tests for the span-level freeze API in ``core``.

The ``frozen_ranges`` and ``frozen_lemmas`` kwargs on ``_render_doc`` and
``build_candidate_sets_from_docs`` let callers exempt character spans or
lemmas from substitution. These tests cover the contract: tokens whose char
span overlaps a frozen range OR whose lemma is in the frozen set are emitted
verbatim and excluded from candidate-set contribution; the default ``None``
leaves substitution unrestricted.
"""
from __future__ import annotations

from typing import List

import pytest
import spacy
from spacy.tokens import Doc

from sambal.core import (
    Occurrence, OccChainLevel,
    _is_token_frozen,
    _render_doc,
    _token_overlaps_any_range,
)


# --------------------------------------------------------------------------
# Stub Augmenter
# --------------------------------------------------------------------------

class _StubInflect:
    def plural_noun(self, s: str) -> str:
        return s + "s" if not s.endswith("s") else s

    def a(self, s: str) -> str:
        return f"an {s}" if s[:1].lower() in "aeiou" else f"a {s}"


class _StubAug:
    def __init__(self):
        self.inflect_engine = _StubInflect()

    def _realize(self, lemma_lc: str, ptb_tag: str) -> str:
        return lemma_lc

    def _fix_indefinite_articles(self, text: str) -> str:
        return text


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def blank_nlp():
    return spacy.blank("en")


def _doc_from_tokens(nlp, words: List[str], spaces: List[bool],
                      lemmas: List[str], pos: List[str],
                      tags: List[str]) -> Doc:
    return Doc(nlp.vocab, words=words, spaces=spaces,
                lemmas=lemmas, pos=pos, tags=tags)


def _mk_occ(doc_tag, tok_i, *, ptb_tag="NN", pos_family="NOUN") -> Occurrence:
    return Occurrence(
        doc_tag=doc_tag, tok_i=tok_i, ptb_tag=ptb_tag,
        pos_family=pos_family, source_surface="",
        chain=[OccChainLevel(cand_set=frozenset(), weights=None)],
    )


# --------------------------------------------------------------------------
# 1. _token_overlaps_any_range — geometric overlap rules
# --------------------------------------------------------------------------

def test_token_overlaps_any_range_semantics(blank_nlp):
    # Tokens "abc def ghi": chars [0,3) [4,7) [8,11)
    doc = _doc_from_tokens(
        blank_nlp,
        ["abc", "def", "ghi"], [True, True, False],
        ["abc", "def", "ghi"], ["NOUN", "NOUN", "NOUN"],
        ["NN", "NN", "NN"],
    )
    abc, def_, ghi = doc[0], doc[1], doc[2]

    # No ranges → never frozen
    assert not _token_overlaps_any_range(abc, None)
    assert not _token_overlaps_any_range(abc, [])

    # Range [0,3) covers exactly abc
    assert _token_overlaps_any_range(abc, [(0, 3)])
    assert not _token_overlaps_any_range(def_, [(0, 3)])
    assert not _token_overlaps_any_range(ghi, [(0, 3)])

    # Range [3,4) is the whitespace between abc/def — no token overlap
    assert not _token_overlaps_any_range(abc, [(3, 4)])
    assert not _token_overlaps_any_range(def_, [(3, 4)])

    # Partial overlap: range [2, 5) clips abc's tail and def's head
    assert _token_overlaps_any_range(abc, [(2, 5)])  # abc[2,3) overlaps
    assert _token_overlaps_any_range(def_, [(2, 5)])  # def[4,5) overlaps
    assert not _token_overlaps_any_range(ghi, [(2, 5)])

    # Multiple ranges: any match wins
    assert _token_overlaps_any_range(ghi, [(0, 3), (8, 11)])


# --------------------------------------------------------------------------
# 2. _is_token_frozen — combines range and lemma checks
# --------------------------------------------------------------------------

def test_is_token_frozen_via_lemma_set(blank_nlp):
    doc = _doc_from_tokens(
        blank_nlp,
        ["Cat", "sleeps"], [True, False],
        ["cat", "sleep"], ["NOUN", "VERB"],
        ["NN", "VBZ"],
    )
    # Lemma check is case-folded — match on lowercase form
    assert _is_token_frozen(doc[0], "x", None, {"cat"})
    assert not _is_token_frozen(doc[1], "x", None, {"cat"})
    # Empty lemma set + no ranges → never frozen
    assert not _is_token_frozen(doc[0], "x", None, None)
    assert not _is_token_frozen(doc[0], "x", None, set())


def test_is_token_frozen_doc_tag_routing(blank_nlp):
    doc = _doc_from_tokens(
        blank_nlp,
        ["abc"], [False],
        ["abc"], ["NOUN"],
        ["NN"],
    )
    # frozen_ranges_per_doc must be looked up by doc_tag — wrong tag ≠ frozen
    assert _is_token_frozen(doc[0], "msg_0",
                              {"msg_0": [(0, 3)]}, None)
    assert not _is_token_frozen(doc[0], "msg_1",
                                  {"msg_0": [(0, 3)]}, None)


# --------------------------------------------------------------------------
# 3. _render_doc with no frozen kwargs = prior behavior (regression)
# --------------------------------------------------------------------------

def test_render_doc_no_freeze_kwargs_replaces_normally(blank_nlp):
    doc = _doc_from_tokens(
        blank_nlp,
        ["The", "cat", "sleeps", "."], [True, True, False, False],
        ["the", "cat", "sleep", "."],
        ["DET", "NOUN", "VERB", "PUNCT"],
        ["DT", "NN", "VBZ", "."],
    )
    occ_lookup = {("msg", 1): _mk_occ("msg", 1, ptb_tag="NN",
                                         pos_family="NOUN")}
    mapping = {"cat": "dog"}
    aug = _StubAug()
    text, _, _ = _render_doc(aug, doc, occ_lookup, "msg", mapping)
    assert "dog" in text and "cat" not in text


# --------------------------------------------------------------------------
# 4. _render_doc with frozen_ranges emits source verbatim
# --------------------------------------------------------------------------

def test_render_doc_frozen_range_emits_verbatim(blank_nlp):
    doc = _doc_from_tokens(
        blank_nlp,
        ["The", "cat", "sleeps", "."], [True, True, False, False],
        ["the", "cat", "sleep", "."],
        ["DET", "NOUN", "VERB", "PUNCT"],
        ["DT", "NN", "VBZ", "."],
    )
    # token "cat" is at chars [4, 7)
    occ_lookup = {("msg", 1): _mk_occ("msg", 1)}
    mapping = {"cat": "dog"}
    aug = _StubAug()
    text, _, _ = _render_doc(aug, doc, occ_lookup, "msg", mapping,
                                frozen_ranges=[(4, 7)])
    assert "cat" in text and "dog" not in text


# --------------------------------------------------------------------------
# 5. _render_doc with frozen_lemmas — works regardless of position
# --------------------------------------------------------------------------

def test_render_doc_frozen_lemmas_emits_verbatim_anywhere(blank_nlp):
    doc = _doc_from_tokens(
        blank_nlp,
        ["A", "cat", "saw", "another", "cat", "."],
        [True, True, True, True, False, False],
        ["a", "cat", "see", "another", "cat", "."],
        ["DET", "NOUN", "VERB", "DET", "NOUN", "PUNCT"],
        ["DT", "NN", "VBD", "DT", "NN", "."],
    )
    # Both "cat" tokens are listed in the occ_lookup with mapping cat→dog.
    occ_lookup = {
        ("msg", 1): _mk_occ("msg", 1),
        ("msg", 4): _mk_occ("msg", 4),
    }
    mapping = {"cat": "dog"}
    aug = _StubAug()

    # Without frozen_lemmas, both occurrences get replaced.
    text_no_freeze, _, _ = _render_doc(aug, doc, occ_lookup, "msg", mapping)
    assert text_no_freeze.count("dog") == 2

    # With frozen_lemmas={"cat"}, neither occurrence is replaced.
    text_freeze, _, _ = _render_doc(aug, doc, occ_lookup, "msg", mapping,
                                       frozen_lemmas={"cat"})
    assert "dog" not in text_freeze
    assert text_freeze.count("cat") == 2


# --------------------------------------------------------------------------
# 6. Partial overlap → entire token frozen
# --------------------------------------------------------------------------

def test_render_doc_partial_overlap_freezes_whole_token(blank_nlp):
    doc = _doc_from_tokens(
        blank_nlp,
        ["function"], [False],
        ["function"], ["NOUN"],
        ["NN"],
    )
    # Token spans [0, 8). Frozen range [4, 8) covers only the second half.
    # Per the conservative rule, the whole "function" token is frozen.
    occ_lookup = {("msg", 0): _mk_occ("msg", 0)}
    mapping = {"function": "method"}
    aug = _StubAug()
    text, _, _ = _render_doc(aug, doc, occ_lookup, "msg", mapping,
                                frozen_ranges=[(4, 8)])
    assert "function" in text and "method" not in text


# --------------------------------------------------------------------------
# 7. End-to-end: range + lemma freeze coexist; non-frozen tokens still ablate
# --------------------------------------------------------------------------

def test_render_doc_range_and_lemma_combined(blank_nlp):
    # "Write a Python function to find the cat in the cat house."
    # Words split as listed below (spaCy-blank).
    words = ["Write", "a", "Python", "function", "to", "find", "the",
             "cat", "in", "the", "cat", "house", "."]
    spaces = [True, True, True, True, True, True, True, True, True,
              True, True, False, False]
    lemmas = ["write", "a", "python", "function", "to", "find", "the",
              "cat", "in", "the", "cat", "house", "."]
    pos = ["VERB", "DET", "PROPN", "NOUN", "PART", "VERB", "DET",
           "NOUN", "ADP", "DET", "NOUN", "NOUN", "PUNCT"]
    tags = ["VB", "DT", "NNP", "NN", "TO", "VB", "DT", "NN", "IN",
            "DT", "NN", "NN", "."]
    doc = _doc_from_tokens(blank_nlp, words, spaces, lemmas, pos, tags)

    # All replaceable tokens get put in occ_lookup.
    rep_indices = [0, 3, 5, 7, 10, 11]   # write, function, find, cat, cat, house
    occ_lookup = {}
    for i in rep_indices:
        occ_lookup[("msg", i)] = _mk_occ("msg", i, ptb_tag=tags[i],
                                            pos_family="VERB" if pos[i] == "VERB"
                                            else "NOUN")
    mapping = {"write": "WROTE", "function": "METHOD", "find": "LOCATE",
               "cat": "DOG", "house": "HOME"}

    # Freeze (a) char range [0, 24) covering "Write a Python function" prefix
    # and (b) lemma {"house"}.
    frozen_ranges = [(0, len("Write a Python function"))]
    frozen_lemmas = {"house"}
    aug = _StubAug()
    text, _, _ = _render_doc(aug, doc, occ_lookup, "msg", mapping,
                                frozen_ranges=frozen_ranges,
                                frozen_lemmas=frozen_lemmas)

    import re as _re
    # The prefix is byte-equal — Write/function not flipped.
    assert text.startswith("Write a Python function ")
    # The lemma {"house"} passes through anywhere.
    assert _re.search(r"\bhouse\b", text)
    assert not _re.search(r"\bhome\b", text.lower())
    # Non-frozen tokens still got ablated (case-matched to source lowercase).
    assert _re.search(r"\blocate\b", text)
    assert not _re.search(r"\bfind\b", text)
    # The "cat" tokens (lemma not frozen, position not frozen) ablated.
    assert _re.search(r"\bdog\b", text)
    assert not _re.search(r"\bcat\b", text)
