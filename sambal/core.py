"""Shared document-scope relexicalization core.

Provides the candidate-set construction, lock-step joint intersection,
sampling, per-doc rendering, chunked output, and Augmenter runtime
construction shared by multi-doc relexicalization pipelines
(e.g. passages + question + answer aliases).
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any, Dict, FrozenSet, List, Optional, Set, Tuple,
)

from spacy.tokens import Doc, Token

from sambal.config import Config, ResourcePaths
from sambal.engine import Augmenter
from sambal.token_features import TokenFeatures
from sambal.token_utils import (
    match_case_like, is_verb_like, is_noun_like, is_particle_dep,
)


POSS_MARKERS = {"'s", "’s"}
CONTRACTION_MARKERS = {"'s", "’s", "'re", "’re", "'m", "’m",
                       "'d", "’d", "'ll", "’ll", "'ve", "’ve"}
SUBJ_DEPS = {"nsubj", "nsubjpass", "nsubj:pass", "csubj", "csubjpass", "csubj:pass"}


# ==========================================================================
# Data types
# ==========================================================================

@dataclass(frozen=True)
class OccChainLevel:
    """One level of a token's ctx-backoff chain.

    `cand_set` is the per-occurrence lexicon pool filtered to lemmas attested
    at this bucket level. `weights` are raw per-lemma bucket counts at this
    level (used for the mean-weight aggregation downstream), or None when
    ctx gating doesn't apply (PROPN path, to-inf verb spine, no ctx stats).

    `is_broad_terminal` marks the final "relaxed" chain level appended when
    a ``broad_terminal_pool`` was supplied (broader countability pool for
    NOUN, union VerbNet for VERB). Lock-step landing here means the joint
    intersection only succeeded via the relaxation.
    """
    cand_set: FrozenSet[str]
    weights: Optional[Counter]
    is_broad_terminal: bool = False


@dataclass
class Occurrence:
    doc_tag: str           # opaque per-pipeline tag, e.g. "ctx" / "q" (squad)
                           # or "title_3" / "text_7" / "ans_1" (rag)
    tok_i: int             # token index within its doc
    ptb_tag: str
    pos_family: str        # "VERB" | "NOUN" | "ADJ" | "ADV" | "PROPN"
    source_surface: str
    # Strictest-first chain of ctx-backoff levels. A non-ctx occurrence
    # (PROPN bare path, no ctx stats, to-inf spine) has a single-level chain.
    chain: List[OccChainLevel]

    @property
    def cand_set(self) -> FrozenSet[str]:
        """Strictest-level candidate set — backward-compat + convenience."""
        return self.chain[0].cand_set if self.chain else frozenset()

    @property
    def weights(self) -> Optional[Counter]:
        return self.chain[0].weights if self.chain else None


@dataclass
class LemmaEntry:
    cand_set: FrozenSet[str]
    weights: Optional[Counter]  # mean across occurrences, or None for uniform
    occs: List[Occurrence]
    pos_families: Set[str]
    pos_conflict: bool
    # bookkeeping:
    mean_cand_size_before: float = 0.0  # mean |Cand(occ_j)| at chain level 0
    size_after: int = 0                 # |Cand(L)| after lock-step intersection
    accepted_level: int = -1            # chain level at which inter was non-empty, -1 if empty


@dataclass
class CandidateIndex:
    lemmas: Dict[str, LemmaEntry]
    # tokens that would be replaceable but had an empty per-occurrence pool
    # (before intersection) — e.g., verb froze due to particle, to-inf spine,
    # etc. Listed for observability only.
    skipped: List[Tuple[str, int, str]]  # (doc_tag, tok_i, reason)
    # Diagnostic: PROPN tokens whose peer-driven NOUN routing had to fall
    # back to the broad noun pool because `_noun_candidates` returned empty
    # (e.g. a plural-only-bucketed source lemma at NN slot).
    broad_noun_fallback_lemmas: List[str] = field(default_factory=list)
    # Diagnostic: lemmas whose lock-step acceptance landed on a broad
    # terminal chain level (NOUN or VERB). Indicates relaxation was
    # critical to mapping this lemma.
    broad_terminal_accept_lemmas: List[str] = field(default_factory=list)


@dataclass
class SampledMapping:
    mapping: Dict[str, str]                    # lemma_lc -> target_lemma_lc
    pos_conflict_lemmas: List[str]             # any lemma with multi-pos_family
    ctx_intersection_empty_lemmas: List[str]   # intersection was empty
    pos_conflict_resolved_lemmas: List[str]    # multi-pos rescued by dual-use lemma
    passthrough_lemmas: List[str]              # passthrough fallback only
    dropped_reason: Optional[str]              # None when kept


# ==========================================================================
# Span-level freeze helpers (additive; default-None preserves prior behavior)
# ==========================================================================

def _token_overlaps_any_range(tok: Token,
                               ranges: Optional[List[Tuple[int, int]]]) -> bool:
    """True if ``[tok.idx, tok.idx + len(tok.text))`` intersects any
    half-open range in ``ranges``. Conservative: any character overlap
    with a frozen range freezes the entire token (avoids partial-token
    rewrites that would break the freeze contract).
    """
    if not ranges:
        return False
    s = tok.idx
    e = s + len(tok.text)
    for rs, re_ in ranges:
        if s < re_ and rs < e:
            return True
    return False


def _is_token_frozen(tok: Token, doc_tag: str,
                      frozen_ranges_per_doc: Optional[Dict[str, List[Tuple[int, int]]]],
                      frozen_lemmas: Optional[Set[str]]) -> bool:
    """True if the token is frozen by either span-overlap or lemma membership.

    Lemma comparison is case-folded to match the casing convention used in
    ``_render_doc`` and ``sample_mapping``.
    """
    if frozen_ranges_per_doc:
        ranges = frozen_ranges_per_doc.get(doc_tag)
        if _token_overlaps_any_range(tok, ranges):
            return True
    if frozen_lemmas:
        lem_lc = (tok.lemma_ or tok.text or "").casefold()
        if lem_lc in frozen_lemmas:
            return True
    return False


# ==========================================================================
# Candidate-set construction
# ==========================================================================

def _compute_reflexive_subject_ids(doc: Doc) -> Set[int]:
    """Mirror the corresponding scan in ``engine.Augmenter``."""
    out: Set[int] = set()
    for t in doc:
        if t.pos_ == "PRON" and t.morph.get("Reflex") == ["Yes"]:
            head = t.head
            if head is None:
                continue
            subs = [c for c in head.children if c.dep_ in SUBJ_DEPS]
            if subs:
                out.add(subs[0].i)
            elif head.head is not None:
                subs2 = [c for c in head.head.children if c.dep_ in SUBJ_DEPS]
                if subs2:
                    out.add(subs2[0].i)
    return out


# Per-Augmenter caches. We attach them to the Augmenter instance itself
# (via private attributes) rather than a global id()-keyed dict; CPython
# is free to recycle ids of garbage-collected objects, which would
# silently alias caches across short-lived test stubs that happen to land
# at the same address. Per-instance attributes are immune to that.
_NOUN_POOL_ATTR = "_sambal_broad_noun_pools"
_VERB_POOL_ATTR = "_sambal_broad_verb_pool"


def _broad_noun_pools(aug: Augmenter) -> Dict[str, frozenset]:
    """Return {slot_kind: pool} keyed on simplified noun-slot class.

    Precomputed once per Augmenter. The pools relax the countability lexicon's partitioning of
    ``count`` / ``mass`` / ``both`` / ``plural_only`` so that a single target
    lemma can satisfy multiple occurrences of the same source lemma when
    those occurrences fall in different (is_bare_singular, tag) slots. Used
    only as the final fallback chain level — lock-step prefers the strict
    slot-specific pool from ``_noun_candidates`` and only advances here
    when the strict raw intersection across occurrences is empty.

    Slot keys:
      "singular" — any singular-capable noun (NN / NNP): mass ∪ count ∪ both
      "plural"   — any pluralizable noun (NNS / NNPS): count ∪ both ∪ plural_only
    """
    cached = getattr(aug, _NOUN_POOL_ATTR, None)
    if cached is not None:
        return cached
    c = getattr(aug, "countability", None)
    if c is None:
        cached = {"singular": frozenset(), "plural": frozenset()}
    else:
        cached = {
            "singular": frozenset(c.mass | c.count | c.both),
            "plural": frozenset(c.count | c.both | c.plural_only),
        }
    try:
        setattr(aug, _NOUN_POOL_ATTR, cached)
    except (AttributeError, TypeError):
        # Augmenter uses __slots__ without our cache slot? Fall back to
        # uncached behaviour — correctness is preserved, just slower.
        pass
    return cached


def _broad_noun_pool_for_tag(aug: Augmenter, ptb_tag: str) -> Optional[frozenset]:
    pools = _broad_noun_pools(aug)
    if ptb_tag in {"NNS", "NNPS"}:
        return pools["plural"]
    return pools["singular"]


def _broad_verb_pool(aug: Augmenter) -> Optional[frozenset]:
    """Union of every lemma VerbNet knows about, regardless of class / frame
    / preps / flags. Precomputed once per Augmenter.

    Used as the terminal chain level for VERB occurrences when their
    frame-specific pools disagree (multi-frame "begin" / "appear" /
    "admit" cases) and ctx/raw terminals still have empty joint intersection.
    """
    cached = getattr(aug, _VERB_POOL_ATTR, None)
    if cached is not None:
        return cached
    vn = getattr(aug, "vn", None)
    all_lemmas: Set[str] = set()
    if vn is not None and hasattr(vn, "_index"):
        for members in vn._index.values():
            try:
                all_lemmas.update(members)
            except Exception:
                continue
    cached = frozenset(all_lemmas)
    try:
        setattr(aug, _VERB_POOL_ATTR, cached)
    except (AttributeError, TypeError):
        pass
    return cached


def _propn_pool_gated(aug: Augmenter, gender: Optional[str]) -> List[str]:
    """Return the gender-matched (or default) PROPN pool, with the same
    candidate-level gates ``_sample_given_name`` applies when the normal
    pipeline would sample one.
    """
    if gender and aug.given_names_subst_by_gender:
        raw = aug.given_names_subst_by_gender.get(gender, []) or []
    else:
        raw = list(aug.allowed_propn_pool or [])

    npi_unigrams = aug._npi_unigrams or set()
    out: List[str] = []
    for cand in raw:
        if not cand or " " in cand:
            continue
        lc = cand.lower()
        if lc in npi_unigrams:
            continue
        if aug._is_functionish_lemma(lc):
            continue
        if (aug._allowed_lemmas_active
                and not aug._override_given_names_active
                and cand not in (aug.allowed_vocab or set())):
            continue
        out.append(cand)
    return out


def _per_token_candidates(aug: Augmenter, tok: Token, feat: TokenFeatures,
                          reflexive_subject_ids: Set[int]
                          ) -> Tuple[Optional[Set[str]], Optional[str], Optional[str]]:
    """Return ``(base_pool, pos_family, ptb_tag)`` for one token, or
    ``(None, None, None)`` if the token is frozen in the base pipeline.

    No context gating or NPI filtering happens here — the base pool is the
    raw lexicon-derived candidate set. ctx gating happens later per
    occurrence via ``_build_occ_chain``, and the joint intersection happens
    per-lemma in ``build_candidate_sets``.
    """
    # VERB
    if feat.is_verb_like:
        if any(is_particle_dep(c.dep_) for c in tok.children):
            return None, None, None
        if tok.text in CONTRACTION_MARKERS:
            return None, None, None
        ptb_tag = feat.ptb_verb or tok.tag_ or "VB"
        if aug._should_freeze_verb(tok, ptb_tag):
            return None, None, None
        cands = aug._verb_candidates(tok, feat=feat)
        if not cands or cands == [tok.lemma_.lower()]:
            return None, None, None
        return set(cands), "VERB", ptb_tag

    # NOUN (respects reflexive-binder and human_to_human overrides)
    if is_noun_like(tok):
        ptb_tag = feat.ptb_noun_forced or tok.tag_ or "NN"
        lem_lc = (tok.lemma_ or "").lower()
        needs_human = (
            bool(aug.cfg.respect_human)
            and tok.dep_ in SUBJ_DEPS
            and tok.i in reflexive_subject_ids
        )
        force_human = (
            bool(aug.cfg.respect_human)
            and bool(aug.cfg.human_to_human)
            and lem_lc in (aug.human_unigrams or set())
        )
        if needs_human or force_human:
            if aug._is_human_np_span(tok, freeze_unigrams=False):
                return None, None, None  # already human: freeze
            base_pool = aug._human_common_pool_for_ptb(ptb_tag)
        else:
            base_pool = aug._noun_candidates(tok, ptb_tag)
        if not base_pool:
            return None, None, None
        return set(base_pool), "NOUN", ptb_tag

    # ADJ
    if tok.pos_ == "ADJ":
        ptb_tag = feat.ptb_adj or tok.tag_ or "JJ"
        cands = aug._adj_candidates(tok)
        if not cands:
            return None, None, None
        return set(cands), "ADJ", ptb_tag

    # ADV
    if tok.pos_ == "ADV":
        ptb_tag = feat.ptb_adv or tok.tag_ or "RB"
        cands = aug._adv_candidates(tok)
        if not cands:
            return None, None, None
        return set(cands), "ADV", ptb_tag

    # PROPN is handled by _propn_token_candidates after non-PROPN peers
    # are collected — see build_candidate_sets' two-pass Phase 1.
    return None, None, None


def _propn_token_candidates(aug: Augmenter, tok: Token, feat: TokenFeatures,
                            peer_families: Set[str]
                            ) -> Tuple[Optional[Set[str]], Optional[str],
                                       Optional[str], bool]:
    """PROPN dispatch: if there's any non-PROPN peer of this lemma in the
    doc, route through the NOUN path. Otherwise use the default PROPN pool.

    A PROPN is syntactically a noun — it fills a noun slot in the sentence
    regardless of what POS its same-lemma peers happen to take elsewhere in
    the doc. So the PROPN always goes through the noun candidate pool.
    Peer constraints (PROPN "House" + VERB "houses", for example) are
    enforced automatically by the lock-step joint intersection.

    Returns ``(base_pool, pos_family, ptb_tag, used_broad_noun_fallback)``.
    The flag is True iff the strict ``_noun_candidates`` returned empty and
    we fell back to the broad noun pool (rescue for plural-only-bucket edge
    cases like lemma "new" at an NN slot).
    """
    if not aug.cfg.replace_propn:
        return None, None, None, False

    propn_tag = tok.tag_ or "NNP"
    used_broad_fallback = False

    # Any non-PROPN peer → NOUN path.
    if peer_families:
        noun_ptb = feat.ptb_noun_forced or ("NNS" if propn_tag == "NNPS" else "NN")
        noun_base = aug._noun_candidates(tok, noun_ptb)
        if not noun_base:
            # `_noun_candidates` can return empty in narrow cases — e.g.
            # the countability lexicon buckets the source lemma as PLURAL_ONLY, the slot is
            # NN (pool_for_slot explicitly returns set() there). Fall back
            # to the broad noun pool so the NOUN-family routing holds and
            # the joint intersection has a fighting chance.
            broad = _broad_noun_pool_for_tag(aug, noun_ptb)
            if broad:
                noun_base = set(broad)
                used_broad_fallback = True
        if noun_base:
            return set(noun_base), "NOUN", noun_ptb, used_broad_fallback

    # No peers (or both strict AND broad noun pools empty) → default PROPN pool.
    gender = aug._gender_of_propn_text(tok.text)
    pool = _propn_pool_gated(aug, gender)
    if not pool:
        return None, None, None, False
    return set(pool), "PROPN", propn_tag, False


def _build_occ_chain(aug: Augmenter, tok: Token, ptb_tag: str,
                     base_pool: Set[str], feat: TokenFeatures,
                     pos_family: str,
                     broad_terminal_pool: Optional[frozenset] = None,
                     ) -> List[OccChainLevel]:
    """Build the per-occurrence ctx-backoff chain against ``base_pool``.

    Returns a list of :class:`OccChainLevel`, strictest first. Occurrences
    where ctx gating doesn't apply (PROPN path, to-inf verb spine, no ctx
    stats loaded) get a single-level chain containing the raw base pool.

    ``broad_terminal_pool`` (optional) is appended as a final chain level
    after the strict raw base pool, giving lock-step a last-resort escape
    hatch for cross-slot-incompatible pool partitions (the bare vs non-bare
    countability conflict in NOUNs). Gated through ``_candidate_allowed_for_tag``
    and with source lemma/surface dropped so NPI/vocab gates still apply.
    Omitted entirely for non-NOUN POS types.

    This replaces the per-occurrence call to
    ``aug._ctx_intersect_candidates`` — instead of the Augmenter picking one
    bucket level in isolation, we expose *all* levels so the joint
    intersection across co-referential occurrences can coarsen in lock-step.
    """
    base_list = list(base_pool)
    if not base_list:
        return []

    # Same short-circuits as _ctx_intersect_candidates (lines 1305-1319):
    if not getattr(aug, "_ctx_bucket2lemma", None):
        return [OccChainLevel(cand_set=frozenset(base_list), weights=None)]
    if pos_family == "PROPN" or ptb_tag in {"NNP", "NNPS"}:
        return [OccChainLevel(cand_set=frozenset(base_list), weights=None)]
    if feat.is_verb_like and getattr(aug, "toinf", None) is not None:
        try:
            if aug.toinf.required_verb_license(tok) is not None:
                return [OccChainLevel(cand_set=frozenset(base_list), weights=None)]
        except Exception:
            pass

    try:
        b = aug._ctx_bucket_for_token(tok, ptb_tag, feat=feat)
    except Exception:
        b = None
    if b is None:
        return [OccChainLevel(cand_set=frozenset(base_list), weights=None)]

    n = aug.cfg.ctx_backoff_min_lemma_count or 1
    use_weights = (getattr(aug, "_ctx_gate_mode", "off") == "freq")

    chain_keys = aug._ctx_backoff_chain(b)
    levels: List[OccChainLevel] = []
    for bk in chain_keys:
        lemma_ctr = aug._ctx_bucket2lemma.get(bk)
        if not lemma_ctr:
            continue
        filtered = {c for c in base_list if lemma_ctr.get(c, 0) >= n}
        if not filtered:
            continue
        weights = None
        if use_weights:
            weights = Counter({c: int(lemma_ctr.get(c, 0)) for c in filtered})
        levels.append(OccChainLevel(cand_set=frozenset(filtered), weights=weights))

    # Escape hatch 1: always make the raw un-ctx-filtered base pool
    # available as a chain level. Lock-step can reach it when every ctx
    # level's joint intersection is empty. For common lemmas this rescues
    # the "begin"/"live"/"develop" pattern — multiple verb occurrences whose
    # strict ctx buckets disagree but whose raw VerbNet pools intersect.
    base_fs = frozenset(base_list)
    if not levels or levels[-1].cand_set != base_fs:
        levels.append(OccChainLevel(cand_set=base_fs, weights=None))

    # Escape hatch 2: optional broader-pool terminal level. Only reached
    # when strict raw pools across occurrences fail to intersect (e.g. the
    # NOUN bare/non-bare countability split, or multi-frame VERB occurrences
    # whose frame-specific VerbNet pools don't overlap). Homogeneous-slot
    # examples never touch this level.
    if broad_terminal_pool is not None:
        broad_gated = frozenset(
            c for c in broad_terminal_pool
            if c not in base_fs  # avoid duplicate of previous level
            and aug._candidate_allowed_for_tag(c, ptb_tag)
        )
        # union so the broad terminal is a superset of strict terminal
        broad_full = base_fs | broad_gated
        # also drop source
        src_lem = (tok.lemma_ or tok.text or "").casefold()
        src_surf = tok.text.casefold()
        broad_full = broad_full - {src_lem, src_surf}
        if broad_full and broad_full != base_fs:
            levels.append(OccChainLevel(cand_set=broad_full, weights=None,
                                         is_broad_terminal=True))

    return levels


def _finalize_occurrence(aug: Augmenter, doc_tag: str, tok: Token,
                          feat: TokenFeatures, base_pool: Set[str],
                          pos_family: str, ptb_tag: str,
                          occs_by_lemma: Dict[str, List[Occurrence]],
                          skipped: List[Tuple[str, int, str]]) -> None:
    """Apply gates + build chain + append to occs_by_lemma. Shared between
    non-PROPN (Phase 1a) and PROPN (Phase 1b) sub-passes.
    """
    gated = {c for c in base_pool
             if aug._candidate_allowed_for_tag(c, ptb_tag)}
    src_lem_lc = (tok.lemma_ or tok.text or "").casefold()
    gated.discard(src_lem_lc)
    gated.discard(tok.text.casefold())
    if not gated:
        skipped.append((doc_tag, tok.i, f"{pos_family}:gate_exhausted"))
        return

    # POS-appropriate broader terminal level for lock-step's escape hatch.
    # Only reached when strict + raw per-occurrence pools fail to intersect.
    broad_terminal: Optional[frozenset] = None
    if pos_family == "NOUN":
        # NOUN: ignore bare/non-bare countability partition.
        broad_terminal = _broad_noun_pool_for_tag(aug, ptb_tag)
    elif pos_family == "VERB":
        # VERB: union of every VerbNet lemma regardless of frame.
        broad_terminal = _broad_verb_pool(aug)

    chain = _build_occ_chain(aug, tok, ptb_tag, gated, feat, pos_family,
                             broad_terminal_pool=broad_terminal)
    if not chain:
        skipped.append((doc_tag, tok.i, f"{pos_family}:chain_empty"))
        return

    occs_by_lemma[src_lem_lc].append(Occurrence(
        doc_tag=doc_tag, tok_i=tok.i, ptb_tag=ptb_tag,
        pos_family=pos_family, source_surface=tok.text,
        chain=chain,
    ))


def build_candidate_sets_from_docs(
        aug: Augmenter,
        doc_pairs: List[Tuple[str, Doc]],
        frozen_ranges_per_doc: Optional[Dict[str, List[Tuple[int, int]]]] = None,
        frozen_lemmas: Optional[Set[str]] = None,
        ) -> CandidateIndex:
    """Two-phase: per-occurrence chain, then joint lock-step intersection.

    ``doc_pairs`` is a list of ``(doc_tag, Doc)`` — one entry per spaCy doc
    that participates in the joint per-example mapping. The doc_tag is
    opaque (the SQuAD pipeline uses ``"ctx"``/``"q"``; the RAG pipeline
    uses ``"title_3"``/``"text_7"``/``"q"``/``"ans_1"``; a chat
    pipeline uses ``"msg_<i>"``).

    ``frozen_ranges_per_doc`` (chat-style template freeze): maps
    ``doc_tag`` to a list of ``(char_start, char_end)`` half-open ranges
    in the source text. Tokens whose char span overlaps any range are
    excluded entirely (treated as if they didn't appear).

    ``frozen_lemmas`` (case-folded set): tokens whose lemma is in the set
    are excluded regardless of position. Used by chat pipelines to propagate
    template-frozen lemmas across all messages in a row.

    Phase 1 has two sub-passes:

    * 1a (non-PROPN tokens): walk every replaceable VERB/NOUN/ADJ/ADV,
      build its ctx-backoff chain, and append to ``occs_by_lemma``. PROPN
      tokens are deferred.
    * 1b (PROPN tokens): now that ``occs_by_lemma`` knows the pos_family
      of every non-PROPN occurrence, each deferred PROPN is dispatched
      using the peer-family info — NOUN peers route the PROPN through the
      NOUN pool (if the lemma has a common-noun reading), ADJ-only peers
      route through the ADJ pool (if the lemma is adjectival), else the
      default PROPN pool. See ``_propn_token_candidates`` for the rules.

    Phase 2: for each case-folded source lemma, walk chain levels in
    lock-step — every occurrence starts at level 0, the joint intersection
    is taken, and if empty all occurrences advance one level together (an
    occurrence that has no more levels stays pinned to its last one). The
    first level at which the joint intersection is non-empty is accepted
    and the lemma is mapped. Mean weights are computed at that accepted
    level only (so per-occurrence counts are scale-consistent).
    """
    occs_by_lemma: Dict[str, List[Occurrence]] = defaultdict(list)
    skipped: List[Tuple[str, int, str]] = []
    # Diagnostic: PROPN lemmas whose peer-driven NOUN path had to fall back
    # to the broad noun pool because `_noun_candidates` returned empty.
    broad_noun_fallback_lemmas: List[str] = []
    # Deferred PROPN tokens — (doc_tag, tok, feat). Dispatched after
    # non-PROPN peer occurrences have been collected.
    deferred_propn: List[Tuple[str, Token, TokenFeatures]] = []

    # --- Phase 1a: non-PROPN token chains + peer collection ---
    for doc_tag, doc in doc_pairs:
        prot = set(aug._protect_tokens(doc))
        licensor_covered: Dict[int, str] = {}
        if aug.licensor_matcher is not None:
            try:
                licensor_covered = aug.licensor_matcher.cover_strengths(
                    doc, conservative_guard=aug.cfg.conservative_guard
                )
            except Exception:
                licensor_covered = {}
        reflexive_subject_ids = _compute_reflexive_subject_ids(doc)

        for tok in doc:
            if tok.i in prot:
                continue
            if tok.text in POSS_MARKERS and (tok.dep_ == "case" or tok.tag_ == "POS"):
                continue
            if aug._is_functionish(tok, licensor_covered):
                continue
            if _is_token_frozen(tok, doc_tag, frozen_ranges_per_doc,
                                 frozen_lemmas):
                skipped.append((doc_tag, tok.i, "frozen"))
                continue

            feat = aug._compute_token_features(tok)

            # Defer PROPN so we can see its non-PROPN peers first.
            if tok.pos_ == "PROPN":
                deferred_propn.append((doc_tag, tok, feat))
                continue

            base_pool, pos_family, ptb_tag = _per_token_candidates(
                aug, tok, feat, reflexive_subject_ids,
            )
            if base_pool is None or not base_pool:
                if pos_family is not None:
                    skipped.append((doc_tag, tok.i, f"{pos_family}:empty_pool"))
                continue
            _finalize_occurrence(aug, doc_tag, tok, feat, base_pool,
                                  pos_family, ptb_tag, occs_by_lemma, skipped)

    # --- Phase 1b: PROPN dispatch driven by peer pos_families ---
    for doc_tag, tok, feat in deferred_propn:
        lem_lc = (tok.lemma_ or tok.text or "").casefold()
        # Only non-PROPN occurrences count as "peers" for dispatch. Earlier
        # PROPN occurrences of the same lemma added in this Phase 1b loop
        # must not retroactively convince later PROPN occurrences to take
        # a different path — that would produce different pool classes
        # across same-lemma PROPN tokens and break the intersection.
        peer_families = {occ.pos_family for occ in occs_by_lemma.get(lem_lc, [])
                         if occ.pos_family != "PROPN"}
        base_pool, pos_family, ptb_tag, used_broad_noun = _propn_token_candidates(
            aug, tok, feat, peer_families,
        )
        if used_broad_noun:
            broad_noun_fallback_lemmas.append(lem_lc)
            print(f"[sambal.core] broad_noun_fallback: lemma={lem_lc!r} "
                  f"tok=[{doc_tag}][{tok.i}] ({tok.text!r}) — "
                  f"strict _noun_candidates empty, used broad noun pool",
                  flush=True)
        if base_pool is None or not base_pool:
            if pos_family is not None:
                skipped.append((doc_tag, tok.i, f"{pos_family}:empty_pool"))
            continue
        _finalize_occurrence(aug, doc_tag, tok, feat, base_pool,
                              pos_family, ptb_tag, occs_by_lemma, skipped)

    # --- Phase 2: lock-step joint backoff + intersection ---
    lemmas: Dict[str, LemmaEntry] = {}
    broad_terminal_accept_lemmas: List[str] = []
    for lem_lc, occs in occs_by_lemma.items():
        pos_families = {o.pos_family for o in occs}
        pos_conflict = len(pos_families) > 1

        max_depth = max(len(o.chain) for o in occs)
        inter: Set[str] = set()
        mean_w: Optional[Counter] = None
        accepted_level = -1
        accepted_on_broad_terminal = False

        for k in range(max_depth):
            # Each occurrence uses min(k, len(chain)-1): once it exhausts its
            # own chain it stays pinned at the coarsest level while others
            # continue to coarsen.
            level_sets: List[FrozenSet[str]] = []
            level_weights: List[Counter] = []
            all_have_weights = True
            any_broad_terminal = False
            for o in occs:
                idx = min(k, len(o.chain) - 1)
                lvl = o.chain[idx]
                level_sets.append(lvl.cand_set)
                if lvl.is_broad_terminal:
                    any_broad_terminal = True
                if lvl.weights is None:
                    all_have_weights = False
                else:
                    level_weights.append(lvl.weights)

            candidate_inter: Set[str] = set(level_sets[0])
            for s in level_sets[1:]:
                candidate_inter &= s
            if not candidate_inter:
                continue

            inter = candidate_inter
            accepted_level = k
            accepted_on_broad_terminal = any_broad_terminal
            if all_have_weights and level_weights:
                k_occs = len(occs)
                mean_w = Counter()
                for c in inter:
                    total = sum(int(w.get(c, 0)) for w in level_weights)
                    mean_w[c] = total / k_occs
            break

        if accepted_on_broad_terminal:
            broad_terminal_accept_lemmas.append(lem_lc)
            # Tag the POS family(ies) of the relaxation for the log.
            fams_csv = "+".join(sorted(pos_families))
            print(f"[sambal.core] broad_terminal_accept: lemma={lem_lc!r} "
                  f"pos_families={fams_csv} level={accepted_level} "
                  f"n_occs={len(occs)} — lock-step rescued via broad terminal",
                  flush=True)

        mean_cand_before = sum(len(o.chain[0].cand_set) for o in occs) / len(occs)
        lemmas[lem_lc] = LemmaEntry(
            cand_set=frozenset(inter),
            weights=mean_w,
            occs=occs,
            pos_families=pos_families,
            pos_conflict=pos_conflict,
            mean_cand_size_before=mean_cand_before,
            size_after=len(inter),
            accepted_level=accepted_level,
        )

    return CandidateIndex(
        lemmas=lemmas,
        skipped=skipped,
        broad_noun_fallback_lemmas=broad_noun_fallback_lemmas,
        broad_terminal_accept_lemmas=broad_terminal_accept_lemmas,
    )


# ==========================================================================
# Sampling
# ==========================================================================

def seed_for_example(global_seed: int, example_id: str) -> int:
    """Per-example deterministic seed from a global seed + opaque ID.

    Same construction as the previous SQuAD-only ``seed_for_example`` —
    keeping the parameter name historical for API compatibility (the
    second arg is conceptually any per-row identifier, e.g. ``qa_id``
    in SQuAD or ``qid`` in RAG).
    """
    import hashlib
    digest = hashlib.sha256(f"{global_seed}:{example_id}".encode("utf-8")).digest()[:8]
    return int.from_bytes(digest, "big")


def _weighted_sample_one(cands: List[str], weights: Counter, rng: random.Random) -> str:
    """Gumbel-top-k with mean weights — same convention as
    ``_sample_from_candidates_inner`` at lines 4279-4294 of the Augmenter.
    """
    def score(c: str) -> float:
        w = float(weights.get(c, 0) or 0)
        u = rng.random()
        if u <= 0.0:
            u = sys.float_info.min
        return math.log(u) / max(w, 1.0)
    return max(cands, key=score)


def sample_mapping(cand_index: CandidateIndex, rng: random.Random,
                   fallback_mode: str) -> SampledMapping:
    fallback_mode = (fallback_mode or "drop").strip().lower()
    if fallback_mode not in {"drop", "coarsen", "passthrough"}:
        raise ValueError(f"unknown fallback_mode={fallback_mode!r}")

    mapping: Dict[str, str] = {}
    # Observational: every lemma that appeared with multiple pos_families,
    # regardless of whether we ended up finding a non-empty intersection.
    pos_conflict_all: List[str] = []
    # Drop-causing: lemmas whose final intersection was empty.
    ctx_empty: List[str] = []
    # Among pos_conflict_all, which ones were "rescued" by a dual-use lemma
    # being present in all per-occurrence pools (non-empty intersection).
    pos_conflict_resolved: List[str] = []
    passthrough: List[str] = []
    dropped: Optional[str] = None

    for lem, entry in cand_index.lemmas.items():
        if entry.pos_conflict:
            pos_conflict_all.append(lem)

        if not entry.cand_set:
            ctx_empty.append(lem)
            if fallback_mode == "drop":
                if dropped is None:
                    dropped = f"ctx_intersection_empty:{lem}"
                continue
            # passthrough / coarsen fallback: leave lemma unmapped, source
            # surface emitted verbatim everywhere.
            passthrough.append(lem)
            continue

        if entry.pos_conflict:
            pos_conflict_resolved.append(lem)

        cands = sorted(entry.cand_set)
        if entry.weights is not None and len(cands) > 1:
            w_counter = Counter({c: entry.weights.get(c, 0.0) for c in cands})
            target = _weighted_sample_one(cands, w_counter, rng)
        else:
            target = rng.choice(cands)
        mapping[lem] = target

    return SampledMapping(
        mapping=mapping,
        pos_conflict_lemmas=pos_conflict_all,
        ctx_intersection_empty_lemmas=ctx_empty,
        pos_conflict_resolved_lemmas=pos_conflict_resolved,
        passthrough_lemmas=passthrough,
        dropped_reason=dropped,
    )


# ==========================================================================
# Rendering
# ==========================================================================

def _cap_first_alpha(s: str) -> str:
    for i, ch in enumerate(s):
        if ch.isalpha():
            return s[:i] + ch.upper() + s[i + 1:]
    return s


_ARTICLE_PATTERN = re.compile(
    r'\b([Aa]n?)(\s+)(["“‘\(\[\{]*)([A-Za-z0-9][A-Za-z0-9.\-]*)'
)


def _fix_articles_with_spans(aug: Augmenter, text: str,
                              per_token_spans: List[Tuple[int, int]]
                              ) -> Tuple[str, List[Tuple[int, int]]]:
    """Span-aware variant of ``Augmenter._fix_indefinite_articles``.

    The base ``_fix_indefinite_articles`` does a regex substitution that
    can change string length (``a`` → ``an`` adds one character), which
    would invalidate any per-token character spans computed from the
    pre-fix rendering. This variant walks the matches explicitly, builds
    the fixed text, AND produces an updated per-token spans list so
    downstream code (answer-span recomputation) stays correct.

    Concretely: for every a↔an flip we record a shift event at the end of
    the original article (``m.start() + len(orig_article)``). Every
    per-token bound at or after that position gets shifted by the cumulative
    delta.
    """
    # First pass: collect (match, suggested) pairs, skipping no-ops.
    events: List[Tuple[int, str, str]] = []  # (article_end_pos, orig_article, suggested)
    for m in _ARTICLE_PATTERN.finditer(text):
        orig_article = m.group(1)
        head = m.group(4)
        try:
            suggested = aug.inflect_engine.a(head).split()[0]
        except Exception:
            continue
        if orig_article[0].isupper():
            suggested = suggested.capitalize()
        if suggested == orig_article:
            continue
        events.append((m.start() + len(orig_article), orig_article, suggested))

    if not events:
        return text, per_token_spans

    # Build the fixed text by splicing in replacements at the article positions.
    out_parts: List[str] = []
    last = 0
    shift_events: List[Tuple[int, int]] = []  # (orig_position, delta)
    for ev_pos, orig_article, suggested in events:
        orig_start = ev_pos - len(orig_article)
        out_parts.append(text[last:orig_start])
        out_parts.append(suggested)
        last = ev_pos
        delta = len(suggested) - len(orig_article)
        if delta != 0:
            shift_events.append((ev_pos, delta))
    out_parts.append(text[last:])
    new_text = "".join(out_parts)

    if not shift_events:
        return new_text, per_token_spans

    def _cum_delta(pos: int) -> int:
        d = 0
        for ev_pos, delta in shift_events:
            if pos >= ev_pos:
                d += delta
        return d

    new_spans = [(s + _cum_delta(s), e + _cum_delta(e))
                  for s, e in per_token_spans]
    return new_text, new_spans


def _render_doc(aug: Augmenter, doc: Doc,
                occ_lookup: Dict[Tuple[str, int], Occurrence],
                doc_tag: str, mapping: Dict[str, str],
                frozen_ranges: Optional[List[Tuple[int, int]]] = None,
                frozen_lemmas: Optional[Set[str]] = None,
                ) -> Tuple[str, List[Tuple[int, int]], List[str]]:
    """Render one doc, returning (final_text, per_token_spans_in_rendered_text,
    per_token_rendered_surface_incl_whitespace).

    Sentence-start capitalization mirrors augment_doc:3968-3972.

    ``frozen_ranges`` (char-spans in the source text) and
    ``frozen_lemmas`` (case-folded lemma set) let callers exempt template
    spans from substitution. A token whose char span overlaps any frozen
    range OR whose lemma is in the frozen set is emitted verbatim — neither
    replaced nor case-flipped on sentence-start. Default ``None`` leaves
    substitution unrestricted.
    """
    out_parts: List[str] = []
    rendered_parts: List[str] = []
    per_token_spans: List[Tuple[int, int]] = []
    cursor = 0
    need_cap = True

    frozen_ranges_per_doc = (
        {doc_tag: frozen_ranges} if frozen_ranges else None
    )

    for tok in doc:
        key = (doc_tag, tok.i)
        replaced = False
        rendered = tok.text
        is_frozen = _is_token_frozen(tok, doc_tag,
                                       frozen_ranges_per_doc,
                                       frozen_lemmas)
        if (not is_frozen) and key in occ_lookup:
            lem_lc = (tok.lemma_ or tok.text or "").casefold()
            tgt = mapping.get(lem_lc)
            if tgt is not None:
                occ = occ_lookup[key]
                ptb = occ.ptb_tag
                if occ.pos_family == "PROPN":
                    new_surf = tgt  # PROPN pool entries are surface forms
                    # NNPS pluralization if the source was tagged NNPS
                    if ptb == "NNPS":
                        try:
                            plural = aug.inflect_engine.plural_noun(new_surf)
                            if plural:
                                new_surf = plural
                        except Exception:
                            pass
                else:
                    new_surf = aug._realize(tgt, ptb)
                rendered = match_case_like(tok.text, new_surf)
                replaced = True

        if tok.is_sent_start:
            need_cap = True
        to_add = rendered
        if need_cap and tok.is_alpha and replaced:
            to_add = _cap_first_alpha(to_add)
        if tok.is_alpha:
            need_cap = False

        start = cursor
        end = start + len(to_add)
        per_token_spans.append((start, end))
        piece = to_add + tok.whitespace_
        out_parts.append(piece)
        rendered_parts.append(piece)
        cursor = end + len(tok.whitespace_)

    raw = "".join(out_parts)
    # IMPORTANT: apply article fixing with span tracking. The base
    # _fix_indefinite_articles regex edit can change string length
    # (a → an adds a character); without shifting the per-token spans the
    # answer-slice recomputation in render_example produces off-by-k
    # truncations.
    fixed, per_token_spans = _fix_articles_with_spans(aug, raw, per_token_spans)
    return fixed, per_token_spans, rendered_parts


# ==========================================================================
# Parts writer (bespoke — simpler than ChunkedOutputWriter for this use case)
# ==========================================================================

class PartsWriter:
    def __init__(self, out_path: str):
        self.out_path = out_path
        self.parts_dir = out_path + ".parts"
        Path(self.parts_dir).mkdir(parents=True, exist_ok=True)

    def reset(self) -> None:
        if os.path.isdir(self.parts_dir):
            shutil.rmtree(self.parts_dir)
        Path(self.parts_dir).mkdir(parents=True, exist_ok=True)
        if os.path.exists(self.out_path):
            os.remove(self.out_path)

    def part_path(self, idx: int) -> str:
        return os.path.join(self.parts_dir, f"part_{idx:06d}.jsonl")

    def write_chunk(self, rows: List[Dict[str, Any]], idx: int) -> None:
        if not rows:
            return
        p = self.part_path(idx)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, p)

    def prune_from(self, idx: int) -> None:
        for name in os.listdir(self.parts_dir):
            if not name.endswith(".jsonl"):
                continue
            try:
                n = int(Path(name).stem.split("_", 1)[1])
            except Exception:
                continue
            if n >= idx:
                os.remove(os.path.join(self.parts_dir, name))

    def finalize(self) -> None:
        parts = sorted(
            os.path.join(self.parts_dir, n)
            for n in os.listdir(self.parts_dir)
            if n.endswith(".jsonl") and n.startswith("part_")
        )
        tmp = self.out_path + ".tmp"
        Path(os.path.dirname(tmp) or ".").mkdir(parents=True, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fout:
            for p in parts:
                with open(p, "r", encoding="utf-8") as fin:
                    shutil.copyfileobj(fin, fout)
        os.replace(tmp, self.out_path)


# ==========================================================================
# Augmenter cache management
# ==========================================================================

# All the "caches" on the Augmenter we MIGHT clear between examples. These
# are all lemma- or bucket-keyed, not example-scoped, so reusing them is
# correct AND much faster — first-example cold runs were ~18s vs ~1.7s
# warm, a ~10× penalty if we cleared every time. The reason the original
# plan called for clearing was to bound memory in long-running pipelines
# (per AGENT_HANDOFF.md's "unbounded cache growth" gotcha); for that we
# clear only periodically.
_AUG_LEMMA_CACHES = (
    "_verb_candidates_cache",
    "_noun_candidates_cache",
    "_inflect_cache",
    "_lemma_tag_allowed",
    "_ctx_bucket_level_stats_cache",
    "_adj_candidates_cache",
    "_pos_cache",
)

# Clear the lemma-keyed caches every N examples to bound RSS. None disables
# periodic clearing entirely.
CACHE_CLEAR_EVERY_N = 500


def _clear_aug_caches(aug: Augmenter) -> None:
    for name in _AUG_LEMMA_CACHES:
        obj = getattr(aug, name, None)
        if isinstance(obj, dict):
            obj.clear()


# ==========================================================================
# Augmenter runtime construction
# ==========================================================================

def build_runtime(overrides: Dict[str, Any]) -> Tuple[Any, ResourcePaths, Config]:
    """Instantiate the Augmenter's runtime config triple from a JSON-shaped
    overrides dict. Identical to the SQuAD pipeline's local helper; lifted
    here so the RAG pipeline can reuse it verbatim.
    """
    args = type("", (), {})()
    args.resources = overrides.get("args", {}).get(
        "resources", str(Path(__file__).resolve().parent / "resources")
    )
    args.spacy_model = "en_core_web_trf"
    args.require_gpu = False
    args.prefer_gpu = True
    args.batch_size = 32
    for k, v in overrides.get("args", {}).items():
        setattr(args, k, v)

    res = Path(args.resources)
    paths_defaults = dict(
        verbnet_source="nltk",
        noun_bucket_dir=str(res),
        function_words_path=str(res / "function_words_ud_ewt.txt"),
        npi_path=str(res / "english_npis_wiktionary.tsv"),
        licensors_path=None,
        noun_lemmas_path=str(res / "wordnet_nouns.jsonl"),
        adj_lemmas_path=str(res / "wordnet_adjectives.jsonl"),
        adv_lemmas_path=str(res / "wordnet_adverbs.jsonl"),
        propn_list_path=str(res / "allowed_proper_nouns.txt"),
        given_names_path=str(res / "given_names_ssa.txt"),
        fixed_mwes_path=str(res / "fixed_mwes_all.txt"),
        mwe_patterns_jsonl_path=str(res / "streusle_vmwe_patterns.jsonl"),
        human_nouns_path=str(res / "human_nouns.jsonl"),
        allowed_vocab_path=str(res / "allowed_vocab.txt"),
        licensor_patterns_jsonl_path=str(res / "licensor_patterns_full.jsonl"),
        toinf_verb_lexicon_jsonl=str(res / "toinf_verbs.jsonl"),
        toinf_adj_lexicon_jsonl=str(res / "toinf_adjs.jsonl"),
        given_names_male_path=str(res / "male_given_names.txt"),
        given_names_female_path=str(res / "female_given_names.txt"),
        given_names_neutral_path=str(res / "neutral_given_names.txt"),
        gendered_words_json_path=str(res / "gendered_words_filtered.json"),
        ctx_lemma_stats_path=str(res / "lemma_stats.pkl"),
    )
    paths_defaults.update(overrides.get("paths", {}))
    resources = ResourcePaths(**paths_defaults)

    cfg_defaults = dict(
        spacy_model=args.spacy_model,
        seed=42,
        replace_propn=True,
        replace_pronouns=False,
        roundtrip_debug=False,
        debug=False,
        debug_vn=False,
        roundtrip_max_tries=4,
        roundtrip_ud=False,
        freeze_aux_hosts=False,
        min_vocab_freq=1,
        respect_human=True,
        respect_gender=True,
        verb_metrics=False,
        prefilter_pools_by_allowed_vocab=True,
        filter_substitution_names_by_allowed_vocab=True,
        ctx_lemma_gate="freq",
        ctx_lemma_gate_min_count=2,
        vn_soft_prep_min_pool=10,
        ctx_backoff_mode="B",
        ctx_backoff_min_lemma_count=2,
        ctx_backoff_min_candidates=2,
        ctx_backoff_min_candidates_pct=0.05,
        skip_toinf_freeze=False,
        require_gpu=args.require_gpu,
    )
    cfg_defaults.update(overrides.get("config", {}))
    cfg = Config(**cfg_defaults)
    return args, resources, cfg


def parse_shard_env() -> Tuple[int, int]:
    """Read SHARD_ID / NUM_SHARDS from the environment.

    Shared here so worker scripts can reuse it.
    """
    sid = int(os.environ.get("SHARD_ID", 0))
    n = int(os.environ.get("NUM_SHARDS", 1))
    if n <= 0 or sid < 0 or sid >= n:
        raise ValueError(f"Invalid SHARD_ID={sid} for NUM_SHARDS={n}")
    return sid, n
