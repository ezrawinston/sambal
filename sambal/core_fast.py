"""Optimized document-scope relexicalization core (parallel to ``core``).

Same input/output contract as ``core.build_candidate_sets_from_docs``
+ ``sample_mapping`` + ``_render_doc``. Bit-identical results on identical
input + same seed (verified by the parity test in
``tests/test_core_fast_parity.py``).

Speedups:

  A1: lazy broad-terminal pool. The 30-50K-element broad NOUN/VERB pool is
      only filtered+materialized when Phase 2 lock-step actually probes
      that level (consumed ~0.5-1.4% of mapped lemmas per empirical audit).
  A1 P2: cross-row chain memo. The chain levels returned by
      `_build_occ_chain_fast` after the early short-circuits are cached on
      `aug._chain_memo_cache` keyed on (ptb_tag, pos_family, is_verb_like,
      ctx_bucket, src_lem, src_surf, frozenset(base_pool)). All chain-build
      dependencies are in the key — bit-equal output. Measured 1.35× wall
      speedup with 77.4% hit rate on a 25-document sample.
  A2: lazy base-pool allow-gate. Replaces eager
      ``{c for c in base_pool if aug._candidate_allowed_for_tag(c, ptb_tag)}``
      Python loop with C-speed ``base_pool & aug._allowed_by_tag(ptb_tag)``.
  A3: singleton-lemma fast path. In Phase 2, when ``len(occs) == 1``, skip
      the lock-step chain walk and accept ``chain[0].cand_set`` directly.
  A4: lazy per-level cand_set on the broad-terminal level only. The chain
      length stays known up-front (Phase 2's ``max_depth = max(len(o.chain))``
      and ``min(k, len(o.chain)-1)`` indexing still work). Only the
      broad-terminal level defers its cand_set materialization via a thunk.
  B1: ``aug._allowed_by_tag(ptb_tag)`` precomputed frozenset (built by
      ``install_fast_caches`` in ``fast_caches``).
  B2: ``aug._bucket_lemmas_at(bucket_key, n)`` precomputed frozenset of
      lemmas at that ctx bucket with count >= n. Chain-level filter becomes
      ``base_set & aug._bucket_lemmas_at(bk, n)`` C-speed intersection.
  B3: realized-surface memoization. ``sample_mapping_fast`` returns a
      ``FastSampledMapping`` with a ``realized[(lemma_lc, ptb_tag)] = surf``
      cache filled lazily on first lookup; ``_render_doc_fast`` reads from
      this cache instead of re-invoking ``aug._realize`` per occurrence.
  B4: gated article fix. ``_render_doc_fast`` skips ``_fix_articles_with_spans``
      regex scan unless the rendered text contains a replaced token adjacent
      to an indefinite article.

Requires ``install_fast_caches(aug)`` before any call.
"""
from __future__ import annotations

import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple, Union

from spacy.tokens import Doc, Token

from sambal.engine import Augmenter
from sambal.token_features import TokenFeatures
from sambal.token_utils import (
    is_verb_like, is_noun_like, is_particle_dep, match_case_like,
)
from sambal.core import (
    POSS_MARKERS, CONTRACTION_MARKERS, SUBJ_DEPS,
    OccChainLevel, Occurrence, LemmaEntry, CandidateIndex, SampledMapping,
    _compute_reflexive_subject_ids,
    _broad_noun_pools, _broad_noun_pool_for_tag, _broad_verb_pool,
    _propn_pool_gated,
    _per_token_candidates, _propn_token_candidates,
    _token_overlaps_any_range, _is_token_frozen,
    seed_for_example, _weighted_sample_one,
    _cap_first_alpha, _ARTICLE_PATTERN, _fix_articles_with_spans,
    PartsWriter,
)

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


# ==========================================================================
# Lazy broad-terminal chain level (A1 + A4)
# ==========================================================================


class _LazyBroadTerminalLevel:
    """Phase-2-compatible chain level whose ``cand_set`` is materialized on
    first access via a stored thunk. Duck-typed to match
    ``OccChainLevel`` (exposes ``cand_set``, ``weights``, ``is_broad_terminal``).

    Only used as the FINAL chain level on NOUN/VERB occurrences; other
    levels (strict + ctx-backoff) are eager ``OccChainLevel`` instances.
    """
    __slots__ = ("_build_fn", "_cand_set", "weights", "is_broad_terminal")

    def __init__(self, build_fn: Callable[[], FrozenSet[str]]) -> None:
        self._build_fn = build_fn
        self._cand_set: Optional[FrozenSet[str]] = None
        self.weights: Optional[Counter] = None
        self.is_broad_terminal: bool = True

    @property
    def cand_set(self) -> FrozenSet[str]:
        cs = self._cand_set
        if cs is None:
            cs = self._build_fn()
            self._cand_set = cs
            # Release the closure to free any captured references.
            self._build_fn = None  # type: ignore[assignment]
        return cs


# ==========================================================================
# Fast SampledMapping (B3)
# ==========================================================================


@dataclass
class FastSampledMapping:
    """Same shape as ``SampledMapping`` plus a realized-surface cache filled
    lazily by ``_render_doc_fast`` (or by ``sample_mapping_fast`` upfront).

    Use ``_render_doc_fast`` together with this dataclass; for legacy
    ``_render_doc`` compatibility, call ``.to_legacy()`` to get a
    ``SampledMapping``.
    """
    mapping: Dict[str, str]
    pos_conflict_lemmas: List[str]
    ctx_intersection_empty_lemmas: List[str]
    pos_conflict_resolved_lemmas: List[str]
    passthrough_lemmas: List[str]
    dropped_reason: Optional[str]
    # NEW: cached realized surfaces keyed by (target_lemma_lc, ptb_tag, pos_family).
    # Filled by ``_render_doc_fast`` on first lookup per (lemma, tag, family).
    realized: Dict[Tuple[str, str, str], str] = field(default_factory=dict)

    def to_legacy(self) -> SampledMapping:
        return SampledMapping(
            mapping=dict(self.mapping),
            pos_conflict_lemmas=list(self.pos_conflict_lemmas),
            ctx_intersection_empty_lemmas=list(self.ctx_intersection_empty_lemmas),
            pos_conflict_resolved_lemmas=list(self.pos_conflict_resolved_lemmas),
            passthrough_lemmas=list(self.passthrough_lemmas),
            dropped_reason=self.dropped_reason,
        )


# ==========================================================================
# Candidate-set construction (fast)
# ==========================================================================


def _finalize_occurrence_fast(
        aug: Augmenter, doc_tag: str, tok: Token,
        feat: TokenFeatures, base_pool: Set[str],
        pos_family: str, ptb_tag: str,
        occs_by_lemma: Dict[str, List[Occurrence]],
        skipped: List[Tuple[str, int, str]]) -> None:
    """Same contract as ``core._finalize_occurrence`` but uses the
    precomputed ``_allowed_by_tag`` frozenset for the allow-gate (A2 + B1).
    Requires ``install_fast_caches(aug)`` to have run.
    """
    allowed = aug._allowed_by_tag(ptb_tag)  # type: ignore[attr-defined]
    src_lem_lc = (tok.lemma_ or tok.text or "").casefold()
    # Single-expression form: avoid `set(base_pool)` copy + two `.discard()`
    # calls. `base_pool & allowed` allocates the gated set directly; the
    # final `- {…}` strips source surface/lemma in one C-level op.
    gated = (base_pool & allowed) - {src_lem_lc, tok.text.casefold()}
    if not gated:
        skipped.append((doc_tag, tok.i, f"{pos_family}:gate_exhausted"))
        return

    chain = _build_occ_chain_fast(aug, tok, ptb_tag, gated, feat, pos_family)
    if not chain:
        skipped.append((doc_tag, tok.i, f"{pos_family}:chain_empty"))
        return

    # `chain` is `List[Union[OccChainLevel, _LazyBroadTerminalLevel]]`; the
    # Occurrence.chain field in core is typed `List[OccChainLevel]`.
    # _LazyBroadTerminalLevel is duck-typed (cand_set/weights/is_broad_terminal)
    # so all downstream Phase 2 consumers handle it. Cast for mypyc compile.
    occs_by_lemma[src_lem_lc].append(Occurrence(
        doc_tag=doc_tag, tok_i=tok.i, ptb_tag=ptb_tag,
        pos_family=pos_family, source_surface=tok.text,
        chain=chain,  # type: ignore[arg-type]
    ))


def _build_occ_chain_fast(
        aug: Augmenter, tok: Token, ptb_tag: str,
        base_pool: Set[str], feat: TokenFeatures, pos_family: str,
        ) -> List[Union[OccChainLevel, _LazyBroadTerminalLevel]]:
    """Same return as ``core._build_occ_chain`` but:
      - B2: chain-level ctx filter uses precomputed ``_bucket_lemmas_at``
        frozensets (C-speed set intersection) instead of a Python comprehension.
      - A1 + A4: broad-terminal level (NOUN/VERB only) is appended as a
        ``_LazyBroadTerminalLevel`` whose cand_set materializes only when
        Phase 2 lock-step probes it.
    """
    base_list = list(base_pool)
    if not base_list:
        return []

    # Same short-circuits as legacy.
    # Pull Optional aug attrs into locals so mypyc can narrow the type.
    ctx_bucket2lemma = aug._ctx_bucket2lemma
    if not ctx_bucket2lemma:
        return [OccChainLevel(cand_set=frozenset(base_list), weights=None)]
    if pos_family == "PROPN" or ptb_tag in {"NNP", "NNPS"}:
        return [OccChainLevel(cand_set=frozenset(base_list), weights=None)]
    toinf = aug.toinf
    if feat.is_verb_like and toinf is not None:
        try:
            if toinf.required_verb_license(tok) is not None:
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
    base_set = frozenset(base_list)

    # A1 P2: cross-row chain memo. Check before doing any chain-build work.
    # Cache key captures every dependency of the chain build below:
    #   - ptb_tag, pos_family — affect short-circuits + which broad pool
    #   - feat.is_verb_like  — caller verified license is None here, but
    #                          is_verb_like still affects broad-terminal append
    #   - b (ctx_bucket)     — drives chain_keys + per-level filtering
    #   - src_lem, src_surf  — broad-terminal level excludes these strings
    #   - base_set           — chain-level intersection AND broad-terminal
    #                          append guard (base_set vs broad_full)
    # has_verb_license is *always* None at this point (the True branch
    # returned above), so it is omitted from the key.
    memo_get = getattr(aug, "_chain_memo_get", None)
    if memo_get is not None:
        src_lem_lc = (tok.lemma_ or tok.text or "").casefold()
        src_surf_lc = tok.text.casefold()
        memo_key = (
            ptb_tag, pos_family, bool(feat.is_verb_like),
            b, src_lem_lc, src_surf_lc, base_set,
        )
        cached = memo_get(memo_key)
        if cached is not None:
            return cached
    else:
        memo_key = None

    chain_keys = aug._ctx_backoff_chain(b)
    levels: List[Union[OccChainLevel, _LazyBroadTerminalLevel]] = []
    have_bucket_cache = (
        getattr(aug, "_bucket_lemmas_at_cache", None) is not None
    )
    for bk in chain_keys:
        # Use the local `ctx_bucket2lemma` (narrowed to non-None above) so
        # mypyc accepts the .get() call.
        lemma_ctr = ctx_bucket2lemma.get(bk)
        if not lemma_ctr:
            continue
        if have_bucket_cache:
            # B2: C-speed intersection.
            bucket_set = aug._bucket_lemmas_at(bk, n)  # type: ignore[attr-defined]
            filtered = base_set & bucket_set
        else:
            # Legacy fallback.
            filtered = frozenset(c for c in base_list if lemma_ctr.get(c, 0) >= n)
        if not filtered:
            continue
        # Skip per-level Counter construction. ``lemma_ctr`` is already a
        # Counter; downstream consumers (Phase 2 mean, sampling) call
        # ``w.get(c, 0)`` only on keys in ``inter ⊆ filtered ⊆ bucket_lemmas``,
        # so the raw bucket Counter is semantically equivalent and avoids
        # building a per-occurrence-per-level dict (~6264 occs × ~3 levels ×
        # ~150 cands of dict.get on the profiled 25-doc sample).
        weights = lemma_ctr if use_weights else None
        levels.append(OccChainLevel(cand_set=filtered, weights=weights))

    # Escape hatch 1: raw un-ctx-filtered base pool as a chain level.
    if not levels or levels[-1].cand_set != base_set:
        levels.append(OccChainLevel(cand_set=base_set, weights=None))

    # Escape hatch 2: broad terminal level (NOUN/VERB only).
    #
    # Profiling showed the eager compute
    #   broad_extra = broad_allowed - base_set
    #   broad_full  = (base_set | broad_extra) - {src_lem, src_surf}
    # was ~30s of `_build_occ_chain_fast` self-time on a 25-document sample:
    # ~150K hash ops × ~4000 NOUN/VERB occurrences. Phase 2 reaches this
    # level for ~1% of lemmas (rest accepted earlier).
    #
    # Lazy form is safe IFF we keep legacy's append guard's semantics:
    # only append a non-empty-and-not-equal-to-base level. We do a
    # short-circuit existence scan to determine that *before* appending,
    # without materializing broad_full. The scan iterates broad_allowed
    # and returns True on the first element that is in (broad_allowed -
    # base_set - {src_lem, src_surf}). Worst case is O(|broad_allowed|)
    # only when broad_allowed ⊆ base_set ∪ {src_lem, src_surf} — vanishingly
    # rare since broad pools (~30-100K) typically dwarf base pools (~1-5K).
    if pos_family in ("NOUN", "VERB"):
        broad_allowed = aug._allowed_broad_by_tag(ptb_tag)  # type: ignore[attr-defined]
        if broad_allowed:
            src_lem = (tok.lemma_ or tok.text or "").casefold()
            src_surf = tok.text.casefold()
            has_ext = False
            for c in broad_allowed:
                if (c not in base_set
                        and c != src_lem and c != src_surf):
                    has_ext = True
                    break
            if has_ext:
                # Defer materialization. The level's cand_set is computed
                # only when Phase 2 lock-step accesses it (typically <2% of
                # NOUN/VERB occurrences per audit). Append guard already
                # satisfied (has_ext means broad_full ⊋ base_set).
                _build_fn = _make_broad_full_thunk(
                    broad_allowed, base_set, src_lem, src_surf,
                )
                levels.append(_LazyBroadTerminalLevel(_build_fn))

    # A1 P2: insert into chain memo. The cached list is shared across
    # callers; the _LazyBroadTerminalLevel (if present) caches its own
    # materialization internally so concurrent cache readers don't repeat
    # the thunk work.
    if memo_key is not None:
        # `_chain_memo_set` is dynamically bound by install_fast_caches
        # (only when enable_chain_memo=True). mypyc can't see the dynamic
        # bind so silence the attr-defined check.
        aug._chain_memo_set(memo_key, levels)  # type: ignore[attr-defined]

    return levels


def _make_broad_full_thunk(broad_allowed, base_set, src_lem, src_surf):
    """Returns a closure that materializes broad_full =
    (base_set | broad_allowed) - {src_lem, src_surf} on first call.
    Pre-bound args avoid attribute lookups at call time.
    """
    def _build():
        # Equivalent to legacy form `(base_set | (broad_allowed - base_set))
        # - {src_lem, src_surf}` but skips the intermediate broad_extra
        # frozenset since `|` already deduplicates.
        return (base_set | broad_allowed) - {src_lem, src_surf}
    return _build


def build_candidate_sets_from_docs_fast(
        aug: Augmenter,
        doc_pairs: List[Tuple[str, Doc]],
        frozen_ranges_per_doc: Optional[Dict[str, List[Tuple[int, int]]]] = None,
        frozen_lemmas: Optional[Set[str]] = None,
        ) -> CandidateIndex:
    """Drop-in fast replacement for
    ``core.build_candidate_sets_from_docs``.

    A3: in Phase 2, singleton-occurrence lemmas (``len(occs) == 1``) skip
    the lock-step walk and accept ``chain[0].cand_set`` directly. Most
    lemmas in a doc appear exactly once, so this is the common case.
    """
    occs_by_lemma: Dict[str, List[Occurrence]] = defaultdict(list)
    skipped: List[Tuple[str, int, str]] = []
    broad_noun_fallback_lemmas: List[str] = []
    deferred_propn: List[Tuple[str, Token, TokenFeatures]] = []

    # --- Phase 1a: non-PROPN ---
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
            # `_per_token_candidates` returns Optional[str] for pos_family/
            # ptb_tag, but when base_pool is non-empty they are guaranteed
            # non-None by construction (core invariant). Assert for
            # mypyc compile.
            assert pos_family is not None and ptb_tag is not None
            _finalize_occurrence_fast(
                aug, doc_tag, tok, feat, base_pool,
                pos_family, ptb_tag, occs_by_lemma, skipped,
            )

    # --- Phase 1b: PROPN ---
    for doc_tag, tok, feat in deferred_propn:
        lem_lc = (tok.lemma_ or tok.text or "").casefold()
        peer_families = {occ.pos_family for occ in occs_by_lemma.get(lem_lc, [])
                         if occ.pos_family != "PROPN"}
        base_pool, pos_family, ptb_tag, used_broad_noun = _propn_token_candidates(
            aug, tok, feat, peer_families,
        )
        if used_broad_noun:
            broad_noun_fallback_lemmas.append(lem_lc)
        if base_pool is None or not base_pool:
            if pos_family is not None:
                skipped.append((doc_tag, tok.i, f"{pos_family}:empty_pool"))
            continue
        assert pos_family is not None and ptb_tag is not None
        _finalize_occurrence_fast(
            aug, doc_tag, tok, feat, base_pool,
            pos_family, ptb_tag, occs_by_lemma, skipped,
        )

    # --- Phase 2: lock-step joint intersection (A3 singleton fast path) ---
    lemmas: Dict[str, LemmaEntry] = {}
    broad_terminal_accept_lemmas: List[str] = []
    for lem_lc, occs in occs_by_lemma.items():
        pos_families = {o.pos_family for o in occs}
        pos_conflict = len(pos_families) > 1

        if len(occs) == 1:
            # A3: singleton. Skip the chain walk; level 0 is the answer.
            o = occs[0]
            lvl0 = o.chain[0]
            inter = set(lvl0.cand_set)
            mean_w = lvl0.weights  # already the single-occurrence weights
            accepted_level = 0
            accepted_on_broad_terminal = bool(getattr(lvl0, "is_broad_terminal", False))
            mean_cand_before: float = float(len(inter))
        else:
            # Multi-occurrence: lock-step walk.
            max_depth = max(len(o.chain) for o in occs)
            inter = set()
            mean_w = None
            accepted_level = -1
            accepted_on_broad_terminal = False

            for k in range(max_depth):
                level_sets: List[FrozenSet[str]] = []
                level_weights: List[Counter] = []
                all_have_weights = True
                any_broad_terminal = False
                for o in occs:
                    idx = min(k, len(o.chain) - 1)
                    lvl = o.chain[idx]
                    # Reading .cand_set on a _LazyBroadTerminalLevel triggers
                    # the thunk; on a regular OccChainLevel it's a direct attr.
                    level_sets.append(lvl.cand_set)
                    if getattr(lvl, "is_broad_terminal", False):
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
                    # Plain for-loop is measurably faster than the genexpr
                    # form (profile showed 9.2M genexpr iterations consuming
                    # 3.5s self-time; loop avoids the genexpr-frame setup).
                    # NOTE: cannot use Counter.__add__ here — level_weights
                    # are raw bucket Counters (O2′), each with ~30K keys,
                    # whereas `inter` is ~100-500 keys. Iterating `inter`
                    # is dramatically faster than summing whole Counters.
                    for c in inter:
                        total = 0
                        for w in level_weights:
                            total += w.get(c, 0) or 0
                        # Counter is typed Counter[T] with int values; we
                        # store float (mean weight). Python tolerates this
                        # at runtime; ignore the type check for mypyc.
                        mean_w[c] = total / k_occs  # type: ignore[assignment]
                break
            mean_cand_before = sum(len(o.chain[0].cand_set) for o in occs) / len(occs)

        if accepted_on_broad_terminal:
            broad_terminal_accept_lemmas.append(lem_lc)

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
# Sampling (fast)
# ==========================================================================


def _weighted_sample_one_fast(cands: List[str], weights,
                               rng: random.Random) -> str:
    """Vectorized Gumbel-top-1 sampling. Bit-identical to
    ``core._weighted_sample_one`` for the same RNG state and inputs:

      legacy:
        def score(c):
            w = float(weights.get(c, 0) or 0)
            u = rng.random()
            if u <= 0.0: u = sys.float_info.min
            return math.log(u) / max(w, 1.0)
        return max(cands, key=score)

    We pre-draw ``n`` uniforms via list comprehension (same RNG order as
    legacy's ``max(..., key=score)`` which iterates cands in order), then
    vectorize log + clamp + division + argmax with numpy. Tie-breaking
    matches: ``np.argmax`` returns the first index of the maximum, same as
    ``max(..., key=...)``.

    Falls back to legacy when numpy is unavailable or ``n == 1`` (RNG
    consumption preserved either way).
    """
    n = len(cands)
    if n == 0:
        raise ValueError("empty cands list")
    if n == 1:
        rng.random()  # match legacy: max(cands, key=score) calls score once
        return cands[0]
    if not _HAS_NUMPY:
        return _weighted_sample_one(cands, weights, rng)

    min_f = sys.float_info.min
    wg = weights.get
    # List comprehensions are faster than for-loops with per-element
    # array assignment for the pre-draw stage.
    us_list = [rng.random() for _ in range(n)]
    us = np.array(
        [u if u > 0.0 else min_f for u in us_list],
        dtype=np.float64,
    )
    ws = np.array(
        [float(wg(c, 0) or 0) for c in cands],
        dtype=np.float64,
    )
    np.maximum(ws, 1.0, out=ws)  # legacy: max(w, 1.0)
    scores = np.log(us) / ws
    return cands[int(np.argmax(scores))]


def sample_mapping_fast(cand_index: CandidateIndex, rng: random.Random,
                        fallback_mode: str) -> FastSampledMapping:
    """Same semantics as ``core.sample_mapping`` — outputs identical
    ``mapping`` for the same RNG state and ``cand_index``. Returns a
    ``FastSampledMapping`` (extends ``SampledMapping`` with a lazy
    ``realized`` cache for B3).
    """
    fallback_mode = (fallback_mode or "drop").strip().lower()
    if fallback_mode not in {"drop", "coarsen", "passthrough"}:
        raise ValueError(f"unknown fallback_mode={fallback_mode!r}")

    mapping: Dict[str, str] = {}
    pos_conflict_all: List[str] = []
    ctx_empty: List[str] = []
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
            passthrough.append(lem)
            continue

        if entry.pos_conflict:
            pos_conflict_resolved.append(lem)

        cands = sorted(entry.cand_set)
        if entry.weights is not None and len(cands) > 1:
            # Pass ``entry.weights`` directly. The legacy code rebuilt a
            # Counter restricted to ``cands``, but ``_weighted_sample_one``
            # only calls ``weights.get(c, 0)`` per cand — and
            # ``cands ⊆ entry.cand_set`` so the values are identical.
            target = _weighted_sample_one_fast(cands, entry.weights, rng)
        else:
            target = rng.choice(cands)
        mapping[lem] = target

    return FastSampledMapping(
        mapping=mapping,
        pos_conflict_lemmas=pos_conflict_all,
        ctx_intersection_empty_lemmas=ctx_empty,
        pos_conflict_resolved_lemmas=pos_conflict_resolved,
        passthrough_lemmas=passthrough,
        dropped_reason=dropped,
        realized={},
    )


# ==========================================================================
# Rendering (fast)
# ==========================================================================


_INDEF_ARTICLE_LC = {"a", "an"}


def _render_doc_fast(aug: Augmenter, doc: Doc,
                     occ_lookup: Dict[Tuple[str, int], Occurrence],
                     doc_tag: str, sampled: FastSampledMapping,
                     frozen_ranges: Optional[List[Tuple[int, int]]] = None,
                     frozen_lemmas: Optional[Set[str]] = None,
                     ) -> Tuple[str, List[Tuple[int, int]], List[str]]:
    """Same contract as ``core._render_doc`` plus:
      - B3: realized surfaces cached in ``sampled.realized[(lemma_lc, ptb_tag)]``.
      - B4: ``_fix_articles_with_spans`` regex scan only runs when an
        indefinite article appears adjacent to a replaced token.

    Output text is bit-identical to ``core._render_doc`` under
    matching inputs (mapping, frozen ranges, frozen lemmas).
    """
    mapping = sampled.mapping
    realized_cache = sampled.realized

    out_parts: List[str] = []
    rendered_parts: List[str] = []
    per_token_spans: List[Tuple[int, int]] = []
    cursor = 0
    need_cap = True

    needs_article_fix = False
    prev_rendered_lc = ""  # tracks the previously emitted rendered token

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
                # B3: realized-surface memoization.
                cache_key = (tgt, ptb, occ.pos_family)
                surf_cached = realized_cache.get(cache_key)
                if surf_cached is None:
                    if occ.pos_family == "PROPN":
                        new_surf = tgt
                        if ptb == "NNPS":
                            try:
                                plural = aug.inflect_engine.plural_noun(new_surf)
                                if plural:
                                    new_surf = plural
                            except Exception:
                                pass
                    else:
                        new_surf = aug._realize(tgt, ptb)
                    realized_cache[cache_key] = new_surf
                    surf_cached = new_surf
                rendered = match_case_like(tok.text, surf_cached)
                replaced = True

        if tok.is_sent_start:
            need_cap = True
        to_add = rendered
        if need_cap and tok.is_alpha and replaced:
            to_add = _cap_first_alpha(to_add)
        if tok.is_alpha:
            need_cap = False

        # B4: track adjacency of an indefinite article + replaced token.
        if replaced and prev_rendered_lc in _INDEF_ARTICLE_LC:
            needs_article_fix = True

        start = cursor
        end = start + len(to_add)
        per_token_spans.append((start, end))
        piece = to_add + tok.whitespace_
        out_parts.append(piece)
        rendered_parts.append(piece)
        cursor = end + len(tok.whitespace_)

        if tok.is_alpha:
            prev_rendered_lc = to_add.casefold()
        else:
            # A non-alpha token (punctuation, number) between two words doesn't
            # change the "previous word" for article-adjacency purposes, BUT
            # it does insert a space — if the previous word was an article and
            # this token is punctuation, articles aren't immediately adjacent
            # anymore. The legacy regex `_ARTICLE_PATTERN` requires whitespace
            # then word, so punctuation breaks adjacency. Conservative
            # behaviour: reset prev_rendered_lc on non-alpha tokens.
            prev_rendered_lc = ""

    raw = "".join(out_parts)

    if needs_article_fix:
        # Article fix only runs when adjacency was actually detected.
        # On rows with no replacement adjacent to a/an, we skip the regex
        # + pyinflect.a() call entirely.
        fixed, per_token_spans = _fix_articles_with_spans(aug, raw, per_token_spans)
        return fixed, per_token_spans, rendered_parts

    return raw, per_token_spans, rendered_parts


# ==========================================================================
# Convenience: build occ_lookup from a cand_index (mirrors core pattern)
# ==========================================================================


def build_occ_lookup(cand_index: CandidateIndex,
                     sampled: Union[FastSampledMapping, SampledMapping]
                     ) -> Dict[Tuple[str, int], Occurrence]:
    """Build a (doc_tag, tok_i) -> Occurrence map from cand_index restricted
    to mapped lemmas. Mirrors the pattern used in
    the document-scope pipeline renderers.
    """
    mapping = sampled.mapping
    occ_lookup: Dict[Tuple[str, int], Occurrence] = {}
    for lem, entry in cand_index.lemmas.items():
        if lem not in mapping:
            continue
        for occ in entry.occs:
            occ_lookup[(occ.doc_tag, occ.tok_i)] = occ
    return occ_lookup
