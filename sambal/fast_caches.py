"""Augmenter lazy-cache extensions for the fast relex engine.

Adds lazy frozenset caches to an Augmenter instance without subclassing
(speedups B1 + B2 + A1P2):

  - `_allowed_by_tag(ptb_tag) -> frozenset[str]` — replaces the eager
    Python loop `{c for c in base_pool if aug._candidate_allowed_for_tag(c, ptb_tag)}`
    with a C-speed set intersection `base_pool & aug._allowed_by_tag(ptb_tag)`.
    Built lazily on first access per ptb_tag (~13 total). Permanent cache.
  - `_allowed_broad_by_tag(ptb_tag) -> frozenset[str]` — same idea applied to
    the broad-terminal pool (noun/verb fallback). Used by the lazy
    broad-terminal level (A1) to keep the materialization cheap when it does
    fire.
  - `_bucket_lemmas_at(bucket_key, n) -> frozenset[str]` — per
    `(bucket_key, n)` frozenset of lemmas with count >= n at that bucket
    (B2). LRU-bounded to keep RSS sane.
  - `_chain_memo_get` / `_chain_memo_set(key, value)` — A1 P2 cross-row chain
    memo. **DEFAULT OFF.** Caches the full chain-levels output of
    `_build_occ_chain_fast` keyed on (ptb_tag, pos_family, is_verb_like,
    ctx_bucket, src_lem, src_surf, frozenset(base_pool)). LRU-bounded.
    Enable via `RELEX_CHAIN_MEMO=1` env var or `enable_chain_memo=True` kwarg
    to install_fast_caches. Verified bit-equal-output parity
    (1.35× speedup, 77.4% hit rate); measured per-entry memory cost is high
    (~670 KB), so it stays off by default.

Activated by a one-shot ``install_fast_caches(aug)`` call from the stage-2
driver, BEFORE any rows are processed. Idempotent.

Cache hygiene:
  - The fast caches are NOT cleared by ``core._clear_aug_caches``;
    they are correctness-stable across rows (allowed-status depends only on
    `(lemma, ptb_tag)` and never on row context). To purge for a long-running
    process where memory is a concern, call ``clear_fast_caches(aug)``
    explicitly. The bucket-level cache is LRU-bounded by default.

Side-effect note: the precomputed-frozenset path bypasses the per-call
`_lemma_tag_allowed` and `_cache_probe` side effects in
`_candidate_allowed_for_tag`. This is correctness-preserving (the gate result
is identical) and the cache probes are diagnostics-only.

Requires `aug._allowed_lemmas_active == True` (i.e., an `allowed_vocab_path`
was provided to the Augmenter). For Augmenters without an active
allowed-lemmas gate, the `_allowed_by_tag` precomputation has no defined
universe and `install_fast_caches` raises. Callers can guard.
"""
from __future__ import annotations

import os
from collections import OrderedDict
from typing import FrozenSet, Optional, Tuple, Any


# Default LRU bound for the bucket-level cache. Each entry is a frozenset of
# lemmas (a few hundred to a few thousand strings). 10K entries × ~5KB ~= 50MB
# ceiling. Increase if RSS budget allows.
DEFAULT_BUCKET_LRU_MAX = 10_000

# Default LRU bound for the chain memo. Each entry is a list of ~4-6
# OccChainLevel objects + optional _LazyBroadTerminalLevel + an
# `frozenset(base_pool)` reference in the key. Empirical per-entry cost
# observed on a production workload: ~670 KB/entry (RSS
# delta 16,737 MB over 25,000 cache fills). A 5,000-entry cap therefore
# bounds the cache at ~3.4 GB per worker — fits comfortably alongside the
# 9-10 GB Augmenter baseline within the 35-36 GB/worker partition budgets.
# Raise via the `chain_memo_max` kwarg to install_fast_caches if your
# worker has more headroom AND you've verified RSS stays bounded.
DEFAULT_CHAIN_MEMO_MAX = 5_000

# ---- Cache attribute names (also used by clear_fast_caches) ----
_ALLOWED_BY_TAG_ATTR = "_allowed_by_tag_cache"
_ALLOWED_BROAD_BY_TAG_ATTR = "_allowed_broad_by_tag_cache"
_BUCKET_LEMMAS_AT_ATTR = "_bucket_lemmas_at_cache"
_CHAIN_MEMO_ATTR = "_chain_memo_cache"
_INSTALLED_MARK = "_fast_caches_installed"

# Names of the fast caches we track. Extends `core._AUG_LEMMA_CACHES`
# semantically but the periodic clear in `_clear_aug_caches` does NOT touch
# them (they're row-invariant). Listed here for explicit reset via
# `clear_fast_caches`.
_FAST_AUG_CACHE_ATTRS: Tuple[str, ...] = (
    _ALLOWED_BY_TAG_ATTR,
    _ALLOWED_BROAD_BY_TAG_ATTR,
    _BUCKET_LEMMAS_AT_ATTR,
    _CHAIN_MEMO_ATTR,
)


# PTB tags expected to be encountered in any real corpus document. Precomputed
# eagerly at install_fast_caches(eager=True) to amortize the universe-walk
# cost (each ptb_tag's first call walks the ~100K-lemma allowed_lemmas
# universe through `_candidate_allowed_for_tag`, which calls `_realize` via
# pyinflect — ~50-100s per tag locally).
_COMMON_PTB_TAGS = (
    "NN", "NNS", "NNP", "NNPS",                  # nouns + proper nouns
    "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",     # verbs
    "JJ", "JJR", "JJS",                          # adjectives
    "RB", "RBR", "RBS",                          # adverbs
)


def install_fast_caches(aug, *, eager: bool = True,
                        bucket_lru_max: int = DEFAULT_BUCKET_LRU_MAX,
                        chain_memo_max: int = DEFAULT_CHAIN_MEMO_MAX,
                        enable_chain_memo: Optional[bool] = None,
                        common_tags=_COMMON_PTB_TAGS,
                        verbose: bool = False) -> None:
    """Attach `_allowed_by_tag`, `_allowed_broad_by_tag`, `_bucket_lemmas_at`
    bound methods to ``aug``. Idempotent — repeat calls are no-ops.

    When ``eager=True`` (default), also precomputes the per-tag
    ``_allowed_by_tag(ptb_tag)`` and ``_allowed_broad_by_tag(ptb_tag)``
    frozensets for every tag in ``common_tags`` upfront. This pays the
    universe-walk cost once at install time (~5-10 minutes total for
    ~16 tags on a typical corpus vocabulary) instead of paying it lazily
    during the first few hundred docs of a shard.

    For long-running stage-2 workers, the upfront cost is amortized cheaply
    against 80K+ docs. For unit tests or short ad-hoc runs that want to
    skip the warmup, pass ``eager=False`` — but then the per-doc latency
    will be high for the first ~30 docs.

    Raises ``RuntimeError`` if ``aug._allowed_lemmas_active`` is False (the
    fast engine relies on a finite candidate universe; without an active
    allowed-vocab gate, the legacy path is the only option).
    """
    if getattr(aug, _INSTALLED_MARK, False):
        return
    if not getattr(aug, "_allowed_lemmas_active", False):
        raise RuntimeError(
            "install_fast_caches requires aug._allowed_lemmas_active=True "
            "(i.e., an allowed_vocab_path must be configured). Without an "
            "active allowed-vocab gate the fast engine's universe-based "
            "precomputation is undefined."
        )

    setattr(aug, _ALLOWED_BY_TAG_ATTR, {})
    setattr(aug, _ALLOWED_BROAD_BY_TAG_ATTR, {})
    setattr(aug, _BUCKET_LEMMAS_AT_ATTR, OrderedDict())
    setattr(aug, "_bucket_lru_max", int(bucket_lru_max))

    # Bind core fast-caches methods (B1/B2; always on).
    aug._allowed_by_tag = _allowed_by_tag.__get__(aug, aug.__class__)
    aug._allowed_broad_by_tag = _allowed_broad_by_tag.__get__(aug, aug.__class__)
    aug._bucket_lemmas_at = _bucket_lemmas_at.__get__(aug, aug.__class__)

    # A1 P2 chain memo — DEFAULT OFF. Resolve enable in order:
    #   explicit kwarg > env var > default False.
    # Measured per-entry cost is ~670 KB, so this stays default-off. Enable
    # via:
    #   RELEX_CHAIN_MEMO=1  (env)
    #   install_fast_caches(..., enable_chain_memo=True)  (kwarg)
    if enable_chain_memo is None:
        enable_chain_memo = (os.environ.get("RELEX_CHAIN_MEMO", "0") == "1")
    if enable_chain_memo:
        setattr(aug, _CHAIN_MEMO_ATTR, OrderedDict())
        setattr(aug, "_chain_memo_max", int(chain_memo_max))
        setattr(aug, "_chain_memo_hits", 0)
        setattr(aug, "_chain_memo_misses", 0)
        aug._chain_memo_get = _chain_memo_get.__get__(aug, aug.__class__)
        aug._chain_memo_set = _chain_memo_set.__get__(aug, aug.__class__)
        if verbose:
            print(f"[install_fast_caches] chain memo ENABLED "
                  f"(max={chain_memo_max})", flush=True)
    # else: do NOT bind _chain_memo_get/_chain_memo_set. The check
    # `getattr(aug, "_chain_memo_get", None)` in core_fast falls to
    # None and the engine takes the original no-cache path (bit-equal to
    # the no-memo path).

    setattr(aug, _INSTALLED_MARK, True)

    if eager:
        warm_allowed_by_tag(aug, common_tags, verbose=verbose)


def warm_allowed_by_tag(aug, tags=_COMMON_PTB_TAGS, *,
                        include_broad: bool = True,
                        verbose: bool = False) -> None:
    """Pre-compute and cache ``_allowed_by_tag(t)`` for each ``t`` in tags.

    Optionally also precompute ``_allowed_broad_by_tag(t)`` (only meaningful
    for NN/NNS/NNP/NNPS/VB* tags; cheap on others as the broad pool is empty).

    Each per-tag precomputation walks the full ``allowed_lemmas`` universe
    once, paying the slow ``_realize`` path for any uncached (lemma, ptb_tag)
    pair. After this completes, every subsequent ``_allowed_by_tag`` /
    ``_allowed_broad_by_tag`` call for these tags is O(1).
    """
    import time
    if verbose:
        print(f"[warm_allowed_by_tag] warming {len(tags)} tags...", flush=True)
    t_total = time.time()
    for tag in tags:
        t0 = time.time()
        fs = aug._allowed_by_tag(tag)
        n = len(fs) if fs is not None else 0
        if include_broad:
            fs_broad = aug._allowed_broad_by_tag(tag)
            n_broad = len(fs_broad)
        else:
            n_broad = 0
        if verbose:
            print(f"  [warm] {tag:<5} |allowed|={n:>6} |broad|={n_broad:>6} "
                  f"  ({time.time()-t0:5.1f}s)", flush=True)
    if verbose:
        print(f"[warm_allowed_by_tag] done in {time.time()-t_total:.1f}s",
              flush=True)


def clear_fast_caches(aug) -> None:
    """Reset all fast caches (universe of allowed lemmas per tag, broad
    terminal per tag, bucket-level lemmas). Safe to call at any time;
    correctness is preserved (caches refill lazily on next access).
    """
    for name in _FAST_AUG_CACHE_ATTRS:
        obj = getattr(aug, name, None)
        if isinstance(obj, dict):
            obj.clear()


# ----------------------------------------------------------------------
# Implementations (bound to aug via install_fast_caches)
# ----------------------------------------------------------------------

def _gate_lemma_for_tag(aug, lemma: str, ptb_tag: str) -> bool:
    """Inlined equivalent of `Augmenter._candidate_allowed_for_tag`.

    Uses only plain attributes of ``engine.Augmenter`` rather than calling
    the method directly, so the fast path stays independent of
    engine-internal details.

    Reproduces ``engine.Augmenter._candidate_allowed_for_tag``:
      - reject NPI lemmas (single-token)
      - reject NPI realized surfaces (single-token)
      - reject function-word lemmas
      - if `_allowed_lemmas_active`:
          - PROPN tags: realized surface must be in `allowed_vocab`
          - other tags: lemma must be in `allowed_lemmas`
    """
    lem_lc = (lemma or "").lower()
    if not lem_lc:
        return False
    npi = aug._npi_unigrams
    if lem_lc in npi:
        return False
    try:
        surf = aug._realize(lem_lc, ptb_tag)
    except Exception:
        surf = ""
    surf_lc = surf.lower() if surf else ""
    if surf_lc and (surf_lc in npi):
        return False
    if aug._is_functionish_lemma(lem_lc):
        return False
    if aug._allowed_lemmas_active:
        if ptb_tag in {"NNP", "NNPS"}:
            return bool(surf and (surf in (aug.allowed_vocab or set())))
        allowed = aug.allowed_lemmas or set()
        return lem_lc in allowed
    return True


def _allowed_by_tag(self, ptb_tag: str) -> FrozenSet[str]:
    """Frozenset of lemmas that pass the per-(lemma, ptb_tag) allow gate.

    Universe:
      - PROPN ptb_tags (NNP, NNPS): `allowed_propn_lemmas` union `allowed_vocab`.
        (The propn-allowed gate checks `surf in allowed_vocab`; we precompute
        over the conservative union so any candidate surface form that could
        realize from a lemma is covered.)
      - Other ptb_tags: `allowed_lemmas`.

    Built lazily once per ptb_tag, cached permanently. ~13 distinct tags.
    """
    cache = getattr(self, _ALLOWED_BY_TAG_ATTR, None)
    if cache is None:
        raise RuntimeError(
            "_allowed_by_tag called without install_fast_caches first; "
            "the fast engine requires the precomputed cache."
        )
    cached = cache.get(ptb_tag)
    if cached is not None:
        return cached

    if ptb_tag in {"NNP", "NNPS"}:
        universe = set(self.allowed_propn_lemmas or set())
        if self.allowed_vocab:
            universe |= set(self.allowed_vocab)
    else:
        universe = set(self.allowed_lemmas or set())

    if not universe:
        out = frozenset()
        cache[ptb_tag] = out
        return out

    out = frozenset(c for c in universe
                    if _gate_lemma_for_tag(self, c, ptb_tag))
    cache[ptb_tag] = out
    return out


def _allowed_broad_by_tag(self, ptb_tag: str) -> FrozenSet[str]:
    """Frozenset of broad-terminal-pool lemmas filtered by
    `_candidate_allowed_for_tag` for this tag. Used by the lazy
    broad-terminal level (A1) — when Phase 2 lock-step actually probes the
    broad terminal, building the gated pool is a single set intersection.

    Universe: union of the broad noun pool (for noun-like tags) or the
    broad verb pool (for verb-like tags). Other tags get empty (broad
    terminal isn't used for ADJ/ADV).

    Built lazily once per ptb_tag.
    """
    cache = getattr(self, _ALLOWED_BROAD_BY_TAG_ATTR, None)
    if cache is None:
        return frozenset()
    cached = cache.get(ptb_tag)
    if cached is not None:
        return cached

    # Defer import to avoid circular: core imports engine.
    from sambal.core import (
        _broad_noun_pool_for_tag,
        _broad_verb_pool,
    )

    if ptb_tag in {"NN", "NNS", "NNP", "NNPS"}:
        broad = _broad_noun_pool_for_tag(self, ptb_tag) or frozenset()
    elif ptb_tag in {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}:
        broad = _broad_verb_pool(self) or frozenset()
    else:
        broad = frozenset()

    if not broad:
        out = frozenset()
        cache[ptb_tag] = out
        return out

    allowed = self._allowed_by_tag(ptb_tag)
    # C-speed intersection. `_allowed_by_tag` always returns a frozenset
    # (or raises) post-install_fast_caches.
    out = frozenset(broad & allowed)
    cache[ptb_tag] = out
    return out


def _chain_memo_get(self, key):
    """A1 P2: look up the cached chain levels for ``key``. Returns None on
    miss. LRU bump on hit. Increments diagnostic counters.

    Cache key is the 7-tuple constructed by `core_fast._build_occ_chain_fast`
    (ptb_tag, pos_family, is_verb_like, ctx_bucket, src_lem, src_surf,
    frozenset(base_pool)). All chain-build dependencies are in the key —
    hits return the chain the no-cache path would produce.
    """
    cache = getattr(self, _CHAIN_MEMO_ATTR, None)
    if cache is None:
        return None
    hit = cache.get(key)
    if hit is not None:
        cache.move_to_end(key)
        self._chain_memo_hits = getattr(self, "_chain_memo_hits", 0) + 1
        return hit
    self._chain_memo_misses = getattr(self, "_chain_memo_misses", 0) + 1
    return None


def _chain_memo_set(self, key, value):
    """A1 P2: insert chain-levels into the cache. LRU-bounded to
    ``self._chain_memo_max``; least-recently-used entries are evicted first."""
    cache = getattr(self, _CHAIN_MEMO_ATTR, None)
    if cache is None:
        return
    cache[key] = value
    cache.move_to_end(key)
    lru_max = int(getattr(self, "_chain_memo_max", DEFAULT_CHAIN_MEMO_MAX))
    while len(cache) > lru_max:
        cache.popitem(last=False)


def _bucket_lemmas_at(self, bucket_key: Tuple[Any, ...], n: int
                     ) -> FrozenSet[str]:
    """Frozenset of lemmas whose count in `_ctx_bucket2lemma[bucket_key]`
    is >= n. Replaces the per-call Python comprehension
    `{c for c in base_list if lemma_ctr.get(c, 0) >= n}` in the chain-level
    construction with a single set intersection on the caller side
    (`base_set & _bucket_lemmas_at(bk, n)`).

    LRU-bounded — bumps to the front on hit, evicts least-recently-used when
    exceeding `_bucket_lru_max`.
    """
    cache = getattr(self, _BUCKET_LEMMAS_AT_ATTR, None)
    if cache is None:
        # Fast caches not installed; return empty to fall back caller-side.
        return frozenset()
    key = (bucket_key, n)
    hit = cache.get(key)
    if hit is not None:
        cache.move_to_end(key)
        return hit

    lemma_ctr = self._ctx_bucket2lemma.get(bucket_key)
    if not lemma_ctr:
        out = frozenset()
    else:
        out = frozenset(c for c, cnt in lemma_ctr.items() if cnt >= n)

    cache[key] = out
    cache.move_to_end(key)
    lru_max = int(getattr(self, "_bucket_lru_max", DEFAULT_BUCKET_LRU_MAX))
    while len(cache) > lru_max:
        cache.popitem(last=False)
    return out
