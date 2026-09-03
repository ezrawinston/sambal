"""Parity tests for ``core_fast`` vs ``core`` (legacy).

Two tiers:

  - **Synthetic-CandidateIndex parity** (always runs, fast): builds a
    ``CandidateIndex`` directly and confirms ``sample_mapping_fast`` and
    ``_render_doc_fast`` produce bit-identical outputs to ``sample_mapping``
    and ``_render_doc`` under the same RNG seed.
  - **End-to-end parity on real text** (requires resources dir + spaCy
    model; gated via ``SAMBAL_TEST_RESOURCES_DIR`` env var or a presence
    check): runs both engines through Phase 1 + Phase 2 + render on a few
    realistic input strings and asserts identical mapping + output text.

The synthetic tier guards against regressions in sample/render code paths;
the integration tier guards against regressions in the candidate-set
construction (A1/A2/A3/A4 + B1/B2) which can't be exercised without a real
Augmenter.

Run all tiers:
    pytest tests/test_core_fast_parity.py -v
Run only synthetic:
    pytest tests/test_core_fast_parity.py -v -k synthetic
Skip integration:
    SAMBAL_TEST_SKIP_INTEGRATION=1 pytest ...
"""
from __future__ import annotations

import os
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import spacy
from spacy.tokens import Doc

from sambal.core import (
    Occurrence, OccChainLevel, LemmaEntry, CandidateIndex, SampledMapping,
    sample_mapping,
    _render_doc as _render_doc_legacy,
)
from sambal.core_fast import (
    FastSampledMapping,
    sample_mapping_fast,
    _render_doc_fast,
    build_occ_lookup,
)


# ==========================================================================
# Stub Augmenter (mirrors test_core_span_freeze._StubAug)
# ==========================================================================

class _StubInflect:
    def plural_noun(self, s: str) -> str:
        return s + "s" if not s.endswith("s") else s

    def a(self, s: str) -> str:
        return f"an {s}" if s[:1].lower() in "aeiou" else f"a {s}"


class _StubAug:
    def __init__(self):
        self.inflect_engine = _StubInflect()

    def _realize(self, lemma_lc: str, ptb_tag: str) -> str:
        # Deterministic surface form: lemma + 's' suffix for plural NN
        if ptb_tag == "NNS":
            return lemma_lc + "s"
        return lemma_lc

    def _fix_indefinite_articles(self, text: str) -> str:
        return text


@pytest.fixture(scope="module")
def blank_nlp():
    return spacy.blank("en")


def _doc_from_tokens(nlp, words, spaces, lemmas, pos, tags):
    return Doc(nlp.vocab, words=words, spaces=spaces,
                lemmas=lemmas, pos=pos, tags=tags)


# ==========================================================================
# 1. Synthetic CandidateIndex parity (sample_mapping + _render_doc)
# ==========================================================================

def _build_cand_index_synthetic() -> CandidateIndex:
    """Build a CandidateIndex by hand with three lemmas — exercises the
    weighted-sample path, the uniform-sample path, and the empty-cand_set
    drop path.
    """
    occ_a = Occurrence(
        doc_tag="doc", tok_i=2, ptb_tag="NN",
        pos_family="NOUN", source_surface="cat",
        chain=[OccChainLevel(cand_set=frozenset({"dog", "bird"}), weights=None)],
    )
    occ_b = Occurrence(
        doc_tag="doc", tok_i=5, ptb_tag="VB",
        pos_family="VERB", source_surface="run",
        chain=[OccChainLevel(
            cand_set=frozenset({"walk", "jog", "sprint"}),
            weights=Counter({"walk": 10, "jog": 5, "sprint": 1}),
        )],
    )
    # Empty cand_set → drop path
    occ_c = Occurrence(
        doc_tag="doc", tok_i=7, ptb_tag="JJ",
        pos_family="ADJ", source_surface="fast",
        chain=[OccChainLevel(cand_set=frozenset(), weights=None)],
    )

    lemmas: Dict[str, LemmaEntry] = {
        "cat": LemmaEntry(
            cand_set=frozenset({"dog", "bird"}),
            weights=None, occs=[occ_a],
            pos_families={"NOUN"}, pos_conflict=False,
            mean_cand_size_before=2.0, size_after=2, accepted_level=0,
        ),
        "run": LemmaEntry(
            cand_set=frozenset({"walk", "jog", "sprint"}),
            weights=Counter({"walk": 10.0, "jog": 5.0, "sprint": 1.0}),
            occs=[occ_b],
            pos_families={"VERB"}, pos_conflict=False,
            mean_cand_size_before=3.0, size_after=3, accepted_level=0,
        ),
        "fast": LemmaEntry(
            cand_set=frozenset(),
            weights=None, occs=[occ_c],
            pos_families={"ADJ"}, pos_conflict=False,
            mean_cand_size_before=0.0, size_after=0, accepted_level=-1,
        ),
    }
    return CandidateIndex(lemmas=lemmas, skipped=[],
                           broad_noun_fallback_lemmas=[],
                           broad_terminal_accept_lemmas=[])


@pytest.mark.parametrize("fallback_mode,seed",
                          [("drop", 1337), ("drop", 42),
                           ("passthrough", 1337), ("passthrough", 42)])
def test_sample_mapping_parity_synthetic(fallback_mode, seed):
    """sample_mapping vs sample_mapping_fast produce identical `mapping`
    dicts and identical drop/passthrough diagnostics under the same RNG.
    """
    cand_index = _build_cand_index_synthetic()

    rng1 = random.Random(seed)
    legacy = sample_mapping(cand_index, rng1, fallback_mode)

    rng2 = random.Random(seed)
    fast = sample_mapping_fast(cand_index, rng2, fallback_mode)

    assert legacy.mapping == fast.mapping
    assert legacy.dropped_reason == fast.dropped_reason
    assert legacy.passthrough_lemmas == fast.passthrough_lemmas
    assert legacy.pos_conflict_lemmas == fast.pos_conflict_lemmas
    assert (legacy.ctx_intersection_empty_lemmas
            == fast.ctx_intersection_empty_lemmas)
    assert (legacy.pos_conflict_resolved_lemmas
            == fast.pos_conflict_resolved_lemmas)


def test_render_doc_parity_synthetic_no_articles(blank_nlp):
    """_render_doc vs _render_doc_fast on a synthetic Doc with NO indefinite
    articles. Both should produce identical text.
    """
    words = ["The", "cat", "and", "the", "dog", "run", "fast", "."]
    spaces = [True, True, True, True, True, True, False, False]
    lemmas = ["the", "cat", "and", "the", "dog", "run", "fast", "."]
    pos = ["DET", "NOUN", "CCONJ", "DET", "NOUN", "VERB", "ADV", "PUNCT"]
    tags = ["DT", "NN", "CC", "DT", "NN", "VB", "RB", "."]
    doc = _doc_from_tokens(blank_nlp, words, spaces, lemmas, pos, tags)

    aug = _StubAug()
    cand_index = _build_cand_index_synthetic()

    rng_legacy = random.Random(0)
    sampled_legacy = sample_mapping(cand_index, rng_legacy, "passthrough")
    rng_fast = random.Random(0)
    sampled_fast = sample_mapping_fast(cand_index, rng_fast, "passthrough")
    assert sampled_legacy.mapping == sampled_fast.mapping

    occ_lookup = build_occ_lookup(cand_index, sampled_fast)

    text_legacy, spans_legacy, _ = _render_doc_legacy(
        aug, doc, occ_lookup, "doc", sampled_legacy.mapping,
    )
    text_fast, spans_fast, _ = _render_doc_fast(
        aug, doc, occ_lookup, "doc", sampled_fast,
    )

    assert text_legacy == text_fast, (
        f"render text differs:\n  legacy={text_legacy!r}\n  fast={text_fast!r}"
    )
    assert spans_legacy == spans_fast


def test_render_doc_parity_synthetic_with_articles(blank_nlp):
    """_render_doc vs _render_doc_fast on a Doc WITH indefinite articles
    adjacent to replaced tokens (forces B4's article-fix path on the fast
    engine).
    """
    # "I saw a cat run." — "a cat" → "a dog" (or "an dog" needing fix)
    words = ["I", "saw", "a", "cat", "run", "."]
    spaces = [True, True, True, True, False, False]
    lemmas = ["I", "see", "a", "cat", "run", "."]
    pos = ["PRON", "VERB", "DET", "NOUN", "VERB", "PUNCT"]
    tags = ["PRP", "VBD", "DT", "NN", "VB", "."]
    doc = _doc_from_tokens(blank_nlp, words, spaces, lemmas, pos, tags)

    aug = _StubAug()

    # Force the mapping: cat -> ostrich (vowel-initial → needs "an")
    occ = Occurrence(
        doc_tag="doc", tok_i=3, ptb_tag="NN",
        pos_family="NOUN", source_surface="cat",
        chain=[OccChainLevel(cand_set=frozenset({"ostrich"}), weights=None)],
    )
    lemmas_dict = {
        "cat": LemmaEntry(
            cand_set=frozenset({"ostrich"}), weights=None, occs=[occ],
            pos_families={"NOUN"}, pos_conflict=False,
            mean_cand_size_before=1.0, size_after=1, accepted_level=0,
        ),
    }
    cand_index = CandidateIndex(lemmas=lemmas_dict, skipped=[],
                                  broad_noun_fallback_lemmas=[],
                                  broad_terminal_accept_lemmas=[])

    rng_legacy = random.Random(0)
    sampled_legacy = sample_mapping(cand_index, rng_legacy, "drop")
    rng_fast = random.Random(0)
    sampled_fast = sample_mapping_fast(cand_index, rng_fast, "drop")
    occ_lookup = build_occ_lookup(cand_index, sampled_fast)

    text_legacy, _, _ = _render_doc_legacy(
        aug, doc, occ_lookup, "doc", sampled_legacy.mapping,
    )
    text_fast, _, _ = _render_doc_fast(
        aug, doc, occ_lookup, "doc", sampled_fast,
    )
    assert text_legacy == text_fast, (
        f"render text differs:\n  legacy={text_legacy!r}\n  fast={text_fast!r}"
    )


# ==========================================================================
# 2. End-to-end parity on real text (integration; gated)
# ==========================================================================

INTEGRATION_TEST_TEXTS = [
    "The Renaissance was an artistic movement that flourished in Italy "
    "during the fourteenth century.",
    "Computers process instructions sequentially, but modern processors "
    "exploit parallelism through multiple cores and SIMD units.",
    "Saint Bernadette of Lourdes reported seeing apparitions of the Virgin "
    "Mary near a grotto in southwestern France in 1858.",
]


def _resources_dir():
    """Locate the relex resources dir locally."""
    # Env var override, else the package's own resources directory.
    pkg_res = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "sambal", "resources",
    )
    for path in (
        os.environ.get("SAMBAL_TEST_RESOURCES_DIR"),
        pkg_res,
    ):
        if path and os.path.isdir(path):
            return path
    return None


def _make_real_augmenter():
    """Construct a real Augmenter with the package resources dir.

    Uses ``engine.Augmenter`` — the document-scope core requires
    `_candidate_allowed_for_tag`, which the engine exposes.

    Caller is responsible for installing fast caches.
    Returns None if resources / spacy model are unavailable.
    """
    rdir = _resources_dir()
    if rdir is None:
        return None
    try:
        from sambal.config import Config, ResourcePaths
        from sambal.engine import Augmenter
    except Exception:
        return None

    res = Path(rdir)
    paths = ResourcePaths(
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
        ctx_lemma_stats_path=str(res / "lemma_stats_top_25k.pkl"),
    )
    cfg = Config(
        spacy_model="en_core_web_trf",
        seed=42, replace_propn=True, roundtrip_debug=False, debug=False,
        debug_vn=False, replace_pronouns=False, roundtrip_max_tries=4,
        roundtrip_ud=False, freeze_aux_hosts=False, min_vocab_freq=1,
        respect_human=True, respect_gender=True, verb_metrics=False,
        prefilter_pools_by_allowed_vocab=True,
        filter_substitution_names_by_allowed_vocab=True,
        ctx_lemma_gate="freq", ctx_lemma_gate_min_count=2,
        vn_soft_prep_min_pool=10, ctx_backoff_mode="B",
        ctx_backoff_min_lemma_count=2, ctx_backoff_min_candidates=2,
        ctx_backoff_min_candidates_pct=0.05, skip_toinf_freeze=False,
        require_gpu=False,
    )
    try:
        return Augmenter(paths, cfg)
    except Exception:
        return None


SKIP_INTEGRATION = (
    os.environ.get("SAMBAL_TEST_SKIP_INTEGRATION") == "1"
    or _resources_dir() is None
)


@pytest.fixture(scope="module")
def real_augmenter():
    """Module-scoped Augmenter so en_core_web_trf is loaded ONCE per pytest
    invocation across all parametrized end-to-end tests. Lazy warmup
    (eager=False) — parity test only exercises the engine on 3 short docs,
    so only ~5-8 tags actually need to be precomputed."""
    aug = _make_real_augmenter()
    if aug is None:
        pytest.skip("Could not instantiate Augmenter (en_core_web_trf missing?)")
    from sambal.fast_caches import install_fast_caches
    install_fast_caches(aug, eager=False)
    return aug


@pytest.mark.skipif(SKIP_INTEGRATION,
                    reason="Skipped: set SAMBAL_TEST_SKIP_INTEGRATION=0 + ensure "
                            "relex resources dir + en_core_web_trf are available")
@pytest.mark.parametrize("text", INTEGRATION_TEST_TEXTS)
def test_end_to_end_parity_real_text(text, real_augmenter):
    """Full pipeline parity: Phase 1 + Phase 2 + sample + render on real
    text, both engines must produce identical results.
    """
    from sambal.core import build_candidate_sets_from_docs
    from sambal.core_fast import (
        build_candidate_sets_from_docs_fast,
    )

    aug = real_augmenter

    doc = aug.nlp(text)

    # Both engines on the same Doc.
    ci_legacy = build_candidate_sets_from_docs(aug, [("doc", doc)])
    ci_fast = build_candidate_sets_from_docs_fast(aug, [("doc", doc)])

    # Per-lemma equality
    assert set(ci_legacy.lemmas.keys()) == set(ci_fast.lemmas.keys()), (
        f"lemma sets differ:\n"
        f"  legacy-only: {set(ci_legacy.lemmas.keys()) - set(ci_fast.lemmas.keys())}\n"
        f"  fast-only:   {set(ci_fast.lemmas.keys()) - set(ci_legacy.lemmas.keys())}"
    )
    for lem, e_legacy in ci_legacy.lemmas.items():
        e_fast = ci_fast.lemmas[lem]
        assert e_legacy.cand_set == e_fast.cand_set, (
            f"cand_set differs for lemma {lem!r}"
        )
        assert e_legacy.accepted_level == e_fast.accepted_level, (
            f"accepted_level differs for lemma {lem!r}"
        )

    # Same seed → same mapping.
    seed = 1337
    rng1 = random.Random(seed)
    sampled_legacy = sample_mapping(ci_legacy, rng1, "drop")
    rng2 = random.Random(seed)
    sampled_fast = sample_mapping_fast(ci_fast, rng2, "drop")
    assert sampled_legacy.mapping == sampled_fast.mapping

    # Render text equality.
    occ_lookup_legacy = {
        (occ.doc_tag, occ.tok_i): occ
        for lem, entry in ci_legacy.lemmas.items()
        if lem in sampled_legacy.mapping
        for occ in entry.occs
    }
    occ_lookup_fast = build_occ_lookup(ci_fast, sampled_fast)
    # Compare on render-relevant Occurrence fields only. ``Occurrence.chain``
    # contains ``OccChainLevel`` instances whose ``weights`` field is a
    # bucket Counter; the fast engine passes the raw bucket Counter while
    # legacy builds a filtered Counter — semantically equivalent for all
    # downstream consumers (Phase 2 mean, sampling) but structurally
    # different, so ``Occurrence.__eq__`` reports a diff. The real semantic
    # check is the rendered-text comparison below.
    def _occ_id(occ):
        return (occ.doc_tag, occ.tok_i, occ.ptb_tag,
                occ.pos_family, occ.source_surface)
    assert set(occ_lookup_legacy) == set(occ_lookup_fast)
    for key in occ_lookup_legacy:
        assert _occ_id(occ_lookup_legacy[key]) == _occ_id(occ_lookup_fast[key])

    text_legacy, _, _ = _render_doc_legacy(
        aug, doc, occ_lookup_legacy, "doc", sampled_legacy.mapping,
    )
    text_fast, _, _ = _render_doc_fast(
        aug, doc, occ_lookup_fast, "doc", sampled_fast,
    )
    assert text_legacy == text_fast, (
        f"render text differs:\n  legacy={text_legacy!r}\n  fast={text_fast!r}"
    )


# ==========================================================================
# 3. A1 P2 chain memo correctness (cache hits return bit-equal chains)
# ==========================================================================

@pytest.mark.skipif(SKIP_INTEGRATION,
                    reason="needs real Augmenter + resources dir")
def test_chain_memo_hits_preserve_parity(real_augmenter):
    """A1 P2: running the same doc twice through the fast engine must yield
    bit-identical CandidateIndex output AND must actually exercise the
    cache (hits > 0 on the second pass).

    The chain memo is DEFAULT OFF — explicitly install it
    here to exercise the code path. Guards against a regression where the
    cache key omits a chain-build dependency (e.g., ctx_bucket, src_surf) —
    a silent drift would surface as cand_set inequality on the second pass.
    """
    from collections import OrderedDict
    from sambal.core_fast import build_candidate_sets_from_docs_fast
    from sambal.fast_caches import (
        _CHAIN_MEMO_ATTR, DEFAULT_CHAIN_MEMO_MAX,
        _chain_memo_get, _chain_memo_set,
    )

    aug = real_augmenter
    # Explicitly install the chain memo for this test (default-off;
    # production callers must opt in via RELEX_CHAIN_MEMO=1 or
    # the enable_chain_memo kwarg).
    setattr(aug, _CHAIN_MEMO_ATTR, OrderedDict())
    setattr(aug, "_chain_memo_max", DEFAULT_CHAIN_MEMO_MAX)
    setattr(aug, "_chain_memo_hits", 0)
    setattr(aug, "_chain_memo_misses", 0)
    aug._chain_memo_get = _chain_memo_get.__get__(aug, aug.__class__)
    aug._chain_memo_set = _chain_memo_set.__get__(aug, aug.__class__)

    text = ("The Renaissance was an artistic movement that flourished in "
            "Italy during the fourteenth century. The Renaissance was also "
            "a period of scientific discovery.")
    doc = aug.nlp(text)

    ci_first = build_candidate_sets_from_docs_fast(aug, [("doc", doc)])
    hits_after_first = aug._chain_memo_hits
    misses_after_first = aug._chain_memo_misses
    ci_second = build_candidate_sets_from_docs_fast(aug, [("doc", doc)])
    hits_after_second = aug._chain_memo_hits

    # Cache MUST have been exercised on the second pass.
    second_pass_hits = hits_after_second - hits_after_first
    assert second_pass_hits > 0, (
        f"chain memo got 0 hits on the second pass; cache key is too "
        f"fine-grained or the cache isn't wired. "
        f"hits_after_first={hits_after_first} misses_after_first={misses_after_first}"
    )

    # Per-lemma cand_set must be bit-identical across the two passes.
    assert set(ci_first.lemmas.keys()) == set(ci_second.lemmas.keys()), (
        "lemma sets diverged between cache-miss and cache-hit passes"
    )
    for lem, e_first in ci_first.lemmas.items():
        e_second = ci_second.lemmas[lem]
        assert e_first.cand_set == e_second.cand_set, (
            f"cand_set diverged for lemma {lem!r}: "
            f"first={sorted(e_first.cand_set)[:5]}..., "
            f"second={sorted(e_second.cand_set)[:5]}..."
        )
        assert e_first.accepted_level == e_second.accepted_level, (
            f"accepted_level diverged for lemma {lem!r}"
        )
