"""Cold start vs warm start of the real engine: same values, same output.

[`test_engine_init_caches.py`](test_engine_init_caches.py) exercises the cache
mechanism on stub augmenters. This file builds the real engine twice against one
cache directory — first with the directory empty, then with the four files the
first build wrote — and asserts that the second engine holds the same values,
augments a fixed sentence set to the same bytes, and rewrote none of the files.

Both engines are built in ONE process, on purpose. The pools, the allowed-lemma
sets and the per-tag frozensets are sets; their iteration order is a per-process
hash artifact (``PYTHONHASHSEED``) and it reaches the sampler through
``tuple(dict.fromkeys(cands))`` in ``engine._sample_from_candidates_inner``.
Across two processes the augmented text agrees only when the hash seed is
pinned, while the values themselves agree either way. One process pins the seed
by construction, so the text comparison tests the caches and not the
environment. The value comparisons are order-independent digests, which hold
across processes as well.

The caches live in ``<noun_bucket_dir>/_augmenter_cache/``, so ``noun_bucket_dir``
points at a temporary directory holding copies of the four countability lists.
Cache keys carry file basenames, not directories, so the cached values are the
ones the package resources produce.

Slow — the cold init, the eager warm and the second init together run into
several minutes. Opt in with:

    SAMBAL_LONG_TESTS=1 pytest tests/test_engine_cache_identity.py
"""
from __future__ import annotations

import gc
import hashlib
import os
import random
import shutil
from pathlib import Path

import pytest

from sambal.fast_caches import _COMMON_PTB_TAGS, install_fast_caches


LONG_TESTS_ENV = "SAMBAL_LONG_TESTS"

CACHE_DIR_NAME = "_augmenter_cache"
CACHE_FILES = ("allowed_lemmas.pkl", "ctx_buckets.pkl", "fast_caches.pkl",
               "filtered_pools.json")
COUNTABILITY_FILES = ("mass.txt", "count.txt", "both.txt", "pluralia_tantum.txt")
COUNTABILITY_POOLS = ("mass", "count", "both", "plural_only",
                      "mass_capable", "count_capable")

SEED = 42

# Written here rather than read from any corpus.
SENTENCES = [
    "The young researcher finished the difficult experiment in the cold laboratory.",
    "A large company bought the small factory near the river last spring.",
    "Several children played in the garden while their parents prepared dinner.",
    "The old teacher explained that every student should read the chapter.",
    "Heavy rain damaged the bridge, so the drivers found another road.",
    "She wrote a long letter to her brother about the strange city.",
    "The doctor asked the patient to describe the pain in her shoulder.",
    "Two engineers repaired the broken engine before the morning shift began.",
    "A quiet street runs behind the market, and few cars ever use it.",
    "The farmer sold most of the wheat and kept the rest for winter.",
    "His younger sister collects old coins and keeps them in a wooden box.",
    "The committee rejected the first proposal but accepted a shorter version.",
    "Cold weather arrived early, and the lake froze before the end of November.",
    "The librarian found the missing book under a pile of newspapers.",
    "Workers painted the fence green and repaired the gate the same afternoon.",
    "A strong wind blew the papers off the table and into the yard.",
    "The captain ordered the crew to lower the sails before the storm.",
    "My neighbour grows tomatoes and gives most of them to his friends.",
    "The lawyer read the contract twice and asked about a single clause.",
    "Students gathered outside the hall and waited for the results.",
    "An old bridge crosses the narrow valley south of the village.",
    "The cook added salt to the soup and tasted it again.",
    "Rescue teams searched the forest for three days without finding anything.",
    "The museum opened a new room for paintings from the last century.",
]


def _resources_dir():
    """The package's resources directory, or an override."""
    pkg_res = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "sambal", "resources",
    )
    for path in (os.environ.get("SAMBAL_TEST_RESOURCES_DIR"), pkg_res):
        if path and os.path.isdir(path):
            return path
    return None


SKIP_REASON = (
    f"long test: set {LONG_TESTS_ENV}=1, and the resources dir plus "
    f"en_core_web_trf must be available"
)

pytestmark = pytest.mark.skipif(
    os.environ.get(LONG_TESTS_ENV) != "1" or _resources_dir() is None,
    reason=SKIP_REASON,
)


def _build_augmenter(noun_bucket_dir):
    """Real Augmenter on the package resources, with the caches in
    ``noun_bucket_dir``. Returns None when the model is unavailable.

    Same construction as ``test_core_fast_parity._make_real_augmenter``, which
    spells out the settings of the ``icml2026`` profile.
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
        noun_bucket_dir=str(noun_bucket_dir),
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
        seed=SEED, replace_propn=True, roundtrip_debug=False, debug=False,
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


def _digest(lines) -> str:
    """Order-independent digest of a collection of strings."""
    h = hashlib.sha256()
    for line in sorted(lines):
        h.update(line.encode("utf-8", "surrogatepass"))
        h.update(b"\x00")
    return h.hexdigest()


def _set_digest(items) -> str:
    return _digest(str(x) for x in (items or ()))


def _ctx_table_digest(table) -> str:
    """Digest of the context bucket table: keys and per-bucket counters."""
    return _digest(
        repr(key) + "\t" + repr(sorted(counter.items()))
        for key, counter in (table or {}).items()
    )


def _augment_all(aug):
    """Augment every sentence, reseeding the module RNG per sentence so the
    two engines sample from the same stream."""
    out = []
    for i, sentence in enumerate(SENTENCES):
        random.seed(SEED + i)
        out.append(aug.augment(sentence))
    return out


def _snapshot(aug) -> dict:
    from sambal.core import _broad_verb_pool

    return {
        "allowed_lemmas_n": len(aug.allowed_lemmas or ()),
        "allowed_lemmas": _set_digest(aug.allowed_lemmas),
        "allowed_propn_lemmas_n": len(aug.allowed_propn_lemmas or ()),
        "allowed_propn_lemmas": _set_digest(aug.allowed_propn_lemmas),
        "countability_n": {f: len(getattr(aug.countability, f))
                           for f in COUNTABILITY_POOLS},
        "countability": {f: _set_digest(getattr(aug.countability, f))
                         for f in COUNTABILITY_POOLS},
        "ctx_buckets_n": len(aug._ctx_bucket2lemma or ()),
        "ctx_buckets": _ctx_table_digest(aug._ctx_bucket2lemma),
        "broad_verb_pool_n": len(_broad_verb_pool(aug) or ()),
        "broad_verb_pool": _set_digest(_broad_verb_pool(aug)),
        "tag_allowed": {t: _set_digest(aug._allowed_by_tag(t))
                        for t in _COMMON_PTB_TAGS},
        "tag_allowed_n": {t: len(aug._allowed_by_tag(t) or ())
                          for t in _COMMON_PTB_TAGS},
        "tag_broad": {t: _set_digest(aug._allowed_broad_by_tag(t))
                      for t in _COMMON_PTB_TAGS},
        "tag_broad_n": {t: len(aug._allowed_broad_by_tag(t) or ())
                        for t in _COMMON_PTB_TAGS},
        "outputs": _augment_all(aug),
    }


def _cache_state(cache_dir: Path) -> dict:
    if not cache_dir.is_dir():
        return {}
    return {p.name: (p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(cache_dir.iterdir()) if p.is_file()}


@pytest.fixture(scope="module")
def cold_and_warm(tmp_path_factory):
    """Build the engine twice against one empty-then-populated cache directory.

    The cold engine is dropped before the warm one is built, so only one spaCy
    pipeline is resident at a time.
    """
    bucket_dir = tmp_path_factory.mktemp("engine_cache_identity")
    res = Path(_resources_dir())
    for name in COUNTABILITY_FILES:
        shutil.copy2(res / name, bucket_dir / name)
    cache_dir = bucket_dir / CACHE_DIR_NAME
    assert not cache_dir.exists(), "the cold pass needs an empty cache directory"

    cold_aug = _build_augmenter(bucket_dir)
    if cold_aug is None:
        pytest.skip(SKIP_REASON)
    install_fast_caches(cold_aug, eager=True)
    cold = _snapshot(cold_aug)
    after_cold = _cache_state(cache_dir)
    del cold_aug
    gc.collect()

    warm_aug = _build_augmenter(bucket_dir)
    assert warm_aug is not None, "the warm pass could not build the engine"
    install_fast_caches(warm_aug, eager=True)
    warm = _snapshot(warm_aug)
    after_warm = _cache_state(cache_dir)
    del warm_aug
    gc.collect()

    return {"cold": cold, "warm": warm,
            "after_cold": after_cold, "after_warm": after_warm}


def test_lexicons_and_pools_identical(cold_and_warm):
    """Allowed lemmas, countability pools, the context bucket table and the
    broad verb pool hold the same values after a warm start."""
    cold, warm = cold_and_warm["cold"], cold_and_warm["warm"]

    for field in ("allowed_lemmas", "allowed_propn_lemmas", "ctx_buckets",
                  "broad_verb_pool"):
        assert cold[field + "_n"] == warm[field + "_n"], (
            f"{field}: {cold[field + '_n']} entries cold, "
            f"{warm[field + '_n']} warm"
        )
        assert cold[field] == warm[field], f"{field}: content digest differs"

    for pool in COUNTABILITY_POOLS:
        assert cold["countability_n"][pool] == warm["countability_n"][pool], (
            f"countability.{pool}: {cold['countability_n'][pool]} entries cold, "
            f"{warm['countability_n'][pool]} warm"
        )
        assert cold["countability"][pool] == warm["countability"][pool], (
            f"countability.{pool}: content digest differs"
        )


def test_tag_pools_identical(cold_and_warm):
    """Every warmed per-tag ``allowed`` / ``broad`` frozenset survives the
    round trip through ``fast_caches.pkl``."""
    cold, warm = cold_and_warm["cold"], cold_and_warm["warm"]

    for tag in _COMMON_PTB_TAGS:
        for kind in ("tag_allowed", "tag_broad"):
            assert cold[kind + "_n"][tag] == warm[kind + "_n"][tag], (
                f"{kind}[{tag}]: {cold[kind + '_n'][tag]} entries cold, "
                f"{warm[kind + '_n'][tag]} warm"
            )
            assert cold[kind][tag] == warm[kind][tag], (
                f"{kind}[{tag}]: content digest differs"
            )


def test_augmented_text_identical(cold_and_warm):
    """Same sentences, same seed, same bytes out."""
    cold, warm = cold_and_warm["cold"], cold_and_warm["warm"]

    assert len(cold["outputs"]) == len(SENTENCES)
    for i, (a, b) in enumerate(zip(cold["outputs"], warm["outputs"])):
        assert a == b, (
            f"sentence {i} differs:\n  cold={a!r}\n  warm={b!r}\n"
            f"  input={SENTENCES[i]!r}"
        )


def test_warm_start_rewrote_no_cache_file(cold_and_warm):
    """Every key hit: the four files are as the cold pass left them."""
    after_cold = cold_and_warm["after_cold"]
    after_warm = cold_and_warm["after_warm"]

    assert sorted(after_cold) == sorted(CACHE_FILES), (
        f"cold pass left {sorted(after_cold)}"
    )
    assert after_warm == after_cold, (
        "warm start rewrote: "
        + ", ".join(sorted(n for n in set(after_cold) | set(after_warm)
                           if after_cold.get(n) != after_warm.get(n)))
    )
