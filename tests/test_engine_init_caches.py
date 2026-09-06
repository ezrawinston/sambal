"""Config-keyed start-up caches: miss, hit, re-key, corruption.

Four values are cached on disk under the Augmenter's cache directory:
``allowed_lemmas.pkl``, ``ctx_buckets.pkl``, ``fast_caches.pkl`` and
``filtered_pools.json``. For each of them these tests check the same four
properties:

  - a miss builds the value and writes the file;
  - a hit returns an object equal to what the build produced, without
    rebuilding;
  - changing a keyed config field, or touching a source file, misses and
    overwrites — the directory still holds one file per kind;
  - a corrupt file prints one line and rebuilds instead of raising.

Plus two properties of the keys themselves: they ignore ``seed``, the debug
flags and ``require_gpu``, and they ignore the directory a tree sits in. And
the off switch: with ``startup_caches=False`` nothing is read, nothing is
written and no cache directory appears, which is how the statistics collector
runs.

Everything runs on small synthetic fixtures and a bare Augmenter (built with
``__new__``, only the attributes each method reads), so the file needs no
spaCy pipeline and finishes in seconds.
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
from collections import Counter
from pathlib import Path

import pytest

from sambal.config import Config, ResourcePaths
from sambal.countability_pools import CountabilityPools
from sambal.engine import Augmenter


CACHE_DIR_NAME = "_augmenter_cache"

VOCAB_LINES = [
    "walked\t9",
    "walking\t7",
    "dogs\t5",
    "happier\t4",
    "quickly\t3",
    "London\t6",
    "rare\t1",
]

COUNTABILITY_FILES = {
    "mass.txt": "water\nsand\n",
    "count.txt": "dog\nchair\n",
    "both.txt": "stone\n",
    "pluralia_tantum.txt": "scissors\n",
}

# {lemma: {bucket_key: count}} in the shape the context statistics ship in.
CTX_STATS = {
    "dog": {("ADV", "RB", "advmod", "VERB", "VBD", "L", "X", "Y", "RB", ".", 0): 5},
    "cat": {("ADV", "RB", "advmod", "VERB", "VBD", "L", "X", "Y", "RB", ".", 0): 3},
    "chair": {("ADV", "RB", "advmod", "VERB", "VBD", "L", "X", "Y", "RB", ",", 0): 1},
}


class _StubPosPipe:
    """Stands in for the tagger-only pipeline: only its meta is keyed on."""
    meta = {"name": "stub_pos", "version": "1.2.3"}


class _StubInflect:
    def plural_noun(self, s: str) -> str:
        return s if s.endswith("s") else s + "s"


def _write_resources(root: Path) -> Path:
    """Lay out the source files the caches key on, return the directory."""
    res = root / "resources"
    res.mkdir(parents=True, exist_ok=True)
    (res / "allowed_vocab.txt").write_text("\n".join(VOCAB_LINES) + "\n",
                                           encoding="utf-8")
    for name, body in COUNTABILITY_FILES.items():
        (res / name).write_text(body, encoding="utf-8")
    (res / "npis.tsv").write_text("any\never\n", encoding="utf-8")
    (res / "function_words.txt").write_text("the\nof\n", encoding="utf-8")
    with open(res / "ctx_stats.pkl", "wb") as f:
        pickle.dump(CTX_STATS, f)
    return res


def _make_augmenter(res: Path, **cfg_overrides) -> Augmenter:
    """A bare Augmenter carrying only what the cached builders read."""
    cfg_values = dict(
        spacy_model="en_core_web_trf",
        spacy_pos_model="en_core_web_sm",
        min_vocab_freq=3,
        ctx_lemma_gate="freq",
        ctx_lemma_gate_min_count=2,
        ctx_backoff_mode="B",
        prefilter_pools_by_allowed_vocab=True,
        seed=7,
        debug=False,
        require_gpu=True,
    )
    cfg_values.update(cfg_overrides)

    aug = Augmenter.__new__(Augmenter)
    aug.cfg = Config(**cfg_values)
    aug.paths = ResourcePaths(
        noun_bucket_dir=str(res),
        allowed_vocab_path=str(res / "allowed_vocab.txt"),
        allow_all_vocab_path=None,
        npi_path=str(res / "npis.tsv"),
        function_words_path=str(res / "function_words.txt"),
        licensors_path=None,
        ctx_lemma_stats_path=str(res / "ctx_stats.pkl"),
    )
    aug._nlp_pos = _StubPosPipe()

    # Attributes the per-tag gate reads.
    aug.allowed_vocab = {line.split("\t")[0] for line in VOCAB_LINES}
    aug.allowed_lemmas = {"walk", "dog", "happy", "quickly", "chair", "stone", "sand"}
    aug.allowed_propn_lemmas = {"london"}
    aug._allowed_lemmas_active = True
    aug._npi_unigrams = {"any", "ever"}
    aug.function_words = {"the", "of"}
    aug.licensors = set()
    aug._inflect_cache = {}
    aug._cache_probe_stats = {}
    aug.inflect_engine = _StubInflect()
    aug.countability = None
    aug.vn = None
    return aug


def _cache_files(res: Path) -> list:
    d = res / CACHE_DIR_NAME
    return sorted(p.name for p in d.iterdir()) if d.is_dir() else []


def _touch_source(path: Path, new_text: str) -> None:
    """Rewrite a source file so its size and mtime both move."""
    path.write_text(new_text, encoding="utf-8")
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000))


def _mark_payload(path: Path, mutate) -> None:
    """Edit a cache's payload in place, keeping its key.

    A later read that returns the edit proves the value came off disk rather
    than being rebuilt to the same answer.
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)
    mutate(obj["payload"])
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# ==========================================================================
# allowed_lemmas
# ==========================================================================

def test_allowed_lemmas_miss_writes_then_hit_returns_equal(tmp_path):
    res = _write_resources(tmp_path)

    cold = _make_augmenter(res)
    cold._build_allowed_lemmas()
    assert cold.allowed_lemmas
    assert _cache_files(res) == ["allowed_lemmas.pkl"]

    warm = _make_augmenter(res)
    warm.allowed_lemmas = None
    warm.allowed_propn_lemmas = None
    warm._build_allowed_lemmas()
    assert warm.allowed_lemmas == cold.allowed_lemmas
    assert warm.allowed_propn_lemmas == cold.allowed_propn_lemmas
    assert _cache_files(res) == ["allowed_lemmas.pkl"]

    # The next read comes off disk, not from a rebuild that agrees.
    _mark_payload(res / CACHE_DIR_NAME / "allowed_lemmas.pkl",
                  lambda p: p["allowed_lemmas"].append("cache-witness"))
    reread = _make_augmenter(res)
    reread._build_allowed_lemmas()
    assert "cache-witness" in reread.allowed_lemmas


def test_allowed_lemmas_rekeys_on_config_change(tmp_path):
    res = _write_resources(tmp_path)
    cold = _make_augmenter(res, min_vocab_freq=3)
    cold._build_allowed_lemmas()
    first = Path(res / CACHE_DIR_NAME / "allowed_lemmas.pkl").read_bytes()

    # A lower threshold lets more surfaces in, so the value must change.
    other = _make_augmenter(res, min_vocab_freq=1)
    other.allowed_vocab = {line.split("\t")[0] for line in VOCAB_LINES} | {"rare"}
    other._build_allowed_lemmas()
    assert "rare" in other.allowed_lemmas
    assert _cache_files(res) == ["allowed_lemmas.pkl"]
    assert Path(res / CACHE_DIR_NAME / "allowed_lemmas.pkl").read_bytes() != first


def test_allowed_lemmas_rekeys_when_source_touched(tmp_path):
    res = _write_resources(tmp_path)
    cold = _make_augmenter(res)
    cold._build_allowed_lemmas()
    first = Path(res / CACHE_DIR_NAME / "allowed_lemmas.pkl").read_bytes()

    _touch_source(res / "allowed_vocab.txt",
                  "\n".join(VOCAB_LINES + ["swimming\t8"]) + "\n")
    after = _make_augmenter(res)
    after.allowed_vocab = cold.allowed_vocab | {"swimming"}
    after._build_allowed_lemmas()
    assert "swim" in after.allowed_lemmas
    assert _cache_files(res) == ["allowed_lemmas.pkl"]
    assert Path(res / CACHE_DIR_NAME / "allowed_lemmas.pkl").read_bytes() != first


def test_allowed_lemmas_corrupt_file_rebuilds(tmp_path, capsys):
    res = _write_resources(tmp_path)
    cold = _make_augmenter(res)
    cold._build_allowed_lemmas()
    expected = cold.allowed_lemmas

    Path(res / CACHE_DIR_NAME / "allowed_lemmas.pkl").write_bytes(b"not a pickle")

    after = _make_augmenter(res)
    after._build_allowed_lemmas()
    assert after.allowed_lemmas == expected
    assert "cannot use allowed_lemmas.pkl" in capsys.readouterr().out
    assert _cache_files(res) == ["allowed_lemmas.pkl"]


# ==========================================================================
# ctx buckets
# ==========================================================================

def test_ctx_buckets_miss_writes_then_hit_returns_equal(tmp_path):
    res = _write_resources(tmp_path)
    stats = str(res / "ctx_stats.pkl")

    cold = _make_augmenter(res)
    built = cold._load_ctx_lemma_buckets(stats)
    assert built
    assert _cache_files(res) == ["ctx_buckets.pkl"]

    warm = _make_augmenter(res)
    assert warm._load_ctx_lemma_buckets(stats) == built
    assert _cache_files(res) == ["ctx_buckets.pkl"]

    _mark_payload(res / CACHE_DIR_NAME / "ctx_buckets.pkl",
                  lambda p: p.__setitem__(("CACHE-WITNESS",), Counter({"x": 1})))
    assert ("CACHE-WITNESS",) in _make_augmenter(res)._load_ctx_lemma_buckets(stats)


def test_ctx_buckets_rekey_on_min_count_change(tmp_path):
    res = _write_resources(tmp_path)
    stats = str(res / "ctx_stats.pkl")

    strict = _make_augmenter(res, ctx_lemma_gate_min_count=2)
    strict_table = strict._load_ctx_lemma_buckets(stats)

    loose = _make_augmenter(res, ctx_lemma_gate_min_count=1)
    loose_table = loose._load_ctx_lemma_buckets(stats)

    # The count-1 lemma survives only under the looser threshold.
    assert loose_table != strict_table
    assert _cache_files(res) == ["ctx_buckets.pkl"]
    # ... and going back to the strict config rebuilds rather than reusing.
    assert _make_augmenter(res)._load_ctx_lemma_buckets(stats) == strict_table


def test_ctx_buckets_rekey_when_source_touched(tmp_path):
    res = _write_resources(tmp_path)
    stats_path = res / "ctx_stats.pkl"
    stats = str(stats_path)

    cold = _make_augmenter(res)
    before = cold._load_ctx_lemma_buckets(stats)

    grown = dict(CTX_STATS)
    grown["horse"] = dict(CTX_STATS["dog"])
    with open(stats_path, "wb") as f:
        pickle.dump(grown, f)
    st = stats_path.stat()
    os.utime(stats_path, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000))

    after = _make_augmenter(res)._load_ctx_lemma_buckets(stats)
    assert after != before
    assert _cache_files(res) == ["ctx_buckets.pkl"]


def test_ctx_buckets_corrupt_file_rebuilds(tmp_path, capsys):
    res = _write_resources(tmp_path)
    stats = str(res / "ctx_stats.pkl")

    expected = _make_augmenter(res)._load_ctx_lemma_buckets(stats)
    Path(res / CACHE_DIR_NAME / "ctx_buckets.pkl").write_bytes(b"\x80\x04 broken")

    assert _make_augmenter(res)._load_ctx_lemma_buckets(stats) == expected
    assert "cannot use ctx_buckets.pkl" in capsys.readouterr().out
    assert _cache_files(res) == ["ctx_buckets.pkl"]


# ==========================================================================
# filtered countability pools
# ==========================================================================

def _pools_snapshot(pools: CountabilityPools) -> dict:
    return {f: set(getattr(pools, f)) for f in
            ("mass", "count", "both", "plural_only",
             "mass_capable", "count_capable")}


def _drop_short(words):
    """Stands in for the single-token filter: deterministic and cheap."""
    return {w for w in words if len(w) > 4}


def test_filtered_pools_miss_writes_then_hit_returns_equal(tmp_path):
    res = _write_resources(tmp_path)

    cold = _make_augmenter(res)
    cold.countability = CountabilityPools.from_dir(str(res))
    assert cold._apply_filtered_countability_pools(_drop_short) is False
    built = _pools_snapshot(cold.countability)
    assert _cache_files(res) == ["filtered_pools.json"]

    warm = _make_augmenter(res)
    warm.countability = CountabilityPools.from_dir(str(res))

    def _must_not_run(words):
        raise AssertionError("a cache hit must not re-run the filter")

    assert warm._apply_filtered_countability_pools(_must_not_run) is True
    assert _pools_snapshot(warm.countability) == built
    assert _cache_files(res) == ["filtered_pools.json"]


def test_filtered_pools_rekey_on_pos_model_change(tmp_path):
    res = _write_resources(tmp_path)
    cold = _make_augmenter(res)
    cold.countability = CountabilityPools.from_dir(str(res))
    cold._apply_filtered_countability_pools(_drop_short)
    first = Path(res / CACHE_DIR_NAME / "filtered_pools.json").read_bytes()

    # The pools are keyed on the POS pipe, which is what the filter runs on —
    # not on the main model, which never sees them.
    other = _make_augmenter(res, spacy_pos_model="other_pos_model")
    other.countability = CountabilityPools.from_dir(str(res))
    assert other._apply_filtered_countability_pools(_drop_short) is False
    assert _cache_files(res) == ["filtered_pools.json"]

    same_key_other_main_model = _make_augmenter(res, spacy_model="some_other_main")
    same_key_other_main_model.countability = CountabilityPools.from_dir(str(res))
    key_a = _make_augmenter(res)._filtered_pools_cache_key()
    assert same_key_other_main_model._filtered_pools_cache_key() == key_a
    assert first  # the first write happened


def test_filtered_pools_rekey_when_source_touched(tmp_path):
    res = _write_resources(tmp_path)
    cold = _make_augmenter(res)
    cold.countability = CountabilityPools.from_dir(str(res))
    cold._apply_filtered_countability_pools(_drop_short)

    _touch_source(res / "count.txt", "dog\nchair\nrabbit\n")

    after = _make_augmenter(res)
    after.countability = CountabilityPools.from_dir(str(res))
    assert after._apply_filtered_countability_pools(_drop_short) is False
    assert "rabbit" in after.countability.count
    assert _cache_files(res) == ["filtered_pools.json"]


def test_filtered_pools_corrupt_file_rebuilds(tmp_path, capsys):
    res = _write_resources(tmp_path)
    cold = _make_augmenter(res)
    cold.countability = CountabilityPools.from_dir(str(res))
    cold._apply_filtered_countability_pools(_drop_short)
    expected = _pools_snapshot(cold.countability)

    Path(res / CACHE_DIR_NAME / "filtered_pools.json").write_text(
        "{ not json", encoding="utf-8")

    after = _make_augmenter(res)
    after.countability = CountabilityPools.from_dir(str(res))
    assert after._apply_filtered_countability_pools(_drop_short) is False
    assert _pools_snapshot(after.countability) == expected
    assert "cannot use filtered_pools.json" in capsys.readouterr().out
    assert _cache_files(res) == ["filtered_pools.json"]


# ==========================================================================
# warmed per-tag pools
# ==========================================================================

WARM_TAGS = ("NN", "NNS", "VB", "JJ")


class _StubVerbNet:
    """Just the attribute `core._broad_verb_pool` reads."""
    def __init__(self, lemmas):
        self._index = {("trans", None, "-"): set(lemmas)}


def _with_pools(aug, res: Path, verbs=("walk", "run")):
    """Give the stub the two things the broad half of the warm reads.

    Without these the broad frozensets are empty for every tag and the round
    trip only ever covers the allowed half.
    """
    aug.countability = CountabilityPools.from_dir(str(res))
    aug.vn = _StubVerbNet(verbs)
    return aug


def _install(aug, tags=WARM_TAGS, **kwargs):
    from sambal.fast_caches import install_fast_caches
    install_fast_caches(aug, eager=True, common_tags=tags, **kwargs)


def _tag_pools(aug, tags=WARM_TAGS) -> dict:
    return {t: (aug._allowed_by_tag(t), aug._allowed_broad_by_tag(t)) for t in tags}


def test_tag_pools_miss_writes_then_hit_returns_equal(tmp_path):
    res = _write_resources(tmp_path)

    cold = _make_augmenter(res)
    _install(cold)
    built = _tag_pools(cold)
    assert any(pools[0] for pools in built.values())
    assert _cache_files(res) == ["fast_caches.pkl"]

    assert cold._inflect_cache, "the warm fills the inflection cache"

    warm = _make_augmenter(res)
    _install(warm)
    assert _tag_pools(warm) == built
    for tag, (allowed, broad) in _tag_pools(warm).items():
        assert isinstance(allowed, frozenset)
        assert isinstance(broad, frozenset)
    assert _cache_files(res) == ["fast_caches.pkl"]
    # A hit skips the warm, so the inflection cache stays cold — see the note
    # on `_warm_or_load`.
    assert warm._inflect_cache == {}

    _mark_payload(res / CACHE_DIR_NAME / "fast_caches.pkl",
                  lambda p: p["allowed"].__setitem__(
                      "NN", frozenset(p["allowed"]["NN"] | {"cache-witness"})))
    reread = _make_augmenter(res)
    _install(reread)
    assert "cache-witness" in reread._allowed_by_tag("NN")


def test_tag_pools_rekey_on_config_change(tmp_path):
    res = _write_resources(tmp_path)
    cold = _make_augmenter(res)
    _install(cold)
    first = Path(res / CACHE_DIR_NAME / "fast_caches.pkl").read_bytes()

    other = _make_augmenter(res, prefilter_pools_by_allowed_vocab=False)
    _install(other)
    assert _cache_files(res) == ["fast_caches.pkl"]
    assert Path(res / CACHE_DIR_NAME / "fast_caches.pkl").read_bytes() != first

    # The warmed tag list is part of the key, so a different list re-keys too.
    fewer = _make_augmenter(res)
    assert fewer._fast_caches_cache_key(("NN",)) != fewer._fast_caches_cache_key(WARM_TAGS)


def test_tag_pools_rekey_when_source_touched(tmp_path):
    res = _write_resources(tmp_path)
    cold = _make_augmenter(res)
    _install(cold)
    before = cold._fast_caches_cache_key(WARM_TAGS)

    _touch_source(res / "function_words.txt", "the\nof\nand\n")
    after = _make_augmenter(res)
    assert after._fast_caches_cache_key(WARM_TAGS) != before

    _install(after)
    assert _cache_files(res) == ["fast_caches.pkl"]


def test_tag_pools_corrupt_file_rebuilds(tmp_path, capsys):
    res = _write_resources(tmp_path)
    cold = _make_augmenter(res)
    _install(cold)
    expected = _tag_pools(cold)

    Path(res / CACHE_DIR_NAME / "fast_caches.pkl").write_bytes(b"garbage")

    after = _make_augmenter(res)
    _install(after)
    assert _tag_pools(after) == expected
    assert "cannot use fast_caches.pkl" in capsys.readouterr().out
    assert _cache_files(res) == ["fast_caches.pkl"]


def test_tag_pools_round_trip_covers_the_broad_half(tmp_path):
    res = _write_resources(tmp_path)

    cold = _with_pools(_make_augmenter(res), res)
    _install(cold)
    built = _tag_pools(cold)
    # The broad pools are only defined for noun-like and verb-like tags.
    assert built["NN"][1] and built["NNS"][1] and built["VB"][1]
    assert built["JJ"][1] == frozenset()
    assert built["NN"][1] <= built["NN"][0]

    warm = _with_pools(_make_augmenter(res), res)
    _install(warm)
    assert _tag_pools(warm) == built
    assert warm._inflect_cache == {}

    _mark_payload(res / CACHE_DIR_NAME / "fast_caches.pkl",
                  lambda p: p["broad"].__setitem__(
                      "NN", frozenset(p["broad"]["NN"] | {"broad-witness"})))
    reread = _with_pools(_make_augmenter(res), res)
    _install(reread)
    assert "broad-witness" in reread._allowed_broad_by_tag("NN")
    assert _cache_files(res) == ["fast_caches.pkl"]


def test_tag_pools_rekey_on_in_memory_pool_content(tmp_path):
    """The prefilter that produces these pools swallows its own exceptions, so
    the key digests the pools themselves and not only the files behind them."""
    res = _write_resources(tmp_path)
    base = _with_pools(_make_augmenter(res), res)
    base_key = base._fast_caches_cache_key(WARM_TAGS)

    for mutate in (
        lambda a: a.allowed_lemmas.discard("dog"),
        lambda a: a.allowed_propn_lemmas.add("paris"),
        lambda a: a.allowed_vocab.add("swimming"),
        lambda a: a.countability.count.discard("chair"),   # broad noun, singular
        lambda a: a.countability.plural_only.add("trousers"),  # broad noun, plural
        lambda a: a.vn._index[("trans", None, "-")].add("jump"),  # broad verb
    ):
        other = _with_pools(_make_augmenter(res), res)
        mutate(other)
        assert other._fast_caches_cache_key(WARM_TAGS) != base_key


# ==========================================================================
# how a damaged or foreign cache is handled
# ==========================================================================

def test_payload_missing_a_field_rebuilds(tmp_path, capsys):
    res = _write_resources(tmp_path)

    cold = _make_augmenter(res)
    cold._build_allowed_lemmas()
    expected = cold.allowed_lemmas
    _mark_payload(res / CACHE_DIR_NAME / "allowed_lemmas.pkl",
                  lambda p: p.pop("allowed_propn_lemmas"))

    after = _make_augmenter(res)
    after._build_allowed_lemmas()
    assert after.allowed_lemmas == expected
    out = capsys.readouterr().out
    assert "cannot use allowed_lemmas.pkl" in out

    pools = _make_augmenter(res)
    pools.countability = CountabilityPools.from_dir(str(res))
    pools._apply_filtered_countability_pools(_drop_short)
    path = res / CACHE_DIR_NAME / "filtered_pools.json"
    obj = json.loads(path.read_text(encoding="utf-8"))
    obj["payload"].pop("mass_capable")
    path.write_text(json.dumps(obj), encoding="utf-8")
    capsys.readouterr()

    again = _make_augmenter(res)
    again.countability = CountabilityPools.from_dir(str(res))
    # A missing field must rebuild, not silently install an empty pool.
    assert again._apply_filtered_countability_pools(_drop_short) is False
    assert again.countability.mass_capable
    assert "cannot use filtered_pools.json" in capsys.readouterr().out


def test_key_mismatch_says_why_it_is_rebuilding(tmp_path, capsys):
    res = _write_resources(tmp_path)
    _make_augmenter(res, min_vocab_freq=3)._build_allowed_lemmas()
    capsys.readouterr()

    other = _make_augmenter(res, min_vocab_freq=1)
    other._build_allowed_lemmas()
    out = capsys.readouterr().out
    assert "allowed_lemmas.pkl was built for a different config" in out


def test_write_sweeps_temporary_files_a_killed_writer_left(tmp_path):
    res = _write_resources(tmp_path)
    cold = _make_augmenter(res)
    cold._build_allowed_lemmas()

    stale = res / CACHE_DIR_NAME / "_tmp.allowed_lemmas.pkl.abcdef.tmp"
    stale.write_bytes(b"half a pickle")
    assert stale.name in _cache_files(res)

    _touch_source(res / "allowed_vocab.txt",
                  "\n".join(VOCAB_LINES + ["running\t8"]) + "\n")
    after = _make_augmenter(res)
    after.allowed_vocab = cold.allowed_vocab | {"running"}
    after._build_allowed_lemmas()

    assert not stale.exists()
    assert _cache_files(res) == ["allowed_lemmas.pkl"]
    # mkstemp opens 0600; a shared cache directory needs a readable file.
    mode = (res / CACHE_DIR_NAME / "allowed_lemmas.pkl").stat().st_mode & 0o777
    assert mode == 0o644


# ==========================================================================
# properties of the keys
# ==========================================================================

def _all_keys(aug) -> tuple:
    _with_pools(aug, Path(aug.paths.noun_bucket_dir))
    return (
        aug._allowed_lemmas_cache_key(),
        aug._ctx_buckets_cache_key(aug.paths.ctx_lemma_stats_path),
        aug._filtered_pools_cache_key(),
        aug._fast_caches_cache_key(WARM_TAGS),
    )


def test_keys_ignore_seed_debug_and_gpu_flags(tmp_path):
    res = _write_resources(tmp_path)
    base = _all_keys(_make_augmenter(res))
    varied = _all_keys(_make_augmenter(
        res, seed=1234, debug=True, roundtrip_debug=True, debug_vn=True,
        require_gpu=False))
    assert varied == base


def test_keys_ignore_the_directory_a_tree_sits_in(tmp_path):
    res_a = _write_resources(tmp_path / "a")
    res_b = tmp_path / "b" / "resources"
    res_b.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(res_a, res_b)  # copytree keeps size and mtime

    assert _all_keys(_make_augmenter(res_a)) == _all_keys(_make_augmenter(res_b))


def test_cache_directory_holds_one_file_per_kind(tmp_path):
    res = _write_resources(tmp_path)
    stats = str(res / "ctx_stats.pkl")

    for _ in range(2):
        aug = _make_augmenter(res)
        aug.countability = CountabilityPools.from_dir(str(res))
        aug._build_allowed_lemmas()
        aug._load_ctx_lemma_buckets(stats)
        aug._apply_filtered_countability_pools(_drop_short)
        _install(aug)

    assert _cache_files(res) == [
        "allowed_lemmas.pkl", "ctx_buckets.pkl",
        "fast_caches.pkl", "filtered_pools.json",
    ]


# ==========================================================================
# startup_caches=False: nothing read, nothing written
# ==========================================================================

ALL_FOUR = ["allowed_lemmas.pkl", "ctx_buckets.pkl",
            "fast_caches.pkl", "filtered_pools.json"]


def _exercise_all_four(aug, res: Path) -> None:
    _with_pools(aug, res)
    aug._build_allowed_lemmas()
    aug._load_ctx_lemma_buckets(str(res / "ctx_stats.pkl"))
    aug._apply_filtered_countability_pools(_drop_short)
    _install(aug)


def test_startup_caches_off_writes_nothing_on_a_clean_tree(tmp_path):
    res = _write_resources(tmp_path)
    off = _make_augmenter(res, startup_caches=False)
    _exercise_all_four(off, res)
    assert off.allowed_lemmas
    assert any(pools[0] for pools in _tag_pools(off).values())
    assert not (res / CACHE_DIR_NAME).exists()


def test_startup_caches_off_reads_nothing_and_leaves_files_alone(tmp_path):
    res = _write_resources(tmp_path)
    on = _make_augmenter(res)
    _exercise_all_four(on, res)
    assert _cache_files(res) == ALL_FOUR
    _mark_payload(res / CACHE_DIR_NAME / "allowed_lemmas.pkl",
                  lambda p: p["allowed_lemmas"].append("cache-witness"))
    before = {n: (res / CACHE_DIR_NAME / n).stat().st_mtime_ns for n in ALL_FOUR}

    off = _make_augmenter(res, startup_caches=False)
    _exercise_all_four(off, res)
    assert "cache-witness" not in off.allowed_lemmas
    assert _cache_files(res) == ALL_FOUR
    assert {n: (res / CACHE_DIR_NAME / n).stat().st_mtime_ns for n in ALL_FOUR} == before


def test_stats_collector_runtime_turns_caching_and_the_context_gate_off():
    from sambal.stats import build_runtime
    _, _, cfg = build_runtime(None)
    assert cfg.startup_caches is False
    assert cfg.ctx_lemma_gate == "off"
