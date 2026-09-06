# augmenter.py
"""
Pipeline: spaCy + VerbNet + countability lexicon + LemmInflect (strict, no fallbacks)

- Function words, licensors, NPIs: kept as-is (loaded from authoritative lists).
- MWEs: protected via spaCy PhraseMatcher (fixed list) and spaCy Matcher (STREUSLE JSONL patterns).
- Verbs: replaced only with lemmas whose VerbNet frames match a coarse signature
  derived from the dependency parse. Verb forms are realized via LemmInflect.
- Nouns: replaced only within the same countability class; number is realized via LemmInflect.
- Adjectives/Adverbs: replaced by lemmas from provided lists; degree realized via LemmInflect.
- Proper nouns: optional replacement via a provided pool.

NOTE: VerbNet is accessed via NLTK’s XML (vn.vnclass(classid)), so this works across
NLPK versions even if helpers like vn.members()/vn.frames() are missing.
"""

from __future__ import annotations

import builtins

# If 'profile' is injected by kernprof, use it.
# Otherwise, use a dummy decorator that does nothing.
if hasattr(builtins, 'profile'):
    profile = builtins.profile
else:
    def profile(func):
        return func

import json
import random
import re
import sys
from typing import Dict, Iterable, List, Tuple, Set, Any, Union
from collections import Counter
import os
from pathlib import Path

from spacy.tokens import Doc, Token
from spacy.language import Language

# Morphological realization
import inflect
import hashlib
import pickle
from functools import lru_cache

# ----------------------- Extracted modules -----------------------
from .ptb_tags import (
    ptb_tag_for_adj,
    ptb_tag_for_adv,
    ptb_tag_for_verb,
    ptb_tag_for_noun,
    force_noun_number_by_context,
)

from .token_utils import (
    VERB_TAGS,
    NOUN_TAGS,
    is_particle_dep,
    is_auxpass_dep,
    is_verb_like,
    is_noun_like,
    is_infinitival_to_as_prep,
    is_governed_prep_child,
    iter_governed_preps,
    match_case_like,
    morph_signature,
    has_expl_there_child,
)

from .verb_frame_analyzer import (
    is_passive_clause,
    extract_frame,
    extract_preps_set,
    is_interrogative_wh_token,
)

from .to_inf_handler import (
    xcomp_has_to,
    xcomp_has_prep_gap,
    adj_subject_license,
    ToInfHandler,
)

from .wordnet_helper import WordNetHelper, _wordnet_helper
from .countability_pools import CountabilityPools
from .licensor_matcher import LicensorMatcher
from .verbnet_adapter import VerbNetAdapter
from .bucket_builders import (
    _build_verb_bucket,
    _build_noun_bucket,
    _build_adj_bucket,
    _build_adv_bucket,
    _build_adv_pcomp_bucket,
    bucket_from_tokinfo,
)

from .config import ResourcePaths, Config
from .token_features import TokenFeatures
from .pronoun_utils import (
    PRON_POOLS,
    GENDER_TAGS,
    pronoun_gender,
    pron_class,
)
from .loaders import (
    load_lines,
    load_first_col,
    load_gendered_words_lexicon,
    load_fixed_mwes,
    load_mwe_patterns,
)
from .verb_frame_analyzer import verb_child_mask

# Debug infrastructure (imported from package)
from . import dbg, set_debug

from typing import Optional, FrozenSet, NamedTuple


# ----------------- Utility Functions -----------------

def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    """Remove duplicates from a sequence while preserving order."""
    seen = set()
    out: List[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


# ----------------- Config-keyed start-up caches -----------------
#
# Four expensive start-up products are kept on disk under
# `<noun_bucket_dir>/_augmenter_cache/` (see `Augmenter._cache_dir`), one fixed
# filename per kind. Every file carries the key it was built under; on a
# mismatch the value is rebuilt and the file atomically replaced, so the
# directory holds at most one file per kind and never grows.
#
# A key is a sha256 over an explicit dependency list: a per-cache
# CODE_VERSION, the config fields the value depends on, a
# `(basename, size, mtime_ns)` signature for each source file it reads, and the
# versions of the libraries whose output it embeds. Deliberately absent: `seed`,
# the debug flags, `require_gpu`, and the directory part of any path — the
# shards of one run differ in seed and may sit under different roots, and they
# must all hit the same cache.
#
# Concurrency: a batch of workers starting together can all miss. Each writes a
# private temporary file and `os.replace`s it into position, so the last writer
# wins with equivalent content and no reader ever observes a partial file. No
# locking.

_CACHE_ALLOWED_LEMMAS_FILE = "allowed_lemmas.pkl"
_CACHE_CTX_BUCKETS_FILE = "ctx_buckets.pkl"
_CACHE_FAST_CACHES_FILE = "fast_caches.pkl"
_CACHE_FILTERED_POOLS_FILE = "filtered_pools.json"

# Bump the matching CODE_VERSION whenever the code that produces a value
# changes what it produces.
_ALLOWED_LEMMAS_CODE_VERSION = "allowed_lemmas.1"
_CTX_BUCKETS_CODE_VERSION = "ctx_buckets.1"
_FAST_CACHES_CODE_VERSION = "fast_caches.1"
_FILTERED_POOLS_CODE_VERSION = "filtered_pools.1"

# The dependency list of each cache, in key order. Every entry names something
# the cached value is read out of, and can be followed to the code path that
# reads it.

# `_build_allowed_lemmas`: allowed_lemmas + allowed_propn_lemmas.
_ALLOWED_LEMMAS_DEPENDENCIES = (
    "code_version",           # _ALLOWED_LEMMAS_CODE_VERSION
    "allowed_vocab_file",     # ResourcePaths.allowed_vocab_path -> allowed_vocab
    "allow_all_vocab_file",   # ResourcePaths.allow_all_vocab_path -> allowed_vocab
    "min_vocab_freq",         # Config: the count a vocab line must reach to enter
    "lemminflect_version",    # every surface is mapped through lemminflect.getLemma,
                              # whose lemma table is library data
)

# `_load_ctx_lemma_buckets`: the finished bucket -> lemma-counter table.
_CTX_BUCKETS_DEPENDENCIES = (
    "code_version",              # covers the inversion and `_ctx_backoff_chain`,
                                 # which shapes the merged keys and reads no config
    "ctx_lemma_stats_file",      # ResourcePaths.ctx_lemma_stats_path
    "ctx_lemma_gate_min_count",  # Config: prunes lemmas below it, once on the
                                 # strict buckets and again on the merged ones
)
# Not dependencies: `ctx_lemma_gate` (the mode decides how the finished table is
# sampled, not how it is built) and the `ctx_backoff_*` acceptance thresholds
# (read per lookup in `_ctx_bucket_level_stats`, never while building).

# `_apply_filtered_countability_pools`: the countability pools after the
# single-token filter.
_FILTERED_POOLS_DEPENDENCIES = (
    "code_version",
    "countability_files",  # mass.txt, count.txt, both.txt, pluralia_tantum.txt
                           # under noun_bucket_dir, read by CountabilityPools.from_dir
    "pos_model",           # Config.spacy_pos_model — the pipe the filter runs on
    "pos_model_version",   # that pipeline's version: the filter keeps a word iff
                           # its tokenizer yields exactly one token
    "spacy_version",
)

# `fast_caches.install_fast_caches(eager=True)`: `_allowed_by_tag_cache` and
# `_allowed_broad_by_tag_cache` for the warmed tags.
_FAST_CACHES_DEPENDENCIES = (
    "code_version",
    "tags",                   # the warmed tags; the payload covers exactly these
    "allowed_vocab_file",     # allowed_vocab is the NNP/NNPS surface gate and the
                              # source of allowed_lemmas / allowed_propn_lemmas
    "allow_all_vocab_file",
    "min_vocab_freq",
    "lemminflect_version",    # allowed_lemmas, and the inflection table `_realize`
                              # asks first
    "inflect_version",        # `_realize`'s plural fallback when lemminflect is silent
    "npi_file",               # `_npi_unigrams` rejects the lemma and its realization
    "function_words_file",    # `_is_functionish_lemma`
    "licensors_file",         # `_is_functionish_lemma`
    "countability_files",     # the broad noun pool is the countability union ...
    "pos_model",              # ... as left by the single-token filter, hence the
    "pos_model_version",      #     same three entries as the pools cache
    "spacy_version",
    "prefilter_pools_by_allowed_vocab",  # Config: narrows the countability pools and
                                         # the VerbNet index before the broad pools
                                         # are formed
    # The entries above derive the value from its sources. The five below are
    # content digests of the in-memory objects the warm actually reads, taken
    # at key time. They are here because the prefilter that produces the pools
    # swallows its own exceptions, so a degraded pool would otherwise be
    # published under a key identical to a healthy one and reused. Digesting
    # the inputs makes the key a function of what was really warmed.
    "allowed_lemmas_digest",             # the universe for every non-PROPN tag
    "allowed_propn_lemmas_digest",       # half the universe for NNP / NNPS
    "allowed_vocab_digest",              # the other half, and the PROPN surface gate
    "broad_noun_pool_singular_digest",   # countability union behind NN / NNP
    "broad_noun_pool_plural_digest",     # countability union behind NNS / NNPS
    "broad_verb_pool_digest",  # the broad verb pool is the union of the VerbNet
                               # index, which ships as corpus data with no version
                               # string, so the pool itself is the dependency
)

_COUNTABILITY_SOURCE_FILES = ("mass.txt", "count.txt", "both.txt", "pluralia_tantum.txt")


def _file_signature(path) -> Optional[list]:
    """`[basename, size, mtime_ns]` for a cache key.

    The directory part is dropped on purpose: the same tree copied to another
    root must hit the same cache. A file that is absent signs as its name alone.
    """
    if not path:
        return None
    name = os.path.basename(str(path))
    try:
        st = os.stat(path)
    except OSError:
        return [name, None, None]
    return [name, st.st_size, st.st_mtime_ns]


_PACKAGE_VERSION_CACHE: Dict[str, str] = {}


def _package_version(name: str) -> str:
    """Installed version of `name`, or "unknown".

    A version that cannot be read drops a real dependency out of the key, so
    say so once per process rather than letting it pass in silence.
    """
    if name in _PACKAGE_VERSION_CACHE:
        return _PACKAGE_VERSION_CACHE[name]
    try:
        from importlib.metadata import version
        out = version(name)
    except Exception as e:
        print(f"[cache] cannot read the installed version of {name} "
              f"({type(e).__name__}); it drops out of the cache keys", flush=True)
        out = "unknown"
    _PACKAGE_VERSION_CACHE[name] = out
    return out


def _string_set_digest(items: Iterable[str]) -> str:
    """Order-independent digest of a collection of strings."""
    h = hashlib.sha256()
    for s in sorted(items):
        h.update(s.encode("utf-8", "surrogatepass"))
        h.update(b"\x00")
    return h.hexdigest()


def _keyed_cache_digest(parts: list) -> str:
    """Cache key from a dependency list. Named apart from the unrelated
    `Augmenter._cache_key` method so neither can be reached by mistake."""
    return hashlib.sha256(
        json.dumps(parts, sort_keys=True, default=repr).encode("utf-8")
    ).hexdigest()


_CACHE_TMP_PREFIX = "_tmp."
_CACHE_TMP_SUFFIX = ".tmp"


def _read_keyed_cache(path: str, key: str, *, kind: str, as_json: bool = False,
                      required: Tuple[str, ...] = ()):
    """Payload of the cache at `path` if it was built under `key`, else None.

    Every rejection says why in one line: a file that cannot be read, an
    envelope that is not the expected one, a payload that is not a dict or is
    missing a field it must carry, and a key that belongs to another config
    (which is the usual reason a start-up suddenly rebuilds for minutes).
    Nothing is used partially and nothing raises.
    """
    if not os.path.exists(path):
        return None
    try:
        if as_json:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        else:
            with open(path, "rb") as f:
                obj = pickle.load(f)
        if not isinstance(obj, dict) or "key" not in obj or "payload" not in obj:
            raise ValueError("unexpected cache envelope")
        if obj["key"] != key:
            print(f"[cache] {kind}: {os.path.basename(path)} was built for a "
                  f"different config; rebuilding", flush=True)
            return None
        payload = obj["payload"]
        if not isinstance(payload, dict):
            raise ValueError(f"payload is {type(payload).__name__}, not a dict")
        missing = [f for f in required if f not in payload]
        if missing:
            raise ValueError(f"payload is missing {missing}")
        return payload
    except Exception as e:
        print(f"[cache] {kind}: cannot use {os.path.basename(path)} "
              f"({type(e).__name__}); rebuilding", flush=True)
        return None


def _sweep_cache_tmp(cache_dir: str) -> None:
    """Drop temporary files a killed writer left behind.

    A process that dies between the write and the replace leaves one; nothing
    else ever reads them, so the only cost of keeping them would be disk.
    """
    try:
        for name in os.listdir(cache_dir):
            if name.startswith(_CACHE_TMP_PREFIX) and name.endswith(_CACHE_TMP_SUFFIX):
                try:
                    os.unlink(os.path.join(cache_dir, name))
                except OSError:
                    pass
    except OSError:
        pass


def _write_keyed_cache(path: str, key: str, payload, *, kind: str,
                       as_json: bool = False) -> None:
    """Publish `payload` under `key`, replacing whatever the file held.

    The write goes to a temporary file in the same directory, created by
    `tempfile.mkstemp` so two writers on different hosts sharing the directory
    cannot pick the same name, and lands with `os.replace`. That is what makes
    concurrent start-ups safe (see the note above).
    """
    import tempfile

    cache_dir = os.path.dirname(path) or "."
    tmp = None
    try:
        os.makedirs(cache_dir, exist_ok=True)
        _sweep_cache_tmp(cache_dir)
        fd, tmp = tempfile.mkstemp(
            dir=cache_dir,
            prefix=f"{_CACHE_TMP_PREFIX}{os.path.basename(path)}.",
            suffix=_CACHE_TMP_SUFFIX,
        )
        if as_json:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"key": key, "payload": payload}, f, ensure_ascii=False)
        else:
            with os.fdopen(fd, "wb") as f:
                pickle.dump({"key": key, "payload": payload}, f,
                            protocol=pickle.HIGHEST_PROTOCOL)
        # mkstemp opens 0600; a cache directory shared by several accounts
        # needs the mode a plain open() would have given.
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except Exception as e:
        print(f"[cache] {kind}: cannot write {os.path.basename(path)} "
              f"({type(e).__name__})", flush=True)
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


# ----------------- Augmenter -----------------


class _VerbCandsCacheEntry(NamedTuple):
    # Base candidate pool for this environment (NO special-casing / NO forced orig_lem add)
    base: FrozenSet[str]

    # Precomputed “passes gate” sets over `base` (so we don’t redo per-candidate work).
    ok_reflexive: Optional[FrozenSet[str]]
    ok_bare_intrans: Optional[FrozenSet[str]]
    ok_wn_trans: Optional[FrozenSet[str]]
    ok_wn_clause: Optional[FrozenSet[str]]
    ok_pp_heads: Optional[FrozenSet[str]]

def chunk_doc_by_words(doc, max_words):
    chunk_start = 0
    current_words = 0

    for sent in doc.sents:
        sent_len = len(sent)

        # Check if adding this sentence exceeds the limit
        if current_words + sent_len > max_words:

            # CRITICAL CHECK:
            # Only split if we already have content in the buffer.
            # If current_words is 0, it means this single sentence is massive
            # (larger than max_words). We must accept it (minimum 1 sentence rule).
            if current_words > 0:
                # 1. Yield the previous chunk
                yield doc[chunk_start : sent.start].as_doc()

                # 2. Reset for the new chunk starting with this sentence
                chunk_start = sent.start
                current_words = 0

        # Add the current sentence length to the accumulator
        # (If this single sentence was huge, current_words will now exceed max_words,
        # forcing a split on the very next loop iteration)
        current_words += sent_len

    # Yield the final chunk if anything remains
    if chunk_start < len(doc):
        yield doc[chunk_start:].as_doc()

class Augmenter:
    def __init__(self, resources: ResourcePaths, config: Config):
        import random
        import spacy
        import json
        from typing import Iterable, Dict, Set, Optional, Tuple, List
        self.paths = resources
        self.cfg = config

        # Initialize optional attributes to default values
        # (eliminates need for getattr checks throughout the code)
        self._allowed_lemmas_active = False
        self._override_given_names_active = False
        self.allowed_vocab = None
        self.allowed_lemmas = None
        self._allowed_vocab_casefold = None
        self.countability = None
        self._adj_whitelist_common = None
        self._adv_rb_whitelist = None
        self.vn = None
        self.gender_lexicon = None
        self.given_names_by_gender = None
        self.given_names_by_gender_cf = None
        self.given_names_subst_by_gender = None
        self.human_unigrams = None
        self.allowed_propn_pool = None
        self._wn_quantity_lemmas = None
        self._npi_unigrams = set()
        self.adj_lemmas = None
        self._verb_candidates_cache = {}
        self._noun_candidates_cache = {}
        self._adj_candidates_cache = {}
        self._inflect_cache = {}
        self._lemma_tag_allowed = {}
        self._pos_cache = {}
        self._ctx_bucket_level_stats_n = None
        self._ctx_bucket_level_stats_cache = {}
        self.licensor_matcher = None
        self.fixed_mwe_matcher = None
        self.mwe_matcher = None
        self._toinf_adj_lex = None
        self._toinf_verb_lex = None
        self._toinf_adj_tags = None
        self._verb_metrics = None
        self._ctx_gate_mode = "off"
        self._ctx_stats = None
        self._nlp_pos = None
        self._human_common_unigrams_nn = []
        self._human_common_unigrams_nns = []
        self._human_common_unigrams_list = []
        self.noun_lemmas = []
        self.toinf = None

        # Set global debug flag via package function
        set_debug(self.cfg.debug)

        if self.cfg.require_gpu:
            spacy.require_gpu()
        random.seed(self.cfg.seed)

        # Main pipeline (parser on; ner off) for full docs
        model_name = self.cfg.spacy_model
        try:
            self.nlp = spacy.load(model_name, disable=["ner"])

            @Language.component("set_custom_boundaries")
            def set_custom_boundaries(doc):
                for token in doc[:-1]:
                    if token.text == ";" or "\n" in token.text:
                        # Find the next non-whitespace token
                        next_idx = token.i + 1
                        while next_idx < len(doc) and doc[next_idx].is_space:
                            next_idx += 1
                        if next_idx < len(doc):
                            doc[next_idx].is_sent_start = True
                return doc

            self.nlp.add_pipe("set_custom_boundaries", before="parser")
            dbg(f"[augmenter] Loaded spaCy pipeline '{model_name}'")
        except Exception as e:
            raise RuntimeError(
                "Failed to load spaCy transformer model. "
                "Install with: pip install spacy spacy-transformers && python -m spacy download en_core_web_trf"
            ) from e

        # Lightweight tagger-only pipe for batch POS checks (prefer a small model for speed)
        pos_model_name = self.cfg.spacy_pos_model
        try:
            self._nlp_pos = spacy.load(pos_model_name, disable=["parser", "senter"])
        except Exception as e:
            raise RuntimeError(
                f"Failed to load spaCy POS model '{pos_model_name}'. "
                "Install it explicitly; fallback to the main pipeline is disabled."
            ) from e

        # ---- minimal local helpers (JSONL-aware; capitalization-based 'proper') ----
        def _case_from(tags: Optional[List[str]], form: str) -> str:
            if tags:
                for t in tags:
                    if isinstance(t, str) and t.startswith("CASE="):
                        return t.split("=", 1)[1].lower()
            # fallback if CASE tag missing
            if form.islower(): return "lower"
            if form.istitle(): return "title"
            if form.isupper(): return "upper"
            return "mixed"

        def _is_proper_like(tags: Optional[List[str]], form: str, include_proper: bool) -> bool:
            if include_proper:
                return False
            # explicit PROPER tag if you ever add it
            if tags and any(t.upper() == "PROPER" for t in tags if isinstance(t, str)):
                return True
            return _case_from(tags, form) in {"title", "upper", "mixed"}

        def _load_lemmas_maybe_jsonl(path: Optional[str], expected_pos: Optional[str], include_proper: bool) -> List[
            str]:
            """
            Load lemmas from .jsonl ({form|orth, lemma, pos, tags?}) or plain text (one per line).
            Proper-like decided ONLY by capitalization (CASE tag or surface).
            Returns lowercase lemmas.
            """
            if not path:
                return []
            if str(path).endswith(".jsonl"):
                out: List[str] = []
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            obj = json.loads(line)
                            # <-- This was the pitfall: accept 'orth' as well as 'form'
                            form = (obj.get("form") or obj.get("orth") or "").strip()
                            if not form:
                                continue
                            pos = obj.get("pos")
                            if expected_pos and pos and pos != expected_pos:
                                continue
                            tags = obj.get("tags") or []
                            if _is_proper_like(tags, form, include_proper) is True:
                                continue
                            lem = (obj.get("lemma") or form).strip().lower()
                            if lem:
                                out.append(lem)
                    return out
                except Exception:
                    # fall back to legacy loader on any parsing error
                    return [w.lower() for w in load_lines(path)]
            # legacy .txt
            return [w.lower() for w in load_lines(path)]

        def _load_humans_any(path: Optional[str], include_proper: bool) -> Tuple[Set[str], Set[str]]:
            """
            Load human nouns from JSONL (capitalization-based) or legacy .txt via _load_human_inventory.
            Returns (unigram_lemmas_lower, mwe_forms_lower).
            """
            if not path:
                return set(), set()
            if str(path).endswith(".jsonl"):
                unigrams: Set[str] = set()
                mwes: Set[str] = set()
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            obj = json.loads(line)
                            form = (obj.get("form") or obj.get("orth") or "").strip()
                            if not form:
                                continue
                            tags = obj.get("tags") or []
                            if _is_proper_like(tags, form, include_proper) is True:
                                continue
                            lem = (obj.get("lemma") or form).strip().lower()
                            if " " in form:
                                mwes.add(form.lower())
                            else:
                                unigrams.add(lem)
                    return unigrams, mwes
                except Exception:
                    return self._load_human_inventory(path)
            # legacy .txt
            return self._load_human_inventory(path)

        # --- Lexicons / lists ---
        self.function_words = set(w.lower() for w in load_lines(self.paths.function_words_path))
        self.npi = set(w.lower() for w in load_first_col(self.paths.npi_path))
        # NPIs: block only single-token items from being sampled as replacements.
        # NOTE: The NPI list may contain surface forms *or* lemmas; later we guard on both.
        self._npi_unigrams = {w for w in self.npi if w and (" " not in w)}
        self.licensors = set(
            w.lower() for w in load_lines(self.paths.licensors_path)) if self.paths.licensors_path else set()
        # Licensor patterns (token + dep). Safe even if path missing (stays None).
        if self.paths.licensor_patterns_jsonl_path:
            self.licensor_matcher = LicensorMatcher(
                self.nlp,
                self.paths.licensor_patterns_jsonl_path,
                self.npi,  # for the NPI-in-clause guard
            )

        # Human inventory (unigrams + MWEs) — capitalization-based proper filtering
        hum_include_proper = self.cfg.include_proper_humans
        self.human_unigrams, self.human_mwes = _load_humans_any(
            self.paths.human_nouns_path,
            include_proper=hum_include_proper,
        )
        self._human_unigrams_list = sorted(self.human_unigrams) if self.human_unigrams else []
        dbg(f"[augmenter] Human list loaded")

        # --- WordNet-derived quantity nouns (no hand-written lists) ---
        # Used to conservatively freeze binominal/degree quantity MWEs like:
        #   'a lot (of) NP', 'plenty (of) NP', 'a number of NP', etc.
        self._wn_quantity_lemmas: Set[str] = _wordnet_helper.get_quantity_lemmas()

        # --- Perf caches ---
        self._inflect_cache: Dict[Tuple[str, Optional[str]], str] = {}
        self.inflect_engine = inflect.engine()

        # Candidate lemma pools
        self.noun_lemmas = _load_lemmas_maybe_jsonl(
            self.paths.noun_lemmas_path,
            expected_pos="NOUN",
            include_proper=self.cfg.include_proper_nouns,
        )

        # Humans: number-specific pools (no re-filtering)
        if self.paths.human_nouns_path and str(self.paths.human_nouns_path).endswith(".jsonl"):
            self._human_common_unigrams_nn = self._human_unigrams_list
            self._human_common_unigrams_nns = self._human_unigrams_list

            # self._human_common_unigrams_list = sorted(self.human_unigrams)
            # self._human_common_unigrams_nn = []
            # self._human_common_unigrams_nns = []
            # for lem in self._human_common_unigrams_list:
            #     sg = self._realize(lem, "NN")
            #     if sg:
            #         t = self._nlp_pos(sg)[0]
            #         if t.pos_ == "NOUN" and t.tag_ == "NN":
            #             self._human_common_unigrams_nn.append(lem)
            #     pl = self._realize(lem, "NNS")
            #     if pl:
            #         t = self._nlp_pos(pl)[0]
            #         if t.pos_ == "NOUN" and t.tag_ == "NNS":
            #             self._human_common_unigrams_nns.append(lem)
        else:
            self._build_common_human_unigrams_cached()
            dbg(f"[augmenter] Building human noun cache")

        # Adjectives
        self.adj_lemmas = _load_lemmas_maybe_jsonl(
            self.paths.adj_lemmas_path,
            expected_pos="ADJ",
            include_proper=self.cfg.include_proper_adjs,
        )
        if self.paths.adj_lemmas_path and str(self.paths.adj_lemmas_path).endswith(".jsonl"):
            self._adj_whitelist_common = set(w for w in self.adj_lemmas if " " not in w)
        else:
            self._build_common_adjectives_cached()
            dbg(f"[augmenter] Building adj cache")

        # Adverbs
        self.adv_lemmas = _load_lemmas_maybe_jsonl(
            self.paths.adv_lemmas_path,
            expected_pos="ADV",
            include_proper=self.cfg.include_proper_advs,
        )
        self.adv_lemmas = [w for w in self.adv_lemmas if " " not in w]
        dbg(f"[augmenter] Adverbs filtered")


        # --- Optional: ERG/ACE-derived to-inf lexicons (delegated to ToInfHandler) ---
        # Conservative default: if a token is in a sensitive to-inf spine but not in the lexicon, freeze it.
        self.toinf = ToInfHandler(
            adj_lexicon_path=self.paths.toinf_adj_lexicon_jsonl,
            verb_lexicon_path=self.paths.toinf_verb_lexicon_jsonl,
            skip_toinf_freeze=self.cfg.skip_toinf_freeze,
        )

        if self.toinf.has_lexicons():
            dbg(f"[augmenter] Loaded to-inf lexicons: adjs={len(self.toinf.adj_lex)} verbs={len(self.toinf.verb_lex)}")


        # --- countability buckets (unchanged) ---
        self.countability = CountabilityPools.from_dir(self.paths.noun_bucket_dir, single_token_only=True)
        dbg(f"[augmenter] Countability pools loaded")

        def _batch_filter_nouns(words: Iterable[str]) -> Set[str]:
            """
            Keep single-token entries; do NOT exclude by POS/NER.
            """
            toks = [w for w in words if w]
            out: Set[str] = set()
            for w, doc_ in zip(toks, self._nlp_pos.pipe(toks, batch_size=4000)):
                if len(doc_) == 1:
                    out.add(w.lower())
            return out

        self._apply_filtered_countability_pools(_batch_filter_nouns)

        # --- Adverb whitelist (fast, one-time) ---
        self._adv_rb_whitelist: Set[str] = set()
        keep_tags = {"RB"} if self.cfg.adv_only_rb else {"RB", "RBR", "RBS"}
        for doc_ in self._nlp_pos.pipe([w for w in self.adv_lemmas if " " not in w], batch_size=4000):
            t = doc_[0]
            if t.tag_ in keep_tags:
                self._adv_rb_whitelist.add(t.text.lower())

        # --- MWEs ---
        self.fixed_mwe_matcher = load_fixed_mwes(self.paths, self.nlp)
        self.mwe_matcher = load_mwe_patterns(self.paths, self.nlp)

        # --- VerbNet (NLTK backend only) ---
        if self.paths.verbnet_source.lower() != "nltk":
            raise RuntimeError("Set verbnet_source='nltk'. Other backends are not supported here.")
        self.vn = VerbNetAdapter()

        # Build allowed vocab (case-sensitive). Each line may be "TOKEN" or "TOKEN<TAB>COUNT".
        self.allowed_vocab: Optional[Set[str]] = None
        if self.paths.allowed_vocab_path:
            av = set()
            with open(self.paths.allowed_vocab_path, "r", encoding="utf-8") as vf:
                for raw in vf:
                    raw = raw.strip()
                    if not raw or raw.startswith("#"):
                        continue
                    # take the first column (token) and ignore any count or extra columns
                    entries = raw.split("\t", 1)
                    tok = entries[0].strip()
                    count = int(entries[1].strip())
                    if tok and count>=self.cfg.min_vocab_freq:
                        av.add(tok)
            self.allowed_vocab = av
            if av:
                try:
                    samp = random.sample(list(av), k=min(12, len(av)))
                    dbg("[init] allowed_vocab sample:", samp)
                except Exception:
                    pass

        # Load allow_all_vocab (include all tokens ignoring counts)
        if self.paths.allow_all_vocab_path:
            allow_all = set()
            with open(self.paths.allow_all_vocab_path, "r", encoding="utf-8") as vf:
                for raw in vf:
                    raw = raw.strip()
                    if not raw or raw.startswith("#"):
                        continue
                    # take the first column (token) only
                    entries = raw.split("\t", 1)
                    tok = entries[0].strip()
                    if tok:
                        allow_all.add(tok)
            
            # Merge with existing allowed_vocab
            if self.allowed_vocab is None:
                self.allowed_vocab = allow_all
            else:
                self.allowed_vocab.update(allow_all)
            
                dbg(f"[init] loaded {len(allow_all)} tokens from allow_all_vocab_path")
                if allow_all:
                    try:
                        samp = random.sample(list(allow_all), k=min(12, len(allow_all)))
                        print("[init] allow_all_vocab sample:", samp)
                    except Exception:
                        pass

        # Treat a non-empty allowed_vocab as an active constraint.
        self._allowed_lemmas_active = bool(self.allowed_vocab)

        # Load proper-noun pool: prefer explicit file, else derive from allowed_vocab.
        self.allowed_propn_pool: List[str] = []
        if self.paths.propn_list_path:
            # Load from file (one token per line, preserve case as-is)
            self.allowed_propn_pool = load_lines(self.paths.propn_list_path)
        elif self.allowed_vocab:
            # Fallback: derive from allowed_vocab using heuristics
            # rule: tokens that are not all-lowercase AND whose lowercase form does not appear in allowed_vocab
            self.allowed_propn_pool = sorted(
                {w for w in self.allowed_vocab
                                if (w != w.lower())
                                and (w.lower() not in self.allowed_vocab)
                                and (w.lower() not in self._npi_unigrams)
                                and not self._is_functionish_lemma(w)})

        # Build allowed_lemmas from allowed_vocab surfaces using lemminflect.
        # This enables lemma-based vocab filtering instead of surface-based.
        self.allowed_lemmas: Optional[Set[str]] = None
        self.allowed_propn_lemmas: Optional[Set[str]] = None
        if self.allowed_vocab:
            self._build_allowed_lemmas()

        # --- Optional: init-time pool prefiltering for small allowed_lemmas ---
        # This is a permissive, cheap filter intended to reduce rejection-sampling failures.
        self._allowed_vocab_casefold: Optional[Set[str]] = None
        if self._allowed_lemmas_active and self.cfg.prefilter_pools_by_allowed_vocab:
            try:
                self._build_allowed_vocab_hint_index()
                self._prefilter_pools_by_allowed_vocab_hint()
            except Exception as e:
                dbg(f"[init] allowed_lemmas prefilter skipped due to error: {e}")


        # --- Gender-aware resources (optional) ---
        # If you provide a gender lexicon (e.g., ecmonsen/gendered_words gendered_words.json),
        # we will preserve lexical gender for gendered tokens during replacement.
        self.gender_lexicon: Dict[str, str] = {}
        if self.paths.gendered_words_json_path:
            self.gender_lexicon = load_gendered_words_lexicon(self.paths.gendered_words_json_path)

        # Given-name pools for gender-aware PROPN replacement (optional).
        # Provide one-per-line first-name lists (single token) for each gender.
        self.given_names_by_gender: Dict[str, List[str]] = {"m": [], "f": [], "n": []}
        self.given_names_by_gender_cf: Dict[str, Set[str]] = {"m": set(), "f": set(), "n": set()}

        def _is_single_token(s: str) -> bool:
            """Check if string is a single token (no spaces)."""
            return bool(s) and ' ' not in s and '\t' not in s

        def _load_name_file(pth: Optional[str]) -> List[str]:
            if not pth:
                return []
            raw = [ln.strip() for ln in load_lines(pth) if ln and ln.strip()]
            # Keep first-names only (single token). Multi-token names require span-level replacement.
            raw = [nm for nm in raw if _is_single_token(nm)]
            return _dedupe_preserve_order(raw)

        male_names = _load_name_file(self.paths.given_names_male_path)
        female_names = _load_name_file(self.paths.given_names_female_path)
        neutral_names = _load_name_file(self.paths.given_names_neutral_path)

        # Back-compat: if you only provide given_names_path (legacy), treat it as neutral.
        if (not male_names) and (not female_names) and (not neutral_names) and self.paths.given_names_path:
            neutral_names = _load_name_file(self.paths.given_names_path)

        self.given_names_by_gender["m"] = male_names
        self.given_names_by_gender["f"] = female_names
        self.given_names_by_gender["n"] = neutral_names
        self.given_names_by_gender_cf["m"] = {n.casefold() for n in male_names}
        self.given_names_by_gender_cf["f"] = {n.casefold() for n in female_names}
        self.given_names_by_gender_cf["n"] = {n.casefold() for n in neutral_names}

        # --- NEW: split name pools into recognition vs substitution ---
        # Recognition stays as-is (can be large/rare). Substitution can optionally be
        # filtered by allowed_lemmas to avoid rejection-sampling failures when vocab is small.

        # Check for override paths first: if provided, these become the definitive
        # substitution pools WITHOUT vocab gating.
        override_male = _load_name_file(self.paths.override_given_names_male_path)
        override_female = _load_name_file(self.paths.override_given_names_female_path)
        override_neutral = _load_name_file(self.paths.override_given_names_neutral_path)
        self._override_given_names_active = bool(override_male or override_female or override_neutral)

        if self._override_given_names_active:
            # Use override pools directly, ignoring original given_names_*_path for substitution
            self.given_names_subst_by_gender: Dict[str, List[str]] = {
                "m": override_male,
                "f": override_female,
                "n": override_neutral,
            }
            dbg(f"[init] using override given-name pools: m={len(override_male)} f={len(override_female)} n={len(override_neutral)}")
        else:
            self.given_names_subst_by_gender: Dict[str, List[str]] = {
                "m": list(male_names),
                "f": list(female_names),
                "n": list(neutral_names),
            }
            if (
                self.cfg.filter_substitution_names_by_allowed_vocab
                and self.allowed_vocab
            ):
                for g in ("m", "f", "n"):
                    pool = self.given_names_subst_by_gender.get(g, []) or []
                    filtered = [nm for nm in pool if (nm and (nm in self.allowed_vocab))]
                    # Be conservative: only narrow if we keep at least one option.
                    if filtered:
                        self.given_names_subst_by_gender[g] = filtered

        # --- tiny sanity log (optional) ---
        dbg(f"[pools] nouns={len(self.noun_lemmas)} adjs={len(self._adj_whitelist_common or [])} "
            f"advs={len(self.adv_lemmas)} humans={len(self._human_unigrams_list)} propn={len(self.allowed_propn_pool)}")

        # === PERFORMANCE CACHES ===
        self._verb_candidates_cache = {}  # cache key -> candidates list
        self._noun_candidates_cache = {}  # (lemma, tag) -> candidates set
        self._adj_candidates_cache = {}

        # Pre-populate POS probe cache for allowed_lemmas

        self._verb_metrics = Counter() if self.cfg.verb_metrics else None
        self._cache_probe_stats = {
            "verb": {"hits": 0, "misses": 0},
            "noun": {"hits": 0, "misses": 0},
            "adj": {"hits": 0, "misses": 0},
            "inflect": {"hits": 0, "misses": 0},
            "ctx_level": {"hits": 0, "misses": 0},
            "lemma_tag": {"hits": 0, "misses": 0},
            "pos": {"hits": 0, "misses": 0},
        }

        self._ctx_gate_mode = (self.cfg.ctx_lemma_gate or "off").lower()
        self._ctx_bucket2lemma = None

        # An enabled context gate must never degrade silently: a missing or
        # unloadable statistics file changes replacement sampling wholesale.
        stats_path = self.paths.ctx_lemma_stats_path
        if self._ctx_gate_mode != "off":
            if not stats_path:
                raise RuntimeError(
                    f"ctx_lemma_gate={self._ctx_gate_mode!r} requires "
                    "ctx_lemma_stats_path in the resource paths; set "
                    "ctx_lemma_gate to 'off' to run without context-conditioned "
                    "sampling."
                )
            try:
                self._ctx_bucket2lemma = self._load_ctx_lemma_buckets(stats_path)
                dbg(f"[ctx-gate] loaded buckets={len(self._ctx_bucket2lemma)} from {stats_path}")
            except Exception as e:
                raise RuntimeError(
                    f"ctx_lemma_gate={self._ctx_gate_mode!r} is enabled but the "
                    f"context statistics could not be loaded from {stats_path}. "
                    "Fetch the statistics file (see sambal/resources/MANIFEST.md) "
                    "or set ctx_lemma_gate to 'off'."
                ) from e

        # The WordNet verb-frame gates have no off switch; running with the
        # corpus data absent would silently make every gate permissive and
        # change the output corpus wholesale.
        if not _wordnet_helper.is_available:
            raise RuntimeError(
                "NLTK WordNet data is required (verb-frame gates). Install it "
                "with: python -c \"import nltk; nltk.download('wordnet')\""
            )


        # self.nlp_time = 0
        # self.the_rest = 0

    def _cache_probe(self, name: str, *, hit: bool) -> None:
        stats = self._cache_probe_stats.get(name)
        if not stats:
            return
        if hit:
            stats["hits"] += 1
        else:
            stats["misses"] += 1

    def _estimate_object_size_bytes(
        self,
        obj: Any,
        *,
        max_nodes: int = 200000,
        max_items_per_container: int = 2000,
    ) -> int:
        """Approximate recursive Python object size in bytes."""
        seen = set()
        stack = [obj]
        total = 0
        visited = 0

        while stack and visited < max_nodes:
            cur = stack.pop()
            oid = id(cur)
            if oid in seen:
                continue
            seen.add(oid)
            visited += 1
            try:
                total += sys.getsizeof(cur)
            except Exception:
                continue

            if isinstance(cur, dict):
                for idx, (k, v) in enumerate(cur.items()):
                    if idx >= max_items_per_container:
                        break
                    stack.append(k)
                    stack.append(v)
            elif isinstance(cur, (list, tuple, set, frozenset)):
                for idx, v in enumerate(cur):
                    if idx >= max_items_per_container:
                        break
                    stack.append(v)

        return int(total)

    def cache_diagnostics_snapshot(self) -> Dict[str, Any]:
        cache_map = {
            "v": ("_verb_candidates_cache", "verb"),
            "n": ("_noun_candidates_cache", "noun"),
            "a": ("_adj_candidates_cache", "adj"),
            "i": ("_inflect_cache", "inflect"),
            "b": ("_ctx_bucket_level_stats_cache", "ctx_level"),
            "l": ("_lemma_tag_allowed", "lemma_tag"),
            "p": ("_pos_cache", "pos"),
        }
        out: Dict[str, Any] = {"caches": {}, "total_bytes": 0}
        for short, (attr, metric_name) in cache_map.items():
            cache_obj = getattr(self, attr, {})
            if not isinstance(cache_obj, dict):
                cache_obj = {}
            entries = len(cache_obj)
            bytes_est = self._estimate_object_size_bytes(cache_obj)
            hits = int(self._cache_probe_stats.get(metric_name, {}).get("hits", 0))
            misses = int(self._cache_probe_stats.get(metric_name, {}).get("misses", 0))
            ops = hits + misses
            hit_rate = (hits / ops) if ops > 0 else 0.0
            out["caches"][short] = {
                "entries": entries,
                "bytes_est": bytes_est,
                "bytes_per_entry": (bytes_est / entries) if entries > 0 else 0.0,
                "hits": hits,
                "misses": misses,
                "ops": ops,
                "hit_rate": hit_rate,
            }
            out["total_bytes"] += bytes_est
        return out

    def _load_ctx_lemma_buckets(self, path: str) -> Dict[Tuple[Any, ...], Counter]:
        import gzip
        from collections import defaultdict

        # The finished table is what the gate reads; the statistics object it
        # was inverted from is a local. Cached under `_CTX_BUCKETS_DEPENDENCIES`.
        cache_path = self._cache_path(_CACHE_CTX_BUCKETS_FILE)
        cache_key = self._ctx_buckets_cache_key(path)
        cached = self._startup_cache_read(cache_path, cache_key, kind="ctx buckets")
        if cached is not None:
            dbg(f"[ctx] loaded {len(cached)} buckets from cache")
            return cached

        def _open(p: str):
            return gzip.open(p, "rb") if p.endswith(".gz") else open(p, "rb")

        with _open(path) as f:
            obj = pickle.load(f)

        min_c = self.cfg.ctx_lemma_gate_min_count or 1

        def _apply_min_count_and_return(base: dict) -> Dict[Tuple[Any, ...], Counter]:
            """Apply min-count pruning and return via backoff builder."""
            if min_c > 1:
                pruned = {}
                for b, c in base.items():
                    c2 = Counter({lem: cnt for lem, cnt in c.items() if cnt >= min_c})
                    if c2:
                        pruned[b] = c2
                base = pruned
            out = self._build_ctx_backoff_buckets(base)
            self._startup_cache_write(cache_path, cache_key, out, kind="ctx buckets")
            return out

        # Stats are {lemma: Counter(bucket)} — invert to {bucket: Counter(lemma)}
        if not isinstance(obj, dict):
            raise ValueError("ctx stats loaded object is not a dict")

        dbg(f"[ctx] Loading lemma-keyed stats from {path} ({len(obj)} lemmas)")

        buckets = defaultdict(Counter)
        for lemma, ctr in obj.items():
            lemma_lc = (lemma or "").lower()
            if not lemma_lc:
                continue
            for bucket_key, n in ctr.items():
                if not n or bucket_key is None:
                    continue
                buckets[bucket_key][lemma_lc] += int(n)

        return _apply_min_count_and_return(dict(buckets))

    # ----------------------- ctx-key backoff / bucket merging -----------------------

    @staticmethod
    def _norm_ctx_neighbor_tag(tag: Any) -> Any:
        """Normalize neighbor PTB tags so punctuation doesn't explode sparsity.

        We keep START/END/OTHER as-is; we map bracket tags and pure-punct tags to "PUNCT".
        """
        if not isinstance(tag, str):
            return tag
        if not tag:
            return tag
        if tag in {"START", "END", "OTHER", "NONE"}:
            return tag
        if tag == "PUNCT":
            return tag
        # PTB bracket tags: -LRB-, -RRB-, etc.
        if tag.startswith("-") and tag.endswith("-"):
            return "PUNCT"
        # If any alnum, treat as a real tag like RB/VBD/etc.
        if any(ch.isalnum() for ch in tag):
            return tag
        # Otherwise it's likely '.', ',', ':', "''", etc.
        return "PUNCT"



    def _ctx_bucket_level_stats(self, bkey: Tuple[Any, ...], lemma_ctr: Counter) -> Tuple[int, int]:
        """Return (S, L) for this bucket level after per-bucket min-count pruning.

        S = sum of counts over lemmas with count>=n in THIS bucket.
        L = number of lemmas with count>=n in THIS bucket.

        Results are cached per-bucket for the current n.
        """
        n = self.cfg.ctx_backoff_min_lemma_count or 1
        # Lazy cache init (and reset when n changes)
        if self._ctx_bucket_level_stats_n != n:
            self._ctx_bucket_level_stats_cache = {}
            self._ctx_bucket_level_stats_n = n
        cache = self._ctx_bucket_level_stats_cache
        if bkey in cache:
            self._cache_probe("ctx_level", hit=True)
            return cache[bkey]
        self._cache_probe("ctx_level", hit=False)

        if not lemma_ctr:
            cache[bkey] = (0, 0)
            return (0, 0)

        if n <= 1:
            S = int(sum(lemma_ctr.values()))
            L = int(len(lemma_ctr))
        else:
            S = 0
            L = 0
            for cnt in lemma_ctr.values():
                try:
                    c = int(cnt)
                except Exception:
                    continue
                if c >= n:
                    S += c
                    L += 1

        cache[bkey] = (S, L)
        return (S, L)
    def _ctx_backoff_chain(self, b: Tuple[Any, ...]) -> List[Tuple[Any, ...]]:
        """Return an ordered list of ctx bucket keys from strict -> looser.

        The looser keys are *merged buckets* built at load time by
        `_build_ctx_backoff_buckets`, and are only used when the strict key
        yields an empty intersection.

        We are intentionally conservative for special cases we've discussed:
        - ADV_PCOMP always keeps the governing preposition (head_prep).
        - VERB keeps frame_key/comp_kind/preps_set/child_mask, etc.
        """
        if not isinstance(b, tuple) or not b:
            return []
        kind = b[0]
        out: List[Tuple[Any, ...]] = [b]

        try:
            if kind == "ADV" and len(b) >= 11:
                # ("ADV", adv_tag, dep, head_pos, head_tag, relpos, prev_b, next_b, prev_tag, next_tag, morph_sig)
                prev_tag, next_tag = b[8], b[9]
                p_prev = self._norm_ctx_neighbor_tag(prev_tag)
                p_next = self._norm_ctx_neighbor_tag(next_tag)
                if (p_prev, p_next) != (prev_tag, next_tag):
                    out.append(b[:8] + (p_prev, p_next) + (b[10],))

                # Drop neighbor PTB tags
                out.append(b[:8] + (None, None) + (b[10],))

                # Drop neighbor coarse POS buckets too
                out.append(b[:6] + (None, None) + (None, None) + (b[10],))

                # Drop head_tag last
                out.append((b[0], b[1], b[2], b[3], None, b[5], None, None, None, None, b[10]))

            elif kind == "ADJ" and len(b) >= 14:
                # ("ADJ", tag, dep, head_pos, head_tag, relpos, prev_tag, next_tag, morph_sig,
                #  adj_has_pp, adj_has_xcomp, adj_has_ccomp, head_has_obj, head_has_aux)
                prev_tag, next_tag = b[6], b[7]
                p_prev = self._norm_ctx_neighbor_tag(prev_tag)
                p_next = self._norm_ctx_neighbor_tag(next_tag)
                if (p_prev, p_next) != (prev_tag, next_tag):
                    out.append(b[:6] + (p_prev, p_next) + b[8:])

                # Drop only next_tag first (often punctuation/phrase boundary)
                out.append(b[:7] + (None,) + b[8:])

                # Drop both neighbor PTB tags
                out.append(b[:6] + (None, None) + b[8:])

                # Drop head_tag
                out.append((b[0], b[1], b[2], b[3], None, b[5], None, None, b[8], b[9], b[10], b[11], b[12], b[13]))

                # Drop relpos (last)
                out.append((b[0], b[1], b[2], b[3], None, None, None, None, b[8], b[9], b[10], b[11], b[12], b[13]))

            elif kind == "ADV_PCOMP" and len(b) >= 9:
                # ("ADV_PCOMP", adv_tag, head_prep, adp_dep, grandhead_pos, grandhead_tag, prev_tag, next_tag, morph_sig)
                prev_tag, next_tag = b[6], b[7]
                p_prev = self._norm_ctx_neighbor_tag(prev_tag)
                p_next = self._norm_ctx_neighbor_tag(next_tag)
                if (p_prev, p_next) != (prev_tag, next_tag):
                    out.append(b[:6] + (p_prev, p_next) + (b[8],))

                # Drop neighbor PTB tags
                out.append(b[:6] + (None, None) + (b[8],))

                # Drop governing grandhead_tag (PTB) to reduce sparsity across tense/number
                out.append((b[0], b[1], b[2], b[3], b[4], None, b[6], b[7], b[8]))

                # Drop grandhead_tag + normalize neighbor tags
                out.append((b[0], b[1], b[2], b[3], b[4], None, p_prev, p_next, b[8]))

                # Drop grandhead_tag + drop neighbor tags
                out.append((b[0], b[1], b[2], b[3], b[4], None, None, None, b[8]))

                # NOTE: We intentionally do **not** drop adp_dep / grandhead_pos / grandhead_tag here.
                # Those fields are doing real work to prevent bad merges like "at first" → "at here".
                # If you later decide you want more ADV_PCOMP coverage, add additional backoff levels
                # only after verifying they don't re-introduce that error mode.

            elif kind == "VERB" and len(b) >= 18:
                # ("VERB", vtag, dep, head_pos, head_tag, head_has_obj, frame_key, frame_has_prt, preps_set,
                #  comp_kind, is_passive, has_expl_there, to_be_xcomp, x_to, x_for, x_ger, verb_has_to_aux, child_mask)
                # If this is NOT an xcomp verb, treat advmod children as low-signal for bucketing.
                # Add a backoff key that clears the adv-bit (bit 3) in child_mask.
                try:
                    dep = b[2]
                    cm = b[17]
                    if dep != 'xcomp' and isinstance(cm, int) and (cm & (1 << 3)):
                        cm2 = cm & ~(1 << 3)
                        out.append(b[:17] + (cm2,))
                except Exception:
                    pass

                out.append((b[0], b[1], b[2], b[3], None, b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15], b[16], b[17]))

        except Exception:
            # If any unexpected format sneaks in, fall back to strict only.
            return [b]

        # De-duplicate while preserving order
        seen = set()
        uniq: List[Tuple[Any, ...]] = []
        for k in out:
            if k in seen:
                continue
            seen.add(k)
            uniq.append(k)
        return uniq

    def _build_ctx_backoff_buckets(self, base: Dict[Tuple[Any, ...], Counter]) -> Dict[Tuple[Any, ...], Counter]:
        """Augment strict buckets with merged backoff buckets.

        This lets `_ctx_intersect_candidates` try progressively less specific
        *ctx keys* (but still grammar-driven) without having to regenerate stats.
        """
        if not base:
            return base

        from collections import defaultdict

        out: Dict[Tuple[Any, ...], Counter] = dict(base)
        extra: Dict[Tuple[Any, ...], Counter] = defaultdict(Counter)

        # Build merged counters for projected keys.
        for k, ctr in base.items():
            if not isinstance(k, tuple) or not k:
                continue
            # Skip if this already looks like a backoff key.
            if any(x is None for x in k):
                continue
            chain = self._ctx_backoff_chain(k)
            if len(chain) <= 1:
                continue
            for bk in chain[1:]:
                extra[bk].update(ctr)

        # Optional prune for added buckets
        min_c = self.cfg.ctx_lemma_gate_min_count or 1
        if min_c > 1:
            for bk in list(extra.keys()):
                c = extra[bk]
                c2 = Counter({lem: cnt for lem, cnt in c.items() if cnt >= min_c})
                if c2:
                    extra[bk] = c2
                else:
                    del extra[bk]

        # Merge into out (additive; do not overwrite existing counters)
        for bk, ctr in extra.items():
            if bk in out and isinstance(out[bk], Counter):
                out[bk].update(ctr)
            else:
                out[bk] = ctr
        return out

    @staticmethod
    def _bucket_from_tokinfo(tokinfo: tuple, *, schema: Dict[str, int]) -> Optional[Tuple[Any, ...]]:
        """Delegate to bucket_from_tokinfo from bucket_builders module."""
        return bucket_from_tokinfo(tokinfo, schema=schema)

    def _xcomp_flags(self, vtok: Token) -> Tuple[bool, bool, bool]:
        xcomps = [c for c in vtok.children if c.dep_ == "xcomp"]
        if not xcomps:
            return False, False, False
        xc = xcomps[0]
        has_to = any(x.lemma_.lower() == "to" and x.pos_ in {"PART", "SCONJ", "AUX"} for x in xc.subtree)
        has_for = any(x.lemma_.lower() == "for" and x.pos_ == "ADP" for x in xc.subtree)
        is_ger = (xc.tag_ == "VBG")
        return has_to, has_for, is_ger

    def _ctx_bucket_for_token(self, tok: Token, ptb_tag: str, *, feat: Optional[TokenFeatures] = None) -> Optional[Tuple[Any, ...]]:
        """Build a context bucket tuple for stats lookup.

        Uses the shared bucket builders to ensure tuple structure matches
        _bucket_from_tokinfo (the invariant for correct stats lookup).
        """
        # --- Shared helper functions ---
        # verb_child_mask is now at module level

        def _pos_bucket(t: Optional[Token]) -> str:
            if t is None:
                return "NONE"
            p = t.pos_ or ""
            if p in {"NOUN","PROPN","PRON","VERB","AUX","ADJ","ADV","ADP","PART","SCONJ","CCONJ","DET","NUM","PUNCT"}:
                return p
            return "OTHER"

        def _get_neighbor_info(t: Token) -> Tuple[str, str, str, str]:
            """Return (prev_b, next_b, prev_tag, next_tag)."""
            try:
                doc = t.doc
                prev_b = "START" if t.i == 0 else _pos_bucket(doc[t.i - 1])
                next_b = "END" if t.i + 1 >= len(doc) else _pos_bucket(doc[t.i + 1])
                prev_tag = "START" if t.i == 0 else (doc[t.i - 1].tag_ or doc[t.i - 1].pos_ or "OTHER")
                next_tag = "END" if t.i + 1 >= len(doc) else (doc[t.i + 1].tag_ or doc[t.i + 1].pos_ or "OTHER")
                return prev_b, next_b, prev_tag, next_tag
            except Exception:
                return "OTHER", "OTHER", "OTHER", "OTHER"

        def _get_morph_sig(t: Token) -> Tuple[str, str, str, str]:
            def _m1(name):
                vals = t.morph.get(name)
                return vals[0] if vals else ""
            return (_m1("Degree"), _m1("NumType"), _m1("PronType"), _m1("Polarity"))

        def _get_head_info(t: Token) -> Tuple[str, str, bool, bool]:
            """Return (head_pos, head_tag, head_has_obj, head_has_aux)."""
            head_pos = t.head.pos_ if t.head is not None else ""
            head_tag = t.head.tag_ if t.head is not None else ""
            head_has_obj = False
            head_has_aux = False
            try:
                if t.head is not None:
                    head_has_obj = any(ch.dep_ in {"dobj", "obj"} for ch in t.head.children)
                    head_has_aux = any(ch.dep_ == "aux" for ch in t.head.children)
            except Exception:
                pass
            return head_pos, head_tag, head_has_obj, head_has_aux

        def _get_relpos(t: Token) -> int:
            try:
                if t.head is not None and t.head.i != t.i:
                    return -1 if t.i < t.head.i else 1
            except Exception:
                pass
            return 0

        # --- Use pre-computed features if available ---
        tok_is_verb_like = feat.is_verb_like if feat else is_verb_like(tok)

        # === VERB ===
        if tok_is_verb_like:
            head_pos, head_tag, head_has_obj, _ = _get_head_info(tok)
            vtag = (feat.ptb_verb if feat else ptb_tag_for_verb(tok)) or ptb_tag

            if feat:
                key = feat.frame_key
                has_prt = feat.frame_has_prt
                preps_set = feat.preps_set
                comp_kind = feat.comp_kind
                is_passive = feat.is_passive
                has_expl_there = feat.has_expl_there
                to_be_xcomp = feat.is_to_be_xcomp
                x_to = feat.xcomp_has_to
                x_for = feat.xcomp_has_for
                x_ger = feat.xcomp_is_ger
            else:
                frame_key, has_prt, req_prep = extract_frame(tok)
                preps_set = tuple(sorted(extract_preps_set(tok)))
                key = self._fix_tough_xcomp_valency(tok, frame_key)
                if frame_key == "intrans" and key == "trans" and preps_set:
                    preps_set = ()
                key = self._fix_wh_gap_valency(tok, key, set(preps_set))
                if key == "intrans" and preps_set:
                    comp_like = {"like", "as", "than"} & set(preps_set)
                    if comp_like:
                        preps_set = tuple(sorted(set(preps_set) - comp_like))
                comp_kind = self._parse_comp_kind(tok, key)
                is_passive = is_passive_clause(tok)
                has_expl_there = has_expl_there_child(tok)
                to_be_xcomp = self._is_to_be_xcomp_head(tok)
                x_to, x_for, x_ger = self._xcomp_flags(tok)

            verb_has_to_aux = any(
                (ch.dep_ == "aux") and ((ch.tag_ == "TO") or ((ch.lemma_ or "").lower() == "to"))
                for ch in tok.children
            )

            return _build_verb_bucket(
                vtag=vtag,
                dep=tok.dep_ or "",
                head_pos=head_pos,
                head_tag=head_tag,
                head_has_obj=head_has_obj,
                frame_key=key,
                frame_has_prt=has_prt,
                preps_set=preps_set,
                comp_kind=comp_kind,
                is_passive=is_passive,
                has_expl_there=has_expl_there,
                to_be_xcomp=to_be_xcomp,
                xcomp_has_to=x_to,
                xcomp_has_for=x_for,
                xcomp_is_ger=x_ger,
                verb_has_to_aux=verb_has_to_aux,
                vtok_child_mask=verb_child_mask(tok),
            )

        # === NOUN ===
        if is_noun_like(tok):
            is_bare = feat.is_bare_singular if feat else self._is_bare_singular_np_head(tok)
            return _build_noun_bucket(ptb_tag, is_bare)

        # === ADJ ===
        if tok.pos_ == "ADJ":
            head_pos, head_tag, head_has_obj, head_has_aux = _get_head_info(tok)
            _, _, prev_tag, next_tag = _get_neighbor_info(tok)
            morph_sig = _get_morph_sig(tok)
            relpos = _get_relpos(tok)

            adj_has_xcomp = any(ch.dep_ == "xcomp" for ch in tok.children)
            adj_has_ccomp = any(ch.dep_ == "ccomp" for ch in tok.children)
            adj_has_pp = any(
                (ch.dep_ == "prep")
                or (ch.dep_ in {"obl", "nmod"} and any(gc.dep_ == "case" and gc.pos_ == "ADP" for gc in ch.children))
                for ch in tok.children
            )

            atag = (feat.ptb_adj if feat else ptb_tag_for_adj(tok)) or ptb_tag
            return _build_adj_bucket(
                atag=atag,
                dep=tok.dep_,
                head_pos=head_pos,
                head_tag=head_tag,
                relpos=relpos,
                prev_tag=prev_tag,
                next_tag=next_tag,
                morph_sig=morph_sig,
                adj_has_pp=adj_has_pp,
                adj_has_xcomp=adj_has_xcomp,
                adj_has_ccomp=adj_has_ccomp,
                head_has_obj=head_has_obj,
                head_has_aux=head_has_aux,
            )

        # === ADV ===
        if tok.pos_ == "ADV":
            head_pos, head_tag, _, _ = _get_head_info(tok)
            prev_b, next_b, prev_tag, next_tag = _get_neighbor_info(tok)
            morph_sig = _get_morph_sig(tok)
            relpos = _get_relpos(tok)
            adv_tag = (feat.ptb_adv if feat else ptb_tag_for_adv(tok)) or ptb_tag

            # PCOMP under ADP: specialize heavily (fix "at first" -> "at here")
            if tok.dep_ == "pcomp" and tok.head is not None and tok.head.pos_ == "ADP":
                head_prep = (tok.head.lemma_ or "").lower()
                if head_prep:
                    adp_dep = tok.head.dep_ or ""
                    grandhead_pos = tok.head.head.pos_ if tok.head.head is not None else ""
                    grandhead_tag = tok.head.head.tag_ if tok.head.head is not None else ""
                    return _build_adv_pcomp_bucket(
                        adv_tag=adv_tag,
                        head_prep=head_prep,
                        adp_dep=adp_dep,
                        grandhead_pos=grandhead_pos,
                        grandhead_tag=grandhead_tag,
                        prev_tag=prev_tag,
                        next_tag=next_tag,
                        morph_sig=morph_sig,
                    )

            return _build_adv_bucket(
                adv_tag=adv_tag,
                dep=tok.dep_,
                head_pos=head_pos,
                head_tag=head_tag,
                relpos=relpos,
                prev_b=prev_b,
                next_b=next_b,
                prev_tag=prev_tag,
                next_tag=next_tag,
                morph_sig=morph_sig,
            )

        return None

    def _ctx_intersect_candidates(
        self,
        tok: Token,
        ptb_tag: str,
        cands: Iterable[str],
        *,
        feat: Optional[TokenFeatures] = None,
    ) -> Tuple[List[str], Optional[Counter]]:
        if not self._ctx_bucket2lemma:
            return list(cands), None

        # Do not ctx-gate proper names: stats collapse PROPN lemmas, and names are handled separately.
        if tok.pos_ == "PROPN" or ptb_tag in {"NNP", "NNPS"}:
            return list(cands), None

        cands_list = list(cands)
        if len(cands_list) <= 1:
            return cands_list, None

        tok_is_verb_like = feat.is_verb_like if feat else is_verb_like(tok)
        if tok_is_verb_like:
            if self.toinf.required_verb_license(tok) is not None:
                return cands_list, None

        # cache_key = (id(tok.doc), tok.i, ptb_tag)
        # if cache_key in self._ctx_bucket_cache:
        #     b = self._ctx_bucket_cache[cache_key]
        # else:
        b = self._ctx_bucket_for_token(tok, ptb_tag, feat=feat)
        # self._ctx_bucket_cache[cache_key] = b
        dbg(f"[ctx sample] lemma: {tok.lemma_}  ptb_tag: {ptb_tag}  ctx_key: {b}")
        if b is None:
            print("======= no bucket for token!!!======")
            return [], None

        # Backoff: if the strict ctx bucket is empty (or yields no intersection),
        # progressively try merged (less-specific) buckets.
        # Strategy: return the smallest (most specific) bucket that meets the threshold,
        # or if none meet the threshold, return the largest bucket found.
        n = self.cfg.ctx_backoff_min_lemma_count or 1
        mode = (self.cfg.ctx_backoff_mode or "off").strip().lower()
        min_total = self.cfg.ctx_backoff_min_total_count or 0
        min_k = self.cfg.ctx_backoff_min_candidates or 0
        min_k_pct = self.cfg.ctx_backoff_min_candidates_pct or 0.0
        n_orig = len(cands_list)

        # Track the best fallback bucket (largest L) in case no bucket meets threshold
        best_fallback = None  # (filtered, lemma_ctr, L, S, b2)

        for b2 in self._ctx_backoff_chain(b):
            lemma_ctr = self._ctx_bucket2lemma.get(b2)
            if not lemma_ctr:
                dbg(f"[ctx backoff] skip bucket (no lemma_ctr): {b2}")
                continue

            # Per-bucket pruning stats (computed after dropping lemmas with cnt<n)
            S, L = self._ctx_bucket_level_stats(b2, lemma_ctr)
            if L <= 0 or S <= 0:
                dbg(f"[ctx backoff] skip bucket (S={S}, L={L} after pruning): {b2}")
                continue

            # Candidate intersection (also applies per-lemma min count)
            if n <= 1:
                filtered = [c for c in cands_list if c in lemma_ctr]
            else:
                filtered = [c for c in cands_list if lemma_ctr.get(c, 0) >= n]

            if not filtered:
                dbg(f"[ctx sample] lemma: {tok.lemma_}  ptb_tag: {ptb_tag}  intersection empty with key {b2}")
                continue

            # Track best fallback (largest L with non-empty intersection)
            if best_fallback is None or L > best_fallback[2]:
                best_fallback = (filtered, lemma_ctr, L, S, b2)

            # Bucket-level acceptance threshold
            # Compute effective_min_k for mode B (used in both threshold check and logging)
            effective_min_k = max(min_k, int(min_k_pct * n_orig)) if mode in {"b", "cands", "candidates", "lemmas"} else 0

            threshold_met = True
            if mode in {"a", "count", "total", "examples"}:
                if min_total > 0 and S < min_total:
                    dbg(f"[ctx backoff] skip bucket (S={S} < {min_total}): {b2}")
                    threshold_met = False
            elif mode in {"b", "cands", "candidates", "lemmas"}:
                if effective_min_k > 0 and L < effective_min_k:
                    dbg(f"[ctx backoff] skip bucket (L={L} < {effective_min_k} [abs={min_k}, pct={min_k_pct}*{n_orig}={int(min_k_pct*n_orig)}]): {b2}")
                    threshold_met = False

            if not threshold_met:
                continue

            # Log successful bucket acceptance (especially backoffs)
            def _log_bucket_accept():
                is_backoff = (b2 != b)
                if is_backoff:
                    print(f"[ctx backoff] ACCEPT backoff (L={L}, S={S}, filtered={len(filtered)}/{n_orig}): {b} -> {b2}")
                elif effective_min_k > 0:
                    print(f"[ctx backoff] ACCEPT strict (L={L} >= {effective_min_k}, filtered={len(filtered)}/{n_orig}): {b2}")
            dbg(_log_bucket_accept)

            # Apply top-p filtering after bucket acceptance
            filtered = self._apply_ctx_top_p_filter(filtered, lemma_ctr)

            if self._ctx_gate_mode == "freq":
                return filtered, Counter({c: int(lemma_ctr[c]) for c in filtered})
            return filtered, None

        # No bucket met the threshold - use best fallback if available
        if best_fallback is not None:
            filtered, lemma_ctr, L, S, b2 = best_fallback
            dbg(f"[ctx backoff] FALLBACK to largest bucket (L={L}, S={S}, filtered={len(filtered)}/{n_orig}): {b2}")

            # Apply top-p filtering after fallback selection
            filtered = self._apply_ctx_top_p_filter(filtered, lemma_ctr)

            if self._ctx_gate_mode == "freq":
                return filtered, Counter({c: int(lemma_ctr[c]) for c in filtered})
            return filtered, None

        return [], None

    def _apply_ctx_top_p_filter(self, filtered: List[str], lemma_ctr: Counter) -> List[str]:
        """Keep only top p fraction of candidates by cumulative frequency.

        After bucket selection, this filters candidates to keep only those
        contributing to the top p fraction of total frequency mass.
        Respects ctx_lemma_gate_min_count as a floor.
        """
        top_p = self.cfg.ctx_lemma_gate_top_p or 0.0
        if top_p <= 0.0 or top_p >= 1.0:
            return filtered
        if not filtered or not lemma_ctr:
            return filtered

        min_keep = self.cfg.ctx_lemma_gate_min_count or 1

        # Sort by frequency descending
        sorted_cands = sorted(filtered, key=lambda c: lemma_ctr.get(c, 0), reverse=True)

        # Calculate total and target
        total = sum(lemma_ctr.get(c, 0) for c in sorted_cands)
        if total <= 0:
            return filtered

        target = top_p * total
        cumsum = 0
        result = []

        for c in sorted_cands:
            result.append(c)
            cumsum += lemma_ctr.get(c, 0)
            # Stop once we've reached the target, but keep at least min_keep
            if cumsum >= target and len(result) >= min_keep:
                break

        # Ensure we have at least min_keep candidates (if available)
        if len(result) < min_keep:
            result = sorted_cands[:min(min_keep, len(sorted_cands))]

        if len(result) < len(filtered):
            dbg(f"[ctx top_p] filtered {len(filtered)} -> {len(result)} (top_p={top_p}, cumsum={cumsum}/{total})")

        return result

    def _realize(self, lemma_lc: str, ptb_tag: str) -> str:
        """Return an inflected surface form for a lowercase lemma and PTB tag.

        We try lemminflect first (if available), then fall back to simple heuristics.
        This is intentionally conservative: if we can't reliably inflect, we return the lemma.
        """
        lemma_lc = (lemma_lc or "").strip()
        ptb_tag = (ptb_tag or "").strip()

        if not lemma_lc:
            return lemma_lc
        if not ptb_tag:
            return lemma_lc

        # Multi-word lemmas: inflect the head (final) token only.
        if " " in lemma_lc:
            parts = lemma_lc.split()
            head = parts[-1]
            realized_head = self._realize(head, ptb_tag)
            return " ".join(parts[:-1] + [realized_head])

        key = (lemma_lc, ptb_tag)
        cached = self._inflect_cache.get(key)
        if cached is not None:
            self._cache_probe("inflect", hit=True)
            return cached
        self._cache_probe("inflect", hit=False)

        out = None

        # Preferred: lemminflect (already an explicit dependency in your environment).
        try:
            from lemminflect import getInflection  # type: ignore
            infl = getInflection(lemma_lc, tag=ptb_tag)
            if infl:
                out = infl[0]
        except Exception:
            out = None

        # Fallback heuristics (kept simple and safe).
        if not out:
            if ptb_tag in {"NN", "NNP"}:
                out = lemma_lc
            elif ptb_tag in {"NNS", "NNPS"}:
                try:
                    out = self.inflect_engine.plural_noun(lemma_lc) or None
                except Exception:
                    out = None
                if not out:
                    # naive pluralization
                    if lemma_lc.endswith(("s", "x", "z", "ch", "sh")):
                        out = lemma_lc + "es"
                    elif lemma_lc.endswith("y") and len(lemma_lc) > 1 and lemma_lc[-2] not in "aeiou":
                        out = lemma_lc[:-1] + "ies"
                    else:
                        out = lemma_lc + "s"
            elif ptb_tag in {"VB", "VBP"}:
                out = lemma_lc
            elif ptb_tag == "VBZ":
                if lemma_lc.endswith(("s", "x", "z", "ch", "sh")):
                    out = lemma_lc + "es"
                elif lemma_lc.endswith("y") and len(lemma_lc) > 1 and lemma_lc[-2] not in "aeiou":
                    out = lemma_lc[:-1] + "ies"
                else:
                    out = lemma_lc + "s"
            elif ptb_tag in {"VBD", "VBN"}:
                if lemma_lc.endswith("e"):
                    out = lemma_lc + "d"
                else:
                    out = lemma_lc + "ed"
            elif ptb_tag == "VBG":
                if lemma_lc.endswith("ie"):
                    out = lemma_lc[:-2] + "ying"
                elif lemma_lc.endswith("e") and len(lemma_lc) > 1:
                    out = lemma_lc[:-1] + "ing"
                else:
                    out = lemma_lc + "ing"
            else:
                # JJR/JJS/RB/etc: be conservative.
                out = lemma_lc

        # Cache
        try:
            self._inflect_cache[key] = out
        except Exception:
            pass
        return out

    def _candidate_allowed_for_tag(self, lemma: str, ptb_tag: str) -> bool:
        """
        Return True iff `lemma` is allowed to be sampled for `ptb_tag`.

        Lifted from the per-call closure in augment_doc so external code
        (the document-scope core) can apply the same gate.

        Enforces:
          - NPI blocklist for single-token NPIs (lemma AND realized surface).
          - Function-word lemma blocklist.
          - Optional allowed_lemmas / allowed_vocab gate (PROPN tags gate on
            realized surface).
        """
        key = (lemma, ptb_tag)
        cached = self._lemma_tag_allowed.get(key)
        if cached is not None:
            self._cache_probe("lemma_tag", hit=True)
            return cached
        self._cache_probe("lemma_tag", hit=False)

        lem_lc = (lemma or "").lower()
        npi_unigrams = self._npi_unigrams

        if lem_lc and (lem_lc in npi_unigrams):
            self._lemma_tag_allowed[key] = False
            return False

        surf = self._realize(lem_lc, ptb_tag)
        surf_lc = surf.lower() if surf else ""

        if surf_lc and (surf_lc in npi_unigrams):
            self._lemma_tag_allowed[key] = False
            return False

        if self._is_functionish_lemma(lem_lc):
            self._lemma_tag_allowed[key] = False
            return False

        if self._allowed_lemmas_active:
            if ptb_tag in {"NNP", "NNPS"}:
                ok = bool(surf and (surf in (self.allowed_vocab or set())))
            else:
                allowed = self.allowed_lemmas or set()
                ok = bool(lem_lc and (lem_lc in allowed))
        else:
            ok = True

        self._lemma_tag_allowed[key] = ok
        return ok

    # --------- Utils ---------

    def _build_allowed_lemmas(self) -> None:
        """Build lemma sets from allowed_vocab surface forms.

        Uses lemminflect.getLemma() to find possible lemmas for each surface form.

        - allowed_lemmas: lemma gate for replacements, derived from allowed_vocab
        - allowed_propn_lemmas: proper-only lemmas (non-lowercase tokens whose
          lowercase form is not present in allowed_vocab), for PROPN contexts

        The result is cached on disk under `_ALLOWED_LEMMAS_DEPENDENCIES`.
        """
        from lemminflect import getLemma

        av = self.allowed_vocab
        if not av:
            self.allowed_lemmas = None
            self.allowed_propn_lemmas = None
            return

        cache_path = self._cache_path(_CACHE_ALLOWED_LEMMAS_FILE)
        cache_key = self._allowed_lemmas_cache_key()
        cached = self._startup_cache_read(cache_path, cache_key, kind="allowed lemmas",
                                   required=("allowed_lemmas", "allowed_propn_lemmas"))
        if cached is not None:
            # The payload holds sorted sequences and both branches build their
            # sets from one, so a hit and a miss agree on iteration order too.
            self.allowed_lemmas = set(cached["allowed_lemmas"])
            self.allowed_propn_lemmas = set(cached["allowed_propn_lemmas"])
            dbg(f"[init] loaded allowed_lemmas from cache "
                f"({len(self.allowed_lemmas)} common, "
                f"{len(self.allowed_propn_lemmas)} proper)")
            return

        lemmas: Set[str] = set()
        propn_lemmas: Set[str] = set()
        upos_common = ("NOUN", "VERB", "ADJ", "ADV")
        upos_propn = ("NOUN", "PROPN")

        for surf in av:
            surf = (surf or "").strip()
            if not surf:
                continue
            surf_lc = surf.lower()
            if not surf_lc:
                continue

            is_proper_only = (surf != surf_lc) and (surf_lc not in av)

            # Track proper-only tokens separately for PROPN contexts.
            if is_proper_only:
                propn_lemmas.add(surf_lc)
                for upos in upos_propn:
                    try:
                        result = getLemma(surf_lc, upos)
                        if result:
                            propn_lemmas.update(lem.lower() for lem in result)
                    except Exception:
                        pass

            # Always include the surface itself (lowercased) as a potential lemma
            # This handles unknown words and edge cases
            lemmas.add(surf_lc)

            # Try to get lemmas for all UPOS tags
            for upos in upos_common:
                try:
                    result = getLemma(surf_lc, upos)
                    if result:
                        lemmas.update(lem.lower() for lem in result)
                except Exception:
                    pass

        lemmas_sorted = sorted(lemmas)
        propn_sorted = sorted(propn_lemmas)
        self.allowed_lemmas = set(lemmas_sorted)
        self.allowed_propn_lemmas = set(propn_sorted)

        dbg(f"[init] built allowed_lemmas: {len(av)} surfaces -> {len(lemmas)} common lemmas "
                  f"+ {len(propn_lemmas)} proper lemmas")

        self._startup_cache_write(cache_path, cache_key, {
            "allowed_lemmas": lemmas_sorted,
            "allowed_propn_lemmas": propn_sorted,
        }, kind="allowed lemmas")

    def _build_allowed_vocab_hint_index(self) -> None:
        """Build a casefolded allowed-lemma index for exact prefiltering."""
        lemmas = self.allowed_lemmas
        if lemmas is None:
            self._allowed_vocab_casefold = None
            return

        av_cf: Set[str] = {w.casefold() for w in lemmas if w}
        self._allowed_vocab_casefold = av_cf

    def _allowed_vocab_hint_ok(self, lemma: str) -> bool:
        """Exact lemma gate for init-time prefiltering."""
        if not lemma:
            return False

        av_cf = self._allowed_vocab_casefold
        if av_cf is None:
            return True
        return lemma.casefold() in av_cf

    def _prefilter_pools_by_allowed_vocab_hint(self) -> None:
        """Init-time prefilter of large candidate pools using _allowed_vocab_hint_ok()."""
        if not self._allowed_lemmas_active:
            return

        def _fset(items: Set[str]) -> Set[str]:
            return {w for w in items if w and self._allowed_vocab_hint_ok(w)}

        # Countability noun pools
        try:
            if self.countability is not None:
                before = (
                    len(self.countability.mass)
                    + len(self.countability.count)
                    + len(self.countability.both)
                    + len(self.countability.plural_only)
                )
                self.countability.filter(_fset)
                after = (
                    len(self.countability.mass)
                    + len(self.countability.count)
                    + len(self.countability.both)
                    + len(self.countability.plural_only)
                )
                dbg(f"[prefilter] countability pools: {before} -> {after}")
        except Exception:
            pass

        # Human replacement pools (respect_human path)
        human_attrs = {
            "_human_common_unigrams_list": self._human_common_unigrams_list,
            "_human_common_unigrams_nn": self._human_common_unigrams_nn,
            "_human_common_unigrams_nns": self._human_common_unigrams_nns,
        }
        for attr, lst in human_attrs.items():
            try:
                if isinstance(lst, list) and lst:
                    before = len(lst)
                    lst2 = [w for w in lst if w and self._allowed_vocab_hint_ok(w)]
                    if lst2:
                        setattr(self, attr, lst2)
                        dbg(f"[prefilter] {attr}: {before} -> {len(lst2)}")
            except Exception:
                continue

        # Adjective whitelist (common)
        try:
            aw = self._adj_whitelist_common
            if isinstance(aw, set) and aw:
                before = len(aw)
                aw2 = {w for w in aw if w and self._allowed_vocab_hint_ok(w)}
                if aw2:
                    self._adj_whitelist_common = aw2
                    # back-compat alias
                    dbg(f"[prefilter] adj whitelist: {before} -> {len(aw2)}")
        except Exception:
            pass

        # Adverb whitelist
        try:
            advw = self._adv_rb_whitelist
            if isinstance(advw, set) and advw:
                before = len(advw)
                adv2 = {w for w in advw if w and self._allowed_vocab_hint_ok(w)}
                if adv2:
                    self._adv_rb_whitelist = adv2
                    dbg(f"[prefilter] adv whitelist: {before} -> {len(adv2)}")
        except Exception:
            pass

        # Optional: filter ERG-derived to-inf lexicons too (keeps them consistent with small vocab)
        try:
            if self.toinf.has_lexicons():
                self.toinf.prefilter_by_allowed_vocab(self._allowed_vocab_hint_ok)
        except Exception:
            pass

        # VerbNet index (largest win for small vocab)
        try:
            if self.vn is not None and hasattr(self.vn, "filter_lemmas"):
                self.vn.filter_lemmas(self._allowed_vocab_hint_ok)
        except Exception:
            pass

        # Any downstream caches that depend on these pools should start empty.
        try:
            self._verb_candidates_cache.clear()
            self._noun_candidates_cache.clear()
        except Exception:
            pass

    # --------- Gender helpers ---------

    def _gender_of_lemma(self, lemma: str) -> Optional[str]:
        """Return lexical gender tag for a lemma in {m,f,n} if known."""
        if not lemma:
            return None
        lex = self.gender_lexicon or {}
        return lex.get(lemma.lower())

    def _gender_of_propn_text(self, text: str) -> Optional[str]:
        """Return gender tag for a first-name PROPN if it matches provided name lists."""
        if not text:
            return None
        pools = self.given_names_by_gender_cf
        if not pools:
            return None
        key = text.casefold()
        if key in pools.get("m", set()):
            return "m"
        if key in pools.get("f", set()):
            return "f"
        if key in pools.get("n", set()):
            return "n"
        return None

    def _sample_given_name(self, gender: str, exclude_casefold: Optional[str] = None) -> Optional[str]:
        """Sample a replacement given name from the requested gender pool."""
        # Use the substitution pools if available (may be pre-filtered by allowed_lemmas),
        # but keep the recognition pools intact for gender detection.
        pools = (
            self.given_names_subst_by_gender
            or self.given_names_by_gender
            or {}
        )
        pool = pools.get(gender, []) or []
        if not pool:
            dbg(f"[sample_given_name] empty pool for gender={gender!r}")
            return None

        npi_unigrams = self._npi_unigrams

        max_attempts = self.cfg.sample_max_attempts
        tries = min(max_attempts, len(pool))

        rejects = []  # Track rejections for debugging
        cand_sample = random.sample(pool, tries)
        for cand in cand_sample:

            if exclude_casefold and cand.casefold() == exclude_casefold:
                if len(rejects) < 5:
                    rejects.append((cand, "same"))
                continue

            # Don't sample NPIs as names (single-token only).
            if npi_unigrams and cand and (" " not in cand):
                if cand.lower() in npi_unigrams:
                    if len(rejects) < 5:
                        rejects.append((cand, "npi"))
                    continue
                if self._is_functionish_lemma(cand.lower()):
                    if len(rejects) < 5:
                        rejects.append((cand, "is_functionish"))
                    continue

            # If allowed_vocab is enabled, keep names within it (case-sensitive gate).
            # Skip this check if override pools are active (they bypass vocab gating).
            if self._allowed_lemmas_active and not self._override_given_names_active:
                if cand not in (self.allowed_vocab or set()):
                    if len(rejects) < 5:
                        rejects.append((cand, "not_allowed_vocab"))
                    continue

            def _log_ok():
                print(f"[sample_given_name] OK gender={gender} cand={cand!r} pool_n={len(pool)} tries={tries}")
                for c, reason in rejects:
                    print(f"  reject cand={c!r} reason={reason}")
            dbg(_log_ok)
            return cand

        def _log_fail():
            print(f"[sample_given_name] FAIL gender={gender} pool_n={len(pool)} tries={tries} exclude={exclude_casefold}")
            for c, reason in rejects:
                print(f"  reject cand={c!r} reason={reason}")
        dbg(_log_fail)
        return None


    def _sample_given_name_any(self, exclude_casefold: Optional[str] = None) -> Optional[str]:
        """Sample a replacement given name from the combined (m/f/n) pools."""
        pools = (
            self.given_names_subst_by_gender
            or self.given_names_by_gender
            or {}
        )

        pool = []
        for g in ("m", "f", "n"):
            pool.extend(pools.get(g, []) or [])
        pool = [nm for nm in dict.fromkeys(pool) if nm]

        if not pool:
            dbg("[sample_given_name_any] empty combined pool")
            return None

        npi_unigrams = self._npi_unigrams

        max_attempts = self.cfg.sample_max_attempts
        tries = min(max_attempts, len(pool))

        rejects = []  # Track rejections for debugging
        cand_sample = random.sample(pool, tries)
        for cand in cand_sample:

            if exclude_casefold and cand.casefold() == exclude_casefold:
                if len(rejects) < 5:
                    rejects.append((cand, "same"))
                continue

            # Don't sample NPIs as names (single-token only).
            if npi_unigrams and cand and (" " not in cand):
                if cand.lower() in npi_unigrams:
                    if len(rejects) < 5:
                        rejects.append((cand, "npi"))
                    continue
                if self._is_functionish_lemma(cand.lower()):
                    if len(rejects) < 5:
                        rejects.append((cand, "is_functionish"))
                    continue

            # If allowed_vocab is enabled, keep names within it (case-sensitive gate).
            # Skip this check if override pools are active (they bypass vocab gating).
            if self._allowed_lemmas_active and not self._override_given_names_active:
                if cand not in (self.allowed_vocab or set()):
                    if len(rejects) < 5:
                        rejects.append((cand, "not_allowed_vocab"))
                    continue

            def _log_ok():
                print(f"[sample_given_name_any] OK cand={cand!r} pool_n={len(pool)} tries={tries}")
                for c, reason in rejects:
                    print(f"  reject cand={c!r} reason={reason}")
            dbg(_log_ok)
            return cand

        def _log_fail():
            print(f"[sample_given_name_any] FAIL pool_n={len(pool)} tries={tries} exclude={exclude_casefold}")
            for c, reason in rejects:
                print(f"  reject cand={c!r} reason={reason}")
        dbg(_log_fail)
        return None

    def _looks_like_proper_or_demonym(self, word: str) -> bool:
        """
        Heuristic without external lists:
          1) If the Title‑cased form is tagged as PROPN (NNP/NNPS) → treat as proper.
          2) If NER sees PERSON or NORP (nationalities/religions/political groups) → treat as proper/demonym.
        Works even if NER is absent; we fall back to (1) only.
        """
        w = (word or "").strip()
        if not w:
            return False
        title = w.title()

        # POS-based test on title-cased token
        pos_tok = self._nlp_pos(title)[0]
        if pos_tok.pos_ == "PROPN" or pos_tok.tag_ in {"NNP", "NNPS"}:
            return True

        # NER-based test (PERSON / NORP)
        for v in (w, w.title(), w.upper()):
            doc = self._nlp_pos(v)
            if doc.ents:  # ← any entity label triggers exclusion
                return True

        return False


    # ---------- Simple JSON list cache ----------
    def _cache_dir(self) -> str:
        # Prefer noun_bucket_dir; else next to human_nouns/adj lists; else cwd
        candidates = []
        nb = self.paths.noun_bucket_dir
        if nb:
            candidates.append(nb)
        for p in [self.paths.human_nouns_path,
                  self.paths.adj_lemmas_path]:
            if p:
                candidates.append(os.path.dirname(p))
        for base in candidates:
            try:
                path = os.path.join(base, "_augmenter_cache")
                os.makedirs(path, exist_ok=True)
                return path
            except Exception:
                continue
        path = os.path.join(os.getcwd(), "_augmenter_cache")
        os.makedirs(path, exist_ok=True)
        return path

    # ---------- Keys for the start-up caches ----------
    # Each method spells out one of the dependency lists declared at module
    # level, in the same order.

    def _cache_path(self, filename: str) -> str:
        if not self._startup_caches_on():
            # Never touch the directory when caching is off; the path is only
            # named, never read or written.
            base = self.paths.noun_bucket_dir or os.getcwd()
            return os.path.join(base, "_augmenter_cache", filename)
        return os.path.join(self._cache_dir(), filename)

    def _startup_caches_on(self) -> bool:
        return bool(getattr(self.cfg, "startup_caches", True))

    def _startup_cache_read(self, path: str, key: str, **kwargs):
        """`_read_keyed_cache`, or None without touching disk when
        `cfg.startup_caches` is off."""
        if not self._startup_caches_on():
            return None
        return _read_keyed_cache(path, key, **kwargs)

    def _startup_cache_write(self, path: str, key: str, payload, **kwargs) -> bool:
        """`_write_keyed_cache` unless `cfg.startup_caches` is off. True when
        a file was written."""
        if not self._startup_caches_on():
            return False
        _write_keyed_cache(path, key, payload, **kwargs)
        return True

    def _countability_source_signatures(self) -> list:
        base = self.paths.noun_bucket_dir or ""
        return [_file_signature(os.path.join(base, name))
                for name in _COUNTABILITY_SOURCE_FILES]

    def _pos_model_signature(self) -> list:
        """Name and pipeline version of the POS pipe, plus the spaCy version."""
        import spacy
        meta = getattr(self._nlp_pos, "meta", None) or {}
        return [self.cfg.spacy_pos_model, meta.get("version"), spacy.__version__]

    def _allowed_lemmas_cache_key(self) -> str:
        return _keyed_cache_digest([
            _ALLOWED_LEMMAS_CODE_VERSION,
            _file_signature(self.paths.allowed_vocab_path),
            _file_signature(self.paths.allow_all_vocab_path),
            self.cfg.min_vocab_freq,
            _package_version("lemminflect"),
        ])

    def _ctx_buckets_cache_key(self, stats_path: str) -> str:
        return _keyed_cache_digest([
            _CTX_BUCKETS_CODE_VERSION,
            _file_signature(stats_path),
            self.cfg.ctx_lemma_gate_min_count,
        ])

    def _filtered_pools_cache_key(self) -> str:
        return _keyed_cache_digest([
            _FILTERED_POOLS_CODE_VERSION,
            self._countability_source_signatures(),
            self._pos_model_signature(),
        ])

    def _fast_caches_cache_key(self, tags) -> str:
        from .core import _broad_noun_pools, _broad_verb_pool
        noun_pools = _broad_noun_pools(self) or {}
        return _keyed_cache_digest([
            _FAST_CACHES_CODE_VERSION,
            sorted(tags),
            _file_signature(self.paths.allowed_vocab_path),
            _file_signature(self.paths.allow_all_vocab_path),
            self.cfg.min_vocab_freq,
            _package_version("lemminflect"),
            _package_version("inflect"),
            _file_signature(self.paths.npi_path),
            _file_signature(self.paths.function_words_path),
            _file_signature(self.paths.licensors_path),
            self._countability_source_signatures(),
            self._pos_model_signature(),
            bool(self.cfg.prefilter_pools_by_allowed_vocab),
            _string_set_digest(self.allowed_lemmas or frozenset()),
            _string_set_digest(self.allowed_propn_lemmas or frozenset()),
            _string_set_digest(self.allowed_vocab or frozenset()),
            _string_set_digest(noun_pools.get("singular") or frozenset()),
            _string_set_digest(noun_pools.get("plural") or frozenset()),
            _string_set_digest(_broad_verb_pool(self) or frozenset()),
        ])

    # ---------- Countability pools, filtered through the POS pipe ----------
    def _apply_filtered_countability_pools(self, batch_filter) -> bool:
        """Narrow the countability pools with `batch_filter`, via the cache.

        Returns True when the pools came from the cache.
        """
        path = self._cache_path(_CACHE_FILTERED_POOLS_FILE)
        key = self._filtered_pools_cache_key()
        fields = ("mass", "count", "both", "plural_only",
                  "mass_capable", "count_capable")

        cached = self._startup_cache_read(path, key, kind="filtered pools", as_json=True,
                                   required=fields)
        if cached is not None:
            for field in fields:
                setattr(self.countability, field, set(sorted(cached[field])))
            dbg("[buckets] loaded filtered pools from cache")
            return True

        self.countability.filter(batch_filter)
        # These pools become candidate sequences downstream, and a set iterates
        # in insertion order-dependent slot order. Filling both branches from a
        # sorted sequence makes a hit and a miss structurally identical, not
        # merely equal.
        for field in fields:
            setattr(self.countability, field, set(sorted(getattr(self.countability, field))))
        if self._startup_cache_write(
            path, key,
            {field: sorted(getattr(self.countability, field)) for field in fields},
            kind="filtered pools", as_json=True,
        ):
            dbg(f"[buckets] wrote filtered pools cache -> {path}")
        return False

    def _cache_key(self, kind: str, sources: list[str]) -> str:
        code_ver = "v1_human_adj_common"
        m = hashlib.md5()
        m.update((self.cfg.spacy_model or "spacy").encode())
        m.update(code_ver.encode())
        for p in sources:
            try:
                st = os.stat(p)
                m.update(str(st.st_mtime_ns).encode())
                m.update(Path(p).name.encode())
            except Exception:
                m.update(("NA:" + str(p)).encode())
        return f"{kind}.{m.hexdigest()[:12]}.json"

    def _load_cache_list(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            data = obj.get("data")
            if isinstance(data, list):
                return data
        except Exception:
            pass
        return None

    def _save_cache_list(self, path: str, data: list[str]):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"data": sorted(list(set(data)))}, f, ensure_ascii=False)
        except Exception:
            pass

    # ---------- Build cached common-only human nouns ----------
    def _build_common_human_unigrams_cached(self):
        src = self.paths.human_nouns_path
        if not src:
            self._human_common_unigrams_list = []
            self._human_common_unigrams_nn = []
            self._human_common_unigrams_nns = []
            return

        cache_dir = self._cache_dir()
        base_key = self._cache_key("human_common_unigrams", [src])
        path_all = os.path.join(cache_dir, base_key)
        path_nn = path_all.replace(".json", ".nn.json")
        path_nns = path_all.replace(".json", ".nns.json")

        cached_all = self._load_cache_list(path_all)
        cached_nn = self._load_cache_list(path_nn)
        cached_nns = self._load_cache_list(path_nns)
        if cached_all is not None and cached_nn is not None and cached_nns is not None:
            self._human_common_unigrams_list = cached_all
            self._human_common_unigrams_nn = cached_nn
            self._human_common_unigrams_nns = cached_nns
            return

        nlp_pos = self._nlp_pos
        # Unigrams only; keep words that are not proper/demonyms
        candidates = [w for w in sorted(self.human_unigrams) if w and " " not in w]
        common_all, nn_ok, nns_ok = [], [], []
        for w in candidates:
            if self._looks_like_proper_or_demonym(w):
                continue
            common_all.append(w)

            # Check that realization yields correct POS tag when needed
            sg = self._realize(w, "NN")
            if sg:
                t = nlp_pos(sg)[0]
                if t.pos_ == "NOUN" and t.tag_ == "NN":
                    nn_ok.append(w)
            pl = self._realize(w, "NNS")
            if pl:
                t = nlp_pos(pl)[0]
                if t.pos_ == "NOUN" and t.tag_ == "NNS":
                    nns_ok.append(w)

        self._save_cache_list(path_all, common_all)
        self._save_cache_list(path_nn, nn_ok)
        self._save_cache_list(path_nns, nns_ok)

        self._human_common_unigrams_list = common_all
        self._human_common_unigrams_nn = nn_ok
        self._human_common_unigrams_nns = nns_ok

    def _human_common_pool_for_ptb(self, ptb: str) -> list[str]:
        if ptb == "NNS":
            return self._human_common_unigrams_nns or self._human_common_unigrams_list
        return self._human_common_unigrams_nn or self._human_common_unigrams_list

    # ---------- Build cached common-only adjectives ----------
    def _build_common_adjectives_cached(self):
        src = self.paths.adj_lemmas_path
        if not src:
            self._adj_whitelist_common = set()
            return

        cache_dir = self._cache_dir()
        path = os.path.join(cache_dir, self._cache_key("adj_common_whitelist", [src]))
        cached = self._load_cache_list(path)
        if cached is not None:
            self._adj_whitelist_common = set(cached)
            return

        nlp_pos = self._nlp_pos
        base = [w for w in self.adj_lemmas or [] if w and " " not in w]
        keep = []
        for doc in nlp_pos.pipe(base, batch_size=4000):
            t = doc[0]
            # ADJ, not numeral, and not proper/demonym (e.g., 'French', 'Prussian')
            if not self._looks_like_proper_or_demonym(t.text):
                keep.append(t.text.lower())

        self._save_cache_list(path, keep)
        self._adj_whitelist_common = set(keep)

    def _load_human_inventory(self, path: str) -> tuple[set[str], set[tuple[str, ...]]]:
        """
        Load human nouns (single and multiword) from file.
        - One entry per line, lowercase.
        - Multiword entries are space-separated.
        Returns (unigrams, mwes) where:
          unigrams: {'student', 'woman', ...}
          mwes: {('police','officer'), ('prime','minister'), ...}
        """
        unigrams: set[str] = set()
        mwes: set[tuple[str, ...]] = set()
        if not path:
            return unigrams, mwes
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip().lower()
                if not s:
                    continue
                toks = s.split()
                if len(toks) == 1:
                    unigrams.add(toks[0])
                else:
                    mwes.add(tuple(toks))
        return unigrams, mwes

    from typing import Tuple

    from spacy.tokens import Token, Span  # make sure this import exists

    # spaCy ↔ UD compatibility helpers imported from token_utils
    # Spine guards (BLiMP-like) ---------
    def _adj_controls_to_inf(self, t: Token) -> bool:
        """
        True iff t behaves like a predicate adjective that selects an infinitival xcomp.

        We treat as 'adj-like':
          - POS=ADJ, or
          - POS in {VERB,AUX} with VerbForm=Part and dep in {acomp, attr} (e.g., 'bound/supposed').

        We return True if:
          (a) t has a child xcomp whose subtree contains 'to' and 'be', OR
          (b) t is the predicate of a verb that (i) has expl=there and (ii) has such an xcomp.
        """
        # Is this token an adjective-like predicate?
        is_adj_like = (t.pos_ == "ADJ") or (
                t.pos_ in {"VERB", "AUX"} and "Part" in set(t.morph.get("VerbForm")) and t.dep_ in {"acomp", "attr"}
        )
        if not is_adj_like:
            return False

        # (a) xcomp attached to the adjective/participle itself
        for xc in (ch for ch in t.children if ch.dep_ == "xcomp"):
            has_to = any(u.lemma_.lower() == "to" and u.pos_ in {"PART", "SCONJ", "AUX"} for u in xc.subtree)
            has_be = any(u.lemma_.lower() == "be" and u.pos_ in {"AUX", "VERB"} for u in xc.subtree)
            if has_to and has_be:
                return True

        # (b) xcomp attached to the host verb (very common with 'there was bound to be ...')
        h = t.head
        if h is not None and h.pos_ in {"VERB", "AUX"} and has_expl_there_child(h):
            for xc in (ch for ch in h.children if ch.dep_ == "xcomp"):
                has_to = any(u.lemma_.lower() == "to" and u.pos_ in {"PART", "SCONJ", "AUX"} for u in xc.subtree)
                has_be = any(u.lemma_.lower() == "be" and u.pos_ in {"AUX", "VERB"} for u in xc.subtree)
                if has_to and has_be:
                    return True

        return False

    def _ecm_expletive_context(self, v: Token) -> bool:
        """
        Detect expletive/ECM-like spines that we should freeze at the MATRIX verb:
          (i) v ... xcomp: (to) be ...
         (ii) v ... ADJ (acomp/attr) ... xcomp: (to) be ...
        Freeze whenever either:
          - 'there' appears as an expletive in the matrix clause, OR
          - matrix object is 'it' (expletive-it object raising).
        """
        if v.pos_ not in {"VERB", "AUX"}:
            return False

        def _has_to_be(xc: Token) -> bool:
            has_be = (xc.lemma_.lower() == "be") or any(
                t.lemma_.lower() == "be" and t.pos_ in {"AUX", "VERB"} for t in xc.subtree
            )
            has_to = any(
                t.lemma_.lower() == "to" and t.pos_ in {"PART", "SCONJ", "AUX"} for t in xc.subtree
            )
            return has_be and has_to

        # xcomp directly under the verb
        xcs = [ch for ch in v.children if ch.dep_ == "xcomp"]

        # xcomp under an ADJ complement of the verb (e.g., 'was bound to be …')
        for adj in (ch for ch in v.children if ch.pos_ == "ADJ" and ch.dep_ in {"acomp", "attr"}):
            xcs.extend(ch for ch in adj.children if ch.dep_ == "xcomp")

        if not xcs:
            return False

        # Evidence for “expletive/raising” flavor
        has_expl_there = any(t.lemma_.lower() == "there" and t.dep_ in {"expl", "nsubj"} for t in v.subtree)
        has_obj_it = any(
            ch.dep_ in {"obj", "dobj"} and ch.pos_ == "PRON" and ch.lemma_.lower() == "it"
            for ch in v.children
        )

        if not (has_expl_there or has_obj_it):
            return False

        # Ensure at least one xcomp is a (to) be‑clause
        return any(_has_to_be(xc) for xc in xcs)

    def _np_span_from_head(self, head_tok: Token) -> Span:
        """
        Conservative NP span from a head NOUN/PROPN:
        take the contiguous span covering its subtree, trim leading/trailing punctuation.
        """
        doc = head_tok.doc
        nodes = list(head_tok.subtree)
        if not nodes:
            return doc[head_tok.i: head_tok.i + 1]
        i0, i1 = min(t.i for t in nodes), max(t.i for t in nodes)
        while i0 <= i1 and doc[i0].is_punct:
            i0 += 1
        while i1 >= i0 and doc[i1].is_punct:
            i1 -= 1
        return doc[i0:i1 + 1]

    def _lemma_tuple(self, span: Span) -> Tuple[str, ...]:
        # Lowercase lemmas, ignore surrounding quotes; keep all tokens otherwise
        return tuple(t.lemma_.lower() for t in span if not t.is_punct)

    def _is_human_np_span(self, span: Span) -> bool:
        """
        True iff the span lemmatizes to a known human MWE tuple OR
        its head lemma is a known human unigram.
        """
        lem_tup = self._lemma_tuple(span)
        if lem_tup in self.human_mwes:
            return True
        head = span.root
        if head.lemma_.lower() in self.human_unigrams:
            return True
        return False

    def _has_safe_conservative_override(self, tok: Token) -> bool:
        """
        Conservative licensor patterns in your JSONL are intended as *fallback freezes*.
        They should not block replacements when we have a structure-aware, license-checked
        replacement route for the token.

        Currently, we treat to-INF selector heads (CRT/raising/tough/ECM verbs/adjs) as
        having a safe route iff:
          * tok is in a to-INF frame, and
          * the corresponding to-INF lexicon lookup yields a non-empty candidate pool.

        This does **not** guarantee an `allowed_lemmas`-compatible lemma exists, but it
        prevents unconditional freezing from conservative patterns.
        """
        # Safe to-INF adjective selector route
        if self.toinf.adj_lex:
            try:
                req = self.toinf.adj_required_licenses(tok)
            except Exception:
                req = None
            if req is not None:
                try:
                    cands = self.toinf.adj_pool_candidates(tok.lemma_, req)
                except Exception:
                    cands = None
                if cands:
                    return True

        # Safe to-INF verb selector route
        if self.toinf.verb_lex:
            try:
                reqv = self.toinf.required_verb_license(tok)
            except Exception:
                reqv = None
            if reqv is not None:
                try:
                    cands = self.toinf.verb_pool_candidates(tok, reqv)
                except Exception:
                    cands = None
                if cands:
                    return True

        return False


    def _is_functionish(self, tok: Token, licensor_covered: Dict[int, str]) -> bool:
        lem = tok.lemma_.lower()
        lw = tok.text.lower()

        # 0) Tokens with no alphabetic characters (bare punctuation, symbols,
        #    digit-only, en/em-dashes) are treated as functionish so the
        #    sampler doesn't replace them with content words. Addresses a
        #    subtle bug where e.g. spaCy tokenizes 'pre–eminent' as
        #    [pre/JJ, –/JJ, eminent/JJ] all with empty whitespace — replacing
        #    the '–' with a sampled adjective yields 'positivetraditionalmutant'.
        if not any(ch.isalpha() for ch in tok.text):
            return True

        # 1) Determiner / numeric modifier tokens are function‑like (no hand lists).
        #    Freeze *the quantifier token itself* (we still randomize the noun it ranges over).
        if tok.dep_ in {"det", "nummod", "predet"}:
            return True

        # 1b) Adjectival quantifiers used as determiners (e.g., 'many/few/much/most' as amod).
        #     Use role + stopword + UD morph features + function_words (union), no tiny hard-coded list.
        if tok.dep_ == "amod" and (
                tok.is_stop
                or lem in self.function_words or lw in self.function_words
                or bool(tok.morph.get("PronType")) or bool(tok.morph.get("NumType"))
        ):
            return True

        # 1c) Standalone quantifier pronouns as arguments (e.g., 'few stayed', 'most left').
        #     Use your function_words lexicon, not a tiny in-code list.
        if tok.dep_ in {"nsubj", "obj", "dobj", "pobj", "attr"} and (
                lem in self.function_words or lw in self.function_words):
            if tok.pos_ in {"DET", "PRON", "ADJ", "ADV", "NUM", "PDT"}:
                return True

        if self.cfg.replace_pronouns:
            # Lock wh/expletive/dummy; allow other pronouns/poss-dets to be replaceable
            if tok.dep_ == "expl" or tok.tag_ in {"WP", "WP$", "WRB", "EX"}:
                return True
            if tok.pos_ == "PRON":
                return False
            if tok.pos_ == "DET" and tok.morph.get("Poss") == ["Yes"]:
                return False

        # 3) NPIs / Licensors from provided lexicons
        if lem in self.npi or lw in self.npi:
            return True
        if lem in self.licensors or lw in self.licensors:
            return True
        # licensor spans (token+dep) coverage
        if self.licensor_matcher and tok.i in licensor_covered:
            # licensor patterns (token+dep) coverage with strength semantics:
            #   - strong/guarded  : always freeze
            #   - conservative    : freeze unless a safe structure-aware replacement route applies
            #   - weak            : treat as conservative
            strength = licensor_covered.get(tok.i) if licensor_covered else None
            if strength in {"strong", "guarded"}:
                return True
            if strength in {"conservative", "weak"}:
                if not self._has_safe_conservative_override(tok):
                    return True
                # else: allow token to be replaceable (continue)

        # 4) Function-word list: freeze only when token is a closed-class POS
        if (lem in self.function_words or lw in self.function_words) and \
                (tok.pos_ in {"AUX", "ADP", "SCONJ", "CCONJ", "PART", "DET", "PRON"} or tok.tag_ == "MD"):
            return True

        # 5) Closed-class POS always frozen
        if tok.pos_ in {"AUX", "ADP", "SCONJ", "CCONJ", "PART", "DET", "PRON"}:
            return True
        if tok.tag_ == "MD":
            return True

        return False

    def _is_functionish_lemma(self, lemma: str) -> bool:
        """
        Lemma-only version of _is_functionish for filtering output candidates.
        Returns True if the lemma belongs to function_words or licensors.
        """
        lem = (lemma or "").lower()
        if lem in self.function_words:
            return True
        if lem in self.licensors:
            return True
        return False

    # --- WordNet- and context-based subcategorization guards ---
    def _wn_prep_after_verb_set(self, lemma: str) -> "set[str]":
        """Delegate to WordNetHelper.prep_after_verb_set."""
        return _wordnet_helper.prep_after_verb_set(lemma)

    def _wn_supports_preps_after_verb(self, lemma: str, preps: "set[str]") -> bool:
        """Delegate to WordNetHelper.supports_preps_after_verb."""
        return _wordnet_helper.supports_preps_after_verb(lemma, preps)

    def _pp_heads_for_verb(self, head) -> "set[str]":
        """
        Collect PP heads (ADP lemmas) that pair directly with the verb in context,
        including stranded prepositions (e.g., '... to talk to.').
        We look for:
          - ADP children of the verb (spaCy 'prep' or UD-like 'case' directly on verb)
          - ADP 'case' on verb's oblique/nominal dependents (UD)
          - A right-adjacent ADP token with 'prep/case' when no pobj is present (stranding)
        """
        preps = set()
        doc = head.doc

        for ch in head.children:
            # direct ADP dependent
            if ch.pos_ == "ADP" and ch.dep_ in ("prep", "case"):
                if is_infinitival_to_as_prep(ch):
                    pass
                else:
                    preps.add(ch.lemma_.lower())

            # UD-style: the ADP is 'case' on a nominal dependent (obl/nmod)
            if ch.dep_ in ("obl", "nmod", "obj", "dobj", "iobj", "pobj", "attr"):
                for g in ch.children:
                    if g.pos_ == "ADP" and g.dep_ in ("case", "prep"):
                        # ultra-safe: only skip if this looks like the same infinitival-to bug
                        try:
                            if (g.lemma_.lower() == "to"
                                    and ch.pos_ in {"VERB", "AUX"}
                                    and (ch.tag_ == "VB" or "Inf" in set(ch.morph.get("VerbForm")))):
                                continue
                        except Exception:
                            pass
                        preps.add(g.lemma_.lower())

            # stranded: ADP child without a pobj
            if ch.pos_ == "ADP" and ch.dep_ in ("prep", "case"):
                if not any(gc.dep_ in ("pobj", "obj") for gc in ch.children):
                    preps.add(ch.lemma_.lower())

        # extra conservative: immediate right token as stranded ADP
        i = head.i
        if i + 1 < len(doc):
            nxt = doc[i + 1]
            if nxt.pos_ == "ADP" and nxt.dep_ in ("prep", "case"):
                if not is_infinitival_to_as_prep(nxt):
                    preps.add(nxt.lemma_.lower())

        return preps

    def _protect_and_mwe_sets(self, doc: Doc) -> Tuple[Set[int], Set[int], Set[int]]:
        prot: Set[int] = set()
        fixed_mwe_idxs: Set[int] = set()
        patt_mwe_idxs: Set[int] = set()

        def _prot_add(i: int, why: str):
            """Add a single token index to prot, with debug print only if it was newly added."""
            if i not in prot:
                prot.add(i)
                def _prot_debug():
                    tok = doc[i] if 0 <= i < len(doc) else None
                    txt = tok.text if tok is not None else "<?>"
                    print(f"[protect] {why}: +{i} '{txt}'")
                dbg(_prot_debug)

        def _prot_update(idxs, why: str):
            """Add many token indices to prot, with debug print only for newly added indices."""
            idxs = list(idxs)
            new = [i for i in idxs if i not in prot]
            if new:
                prot.update(new)
                added = dbg(lambda: [(i, doc[i].text) for i in sorted(new) if 0 <= i < len(doc)])
                dbg(f"[protect] {why}: +{added}")

        # ... existing licensor/MWE protections ...
        # (make sure any prot.add/prot.update in that omitted section uses _prot_add/_prot_update too)

        # (a) Freeze any MWEs matched by your provided lexicons/patterns
        if self.fixed_mwe_matcher:
            for mid, s, e in self.fixed_mwe_matcher(doc):
                fixed_mwe_idxs.update(range(s, e))
                _prot_update(range(s, e), f"fixed_mwe_matcher mid={mid} span=({s},{e})")
        if self.mwe_matcher:
            for mid, s, e in self.mwe_matcher(doc):
                patt_mwe_idxs.update(range(s, e))
                _prot_update(range(s, e), f"mwe_matcher mid={mid} span=({s},{e})")

        # # (a.1) Keep content NP replaceable even if an external MWE matched the whole 'X of Y' span
        # # Unfreeze the 'of'-complement head under any matched MWE spans so the content noun can vary.
        # try:
        #     _mwe_spans = []
        #     if self.fixed_mwe_matcher:
        #         _mwe_spans.extend([(s, e) for _, s, e in self.fixed_mwe_matcher(doc)])
        #     if self.mwe_matcher:
        #         _mwe_spans.extend([(s, e) for _, s, e in self.mwe_matcher(doc)])
        #     for (s, e) in _mwe_spans:
        #         for tok in doc[s:e]:
        #             if tok.lemma_.lower() == "of" and tok.pos_ == "ADP":
        #                 # spaCy style: 'of' with pobj
        #                 for ch in tok.children:
        #                     if ch.dep_ in {"pobj", "pcomp"} and ch.i in prot:
        #                         prot.remove(ch.i)
        #                 # UD style: 'of' as case under an nmod head
        #                 if tok.dep_ == "case" and tok.head.dep_.startswith("nmod") and tok.head.i in prot:
        #                     prot.remove(tok.head.i)
        # except Exception:
        #     pass

        # (b) Structural, list‑free detection of binominal/partitive quantifiers:
        #     Freeze the *quantifier side* [DET? + NOUN(quantity) + 'of'], but leave the 'of'-complement NP replaceable.
        for h in doc:
            if h.pos_ != "NOUN":
                continue

            # --- UD-style: head NOUN with nmod(:of) child that has case='of' (defensive)
            ud_of_cases = []
            for c in h.children:
                if c.dep_.startswith("nmod"):
                    ud_of_cases.extend([cc for cc in c.children if cc.dep_ == "case" and cc.lemma_.lower() == "of"])

            # --- spaCy-style: head NOUN with 'prep' child lemma='of'
            spacy_of_preps = [c for c in h.children if c.dep_ == "prep" and c.lemma_.lower() == "of"]

            has_of = bool(ud_of_cases or spacy_of_preps)
            is_quantity_head = bool(self._wn_quantity_lemmas or set()) and (
                    h.lemma_.lower() in self._wn_quantity_lemmas
            )
            old_prot = prot.copy()  # keep your summary prints working as-is

            # --- UD case: 'h' is the content head, quantity noun is an nmod child with case='of'
            quantity_children = []
            for c in h.children:
                if c.pos_ == "NOUN" and self._wn_quantity_lemmas:
                    if c.lemma_.lower() in self._wn_quantity_lemmas:
                        if any(cc.dep_ == "case" and cc.lemma_.lower() == "of" for cc in c.children):
                            quantity_children.append(c)

            if quantity_children:
                for q in quantity_children:
                    # freeze det/predet on the QUANTITY noun
                    for d in q.children:
                        if d.dep_ in {"det", "predet"}:
                            _prot_add(d.i, "quantity-of (UD) det/predet")
                    _prot_add(q.i, "quantity-of (UD) quantity head")

                    # freeze the 'of' case marker
                    for cc in q.children:
                        if cc.dep_ == "case" and cc.lemma_.lower() == "of":
                            _prot_add(cc.i, "quantity-of (UD) case=of")

                    # also freeze coordination on the QUANTITY noun (e.g., 'lots and lots')
                    for ch in q.children:
                        if ch.dep_ in {"conj", "cc"}:
                            _prot_add(ch.i, "quantity-of (UD) coord on quantity")
                            for ch2 in ch.children:
                                if ch2.dep_ == "cc":
                                    _prot_add(ch2.i, "quantity-of (UD) nested cc")

                dbg(f"[protect] quantity-of (UD) heads={[q.text for q in quantity_children]} add={sorted(list(prot - old_prot))}")
                continue

            # --- spaCy case: 'h' is the quantity noun; 'of' is a 'prep' child under 'h'
            if has_of and is_quantity_head:
                for d in h.children:
                    if d.dep_ in {"det", "predet"}:
                        _prot_add(d.i, "quantity-of (spaCy) det/predet")
                _prot_add(h.i, "quantity-of (spaCy) quantity head")

                # freeze the 'of' preposition(s) AND actively unfreeze its complement head
                for prep in spacy_of_preps:
                    _prot_add(prep.i, "quantity-of (spaCy) prep=of")
                    for ch in prep.children:
                        if ch.dep_ in {"pobj", "pcomp"} and ch.pos_ in {"NOUN", "PROPN"} and ch.i in prot:
                            prot.remove(ch.i)  # removal: not requested, but keep behavior
                            dbg(f"[protect] quantity-of (spaCy) unfreeze complement head: -{ch.i} '{ch.text}'")

                for ch in h.children:
                    if ch.dep_ in {"conj", "cc"}:
                        _prot_add(ch.i, "quantity-of (spaCy) coord on quantity")
                        for ch2 in ch.children:
                            if ch2.dep_ == "cc":
                                _prot_add(ch2.i, "quantity-of (spaCy) nested cc")

                dbg(f"[protect] quantity-of (spaCy) head={h.text!r} add={sorted(list(prot - old_prot))}")
                continue

            # No 'of': catch degree 'a lot' / 'a great deal' etc.
            if is_quantity_head and any(d.dep_ == "det" and d.lemma_.lower() in {"a", "an"} for d in h.children):
                _prot_add(h.i, "quantity-degree (a/an) head")
                for d in h.children:
                    if d.dep_ == "det":
                        _prot_add(d.i, "quantity-degree (a/an) det")

        # (c) Negation handling...
        for t in doc:
            low_t = t.text.lower()

            def _freeze_aux_hosts(head_tok) -> bool:
                froze = False
                aux_hosts = [
                    a for a in head_tok.children
                    if a.dep_ in {"aux", "auxpass", "cop"} and (
                            a.pos_ == "AUX" or a.tag_ == "MD" or "Fin" in a.morph.get("VerbForm")
                    )
                ]
                for a in aux_hosts:
                    _prot_add(a.i, "freeze neg AUX host")
                    froze = True
                    # (existing print is now redundant, but harmless if you want to keep it)
                if head_tok.pos_ == "AUX" or head_tok.tag_ == "MD":
                    _prot_add(head_tok.i, "freeze neg head AUX/MD")
                    froze = True
                return froze

            # Case A: the head token has a neg child ('n't' / 'n’t' / 'not')
            neg_kids = [
                c for c in t.children
                if c.dep_ == "neg" and (c.lemma_.lower() == "not" or c.text.lower() in {"n't", "n’t"})
            ]
            if neg_kids:
                for c in neg_kids:
                    _prot_add(c.i, "freeze neg token")
                if not _freeze_aux_hosts(t):
                    _prot_add(t.i, "freeze neg lexical head (no aux host)")

            # Case B: the neg token itself — ensure head's AUX/MODAL host is frozen too
            if low_t in {"n't", "n’t", "not"}:
                head = t.head
                _prot_add(t.i, "freeze neg self")
                if not _freeze_aux_hosts(head):
                    _prot_add(head.i, "freeze neg head (no aux host)")

            # Case C: unsplit contraction edge (rare) like a single token ending in n't/n’t
            if low_t.endswith("n't") or low_t.endswith("n’t"):
                _prot_add(t.i, "freeze neg unsplit")

            # (i) Already present: freeze 'to be' xcomp head
            if self._is_to_be_xcomp_head(t):
                _prot_add(t.i, "freeze to-be xcomp head")
            # (ii)/(iii)/(iv) previously froze to-INF selector heads (raising/control/tough/ECM)
            # to preserve BLiMP-style spine invariants. For the ablation, we want these heads
            # to remain replaceable via the to-INF lexicon pools, so this is OFF by default.
            # (Set cfg.freeze_toinf_selectors=True to restore the old behavior.)
            if self.cfg.freeze_toinf_selectors:
                # (ii) NEW: freeze matrix verb when it has expl=there
                if t.pos_ in {"VERB", "AUX"} and has_expl_there_child(t):
                    _prot_add(t.i, "freeze expl-there host")

                    # (ii.a) also freeze its predicate ADJ/participial if this verb has an xcomp with 'to'+'be'
                    xcomps = [xc for xc in t.children if xc.dep_ == "xcomp"]
                    xcomp_has_to_be = any(
                        any(u.lemma_.lower() == "to" and u.pos_ in {"PART", "SCONJ", "AUX"} for u in xc.subtree)
                        and any(u.lemma_.lower() == "be" and u.pos_ in {"AUX", "VERB"} for u in xc.subtree)
                        for xc in xcomps
                    )
                    if xcomps and xcomp_has_to_be:
                        for ch in t.children:
                            if ch.dep_ in {"nsubj", "attr", "acomp"} and (
                                    ch.pos_ == "ADJ" or "Part" in set(ch.morph.get("VerbForm"))
                            ):
                                _prot_add(ch.i, "freeze raising predicate")

                # (iii) Existing: ECM/expletive contexts when xcomp is under the verb
                if t.pos_ in {"VERB", "AUX"} and self._ecm_expletive_context(t):
                    _prot_add(t.i, "freeze ECM/expletive context host")

                # (iv) Existing: adjective controlling 'to'-xcomp (now broader)
                if self._adj_controls_to_inf(t):
                    _prot_add(t.i, "freeze raising/tough predicate")

        dbg(f"[protected]: {sorted(list(prot))}")
        return prot, fixed_mwe_idxs, patt_mwe_idxs

    def _protect_tokens(self, doc: Doc) -> Set[int]:
        prot, _, _ = self._protect_and_mwe_sets(doc)
        return prot

    def _guess_noun_number(self, tok) -> str:
        """
        Return 'Plur' or 'Sing' for a noun token using robust fallbacks that
        work under spaCy‑Stanza (where PTB tags and morph may be missing).
        """
        # 1) PTB tag if present
        if tok.tag_ in {"NNS", "NNPS"}:
            return "Plur"
        if tok.tag_ in {"NN", "NNP"}:
            return "Sing"

        # 2) UD morph if present
        num = tok.morph.get("Number")
        if "Plur" in num:
            return "Plur"
        if "Sing" in num:
            return "Sing"

        # 3) Determiner heuristic
        det_lemmas = {c.lemma_.lower() for c in tok.children if c.dep_ == "det"}
        if det_lemmas & {"these", "those", "many", "several", "few", "both"}:
            return "Plur"
        if det_lemmas & {"this", "that", "each", "every", "either", "neither"}:
            return "Sing"

        # 4) Lemma vs surface heuristic (captures irregular plurals like 'children')
        txt = tok.text.lower()
        lem = tok.lemma_.lower()
        if txt != lem:
            # If tokenized surface != lemma and we're looking at NOUN, it is often plural
            return "Plur"

        # 5) S-ending heuristic
        if txt.endswith("s") and not txt.endswith("'s") and not txt.endswith("’s"):
            return "Plur"

        return "Sing"

    def _is_bare_singular_np_head(self, tok: Token) -> bool:
        """
        True iff this NOUN is a singular NP head with no article/possessor/quantifier.
        Used to conservatively block replacements that would create ungrammatical
        bare singular count-noun uses.
        """
        if tok.pos_ != "NOUN":
            return False
        if tok.morph.get("Number") != ["Sing"]:
            return False

        # Not a nominal modifier of another noun
        if tok.dep_ in {"compound", "amod", "nmod", "flat", "appos"}:
            return False

        # Consider determiners, predeterminers, possessors, numerals & quantifiers
        for ch in tok.children:
            dep = ch.dep_.lower()
            if (
                    dep == "det" or
                    dep == "det:predet" or
                    dep == "poss" or
                    dep.startswith("nmod:poss") or
                    dep == "nummod" or
                    dep == "quantmod"
            ):
                return False

        return True

    def _compute_token_features(self, tok: Token) -> TokenFeatures:
        """
        Compute all linguistic features for a token.

        This consolidates feature computation that was previously duplicated across
        tokeninfo_key, _ctx_bucket_for_token, and augment_doc/_verb_candidates.
        """
        # Core spacy attrs
        pos = str(tok.pos_)
        tag = str(tok.tag_)
        dep = str(tok.dep_)
        msig = morph_signature(tok)

        # Head info
        head = tok.head
        head_pos = str(head.pos_)
        head_tag = str(head.tag_)
        head_dep = str(head.dep_)
        head_is_self = head.i == tok.i

        # PTB tags (computed per POS type)
        ptb_noun = ptb_tag_for_noun(tok, guess_number_fn=self._guess_noun_number) if (tag in NOUN_TAGS or pos == "NOUN") else ""
        ptb_noun_forced = force_noun_number_by_context(tok, ptb_noun) if ptb_noun else ""
        ptb_adj = ptb_tag_for_adj(tok) if pos == "ADJ" else ""
        ptb_adv = ptb_tag_for_adv(tok) if pos == "ADV" else ""

        # Verb-like detection (matches Augmenter._is_verb_like)
        is_verb_like = (tag in VERB_TAGS) or bool(tok.morph.get("VerbForm"))
        ptb_verb = ptb_tag_for_verb(tok) if is_verb_like else ""

        # Noun features
        is_bare_singular = self._is_bare_singular_np_head(tok)

        # Verb frame/valency (only for verb-like tokens)
        frame_key_raw = ""
        frame_key = ""
        frame_has_prt = False
        frame_req_prep = ""
        preps_set: Tuple[str, ...] = ()
        comp_kind = ""
        comp_kind_simple = ""
        is_passive = False
        has_aux = False
        has_heavy_aux = False
        has_expl_there = False
        is_to_be_xcomp = False
        ccomp_marks: Tuple[str, ...] = ()
        xcomp_has_to = False
        xcomp_has_for = False
        xcomp_is_ger = False

        if is_verb_like:
            k_raw, has_prt, req_prep = extract_frame(tok)
            frame_key_raw = str(k_raw)
            frame_has_prt = bool(has_prt)
            preps_set_raw = tuple(sorted(extract_preps_set(tok)))

            # Apply valency fixes (same logic as in _verb_candidates and tokeninfo_key)
            key = frame_key_raw
            try:
                key = self._fix_tough_xcomp_valency(tok, key)
                tough_forced_trans = (frame_key_raw == "intrans" and key == "trans")
                if tough_forced_trans and preps_set_raw:
                    preps_set_raw = ()
                    req_prep = None
                key = self._fix_wh_gap_valency(tok, key, set(preps_set_raw))
            except Exception:
                pass

            # Drop comparative adjunct preps for intrans (matches tokeninfo_key logic)
            if key == "intrans" and preps_set_raw:
                comp_like = {"like", "as", "than"} & set(preps_set_raw)
                if comp_like:
                    preps_set_raw = tuple(sorted(set(preps_set_raw) - comp_like))

            frame_key = str(key)
            frame_req_prep = str(req_prep or "")
            preps_set = preps_set_raw

            comp_kind = str(self._parse_comp_kind(tok, frame_key or "intrans"))
            comp_kind_simple = str(self._comp_kind_from_parse(tok))
            is_passive = is_passive_clause(tok)
            has_aux = self._has_aux_chain(tok)
            has_heavy_aux = self._has_heavy_aux_chain(tok)
            has_expl_there = has_expl_there_child(tok)
            is_to_be_xcomp = self._is_to_be_xcomp_head(tok)

            # Collect ccomp markers
            ccomps = [c for c in tok.children if c.dep_ == "ccomp"]
            if ccomps:
                marks = set()
                for c in ccomps:
                    for m in c.children:
                        if m.dep_ == "mark":
                            marks.add(m.lemma_.lower())
                ccomp_marks = tuple(sorted(marks))

            # Collect xcomp flags
            xcomps = [c for c in tok.children if c.dep_ == "xcomp"]
            if xcomps:
                xcomp_has_for = any(m.dep_ == "mark" and m.lemma_.lower() == "for"
                                    for c in xcomps for m in c.children)
                xcomp_has_to = any((t.lemma_.lower() == "to" and t.pos_ in {"PART", "SCONJ", "AUX"})
                                   for c in xcomps for t in c.subtree)
                xcomp_is_ger = any((c.tag_ == "VBG") or any(t.tag_ == "VBG" for t in c.subtree)
                                   for c in xcomps)

        return TokenFeatures(
            pos=pos, tag=tag, dep=dep, morph_sig=msig,
            head_pos=head_pos, head_tag=head_tag, head_dep=head_dep, head_is_self=head_is_self,
            ptb_noun=ptb_noun, ptb_noun_forced=ptb_noun_forced,
            ptb_adj=ptb_adj, ptb_adv=ptb_adv, ptb_verb=ptb_verb,
            is_bare_singular=is_bare_singular,
            is_verb_like=is_verb_like, frame_key=frame_key, frame_key_raw=frame_key_raw,
            frame_has_prt=frame_has_prt, frame_req_prep=frame_req_prep, preps_set=preps_set,
            comp_kind=comp_kind, comp_kind_simple=comp_kind_simple,
            is_passive=is_passive, has_aux=has_aux, has_heavy_aux=has_heavy_aux,
            has_expl_there=has_expl_there, is_to_be_xcomp=is_to_be_xcomp,
            ccomp_marks=ccomp_marks, xcomp_has_to=xcomp_has_to,
            xcomp_has_for=xcomp_has_for, xcomp_is_ger=xcomp_is_ger,
        )

    def _should_freeze_verb(self, tok, inferred_tag: str) -> bool:
        """
        Return True when we should *not replace* this token due to lack of reliable
        verb/inflection cues. Freezing keeps the original surface.
        """
        if tok.tag_ in VERB_TAGS:
            return False  # parser already says it's a verb form

        # Local cues
        auxes = [ch for ch in tok.children if ch.dep_ in {"aux", "auxpass"}]
        cops = [ch for ch in tok.children if ch.dep_ == "cop"]
        has_aux = len(auxes) > 0
        has_cop_be = any(c.lemma_.lower() == "be" for c in cops)

        comp_deps = {"obj", "dobj", "ccomp", "xcomp", "obl", "pobj"}
        has_obj_or_comp = any(ch.dep_ in comp_deps for ch in tok.children)
        in_relcl = ("relcl" in tok.dep_.lower())
        looks_participial = tok.text.lower().endswith("ing") or ("Part" in tok.morph.get("VerbForm")) or (
                    "Prog" in tok.morph.get("Aspect"))

        # If it's 'be' as cop but we DON'T have object/comp/relcl + participial look, treat as adjectival → freeze
        if has_cop_be and not ((has_obj_or_comp or in_relcl) and looks_participial):
            return True

        confident_from_tag = inferred_tag in {"VBG", "VBN", "VB"}
        confident = has_aux or has_obj_or_comp or in_relcl or looks_participial or confident_from_tag
        return not confident

    def _verb_frame_profile(self, doc: Doc) -> Counter:
        """
        Summarize coarse verbal frames across the sentence for robustness checks.
        We normalize 'trans_prt' → 'trans' to avoid spurious rejects on phrasal-verb tagging.
        """
        keys = []
        for t in doc:
            if t.pos_ in {"VERB", "AUX"}:
                k, has_prt, _ = extract_frame(t)
                k = "trans" if k in {"trans", "trans_prt"} else k
                keys.append(k)
        return Counter(keys)

    def _expl_profile(self, doc: Doc) -> Tuple[int, int]:
        """Return (#expl=there, #expl=it)."""
        there = 0; it = 0
        for t in doc:
            if t.dep_ == "expl":
                if t.lemma_.lower() == "there":
                    there += 1
                elif t.lemma_.lower() == "it":
                    it += 1
        return (there, it)

    # =============== DEBUG HELPERS (no behavior change) ===============
    def _tough_raising_profile(self, doc: Doc) -> Tuple[int, int]:
        """
        Count ('tough-like', 'raising-like') patterns of the form:
          be + ADJ + xcomp(to V ...)
        We classify 'tough-like' if the embedded predicate looks unsaturated
        (no obj/pobj under the xcomp head), otherwise 'raising-like'.
        This is heuristic but stable across our augmentations.
        """
        tough = 0; raising = 0
        for be in doc:
            if be.lemma_.lower() != "be" or be.pos_ not in {"AUX", "VERB"}:
                continue
            adjs = [c for c in be.children if c.pos_ == "ADJ" and c.dep_ in {"acomp", "attr"}]
            if not adjs:
                continue
            for adj in adjs:
                xcs = [c for c in adj.children if c.dep_ == "xcomp"] or [c for c in be.children if c.dep_ == "xcomp"]
                for xc in xcs:
                    has_to = any(x.lemma_.lower() == "to" and x.pos_ in {"PART", "SCONJ", "AUX"} for x in xc.subtree)
                    if not has_to:
                        continue
                    has_obj = any(ch.dep_ in {"obj", "dobj"} for ch in xc.children)
                    has_pobj = any(gc.dep_ == "pobj" for ch in xc.children if ch.dep_ == "prep" for gc in ch.children)
                    if has_obj or has_pobj:
                        raising += 1
                    else:
                        tough += 1
        return tough, raising

    # Freeze the embedded 'to be' xcomp head (prevents breaking expletive spines)
    def _is_to_be_xcomp_head(self, t: Token) -> bool:
        if t.lemma_.lower() != "be" or t.pos_ not in {"AUX", "VERB"}:
            return False
        if t.dep_ != "xcomp":
            return False
        # require an infinitival marker 'to' under this xcomp
        return any(x.lemma_.lower() == "to" and x.pos_ in {"PART", "SCONJ", "AUX"} for x in t.subtree)

    def _is_human_np_span(self, head_tok: Token, *, freeze_unigrams: bool = False) -> bool:
        """
        True iff the NP headed by head_tok is a known multi-word human MWE
        (e.g., 'police officer', 'prime minister'). Single-token human nouns
        are NOT frozen by default so they can be replaced human→human.
        Set freeze_unigrams=True to also freeze single-token human heads.
        """
        span = self._np_span_from_head(head_tok)
        lem_tup = self._lemma_tuple(span)
        # freeze only multiword human MWEs by default
        if len(lem_tup) > 1 and lem_tup in self.human_mwes:
            return True
        if freeze_unigrams and len(lem_tup) == 1 and head_tok.lemma_.lower() in self.human_unigrams:
            return True
        return False

    # Frame extraction functions imported from verb_frame_analyzer

    # --- NEW: compact profiles & comparison for round-trip debug ---
    def _profiles(self, doc: Doc) -> Dict[str, Any]:
        return {
            "verb_frames": dict(self._verb_frame_profile(doc)),
            "expletives": {
                "there": self._expl_profile(doc)[0],
                "it": self._expl_profile(doc)[1],
            },
            "tough_raising": {
                "tough": self._tough_raising_profile(doc)[0],
                "raising": self._tough_raising_profile(doc)[1],
            },
            "ecm_count": sum(1 for t in doc if self._ecm_expletive_context(t)),
        }

    def _roundtrip_compare(self, src: Doc, tgt: Doc) -> Tuple[bool, Dict[str, Any]]:
        """Return (ok, report) comparing src vs tgt structural profiles."""
        src_p = self._profiles(src)
        tgt_p = self._profiles(tgt)
        violations: List[str] = []
        if src_p["verb_frames"] != tgt_p["verb_frames"]:
            violations.append(f"verb_frames: {src_p['verb_frames']} -> {tgt_p['verb_frames']}")
        if src_p["expletives"] != tgt_p["expletives"]:
            violations.append(f"expletives: {src_p['expletives']} -> {tgt_p['expletives']}")
        if src_p["tough_raising"] != tgt_p["tough_raising"]:
            violations.append(f"tough_raising: {src_p['tough_raising']} -> {tgt_p['tough_raising']}")
        if src_p["ecm_count"] != tgt_p["ecm_count"]:
            violations.append(f"ecm_count: {src_p['ecm_count']} -> {tgt_p['ecm_count']}")
        ok = not violations
        report = {
            "ok": ok,
            "violations": violations,
            "profiles_src": src_p,
            "profiles_tgt": tgt_p,
        }
        if self.cfg.roundtrip_debug and violations:
            dbg(f"[roundtrip][reject] report: {report}")
            dbg(f"[roundtrip][reject] proposed text: {tgt.text}")
        return ok, report


    # --------- Replacement pools ---------

    def _noun_candidates(self, tok: Token, ptb: str) -> Set[str]:
        lem = tok.lemma_.lower()

        # decide candidate pool via CountabilityPools (handles bare NN)
        tag = tok.tag_  # spaCy tag: 'NN' or 'NNS' etc.

        try:
            is_bare = self._is_bare_singular_np_head(tok)
        except AttributeError:
            is_bare = False

        cache_key = (lem, tag, ptb, is_bare)
        if cache_key in self._noun_candidates_cache:
            self._cache_probe("noun", hit=True)
            return self._noun_candidates_cache[cache_key]
        self._cache_probe("noun", hit=False)

        pool = self.countability.pool_for_slot(
            source_lemma=lem,
            tag=tag,
            is_bare_singular=is_bare,
        )

        if not pool:
            dbg(
                f"[noun-cands] EMPTY for i={tok.i} '{tok.text}' lemma={lem} "
                f"is_bare_singular={is_bare}, tag={tag}, ptb={ptb}"
            )

        self._noun_candidates_cache[cache_key] = pool
        return pool

    @lru_cache(maxsize=50000)
    def _has_vn_bare_intrans(self, lemma: str) -> bool:
        """True if VN shows an intransitive frame with no PREP and no flags for lemma."""
        try:
            for (k, prep_t, flags_t) in self.vn.explain_lemma(lemma):
                if k == "intrans" and not prep_t and not flags_t:
                    return True
        except Exception:
            pass
        return False

    def _comp_kind_from_parse(self, v: Token) -> str:
        # order matters: classify CPs before NP object to avoid double counts in noisy parses
        # CP_THAT / CP_WH via ccomp
        for ch in v.children:
            if ch.dep_ == "ccomp":
                marks = {m.lemma_.lower() for m in ch.children if m.dep_ == "mark"}
                has_that = bool({"that", "if", "whether"} & marks)
                return "CP_THAT" if has_that else "CP_WH"

        # CP_TO / CP_FOR_TO / CP_ING via xcomp
        for ch in v.children:
            if ch.dep_ == "xcomp":
                has_to = any(t.lemma_.lower() == "to" and t.pos_ in {"PART", "SCONJ", "AUX"} for t in ch.subtree)
                has_for = any(t.lemma_.lower() == "for" and t.dep_ == "mark" for t in ch.children)
                if has_for and has_to:
                    return "CP_FOR_TO"
                if has_to:
                    return "CP_TO"
                if ch.tag_ == "VBG" or any(t.tag_ == "VBG" for t in ch.subtree):
                    return "CP_ING"
                return "NONE"

        # DOUBLE_OBJ if iobj present
        if any(ch.dep_ == "iobj" for ch in v.children):
            return "DOUBLE_OBJ"

        # NP object
        if any(ch.dep_ in {"obj", "dobj"} for ch in v.children):
            return "NP_OBJ"

        return "NONE"

    # WH/Q detection helpers imported from verb_frame_analyzer

    # --- parse → complement-kind mapping (used by verbs_for_shape) -------------
    def _parse_comp_kind(self, v: Token, key: str) -> str:
        children = list(v.children)
        ccomps = [c for c in children if c.dep_ == "ccomp"]
        xcomps = [c for c in children if c.dep_ == "xcomp"]
        has_iobj = any(c.dep_ == "iobj" for c in children)
        has_obj = any(c.dep_ in {"obj", "dobj"} for c in children)

        # finite CP
        if ccomps:
            # spaCy sometimes attaches infinitival complements as `ccomp` (esp. ECM / control),
            # e.g. "I expected it to rain."  If a `ccomp` subtree contains infinitival *to*
            # (PART/SCONJ/AUX), treat it as CP_TO / CP_FOR_TO rather than finite CP_THAT/WH.
            has_inf_to = any(
                (t.lemma_.lower() == "to" and t.pos_ in {"PART", "SCONJ", "AUX"})
                for c in ccomps for t in c.subtree
            )
            if has_inf_to:
                has_for = any(
                    (m.dep_ == "mark" and m.lemma_.lower() == "for")
                    for c in ccomps for m in c.children
                )
                return "CP_FOR_TO" if has_for else "CP_TO"
            # Stricter WH detection: require an interrogative WH token (PronType=Int)
            # or an explicit complementizer mark ('if'/'whether').
            has_wh = False
            for c in ccomps:
                mark_lemmas = {m.lemma_.lower() for m in c.children if m.dep_ == "mark"}
                if {"whether", "if"} & mark_lemmas:
                    has_wh = True
                    break
                fronted_comp = c.i < v.i
                if any(is_interrogative_wh_token(t) and (fronted_comp or (t.i > v.i)) for t in c.subtree):
                    has_wh = True
                    break
            return "CP_WH" if has_wh else "CP_THAT"

        # non‑finite CP / ECM
        if xcomps:
            has_for = any(m.dep_ == "mark" and m.lemma_.lower() == "for"
                          for c in xcomps for m in c.children)
            has_to = any(t.lemma_.lower() == "to" and t.pos_ in {"PART", "SCONJ", "AUX"}
                         for c in xcomps for t in c.subtree)
            is_ger = any((c.tag_ == "VBG") or any(t.tag_ == "VBG" for t in c.subtree) for c in xcomps)

            if has_for and has_to: return "CP_FOR_TO"
            if has_to:             return "CP_TO"
            if is_ger:             return "CP_ING"

            # *** change begins: bare xcomp with an NP object → ECM/permissive ***
            if has_iobj:
                return "DOUBLE_OBJ"
            if has_obj or key in {"trans", "ditrans"}:
                return "NP_OBJ"
            # otherwise truly non‑diagnostic
            return "NONE"

        # no CP observed → fall back to valency
        if key == "ditrans" or has_iobj: return "DOUBLE_OBJ"
        if key == "trans":               return "NP_OBJ"
        return "NONE"

    def _has_aux_chain(self, v):
        """True if v hosts an aux/auxpass (perfect/prog/periphrasis)."""
        return any(ch.dep_ in {"aux", "auxpass"} for ch in v.children)

    def _has_heavy_aux_chain(self, v):
        """
        Return True only for perfect/passive periphrasis on the lexical head:
          - PERFECT: have + VBN on the head
          - PASSIVE: auxpass present, or be/get + VBN with passive subject cues
        Modals, do-support, and simple progressive (be + VBG) are ignored.
        """
        auxes = [ch for ch in v.children if ch.dep_ in {"aux", "auxpass"}]
        if not auxes:
            return False

        lemmas = {a.lemma_.lower() for a in auxes}
        has_have = "have" in lemmas
        has_be = "be" in lemmas
        has_get = "get" in lemmas
        has_auxpass = any(is_auxpass_dep(a.dep_) for a in auxes)

        # head is a past participle (non-progressive)
        is_vbn = (v.tag_ == "VBN") or (
                v.morph.get("VerbForm") == ["Part"] and "Prog" not in v.morph.get("Aspect")
        )

        # perfect
        if has_have and is_vbn:
            return True

        # passive
        if has_auxpass:
            return True
        has_passive_subj = any(ch.dep_ == "nsubjpass" for ch in v.children)
        if is_vbn and (has_be or has_get) and has_passive_subj:
            return True

        return False

    def _is_light_like(self, lemma: str) -> bool:
        # Allow override via config; else use a conservative default
        cfg_set = self.cfg.light_verb_lemmas
        base = {"have", "make", "take", "give", "get", "keep", "let", "put", "set", "do", "go", "come"}
        return (lemma in (cfg_set if cfg_set else base))

    def _fix_tough_xcomp_valency(self, v: Token, key: str) -> str:
        """
        UD parses for tough-to-INF adjectives (e.g., "a book that was easy to read")
        typically contain no explicit object on the embedded infinitive. That makes a
        truly transitive verb like `read` look "intrans" to `_extract_frame`, which
        then causes us to sample from the bare-intransitive pool.

        When we can confidently identify a *tough* adjective environment with an NP
        surface subject and no PP-gap ("easy to talk to" is handled via preposition
        matching), treat the embedded xcomp verb as TRANSITIVE for candidate selection.
        """
        if key != "intrans":
            return key
        if v.dep_ != "xcomp":
            return key

        # If the embedded infinitive itself takes clausal complements (xcomp/ccomp),
        # don't override: this is usually not the simple tough-object-gap pattern.
        if any(ch.dep_ in {"ccomp", "xcomp"} for ch in v.children):
            return key

        # Only for to-INF clauses
        if not xcomp_has_to(v):
            return key

        # Only block override for *true* PP-gap/stranding cases
        try:
            if xcomp_has_prep_gap(v):
                return key
        except Exception:
            return key

        # Find the predicate ADJ that licenses this to-INF clause.
        # Walk UP the xcomp chain to handle nested infinitives like
        # "The paper was difficult to decide to summarize."
        # where "summarize" is 2 levels deep from the adjective "difficult".
        adj: Optional[Token] = None
        cur = v
        seen = {v.i}  # cycle guard
        while cur.head is not None and cur.head.i not in seen:
            seen.add(cur.head.i)
            head = cur.head
            if head.pos_ == "ADJ":
                adj = head
                break
            elif head.pos_ in {"AUX", "VERB"}:
                # Check if the AUX/VERB has an ADJ child (copular structure)
                adjs = [ch for ch in head.children if ch.pos_ == "ADJ" and ch.dep_ in {"acomp", "attr"}]
                if len(adjs) == 1:
                    adj = adjs[0]
                    break
            # Continue up the chain only if this was an xcomp link
            if cur.dep_ != "xcomp":
                break
            cur = head

        if adj is None:
            return key

        # Must be an NP-subject configuration (not expletive it/there).
        try:
            if adj_subject_license(adj) != "np":
                return key
        except Exception:
            return key

        adj_lem = adj.lemma_.lower()

        # Prefer ERG-derived tags when available.
        tags: Set[str] = set()
        if self.toinf.adj_tags is not None:
            tags = set(self.toinf.adj_tags.get(adj_lem, set()))
        elif self.toinf.adj_lex:
            tags = set((self.toinf.adj_lex.get(adj_lem) or {}).get("tags") or [])

        is_tough = ("toinf:tough" in tags)

        # Minimal fallback: only the most canonical tough adjectives (keeps false positives low).
        # Bypassed when cfg.skip_toinf_freeze=True.
        if not is_tough and not tags and not self.cfg.skip_toinf_freeze:
            is_tough = adj_lem in {"easy", "hard", "tough", "difficult"}

        if is_tough:
            dbg(f"[VERB-CANDS] lemma='{v.lemma_}' tough-toinf object gap under ADJ='{adj.text}' → treat as trans")
            return "trans"

        return key

    def _fix_wh_gap_valency(self, tok, key: str, req_preps: Set[str]) -> str:
        """Heuristic: in wh-object extraction contexts, treat the gap-site verb as transitive.

        We mainly want to catch cases like:
            "Which book did Mary say John bought ___ ?"
        where spaCy often fails to attach the extracted object to the deepest verb, making it look intransitive.

        Guardrails:
          - Only applies when the verb has an overt (non-wh) subject, no direct object, and no governed PP.
          - Requires a wh-word in the sentence AND do-support (aux 'do') to reduce false positives.
        """
        try:
            if key != "intrans":
                return key
            if req_preps:
                return key

            # If this verb itself selects a clausal complement (ccomp/xcomp/csubj),
            # the wh-gap is almost always *inside* that complement (e.g., extraction from an
            # embedded clause). Avoid forcing transitivity on the matrix verb in cases like:
            #   "Which book did Maya say she bought?"
            if any(ch.dep_ in {"ccomp", "xcomp", "csubj"} for ch in tok.children):
                dbg(f"not enforcing trans on {tok}")
                return key

            if tok.pos_ != "VERB":
                return key

            has_obj = any(ch.dep_ in {"dobj", "obj"} for ch in tok.children)
            if has_obj:
                return key

            subj = next((ch for ch in tok.children if ch.dep_ == "nsubj"), None)
            if subj is None:
                return key

            # If the subject itself is wh (subject extraction), do NOT force transitivity.
            if subj.tag_.startswith("W") or subj.lower_ in {"what", "which", "who", "whom", "whose"}:
                return key

            sent = tok.sent
            has_wh = any(
                (t.tag_.startswith("W") or t.lower_ in {"what", "which", "who", "whom", "whose"})
                for t in sent
            )
            if not has_wh:
                return key

            has_do_aux = any((t.dep_ == "aux" and t.lemma_ == "do") for t in sent)
            if not has_do_aux:
                return key

                dbg(f"[WH-GAP] Forcing transitivity at gap site: {tok.text} (key {key} -> trans)")

            return "trans"
        except Exception:
            return key

    def _verb_candidates(self, tok, *, feat: Optional[TokenFeatures] = None):
        orig_lem = tok.lemma_.lower()

        # 0) Global off switch
        if not self.cfg.swap_heads_via_verbnet:
            dbg("[VERB-CANDS] swap_heads_via_verbnet=False → freeze verb")
            return [orig_lem]

        # A) Never touch auxiliaries or copulas in this slot
        if tok.dep_ in {"aux", "auxpass", "cop"}:
            dbg(f"[VERB-CANDS] {tok.dep_} → freeze verb")
            return [orig_lem]

        # B) Never touch lexical BE (structural backbone)
        if orig_lem == "be":
            dbg("[VERB-CANDS] lemma=be → freeze verb")
            return [orig_lem]

        # --- Option A: to-INF lexicon runtime filtering ---
        # If this verb sits in a to-INF spine (raising/control/ECM), then:
        #   * with lexicon: swap only within a compatible pool and only if the
        #     candidate's ERG licenses cover the local surface pattern;
        #   * without lexicon or unknown lemma: freeze (conservative).
        # Set cfg.skip_toinf_freeze=True to bypass freezing and fall through to normal candidates.
        req_license = self.toinf.required_verb_license(tok)
        if req_license is not None and not self.cfg.skip_toinf_freeze:
            dbg( f"[VERB-CANDS][TOINF] tok='{tok.text}' lem='{orig_lem}' req_license={req_license}")
            if self.toinf.verb_lex:
                cands = self.toinf.verb_pool_candidates(tok, req_license)
                if cands:
                    return cands
                    dbg(f"[VERB-CANDS] to-inf spine ({req_license}) but no safe pool → freeze")
                return [orig_lem]
                dbg(f"[VERB-CANDS] to-inf spine ({req_license}) and no lexicon → freeze")
            return [orig_lem]

        # C) Expletive 'there' under this head → freeze
        # Use pre-computed has_expl_there if available
        has_expl_there = feat.has_expl_there if feat else any(n.lemma_.lower() == "there" and n.dep_ in {"expl", "nsubj"} for n in tok.subtree)
        if has_expl_there:
            dbg("[VERB-CANDS] expl=there under head → freeze verb")
            return [orig_lem]

        # D) ECM / raising contexts (per the ECM/raising detector) → freeze
        try:
            if self._ecm_expletive_context(tok):
                dbg("[VERB-CANDS] ECM/expletive context → freeze verb")
                return [orig_lem]
        except Exception:
            pass

        # E)  Freeze only heavy aux chains (perfect/passive); keep modals/progressive swappable
        # Use pre-computed has_heavy_aux if available
        has_heavy_aux = feat.has_heavy_aux if feat else self._has_heavy_aux_chain(tok)
        if self.cfg.freeze_aux_hosts and has_heavy_aux:
            dbg("[VERB-CANDS] heavy-aux(host: perfect/passive) → freeze verb")
            return [orig_lem]

        # F) Light-verb matrix with clausal complement → freeze
        #    (keeps possessive 'have NP' etc. swappable; only freezes when CP present)
        has_clausal_comp = any(ch.dep_ in {"ccomp", "xcomp"} for ch in tok.children)
        if has_clausal_comp and self._is_light_like(orig_lem):
            # Also freeze generic DO-support lightness
            dbg("[VERB-CANDS] light-verb + ccomp/xcomp → freeze verb")
            return [orig_lem]

        # --- continue with the standard flow ---
        # Use pre-computed frame info if available
        if feat:
            key0 = feat.frame_key_raw
            has_prt = feat.frame_has_prt
            req_prep = feat.frame_req_prep or None
            # Note: feat.frame_key already has tough_xcomp_valency applied, but we need
            # to track tough_forced_trans, so we use frame_key_raw and reapply
            key = self._fix_tough_xcomp_valency(tok, key0)
        else:
            key0, has_prt, req_prep = extract_frame(tok)
            key = self._fix_tough_xcomp_valency(tok, key0)
        tough_forced_trans = (key0 == "intrans" and key == "trans")

        # Phrasal verb heads → freeze (you already had this)
        if has_prt:
            dbg("[VERB-CANDS] has particle → freeze verb")
            return [orig_lem]

        # Copulas are structural – never swap them
        if tok.dep_ == "cop" or tok.lemma_.lower() == "be":
            dbg("[VERB-CANDS] cop/BE → freeze verb")
            return [orig_lem]

        # If this head governs an expletive 'there', don't swap it either
        if any(n.lemma_.lower() == "there" and n.dep_ in {"expl", "nsubj"} for n in tok.subtree):
            dbg("[VERB-CANDS] expl=there under head → freeze verb")
            return [orig_lem]

        # --- Global switch: disable verb head swaps entirely if requested
        if not self.cfg.swap_heads_via_verbnet:
            dbg("[VERB-CANDS] swap_heads_via_verbnet=False → freeze verb")
            return [orig_lem]

        # --- ECM / expletive-raising spines: freeze matrix verb
        try:
            if self._ecm_expletive_context(tok):
                dbg("[VERB-CANDS] ECM/expletive context → freeze verb")
                return [orig_lem]
        except Exception:
            pass

        # --- Phrasal verb heads: safest is to freeze (particle compatibility not guaranteed)
        if has_prt:
            dbg("[VERB-CANDS] has particle → freeze verb")
            return [orig_lem]

        # --- Collect ALL governed prepositions from the current parse (ignore passive 'by')
        def _observed_preps(v: Token) -> set[str]:
            preps = set(iter_governed_preps(v))
            # Use pre-computed is_passive if available and v is the same token
            is_passive = feat.is_passive if (feat and v.i == tok.i) else is_passive_clause(v)
            if is_passive and "by" in preps:
                preps.discard("by")
            return preps
        def _observed_pcomp_preps(v: Token) -> set[str]:
            """Return governed preps that take a gerund pcomp (prep + V-ing) complement."""
            out: set[str] = set()
            for child in v.children:
                if not is_governed_prep_child(child):
                    continue
                # spaCy marks prep+V-ing complements as `pcomp` on the preposition.
                if any((gc.dep_ == "pcomp" and gc.tag_ == "VBG") for gc in child.children):
                    out.add(child.lemma_.lower())
            return out


        req_preps = _observed_preps(tok)
        pcomp_preps = _observed_pcomp_preps(tok) if req_preps else set()

        # If we forced transitivity due to a tough-object-gap, any observed preps here
        # are (by construction) *not* stranded gaps. They are usually adjuncts like
        # "in bed", "with a fork". Ignore them for VN matching to preserve diversity.
        if tough_forced_trans and req_preps:
            dbg(f"[VERB-CANDS] tough-toinf override: ignore preps for VN match {sorted(req_preps)}")
            req_preps = set()
            pcomp_preps = set()

        # --- Optional: treat governed preps as SOFT for plain intransitive verbs (adjunct PPs) ---
        # This helps cases like 'sail under the bridge' where 'under' is an adjunct PP: VN rarely lists
        # such preps as required, so matching on them collapses the VN pool. When enabled, we *probe*
        # the strict VN pool size and, if it is tiny, we drop req_preps/pcomp_preps for VN lookup.
        #
        # Safety:
        #   - Only applies to key=='intrans' and comp_kind=='NONE'.
        #   - Leaves ctx-key preps_set untouched, so ctx-gating can still recover distributional fit.
        vn_soft_min = self.cfg.vn_soft_prep_min_pool or 0
        if vn_soft_min > 0 and key == 'intrans' and req_preps:
            try:
                _ck = self._parse_comp_kind(tok, key)
            except Exception:
                _ck = ""
            if _ck == 'NONE':
                try:
                    # Probe strict VN pool size via the cache worker (no flags; no pp-head filtering).
                    _probe_cache_key = (
                        key,                      # intrans/trans/ditrans
                        'NONE',                   # comp_kind
                        tuple(sorted(req_preps)), # req_preps
                        tuple(),                  # flags
                        False,                    # obj_is_reflexive
                        False,                    # is_passive_clause
                        tuple(),                  # pp_heads
                        tok.tag_,                 # tag
                        False,                    # want_trans_slot
                        False,                    # reflexive_frame_check
                    )
                    entry_strict = self._verb_candidates_cached(_probe_cache_key)
                    if len(entry_strict.base) < vn_soft_min:
                        dbg(f"[VERB-CANDS] soft-prep fallback: strict VN pool {len(entry_strict.base)} < {vn_soft_min}; drop req_preps={sorted(req_preps)}")
                        req_preps = set()
                        pcomp_preps = set()
                except Exception:
                    pass


        key = self._fix_wh_gap_valency(tok, key, req_preps)

        # --- collect clausal-complement flags: {'that','wh','to','for-to'}
        def _clausal_flags(head):
            """
            Map the current clause's complement type to VN-style flags:
              - 'that' for declarative CPs marked by 'that'
              - 'q'    for Q/wh CPs (whether/if or any wh-word in the clause)
              - 'to'   for infinitival xcomps (detect 'to' as AUX/PART/SCONJ in the subtree)
            """
            flags = set()
            for ch in head.children:
                # Only treat true complements as CP cues (avoid advcl relatives like "when/where" clauses)
                if ch.dep_ in {"ccomp", "xcomp", "csubj"}:
                    marks = [m for m in ch.children if m.dep_ == "mark"]
                    mark_lemmas = {m.lemma_.lower() for m in marks}

                    # declarative 'that'
                    if "that" in mark_lemmas:
                        flags.add("that")

                    # Q/wh: whether/if OR interrogative WH token (PronType=Int)
                    # Important: ignore matrix-extracted WH items (fronted before the matrix head),
                    # e.g. "Which book did Maya say she bought?" — here 'say' does NOT select a WH-CP.
                    fronted_comp = ch.i < head.i
                    has_q = bool({"whether", "if"} & mark_lemmas)
                    if not has_q:
                        if fronted_comp:
                            has_q = any(is_interrogative_wh_token(t) for t in ch.subtree)
                        else:
                            has_q = any(is_interrogative_wh_token(t) and (t.i > head.i) for t in ch.subtree)

                    # Fallback: some pipelines omit PronType features. If the complement clause begins
                    # with a WH-word, treat it as interrogative (embedded question) — but avoid triggering
                    # on copular "be" contexts (clefts / extraposition).
                    # NOTE: 'head' here is the matrix verb taking the complement.
                    if (not has_q) and (head.lemma_.lower() != "be"):
                        left_i = min((t.i for t in ch.subtree if not t.is_punct), default=ch.i)
                        left = next((t for t in ch.subtree if t.i == left_i), None)
                        if left is not None and (fronted_comp or left.i > head.i):
                            WH_TAGS = {"WDT", "WP", "WP$", "WRB"}
                            WH_FORMS = {"who", "whom", "whose", "what", "which", "where", "when", "why", "how"}
                            if left.tag_ in WH_TAGS and left.lower_ in WH_FORMS and left.dep_ != "det":
                                if "Rel" not in left.morph.get("PronType"):
                                    has_q = True

                    if has_q:
                        flags.add("q")

                    # infinitival 'to' (UD may tag it as AUX/PART/SCONJ)
                    has_to = any(t.lemma_.lower() == "to" and t.pos_ in {"PART", "SCONJ", "AUX"} for t in ch.subtree)
                    if has_to:
                        flags.add("to")
            return flags

        flags = _clausal_flags(tok)
        if pcomp_preps:
            # Mark this as a prep+V-ing (pcomp) environment for specific prepositions.
            # NOTE: VerbNet's PP inventory often does *not* distinguish NP vs gerund-clause complements
            # for a given preposition. We record these flags for debugging and (optionally) strict
            # matching, but we may drop them if they zero out the VN candidate set.
            pcomp_flags = {f"pcomp:{p}" for p in pcomp_preps}
            flags.update(pcomp_flags)


        # --- choose complement kind from the current parse
        # --- choose complement kind from the current parse
        comp_kind = self._parse_comp_kind(tok, key)
        if self.cfg.debug_vn:
            dbg(f"[VN shape] key={key} comp_kind={comp_kind} preps={sorted(req_preps)} flags={sorted(flags)}")

        obj_is_reflexive = any(
            ch.dep_ in {"obj", "dobj"} and ch.pos_ == "PRON" and ch.morph.get("Reflex") == ["Yes"]
            for ch in tok.children
        )

        # Detect passive once here for gating
        tok_is_passive = is_passive_clause(tok)

        # Build cache key from all relevant features
        pp_heads = self._pp_heads_for_verb(tok)

        want_trans_slot = bool(key in {"trans", "ditrans"} or tok_is_passive or tough_forced_trans)

        # NEW cache key: NO orig_lem, but includes pp_heads + tag (so pp/pos work is cached)
        cache_key = (
            key,
            comp_kind,
            tuple(sorted(req_preps)),
            tuple(sorted(flags)),
            bool(obj_is_reflexive),
            bool(tok_is_passive),
            tuple(sorted(pp_heads)),
            tok.tag_,
            bool(want_trans_slot),
            bool(self.cfg.reflexive_frame_check),
        )

        # Cache lookup
        entry = self._verb_candidates_cache.get(cache_key)
        if entry is None:
            self._cache_probe("verb", hit=False)
            entry = self._verb_candidates_cached(cache_key)
            self._verb_candidates_cache[cache_key] = entry
        else:
            self._cache_probe("verb", hit=True)
            dbg(f"[VERB-CANDS] cache hit for key: {cache_key}")

        # Materialize: start from cached base pool, then add orig_lem AFTER cache
        cands = set(entry.base)
        if orig_lem:
            cands.add(orig_lem)

        # --- Reflexive direct object guard (preserve orig_lem fallback semantics) ---
        if obj_is_reflexive and cands and self.cfg.reflexive_frame_check:
            before = len(cands)

            kept = set()
            if entry.ok_reflexive is not None:
                kept |= (cands & entry.ok_reflexive)

            # Ensure the SINGLE item (orig_lem) is checked even if not in cached base
            orig_in = bool(orig_lem) and (orig_lem in cands)
            if orig_in and orig_lem not in entry.base:
                try:
                    if _wordnet_helper.has_reflexive_or_animate_do(orig_lem):
                        kept.add(orig_lem)
                except Exception:
                    pass

            if kept:
                cands = kept
            else:
                # preserve your fail-safe: never dead-end augmentation; keep original lemma
                cands = {orig_lem} if orig_lem else set()

            dbg(f"[VERB-CANDS] reflexive obj → kept {len(cands)} / {before} (WN animate/oneself DO)")

        # --- Bare intransitive gate (unconditional, like your code) ---
        is_truly_plain_intrans = (
                key == "intrans"
                and comp_kind == "NONE"
                and not req_preps
                and not flags
        )
        if is_truly_plain_intrans and cands:
            kept = set()
            if entry.ok_bare_intrans is not None:
                kept |= (cands & entry.ok_bare_intrans)

            orig_in = bool(orig_lem) and (orig_lem in cands)
            if orig_in and orig_lem not in entry.base:
                try:
                    wn_ok = _wordnet_helper.allows_bare_intrans(orig_lem)
                except Exception:
                    wn_ok = False
                try:
                    vn_ok = self._has_vn_bare_intrans(orig_lem)
                except Exception:
                    vn_ok = True
                if wn_ok and vn_ok:
                    kept.add(orig_lem)

            if kept != cands:
                dropped = sorted(list(cands - kept))[:15]
                dbg(f"[VERB-CANDS] bare-intrans gate dropped {len(cands - kept)} e.g. {dropped}")

            cands = kept

        # --- WordNet transitivity gate (apply only if non-empty result, like your code) ---
        if key in {"trans", "ditrans"} or tok_is_passive:
            kept = set()
            if entry.ok_wn_trans is not None:
                kept |= (cands & entry.ok_wn_trans)

            orig_in = bool(orig_lem) and (orig_lem in cands)
            if orig_in and orig_lem not in entry.base:
                try:
                    orig_min = _wordnet_helper.min_object_count(orig_lem)
                except Exception:
                    orig_min = -1
                if orig_min >= 1:
                    kept.add(orig_lem)

            if kept:
                if kept != cands:
                    dropped = sorted(list(cands - kept))[:15]
                    dbg(f"[VERB-CANDS] WN strict trans gate dropped {len(cands - kept)} (e.g., {dropped})")
                cands = kept
            # else: preserve original behavior (do NOT narrow to empty)

        # --- Clause-type gate (apply only if non-empty result, like your code) ---
        if cands and flags:
            needs_that = "that" in flags
            needs_q = "q" in flags
            needs_to = "to" in flags

            if needs_that or needs_q or needs_to:
                kept = set()
                if entry.ok_wn_clause is not None:
                    kept |= (cands & entry.ok_wn_clause)

                orig_in = bool(orig_lem) and (orig_lem in cands)
                if orig_in and orig_lem not in entry.base:
                    try:
                        if needs_that and not _wordnet_helper.allows_that_cp(orig_lem):
                            pass
                        elif needs_q and not _wordnet_helper.allows_q_cp(orig_lem):
                            pass
                        elif needs_to and not _wordnet_helper.allows_to_inf(orig_lem):
                            pass
                        else:
                            kept.add(orig_lem)
                    except Exception:
                        # be conservative: if WN probe fails, don't over-block
                        kept.add(orig_lem)

                if kept:
                    if kept != cands:
                        def _log_clause_gate():
                            dropped = sorted(list(cands - kept))[:15]
                            clause_desc = []
                            if needs_that: clause_desc.append("that")
                            if needs_q: clause_desc.append("wh")
                            if needs_to: clause_desc.append("to-inf")
                            print(f"[VERB-CANDS] WN clause-type gate ({'+'.join(clause_desc)}) "
                                  f"dropped {len(cands - kept)} (e.g., {dropped})")
                        dbg(_log_clause_gate)
                    cands = kept
                # else: preserve original behavior (do NOT narrow to empty)

        # --- WN preps filter using pp_heads (cached for base; single-check for orig_lem) ---
        if pp_heads and cands:
            kept = set()
            if entry.ok_pp_heads is not None:
                kept |= (cands & entry.ok_pp_heads)

            orig_in = bool(orig_lem) and (orig_lem in cands)
            if orig_in and orig_lem not in entry.base:
                try:
                    if self._wn_supports_preps_after_verb(orig_lem, pp_heads):
                        kept.add(orig_lem)
                except Exception:
                    # match your overall “be conservative” stance
                    kept.add(orig_lem)

            # Only narrow if we preserve at least one option.
            if kept:
                if kept != cands:
                    dropped = sorted(list(cands - kept))[:12]
                    dbg(f"[WN preps] required={sorted(pp_heads)} kept={len(kept)} "
                        f"dropped={len(cands - kept)} ex={dropped}")
                cands = kept

        # --- Debug sample ---
        dbg(lambda: f"[VERB-CANDS] {len(cands)} candidates (sample): {sorted(list(cands))[:20]}")

        return sorted(cands)

    def _verb_candidates_cached(self, cache_key) -> _VerbCandsCacheEntry:
        """
        Cached worker: MUST depend ONLY on cache_key (no tok, no orig_lem).
        It computes the VN pool and precomputes expensive per-candidate gates
        (pp_heads WordNet check + slot POS probe, plus the other WN gates).
        """
        (
            key,  # str: intrans/trans/ditrans
            comp_kind,  # str
            req_preps_t,  # Tuple[str, ...]
            flags_t,  # Tuple[str, ...]
            obj_is_reflexive,  # bool
            tok_is_passive,  # bool
            pp_heads_t,  # Tuple[str, ...]
            tag,  # str (tok._tag / tok.tag_)
            want_trans_slot,  # bool
            reflexive_frame_check,  # bool
        ) = cache_key

        req_preps = set(req_preps_t)
        flags = set(flags_t)
        pp_heads = set(pp_heads_t)

        # comparative-like PP relaxor
        relaxed_preps = req_preps
        if key == "intrans":
            comp_like = {"like", "as", "than"} & req_preps
            if comp_like:
                relaxed_preps = set(req_preps) - comp_like

        # VN query (with pcomp-flag relaxation retry)
        query_flags = set(flags)
        pcomp_flags = {f for f in query_flags if f.startswith("pcomp:")}

        cands = set(self.vn.verbs_for_shape(key, comp_kind, relaxed_preps, query_flags))

        if (not cands) and pcomp_flags:
            query_flags = query_flags - pcomp_flags
            cands = set(self.vn.verbs_for_shape(key, comp_kind, relaxed_preps, query_flags))

        if self._verb_metrics is not None:
            self._verb_metrics["vn_query_calls"] += 1
            if not cands:
                self._verb_metrics["vn_empty"] += 1

        # Question complements: allow CP_THAT/CP_WH interchange if q is present
        if "q" in query_flags and comp_kind in {"CP_THAT", "CP_WH"}:
            other_ck = "CP_THAT" if comp_kind == "CP_WH" else "CP_WH"
            cands |= set(self.vn.verbs_for_shape(key, other_ck, relaxed_preps, query_flags))

        # NP_OBJ WordNet backstop
        if comp_kind == "NP_OBJ":
            cands = {l for l in cands if _wordnet_helper.supports_np_object(l)}

        # Clean odd VN entries
        if cands:
            cands = {l for l in cands if l and not l.startswith("?")}

        base = frozenset(cands)

        # --- Precompute gate-pass sets over `base` ---
        ok_reflexive = None
        if obj_is_reflexive and base and reflexive_frame_check:
            ok_reflexive = frozenset({l for l in base if _wordnet_helper.has_reflexive_or_animate_do(l)})

        is_truly_plain_intrans = (
                key == "intrans"
                and comp_kind == "NONE"
                and not req_preps
                and not flags
        )
        ok_bare_intrans = None
        if is_truly_plain_intrans and base:
            kept = set()
            for l in base:
                try:
                    wn_ok = _wordnet_helper.allows_bare_intrans(l)
                except Exception:
                    wn_ok = False
                try:
                    vn_ok = self._has_vn_bare_intrans(l)
                except Exception:
                    vn_ok = True  # be conservative like your code
                if wn_ok and vn_ok:
                    kept.add(l)
            ok_bare_intrans = frozenset(kept)

        ok_wn_trans = None
        if (key in {"trans", "ditrans"} or tok_is_passive) and base:
            def _wn_min_obj(lemma: str) -> int:
                try:
                    return _wordnet_helper.min_object_count(lemma)
                except Exception:
                    return -1

            ok_wn_trans = frozenset({l for l in base if _wn_min_obj(l) >= 1})

        ok_wn_clause = None
        if flags and base:
            needs_that = "that" in flags
            needs_q = "q" in flags
            needs_to = "to" in flags
            if needs_that or needs_q or needs_to:
                def _wn_allows_clause_types(lemma: str) -> bool:
                    if needs_that and not _wordnet_helper.allows_that_cp(lemma):
                        return False
                    if needs_q and not _wordnet_helper.allows_q_cp(lemma):
                        return False
                    if needs_to and not _wordnet_helper.allows_to_inf(lemma):
                        return False
                    return True

                ok_wn_clause = frozenset({l for l in base if _wn_allows_clause_types(l)})

        ok_pp_heads = None
        if pp_heads and base:
            ok_pp_heads = frozenset({l for l in base if self._wn_supports_preps_after_verb(l, pp_heads)})

        return _VerbCandsCacheEntry(
            base=base,
            ok_reflexive=ok_reflexive,
            ok_bare_intrans=ok_bare_intrans,
            ok_wn_trans=ok_wn_trans,
            ok_wn_clause=ok_wn_clause,
            ok_pp_heads=ok_pp_heads,
        )

    def _adv_candidates(self, tok: Token) -> List[str]:
        """
        Adverb substitution:
          - draw from pre-tagged RB(/RBR/RBS) whitelist
          - optionally intersect with provided manner_adverbs
        """
        base = [w for w in self._adv_rb_whitelist if w != tok.lemma_.lower()]
        # if self.manner_adverbs:
        #     base = [w for w in base if w in self.manner_adverbs]
        return sorted(base)

    def _adj_candidates(self, tok) -> List[str]:
        # Pull from adjective lexicon / whitelist.
        # For *to-inf adjective* slots, constrain replacements to other to-inf adjectives
        # with compatible licenses; if anything goes wrong, freeze rather than silently
        # falling back to generic JJ sampling (which can break grammaticality).
        try:
            req = self.toinf.adj_required_licenses(tok)
        except Exception as e:
            dbg(f"  [debug] toinf-adj required-licenses failed for {tok.text!r}/{tok.lemma_}: {e}")
            req = None

        lem = tok.lemma_.lower()

        if req is not None:
            dbg(f"[ADJ-CANDS][TOINF] tok='{tok.text}' lem='{lem}' req_licenses={sorted(req)}")

        req_for_key = tuple(sorted(req)) if req else None
        cache_key = (lem, req_for_key)
        cands = self._adj_candidates_cache.get(cache_key, None)
        if cands:
            self._cache_probe("adj", hit=True)
            dbg(f"[ADJ-CANDS] cache hit for key: {cache_key}")
            return cands
        else:
            self._cache_probe("adj", hit=False)
            cands = self._adj_candidates_inner_cached(lem, req)
            self._adj_candidates_cache[cache_key] = cands
            return cands

    def _adj_candidates_inner_cached(self, lem, req):
        # to-inf adjective gating (skip with cfg.skip_toinf_freeze=True)
        if req is not None and not self.cfg.skip_toinf_freeze and self.toinf.adj_lex and self.toinf.adj_by_tag:
            cands = self.toinf.adj_pool_candidates(lem, req)
            dbg(f"[ADJ-CANDS][TOINF] pool_candidates={len(cands) if cands else 'none'}")
            if cands:
                return cands
                dbg(f"[ADJ-CANDS][TOINF] empty pool; fallback='{lem}'")
            return [lem]

        # Fallback: common adjective whitelist (kept broad; not all contexts need strict gating)
        base = []
        for x in self._adj_whitelist_common:
            if x == lem:
                continue
            base.append(x)
        return base

    def _fix_indefinite_articles(self, text: str) -> str:
        """
        Adjust 'a'/'an' determiners to match the following word's pronunciation
        using the inflect library. Capitalization is preserved.
        """

        # Matches: (a|an) + spaces + optional opening quote/bracket + next token
        # Examples handled: a apple → an apple, an unicorn → a unicorn,
        #                   a "hour" → an "hour", a 8-year-old → an 8-year-old, An FBI → An FBI
        pattern = re.compile(r'\b([Aa]n?)(\s+)(["“‘\(\[\{]*)([A-Za-z0-9][A-Za-z0-9.\-]*)')

        def _repl(m: re.Match) -> str:
            orig_article = m.group(1)
            space = m.group(2)
            punct = m.group(3) or ""
            head = m.group(4)

            # inflect returns e.g. "an FBI" or "a unicorn"; we only need the article
            suggested = self.inflect_engine.a(head).split()[0]
            if orig_article[0].isupper():
                suggested = suggested.capitalize()
            return f"{suggested}{space}{punct}{head}"

        return pattern.sub(_repl, text)

    @profile
    def augment_doc(self, doc: Doc, *, extra_protected: Optional[Set[int]] = None, return_meta: bool = False) -> Union[str, Tuple[str, List[Dict[str, Any]]]]:
        prot = set(self._protect_tokens(doc))
        if extra_protected:
            prot |= set(extra_protected)

        dbg(f"[augment_doc] n_toks={len(doc)} protected={sorted(list(prot))} "
            f"allowed_lemmas={'ON' if self._allowed_lemmas_active else 'OFF'}")
        dbg(lambda: [print(f"[tok] i={t.i:02d} text={t.text!r:12} lemma={t.lemma_!r:12} pos={t.pos_:5} tag={t.tag_:5} "
                           f"dep={t.dep_:10} head={t.head.i:02d}:{t.head.text!r} prot={'Y' if t.i in prot else 'N'}")
                     for t in doc])

        out: List[str] = []
        meta: List[Dict[str, Any]] = [dict() for _ in range(len(doc))]
        out_cursor = 0

        licensor_covered: Dict[int, str] = {}
        if self.licensor_matcher:
            conservative_guard = self.cfg.conservative_guard
            licensor_covered = self.licensor_matcher.cover_strengths(doc, conservative_guard=conservative_guard)

        # sentence-start capitalization
        need_cap_at_sent_start = True

        def _cap_first_alpha(s: str) -> str:
            for i, ch in enumerate(s):
                if ch.isalpha():
                    return s[:i] + ch.upper() + s[i + 1:]
            return s

        def _append_token(text: str, tok: Token, replaced: bool) -> None:
            nonlocal need_cap_at_sent_start, out_cursor
            if tok.is_sent_start:
                need_cap_at_sent_start = True
            to_add = text
            if need_cap_at_sent_start and tok.is_alpha and replaced:
                to_add = _cap_first_alpha(to_add)
            if tok.is_alpha:
                need_cap_at_sent_start = False

            start = out_cursor
            end = start + len(to_add)
            meta[tok.i] = {
                "i": tok.i,
                "src": tok.text,
                "out": to_add,
                "start": start,
                "end": end,
                "replaced": bool(replaced),
                "pos": tok.pos_,
                "tag": tok.tag_,
            }

            out.append(to_add + tok.whitespace_)
            out_cursor = end + len(tok.whitespace_)

        _SUBJ_DEPS = {"nsubj", "nsubjpass", "nsubj:pass", "csubj", "csubjpass", "csubj:pass"}
        _OBJ_DEPS = {"dobj", "obj", "iobj", "pobj", "obl"}
        _SUBJ_POOL = ["i", "you", "he", "she", "it", "we", "they"]


        # ------- pronoun agreement helpers --------
        def _person_number(pron: str):
            p = pron.lower()
            if p == "i": return 1, "Sing"
            if p == "you": return 2, None
            if p in {"he", "she", "it"}: return 3, "Sing"
            if p == "we": return 1, "Plur"
            if p == "they": return 3, "Plur"
            return None, None

        def _lex_present_ok(tag, person, number):
            if tag == "VBZ": return person == 3 and number == "Sing"
            if tag == "VBP": return not (person == 3 and number == "Sing")
            return True

        def _be_ok(surf, person, number):
            s = surf.lower()
            if s in {"am", "is", "are"}:
                if person == 1 and number == "Sing": return s == "am"
                if person == 3 and number == "Sing": return s == "is"
                return s == "are"
            if s in {"was", "were"}:
                if person in {1, 3} and number == "Sing": return s == "was"
                return s == "were"
            return True

        def _have_ok(surf, person, number):
            s = surf.lower()
            if s in {"has", "have"}:
                return s == ("has" if (person == 3 and number == "Sing") else "have")
            return True

        def _do_ok(surf, person, number):
            s = surf.lower()
            if s in {"does", "do"}:
                return s == ("does" if (person == 3 and number == "Sing") else "do")
            return True

        def _agrees_without_changes(subj_tok, cand_pron):
            person, number = _person_number(cand_pron)
            if person is None:
                return False
            head = subj_tok.head
            if head.pos_ not in {"VERB", "AUX"}:
                return True
            targets = [head] + [c for c in head.children if c.dep_ in {"aux", "auxpass", "cop"}]
            for v in targets:
                if v.tag_ == "MD":
                    continue
                lem = v.lemma_.lower()
                if lem == "be" and not _be_ok(v.text, person, number): return False
                if lem == "have" and not _have_ok(v.text, person, number): return False
                if lem == "do" and not _do_ok(v.text, person, number): return False
                if v.i == head.i and not _lex_present_ok(v.tag_, person, number): return False
            return True

        # ---- reflexives / nearest subject ----
        _SUBJ_TO_REFL = {
            "i": "myself", "you": "yourself", "he": "himself", "she": "herself",
            "it": "itself", "we": "ourselves", "they": "themselves",
        }

        def _nearest_subject(t: Token):
            head = t.head
            if head is None: return None
            subs = [c for c in head.children if c.dep_ in _SUBJ_DEPS]
            if subs: return subs[0]
            if head.head is not None:
                subs2 = [c for c in head.head.children if c.dep_ in _SUBJ_DEPS]
                return subs2[0] if subs2 else None
            return None


        # ---- POS mini-pipeline for adj guard ----
        if not hasattr(self, "_nlp_pos") or self._nlp_pos is None:
            try:
                import spacy
                self._nlp_pos = spacy.load(self.cfg.spacy_pos_model, disable=["parser", "senter", "ner"])
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load spaCy POS model '{self.cfg.spacy_pos_model}'. "
                    "Fallback to the main pipeline is disabled."
                ) from e
        if not hasattr(self, "_pos_cache"):
            self._pos_cache = {}

        # ---- NEW: allowed-vocab + NPI cache per (lemma, canonical PTB tag) ----
        if not hasattr(self, "_lemma_tag_allowed"):
            self._lemma_tag_allowed = {}  # (lemma, canon_tag) -> bool

        # Block swapping-in of *single-token* NPIs.
        # We ignore multi-word NPIs here (those are handled via span protection).
        npi_unigrams = self._npi_unigrams

        def _probe_ptb_tags(surf: str, target_ptb: str) -> Tuple[str, str, str]:
            """
            Debug-only: return (iso_tag, ctx_tag, ctx_sent).

            iso_tag  = spaCy tag when `surf` is parsed alone (often noisy/ambiguous).
            ctx_tag  = spaCy tag when `surf` is placed into a minimal forcing context
                       for `target_ptb` (much more reliable for verbs).
            ctx_sent = the context sentence used for ctx_tag.
            """
            nlp_pos = self._nlp_pos

            iso_tag = ""
            try:
                d0 = nlp_pos(surf)
                if len(d0) > 0:
                    iso_tag = d0[0].tag_ or ""
            except Exception:
                iso_tag = ""

            # Choose a tiny forcing context.
            ctx_sent = surf
            idx = 0
            t = target_ptb or ""

            if t.startswith("VB"):
                if t == "VBG":
                    ctx_sent = f"They are {surf}."
                    idx = 2
                elif t == "VBN":
                    ctx_sent = f"They have {surf}."
                    idx = 2
                elif t == "VBZ":
                    ctx_sent = f"He {surf}."
                    idx = 1
                else:
                    # VB / VBP / VBD
                    ctx_sent = f"They {surf}."
                    idx = 1
            elif t == "NN":
                ctx_sent = f"a {surf}."
                idx = 1
            elif t == "NNS":
                ctx_sent = f"many {surf}."
                idx = 1
            elif t == "RB":
                ctx_sent = f"They moved {surf}."
                idx = 2
            elif t == "JJ":
                ctx_sent = f"the {surf} thing."
                idx = 1

            ctx_tag = ""
            try:
                d1 = nlp_pos(ctx_sent)
                if len(d1) > idx:
                    ctx_tag = d1[idx].tag_ or ""
            except Exception:
                ctx_tag = ""

            return iso_tag, ctx_tag, ctx_sent

        def _sample_from_candidates(
                cands: Iterable[str],
                ptb_tag: str,
                tok: Token,
                want_gender: Optional[str] = None,
                feat: Optional[TokenFeatures] = None,
        ) -> Optional[str]:
            cands_ctx, ctx_w = self._ctx_intersect_candidates(tok, ptb_tag, cands, feat=feat)
            dbg(f"[ctx sample] n_orig={len(cands)} n_for_ctx={len(cands_ctx)}")
            return _sample_from_candidates_inner(cands_ctx, ptb_tag, exclude=[tok.lemma_.lower()],
                                                 want_gender=want_gender, cand_weights=ctx_w)

        @profile
        def _sample_from_candidates_inner(
                cands: Iterable[str],
                ptb_tag: str,
                exclude: Iterable[str] = (),
                want_gender: Optional[str] = None,
                cand_weights: Optional[Counter] = None,
        ) -> Optional[str]:
            """
            Sample with:
              - optional exclusion (usually the source lemma),
              - limited attempts per token,
              - per-(lemma,tag) caching for realize+constraints.

            Constraints checked in self._candidate_allowed_for_tag():
              - NPI blocklist (single-token only; lemma OR realized surface)
              - allowed_lemmas (optional)

            Optional (cfg.respect_gender):
              - if want_gender in {"m","f","n"}, restrict to candidates whose lemma has the same
                gender tag in self.gender_lexicon (e.g., from ecmonsen/gendered_words).
            """
            base = tuple(dict.fromkeys(cands))
            if not base:
                return None

            excl = {e.lower() for e in (exclude or []) if e}

            respect_gender = (
                bool(self.cfg.respect_gender)
                and bool(want_gender)
                and (want_gender in GENDER_TAGS)
                and bool(self.gender_lexicon)
            )

            # If we’re enforcing gender, sample *from the gender-matching pool*,
            # instead of rejection-sampling from the full pool.
            if respect_gender:
                base = tuple(
                    cand for cand in base
                    if self._gender_of_lemma((cand or "").lower()) == want_gender
                )
                if not base:
                    return None

            dbg(f"[sample] ptb={ptb_tag} base_n={len(base)} exclude_n={len(excl)} "
                f"gender={'ON:'+want_gender if respect_gender else 'OFF'} "
                f"lemmas={'ON' if self._allowed_lemmas_active else 'OFF'}")

            # Keep per-token work bounded
            max_attempts = self.cfg.sample_max_attempts
            tries = min(max_attempts, len(base))

            attempts = []

            if cand_weights:
                import math
                def w(c: str) -> float:
                    return float(cand_weights.get(c, 0) or 0)

                if any(w(c) > 0 for c in base):
                    ordered = sorted(
                        base,
                        key=lambda c: math.log(random.random()) / max(w(c), 1.0),
                        reverse=True,
                    )
                else:
                    ordered = list(base)
                    random.shuffle(ordered)

                ordered = ordered[:tries]
            else:
                ordered = random.sample(base, tries)

            for cand in ordered:
                cand_lc = (cand or "").lower()
                if not cand_lc:
                    continue
                if cand_lc in excl:
                    continue

                if self._candidate_allowed_for_tag(cand_lc, ptb_tag):
                    def _log_ok():
                        surf = self._realize(cand_lc, ptb_tag)
                        print(f"[sample] OK ptb={ptb_tag} cand={cand_lc} surf={surf!r}")
                    dbg(_log_ok)
                    return cand_lc

                else:
                    def _track_attempt():
                        if len(attempts) >= 6:
                            return
                        surf = self._realize(cand_lc, ptb_tag)
                        in_vocab = True
                        if self._allowed_lemmas_active:
                            if ptb_tag in {"NNP", "NNPS"}:
                                in_vocab = bool(surf and (surf in (self.allowed_vocab or set())))
                            else:
                                in_vocab = bool(cand_lc and (cand_lc in (self.allowed_lemmas or set())))
                        is_npi = False
                        if npi_unigrams:
                            is_npi = (cand_lc in npi_unigrams) or (surf and (surf.lower() in npi_unigrams))

                        iso_tag, probe_tag, probe_sent = ("", "", "")
                        if surf:
                            iso_tag, probe_tag, probe_sent = _probe_ptb_tags(surf, ptb_tag)

                        attempts.append((
                            cand_lc,
                            surf,
                            in_vocab,
                            is_npi,
                            self._gender_of_lemma(cand_lc),
                            iso_tag,
                            probe_tag,
                            probe_sent,
                        ))
                    dbg(_track_attempt)

            def _log_fail():
                if tries == len(base):
                    print(f"[sample] SKIP ptb={ptb_tag} base_n={len(base)} tries={tries} (exhausted pool)")
                else:
                    print(f"[sample] FAIL ptb={ptb_tag} base_n={len(base)} tries={tries} (constraints)")
                for cand_lc, surf, in_vocab, is_npi, g, iso_tag, probe_tag, probe_sent in attempts:
                    extra = ""
                    if iso_tag:
                        extra += f" iso_tag={iso_tag}"
                    if probe_tag:
                        extra += f" probe_tag={probe_tag}"
                    if probe_sent and probe_sent != (surf or ""):
                        extra += f" probe_sent={probe_sent!r}"
                    print(f"         cand={cand_lc!r:14} surf={surf!r:16} in_vocab={in_vocab} npi={is_npi} gender={g}{extra}")
            dbg(_log_fail)

            return None
        # ---- collect subjects that bind a reflexive ----
        reflexive_subject_ids: set[int] = set()
        for t in doc:
            if t.pos_ == "PRON" and t.morph.get("Reflex") == ["Yes"]:
                head = t.head
                if head is None:
                    continue
                subs = [c for c in head.children if c.dep_ in _SUBJ_DEPS]
                if subs:
                    reflexive_subject_ids.add(subs[0].i)
                elif head.head is not None:
                    subs2 = [c for c in head.head.children if c.dep_ in _SUBJ_DEPS]
                    if subs2:
                        reflexive_subject_ids.add(subs2[0].i)

        # ---------------- main token loop ----------------
        for i, tok in enumerate(doc):
            if i in prot:
                _append_token(tok.text, tok, replaced=False)
                continue

            if tok.text in {"'s", "’s"} and (tok.dep_ == "case" or tok.tag_ == "POS"):
                _append_token(tok.text, tok, replaced=False)
                continue

            # ---- pronouns ----
            if self.cfg.replace_pronouns:
                # possessive determiners
                if tok.pos_ == "DET" and tok.morph.get("Poss") == ["Yes"]:
                    cls = "POSSDET"
                    pool = [p for p in PRON_POOLS.get(cls, []) if p != tok.lemma_.lower()]
                    if self._allowed_lemmas_active:
                        pool = [p for p in pool if p in (self.allowed_lemmas or set())]
                    if self.cfg.respect_gender:
                        want_g = pronoun_gender(tok.lemma_.lower() or tok.text.lower())
                        if want_g:
                            pool = [p for p in pool if pronoun_gender(p) == want_g]
                    if pool:
                        rep = random.choice(pool)
                        _append_token(rep, tok, replaced=True)
                        continue

                # personal pronouns
                if tok.pos_ == "PRON" and tok.tag_ not in {"WP", "WP$", "WRB", "EX"} and tok.dep_ != "expl":
                    # reflexives
                    if tok.morph.get("Reflex") == ["Yes"]:
                        subj = _nearest_subject(tok)
                        if subj is not None and subj.pos_ == "PRON":
                            want = _SUBJ_TO_REFL.get(subj.lemma_.lower())
                            if want and want != tok.text.lower():
                                if (not self._allowed_lemmas_active) or (want in (self.allowed_lemmas or set())):
                                    rep = want
                                    if tok.i == 0 and tok.text[:1].isupper():
                                        rep = rep.capitalize()
                                    _append_token(rep, tok, replaced=True)
                                    continue
                        _append_token(tok.text, tok, replaced=False)
                        continue

                    cls = pron_class(tok)
                    if "Acc" in tok.morph.get("Case"):
                        cls = "OBJ"
                    elif tok.dep_.lower() in _OBJ_DEPS:
                        cls = "OBJ"

                    if cls and cls in PRON_POOLS:
                        candidates = [p for p in PRON_POOLS[cls] if p != tok.lemma_.lower()]
                        if tok.dep_ in _SUBJ_DEPS and cls == "SUBJ":
                            candidates = [
                                p for p in _SUBJ_POOL
                                if p != tok.lemma_.lower() and _agrees_without_changes(tok, p)
                            ]
                            if tok.lemma_.lower() != "it":
                                candidates = [p for p in candidates if p != "it"]
                        if self._allowed_lemmas_active:
                            candidates = [p for p in candidates if p in (self.allowed_lemmas or set())]
                        if self.cfg.respect_gender:
                            want_g = pronoun_gender(tok.lemma_.lower() or tok.text.lower())
                            if want_g:
                                candidates = [p for p in candidates if pronoun_gender(p) == want_g]
                        if candidates:
                            rep = random.choice(candidates)
                            if rep == "i":
                                rep = "I"
                            if tok.i == 0 and tok.text[:1].isupper():
                                rep = rep[0].upper() + rep[1:]
                            _append_token(rep, tok, replaced=True)
                            continue

            # Apostrophe-initial clitic tokens ('s, 're, 'm, 'll and their
            # curly-quote variants) are contraction halves, never standalone
            # content words — substituting one glues its host to a sampled
            # verb ("it's" -> "itlooks"). Emit the whole class verbatim, both
            # apostrophe codepoints. (Substitution-side only: sensing and
            # statistics collection are unaffected.)
            if tok.text[0] in ("'", "’"):
                _append_token(tok.text, tok, replaced=False)
                continue

            # function words
            if self._is_functionish(tok, licensor_covered):
                def _log_functionish():
                    lem = tok.lemma_.lower()
                    reason = []
                    if tok.is_stop: reason.append("spacy_stop")
                    if tok.pos_ in {"DET", "ADP", "AUX", "PART", "PRON", "SCONJ", "CCONJ"}: reason.append(
                        f"closed:{tok.pos_}")
                    if lem in self.function_words: reason.append("function_words")
                    if lem in self.licensors: reason.append("licensors_list")
                    if tok.i in (licensor_covered or {}): reason.append(f"licensor_matcher:{licensor_covered[tok.i]}")
                    print(f"[freeze] functionish i={tok.i} text='{tok.text}' lem='{lem}' pos={tok.pos_} :: {','.join(reason) or 'unknown'}")
                dbg(_log_functionish)
                _append_token(tok.text, tok, replaced=False)
                continue

            # PROPN
            if tok.pos_ == "PROPN" and self.cfg.replace_propn:
                dbg(f"[PROPN] enter i={tok.i} text={tok.text!r} lemma={tok.lemma_!r} tag={tok.tag_}")
                new_txt: Optional[str] = None

                # If this looks like a first-name PROPN we can infer gender from, replace using the given-name pools.
                g = self._gender_of_propn_text(tok.text)
                dbg(f"[PROPN] gender_detected={g!r} from text={tok.text!r}")
                if g:
                    if self.cfg.respect_gender:
                        cand = self._sample_given_name(g, exclude_casefold=tok.text.casefold())
                        dbg(f"[PROPN] sample_given_name gender={g!r} -> {cand!r}")
                    else:
                        cand = self._sample_given_name_any(exclude_casefold=tok.text.casefold())
                        dbg(f"[PROPN] sample_given_name_any -> {cand!r}")

                    if cand:
                        new_txt = match_case_like(tok.text, cand)

                # Fallback: replace with any allowed proper noun token (from allowed_vocab-derived pool).
                if new_txt is None:
                    pool = self.allowed_propn_pool
                    dbg(f"[PROPN] fallback allowed_propn_pool n={0 if not pool else len(pool)}")
                    if pool:
                        tries = min(self.cfg.sample_max_attempts, len(pool))
                        for _ in range(tries):
                            cand = random.choice(pool)
                            if cand.casefold() == tok.text.casefold():
                                continue
                            new_txt = cand
                            break
                        dbg(f"[PROPN] fallback sampled -> {new_txt!r}")

                if new_txt:
                    # Pluralize if original was NNPS (plural proper noun)
                    if tok.tag_ == "NNPS":
                        plural = self.inflect_engine.plural_noun(new_txt)
                        if plural:
                            new_txt = plural
                        dbg(f"[PROPN] pluralized NNPS -> {new_txt!r}")
                    _append_token(new_txt, tok, replaced=True)
                else:
                    dbg("[PROPN] no replacement, keep original")
                    _append_token(tok.text, tok, replaced=False)
                continue

            # Compute token features once for content word handling
            feat = self._compute_token_features(tok)

            # VERB
            if feat.is_verb_like:
                if any(is_particle_dep(c.dep_) for c in tok.children):
                    _append_token(tok.text, tok, replaced=False)
                    continue

                # Freeze single-token AUX/VERB contractions (e.g., "'s", "'re") that spaCy may tag as verbs.
                # These often have irregular lemmas (e.g., "'") and are not safe to replace in isolation.
                if tok.text in {"'s", "'s", "'re", "'re", "'m", "'m", "'d", "'d", "'ll", "'ll", "'ve", "'ve"}:
                    _append_token(tok.text, tok, replaced=False)
                    continue

                tag = feat.ptb_verb
                if self._should_freeze_verb(tok, tag):
                    _append_token(tok.text, tok, replaced=False)
                    continue
                cands = self._verb_candidates(tok, feat=feat)
                rep_lemma = _sample_from_candidates(cands, tag, tok, feat=feat)
                if self._verb_metrics is not None:
                    self._verb_metrics["verb_slots"] += 1
                    non_orig = [c for c in cands if (c or "").lower() != tok.lemma_.lower()]
                    if rep_lemma is None:
                        self._verb_metrics["verb_frozen"] += 1
                        reason = "only_orig" if not non_orig else "sample_rejected"
                        self._verb_metrics[f"verb_frozen_{reason}"] += 1
                    else:
                        self._verb_metrics["verb_replaced"] += 1
                if rep_lemma is None:
                    _append_token(tok.text, tok, replaced=False)
                    continue
                _append_token(self._realize(rep_lemma, tag), tok, replaced=True)
                continue

            # subject that binds a reflexive → enforce human NP
            if self.cfg.respect_human:
                if (tok.pos_ in {"NOUN", "PROPN"}) and (tok.dep_ in _SUBJ_DEPS) and (tok.i in reflexive_subject_ids):
                    if self._is_human_np_span(tok, freeze_unigrams=False):
                        _append_token(tok.text, tok, replaced=False)
                        continue

                    if tok.pos_ == "NOUN":
                        ptb = feat.ptb_noun_forced
                        lem = tok.lemma_.lower()
                        base_pool = self._human_common_pool_for_ptb(ptb)
                        want_g = self._gender_of_lemma(lem) if self.cfg.respect_gender else None
                        rep_lemma = _sample_from_candidates(base_pool, ptb, tok, want_gender=want_g, feat=feat)
                        if rep_lemma:
                            _append_token(self._realize(rep_lemma, ptb), tok, replaced=True)
                            continue
                        _append_token(tok.text, tok, replaced=False)
                        continue

                    _append_token(tok.text, tok, replaced=False)
                    continue

            # NOUN (human + default path)
            if is_noun_like(tok):
                lem = tok.lemma_.lower()
                ptb = feat.ptb_noun_forced

                if self.cfg.respect_human:
                    needs_human = (tok.dep_ in _SUBJ_DEPS) and (tok.i in reflexive_subject_ids)
                    force_human = bool(self.cfg.human_to_human) and (
                            lem in self.human_unigrams or set()
                    )
                    if needs_human or force_human:
                        if self._is_human_np_span(tok, freeze_unigrams=False):
                            _append_token(tok.text, tok, replaced=False)
                            continue
                        base_pool = self._human_common_pool_for_ptb(ptb)
                        want_g = self._gender_of_lemma(lem) if self.cfg.respect_gender else None
                        rep_lemma = _sample_from_candidates(base_pool, ptb, tok, want_gender=want_g, feat=feat)
                        if rep_lemma:
                            _append_token(self._realize(rep_lemma, ptb), tok, replaced=True)
                            continue

                cands = self._noun_candidates(tok, ptb)
                want_g = self._gender_of_lemma(lem) if self.cfg.respect_gender else None
                rep_lemma = _sample_from_candidates(cands, ptb, tok, want_gender=want_g, feat=feat)
                if rep_lemma is None:
                    _append_token(tok.text, tok, replaced=False)
                    continue
                _append_token(self._realize(rep_lemma, ptb), tok, replaced=True)
                continue

            # ADJ
            if tok.pos_ == "ADJ":
                tag = feat.ptb_adj
                cands = self._adj_candidates(tok)  # assumed already true-ADJ filtered
                # want_g = self._gender_of_lemma(tok.lemma_.lower()) if self.cfg.respect_gender else None
                rep_lemma = _sample_from_candidates(cands, tag, tok, feat=feat)#, want_gender=want_g)
                if rep_lemma is None:
                    _append_token(tok.text, tok, replaced=False)
                    continue
                _append_token(self._realize(rep_lemma, tag), tok, replaced=True)
                continue

            # ADV
            if tok.pos_ == "ADV":
                tag = feat.ptb_adv
                cands = self._adv_candidates(tok)
                rep_lemma = _sample_from_candidates(cands, tag, tok, feat=feat)
                if rep_lemma is None:
                    _append_token(tok.text, tok, replaced=False)
                    continue
                _append_token(self._realize(rep_lemma, tag), tok, replaced=True)
                continue

            # default passthrough
            _append_token(tok.text, tok, replaced=False)

        raw_text = "".join(out)
        if return_meta:
            return raw_text, meta
        return self._fix_indefinite_articles(raw_text)

    @profile
    def _augment_single_doc(self, doc: Doc) -> str:
        """Augment a *single* doc (ideally one sentence), with optional UD round-trip validation.

        NOTE: This version intentionally does *not* do any 'repair' (no incremental freezing of
        suspect slots). It just resamples up to roundtrip_max_tries and falls back to the
        original sentence if nothing passes.
        """
        # Skip empty or whitespace-only docs
        if not doc.text or not doc.text.strip():
            return doc.text

        if not self.cfg.roundtrip_ud:
            return self.augment_doc(doc)

        tries = max(0, self.cfg.roundtrip_max_tries)
        last_attempt = None  # Track the last attempt in case all tries fail
        for t in range(tries + 1):
            out_raw = self.augment_doc(doc)
            last_attempt = out_raw  # Remember this attempt

            # Skip validation for empty or whitespace-only output
            if not out_raw or not out_raw.strip():
                return out_raw

            tgt_doc = self.nlp(out_raw)
            ok, _report = self._roundtrip_compare(doc, tgt_doc)
            if ok:
                fixed = self._fix_indefinite_articles(out_raw)
                if fixed != out_raw:
                    tgt_fixed = self.nlp(fixed)
                    ok_fixed, _ = self._roundtrip_compare(doc, tgt_fixed)
                    if ok_fixed:
                        return fixed
                    # Extremely rare: article fix changes the parse profile. Keep the verified text.
                    dbg(f"[roundtrip] succeeded after {t+1} tries")
                return out_raw
            # Resample by perturbing local RNG slightly.
            try:
                random.randint(0, 2**31 - 1)
            except Exception:
                pass
        # print(t+1)
        dbg("[augmenter] round-trip validation failed; returning last attempt")
        return last_attempt

    def augment(self, text: str) -> str:
        # Fast path
        if not text:
            return text

        # If you *really* want whole-block augmentation, disable this in Config.
        if not self.cfg.split_long_text_by_sentence:
            doc = self.nlp(text)
            return self._augment_single_doc(doc)

        # Parse the entire text and let custom sentence boundaries handle newlines
        # import time
        # start = time.time()
        doc = self.nlp(text)
        # self.nlp_time += time.time() - start
        out_parts: List[str] = []
        # print_out = ""
        # start = time.time()
        for sent_doc in chunk_doc_by_words(doc, max_words=20):
            to_append = self._augment_single_doc(sent_doc)
            out_parts.append(to_append if to_append else "")
            # print_out += out_parts[-1]
            # print(f"ORIG: {sent_doc.text}\nAUG : {print_out}\n" + "-" * 60)
        # self.the_rest += time.time() - start
        # print(self.nlp_time, self.the_rest)
        return "".join(out_parts)

    def augment_parsed_doc(self, doc: Doc) -> str:
        if not doc.text:
            return doc.text

        if not self.cfg.split_long_text_by_sentence:
            return self._augment_single_doc(doc)

        out_parts: List[str] = []
        for sent_doc in chunk_doc_by_words(doc, max_words=20):
            to_append = self._augment_single_doc(sent_doc)
            out_parts.append(to_append if to_append else "")
        return "".join(out_parts)
