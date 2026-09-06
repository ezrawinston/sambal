#!/usr/bin/env python3
"""
Collect lemma-conditioned contextual stats from a corpus using spaCy parses.

Features:
- processes one whole input file
- supports jsonl or parquet input
- uses smart length-aware batching for parsing
- supports real resume via a sidecar state file
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple

from spacy.tokens import Token
from wordfreq import top_n_list

from sambal.config import Config, ResourcePaths
from sambal.data_utils import (
    iter_smart_parsed_records,
    make_resume_state,
    validate_resume_state,
)
from sambal.engine import Augmenter
from sambal.token_features import TOKENINFO_SCHEMA, TokenFeatures
from sambal.verb_frame_analyzer import verb_child_mask


def atomic_pickle_dump(obj: Any, out_path: str) -> None:
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, out_path)


def atomic_pickle_dump_gz(obj: Any, out_path: str) -> None:
    tmp = out_path + ".tmp"
    with gzip.open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, out_path)


def atomic_json_dump(obj: Dict[str, Any], out_path: str) -> None:
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, out_path)


VERB_TAGS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
NOUN_TAGS = {"NN", "NNS", "NNP", "NNPS"}
PROPN_TAGS = {"NNP", "NNPS"}

_POS_BUCKET_KEEP = {
    "NOUN", "PROPN", "PRON", "VERB", "AUX", "ADJ", "ADV",
    "ADP", "PART", "SCONJ", "CCONJ", "DET", "NUM", "PUNCT",
}


def morph_signature(tok) -> Tuple[Tuple[str, str], ...]:
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


def child_dep_counts(tok) -> Tuple[Tuple[str, int], ...]:
    counts = Counter()
    try:
        for ch in tok.children:
            counts[str(ch.dep_)] += 1
    except Exception:
        pass
    return tuple(sorted(counts.items()))


def safe_lemma(tok) -> str:
    lem = (tok.lemma_ or tok.text or "").strip().lower()
    return lem if lem else (tok.text or "").strip().lower()


def pos_bucket(tok: Token | None) -> str:
    if tok is None:
        return "NONE"
    pos = tok.pos_ or ""
    return pos if pos in _POS_BUCKET_KEEP else "OTHER"


def _compute_ctx_bits(tok, feat: TokenFeatures) -> Tuple[Any, ...]:
    head = tok.head
    relpos = 0
    adv_head_prep = ""
    adj_has_pp = adj_has_xcomp = adj_has_ccomp = 0
    verb_has_to_aux = 0
    head_has_obj = 0
    head_has_aux = 0

    try:
        if head is not None and head.i != tok.i:
            relpos = -1 if tok.i < head.i else 1
    except Exception:
        relpos = 0

    try:
        prev_b = "START" if tok.i == 0 else pos_bucket(tok.doc[tok.i - 1])
        next_b = "END" if tok.i + 1 >= len(tok.doc) else pos_bucket(tok.doc[tok.i + 1])
    except Exception:
        prev_b = next_b = "OTHER"

    if feat.pos == "ADV" and feat.dep == "pcomp" and feat.head_pos == "ADP":
        try:
            adv_head_prep = str((head.lemma_ or "") if head is not None else "").lower()
        except Exception:
            adv_head_prep = ""

    if feat.pos == "ADJ":
        try:
            adj_has_xcomp = int(any(ch.dep_ == "xcomp" for ch in tok.children))
            adj_has_ccomp = int(any(ch.dep_ == "ccomp" for ch in tok.children))
            adj_has_pp = int(any(
                (ch.dep_ == "prep")
                or (
                    ch.dep_ in {"obl", "nmod"}
                    and any(gc.dep_ == "case" and gc.pos_ == "ADP" for gc in ch.children)
                )
                for ch in tok.children
            ))
        except Exception:
            pass

    if feat.is_verb_like:
        try:
            verb_has_to_aux = int(any(
                (ch.dep_ == "aux") and ((ch.tag_ == "TO") or ((ch.lemma_ or "").lower() == "to"))
                for ch in tok.children
            ))
        except Exception:
            verb_has_to_aux = 0

    try:
        if head is not None:
            head_has_obj = int(any(ch.dep_ in {"dobj", "obj"} for ch in head.children))
            head_has_aux = int(any(ch.dep_ == "aux" for ch in head.children))
    except Exception:
        pass

    vtok_child_mask = 0
    if feat.is_verb_like:
        try:
            vtok_child_mask = verb_child_mask(tok)
        except Exception:
            vtok_child_mask = 0

    prev_tag = "START"
    next_tag = "END"
    try:
        if tok.i > 0:
            prev_tag = tok.doc[tok.i - 1].tag_ or tok.doc[tok.i - 1].pos_ or "OTHER"
        if tok.i + 1 < len(tok.doc):
            next_tag = tok.doc[tok.i + 1].tag_ or tok.doc[tok.i + 1].pos_ or "OTHER"
    except Exception:
        prev_tag, next_tag = "OTHER", "OTHER"

    def _m1(name: str) -> str:
        try:
            vals = tok.morph.get(name)
            return vals[0] if vals else ""
        except Exception:
            return ""

    morph_sig = (_m1("Degree"), _m1("NumType"), _m1("PronType"), _m1("Polarity"))

    adp_dep = ""
    grandhead_pos = ""
    grandhead_tag = ""
    if tok.pos_ == "ADV" and tok.dep_ == "pcomp" and tok.head is not None and tok.head.pos_ == "ADP":
        try:
            adp_dep = tok.head.dep_ or ""
            if tok.head.head is not None:
                grandhead_pos = tok.head.head.pos_ or ""
                grandhead_tag = tok.head.head.tag_ or ""
        except Exception:
            pass

    return (
        "ctx_v3",
        relpos, prev_b, next_b, adv_head_prep,
        adj_has_pp, adj_has_xcomp, adj_has_ccomp,
        verb_has_to_aux,
        head_has_obj, head_has_aux,
        vtok_child_mask,
        prev_tag, next_tag,
        morph_sig,
        adp_dep, grandhead_pos, grandhead_tag,
    )


def tokeninfo_key(
    tok,
    *,
    aug: Augmenter,
    protected: bool,
    licensor_strength: str,
    in_fixed_mwe: bool,
    in_pattern_mwe: bool,
    functionish: bool,
) -> Tuple[Any, ...]:
    feat = aug._compute_token_features(tok)
    flags = (
        int(tok.is_sent_start),
        int(tok.is_alpha),
        int(tok.is_stop),
        int(tok.is_title),
        int(tok.is_upper),
        int(tok.is_lower),
        int(tok.like_num),
        int(tok.is_punct),
    )
    cdeps = child_dep_counts(tok)
    lem = safe_lemma(tok)
    prot_bits = (
        int(protected),
        int(functionish),
        str(licensor_strength or ""),
        int(in_fixed_mwe),
        int(in_pattern_mwe),
    )
    ctx_bits = _compute_ctx_bits(tok, feat)
    return (
        "tokinfo_v2",
        lem, int(feat.is_bare_singular),
        feat.pos, feat.tag, feat.dep, feat.morph_sig,
        feat.head_pos, feat.head_tag, feat.head_dep, int(feat.head_is_self),
        flags,
        cdeps,
        feat.ptb_noun, feat.ptb_noun_forced, feat.ptb_adj, feat.ptb_adv,
        int(feat.is_verb_like), feat.frame_key, int(feat.frame_has_prt), feat.frame_req_prep, feat.preps_set,
        feat.comp_kind, feat.comp_kind_simple,
        int(feat.is_passive), int(feat.has_aux), int(feat.has_heavy_aux), int(feat.has_expl_there), int(feat.is_to_be_xcomp),
        feat.ccomp_marks, int(feat.xcomp_has_to), int(feat.xcomp_has_for), int(feat.xcomp_is_ger), feat.ptb_verb,
        prot_bits,
        ctx_bits,
    )
def build_runtime(config_path: str | None):
    overrides = {}
    if config_path:
        with open(config_path) as f:
            overrides = json.load(f)
    from sambal.profile_loader import apply_profile
    overrides = apply_profile(overrides)

    args = type("", (), {})()
    args.resources = str(Path(__file__).resolve().parent / "resources")
    args.spacy_model = "en_core_web_trf"
    args.input = ""
    args.fmt = "jsonl"
    args.jsonl_field = "text"
    args.parquet_field = "text"
    args.out = "lemma_stats.pkl"
    args.flush_every_docs = 10000
    args.flush_every_secs = 1200
    args.max_docs = 0
    args.require_gpu = True
    args.prefer_gpu = True
    args.batch_size = 128
    args.resume = False
    args.no_chunk = False

    for key, value in overrides.get("args", {}).items():
        setattr(args, key, value)

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
        propn_list_path=None,
        given_names_path=str(res / "given_names_ssa.txt"),
        fixed_mwes_path=str(res / "fixed_mwes_all.txt"),
        mwe_patterns_jsonl_path=str(res / "streusle_vmwe_patterns.jsonl"),
        human_nouns_path=str(res / "human_nouns.jsonl"),
        allowed_vocab_path=str(res / "allowed_vocab.txt"),
        allow_all_vocab_path=None,
        licensor_patterns_jsonl_path=str(res / "licensor_patterns_full.jsonl"),
        toinf_verb_lexicon_jsonl=str(res / "toinf_verbs.jsonl"),
        toinf_adj_lexicon_jsonl=str(res / "toinf_adjs.jsonl"),
        given_names_male_path=str(res / "male_given_names.txt"),
        given_names_female_path=str(res / "female_given_names.txt"),
        given_names_neutral_path=str(res / "neutral_given_names.txt"),
        gendered_words_json_path=str(res / "gendered_words_filtered.json"),
    )
    paths_defaults.update(overrides.get("paths", {}))
    resources = ResourcePaths(**paths_defaults)

    cfg_defaults = dict(
        spacy_model=args.spacy_model,
        seed=42,
        replace_propn=True,
        roundtrip_debug=True,
        debug=False,
        replace_pronouns=False,
        roundtrip_max_tries=10,
        roundtrip_ud=True,
        freeze_aux_hosts=False,
        min_vocab_freq=300000,
        respect_human=True,
        respect_gender=True,
        debug_vn=True,
        verb_metrics=True,
        prefilter_pools_by_allowed_vocab=True,
        filter_substitution_names_by_allowed_vocab=True,
    )
    cfg_defaults.update(overrides.get("config", {}))
    # The collector reads the parse, the protection sets and the function-word
    # test; it never samples. So the context gate, whose table is built from
    # the file this collector produces, stays off, and the start-up caches are
    # neither read nor written.
    cfg_defaults["ctx_lemma_gate"] = "off"
    cfg_defaults["startup_caches"] = False
    cfg = Config(**cfg_defaults)
    return args, resources, cfg


def maybe_enable_gpu(args) -> None:
    if not (args.prefer_gpu or args.require_gpu):
        return

    import spacy

    if args.require_gpu:
        spacy.require_gpu()
        print("[sambal.stats] GPU REQUIRED and enabled.")
    else:
        ok = spacy.prefer_gpu()
        print(f"[sambal.stats] prefer_gpu() -> {ok}")

    try:
        import torch
        torch.set_grad_enabled(False)
    except Exception:
        pass


def load_resume_state(state_path: str, out_path: str, args) -> Tuple[Dict[str, Counter], int]:
    lemma2counter: Dict[str, Counter] = defaultdict(Counter)
    next_record_index = 0

    if not args.resume:
        return lemma2counter, next_record_index

    out_exists = os.path.exists(out_path)
    state_exists = os.path.exists(state_path)
    if not out_exists and not state_exists:
        return lemma2counter, next_record_index
    if out_exists != state_exists:
        raise ValueError(
            f"Resume requires both output and state file to exist. out_exists={out_exists} state_exists={state_exists}"
        )

    print(f"[sambal.stats] resuming from {out_path}")
    if out_path.endswith(".gz"):
        with gzip.open(out_path, "rb") as f:
            lemma2counter = pickle.load(f)
    else:
        with open(out_path, "rb") as f:
            lemma2counter = pickle.load(f)
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    next_record_index = validate_resume_state(
        state,
        input_path=args.input,
        fmt=args.fmt,
        jsonl_field=args.jsonl_field,
        parquet_field=args.parquet_field,
    )
    return lemma2counter, next_record_index


def write_checkpoint(
    lemma2counter: Dict[str, Counter],
    out_path: str,
    state_path: str,
    args,
    *,
    next_record_index: int,
) -> None:
    if out_path.endswith(".gz"):
        atomic_pickle_dump_gz(lemma2counter, out_path)
    else:
        atomic_pickle_dump(lemma2counter, out_path)
    state = make_resume_state(
        input_path=args.input,
        fmt=args.fmt,
        jsonl_field=args.jsonl_field,
        parquet_field=args.parquet_field,
        next_record_index=next_record_index,
    )
    atomic_json_dump(state, state_path)


def _format_runtime_metrics(
    *,
    start_t: float,
    records_done: int,
    docs_done: int,
    parse_seconds: float,
    stats_seconds: float,
) -> str:
    elapsed = max(1e-9, time.time() - start_t)
    return (
        f"elapsed={elapsed:.1f}s "
        f"rec/s={records_done / elapsed:.2f} "
        f"docs/s={docs_done / elapsed:.2f} "
        f"parse={parse_seconds:.1f}s ({100.0 * parse_seconds / elapsed:.1f}%) "
        f"stats={stats_seconds:.1f}s ({100.0 * stats_seconds / elapsed:.1f}%)"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to JSON config overrides")
    cli_args = parser.parse_args()

    args, resources, cfg = build_runtime(cli_args.config)
    maybe_enable_gpu(args)

    if getattr(args, "no_chunk", False):
        # Whole-document parsing: disable the long-document chunking so very
        # long documents are parsed in one pass. This is the faithful setting
        # for regenerating the ICML 2026 paper's context statistics.
        from sambal import data_utils
        data_utils.LONG_DOC_WORD_THRESHOLD = 10 ** 9
        print("[sambal.stats] long-document chunking disabled (whole-document parsing)")

    if not cfg.spacy_model:
        cfg.spacy_model = "en_core_web_trf"
    aug = Augmenter(resources, cfg)

    schema = TOKENINFO_SCHEMA
    top_n = set(top_n_list("en", 25000))

    out_path = str(Path(args.out))
    state_path = out_path + ".state.json"
    lemma2counter, next_record_index = load_resume_state(state_path, out_path, args)

    print(f"[sambal.stats] input={args.input} fmt={args.fmt} out={out_path}")
    print(f"[sambal.stats] starting at record_index={next_record_index}")

    start_t = time.time()
    records_done = next_record_index
    docs_done = 0
    parse_seconds = 0.0
    stats_seconds = 0.0
    last_flush_t = time.time()
    last_report_t = time.time()

    parsed_iter = iter_smart_parsed_records(
        aug.nlp,
        args.input,
        fmt=args.fmt,
        jsonl_field=args.jsonl_field,
        parquet_field=args.parquet_field,
        start_index=next_record_index,
    )

    while True:
        parse_t0 = time.perf_counter()
        try:
            parsed_record = next(parsed_iter)
        except StopIteration:
            parse_seconds += time.perf_counter() - parse_t0
            break
        parse_seconds += time.perf_counter() - parse_t0

        if args.max_docs and records_done >= args.max_docs:
            break

        stats_t0 = time.perf_counter()
        for doc in parsed_record.docs:
            docs_done += 1
            try:
                prot, fixed_mwe_idxs, patt_mwe_idxs = aug._protect_and_mwe_sets(doc)
            except Exception:
                prot = set()
                fixed_mwe_idxs = set()
                patt_mwe_idxs = set()

            if aug.licensor_matcher:
                try:
                    conservative_guard = aug.cfg.conservative_guard
                    licensor_covered = aug.licensor_matcher.cover_strengths(
                        doc, conservative_guard=conservative_guard
                    )
                except Exception:
                    licensor_covered = {}
            else:
                licensor_covered = {}

            for tok in doc:
                lem = safe_lemma(tok)
                is_propn = tok.pos_ == "PROPN" or tok.tag_ in PROPN_TAGS
                if is_propn:
                    continue
                if lem not in top_n:
                    continue

                lic_strength = str(licensor_covered.get(tok.i, "")) if licensor_covered else ""
                is_func = bool(aug._is_functionish(tok, licensor_covered))
                if is_func:
                    continue

                key = tokeninfo_key(
                    tok,
                    aug=aug,
                    protected=(tok.i in prot),
                    licensor_strength=lic_strength,
                    in_fixed_mwe=(tok.i in fixed_mwe_idxs),
                    in_pattern_mwe=(tok.i in patt_mwe_idxs),
                    functionish=is_func,
                )
                bucket = Augmenter._bucket_from_tokinfo(key, schema=schema)
                lemma2counter[lem][bucket] += 1
        stats_seconds += time.perf_counter() - stats_t0

        records_done = parsed_record.record_index + 1
        next_record_index = records_done
        now = time.time()
        if (records_done % args.flush_every_docs == 0) or ((now - last_flush_t) >= args.flush_every_secs):
            print(
                f"[sambal.stats] flush @record={records_done} parsed_docs={docs_done} "
                f"lemmas={len(lemma2counter)} out={out_path} "
                f"{_format_runtime_metrics(start_t=start_t, records_done=records_done, docs_done=docs_done, parse_seconds=parse_seconds, stats_seconds=stats_seconds)}",
                flush=True,
            )
            write_checkpoint(
                lemma2counter,
                out_path,
                state_path,
                args,
                next_record_index=next_record_index,
            )
            last_flush_t = now

        if (now - last_report_t) >= 30:
            print(
                f"[sambal.stats] progress records={records_done} parsed_docs={docs_done} "
                f"lemmas={len(lemma2counter)} "
                f"{_format_runtime_metrics(start_t=start_t, records_done=records_done, docs_done=docs_done, parse_seconds=parse_seconds, stats_seconds=stats_seconds)}",
                flush=True,
            )
            last_report_t = now

    print(
        f"[sambal.stats] done records={records_done} parsed_docs={docs_done} "
        f"lemmas={len(lemma2counter)} writing {out_path} "
        f"{_format_runtime_metrics(start_t=start_t, records_done=records_done, docs_done=docs_done, parse_seconds=parse_seconds, stats_seconds=stats_seconds)}",
        flush=True,
    )
    write_checkpoint(
        lemma2counter,
        out_path,
        state_path,
        args,
        next_record_index=next_record_index,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
