#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import tracemalloc
import time
from pathlib import Path
from typing import Dict, List

from sambal.aug_data_utils import (
    DOCBIN_FORMAT,
    ChunkedOutputWriter,
    atomic_json_dump,
    iter_docbin_docs,
    load_docbin_meta,
    make_resume_state,
    validate_resume_state,
)
from sambal.augment import (
    add_row_range_arguments,
    apply_row_range_arguments,
    format_row_range,
    resolve_row_range,
)
from sambal.config import Config, ResourcePaths

_CACHE_SNAPSHOT_PREV = {"rows": None, "total_bytes": None}
_MEM_SNAPSHOT_PREV = {"rows": None, "py_cur": None, "trace": None}
_PSUTIL_PROCESS = None

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None


def build_runtime(config_path: str | None):
    overrides = {}
    if config_path:
        with open(config_path) as f:
            overrides = json.load(f)
    from sambal.profile_loader import apply_profile
    # Driver default: this stage consumes pre-parsed DocBins, so no GPU is
    # required by default even under a GPU-parsed profile (the UD-roundtrip
    # reparse, when enabled, runs fine on CPU). An explicit config value
    # still wins.
    overrides = apply_profile(overrides, driver_defaults={"require_gpu": False})

    args = type("", (), {})()
    args.resources = str(Path(__file__).resolve().parent / "resources")
    args.spacy_model = "en_core_web_trf"
    args.input = ""
    args.text_field = "text"
    args.out = "relexed_shard.parquet"
    args.output_fmt = "parquet"
    args.flush_every_rows = 2000
    args.flush_every_secs = 1200
    args.cache_diag = False
    args.cache_diag_every_rows = 0
    args.max_rows = 0
    args.resume = True
    args.augmenter_module = "sambal.engine"
    args.mem_diag = False
    args.mem_diag_tracemalloc = False
    args.mem_diag_every_rows = 64
    args.mem_diag_top_structs = 8
    args.mem_diag_deep = False
    args.ablate_no_candidate_cache = False
    args.ablate_no_ctx_level_cache = False
    args.passthrough_text = False
    args.start_row = 0
    args.end_row = 0
    args.shard = 0
    args.num_shards = 1

    args_overrides = overrides.get("args", {})
    for key, value in args_overrides.items():
        setattr(args, key, value)
    if "cache_diag" not in args_overrides:
        args.cache_diag = bool(int(getattr(args, "cache_diag_every_rows", 0) or 0))

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
        ctx_lemma_stats_path=str(res / "lemma_stats_top_25k.pkl"),
    )
    paths_defaults.update(overrides.get("paths", {}))
    resources = ResourcePaths(**paths_defaults)

    cfg_defaults = dict(
        spacy_model=args.spacy_model,
        seed=42,
        replace_propn=True,
        roundtrip_debug=False,
        debug=False,
        debug_vn=False,
        replace_pronouns=False,
        roundtrip_max_tries=4,
        roundtrip_ud=False,
        freeze_aux_hosts=False,
        min_vocab_freq=2,
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
        require_gpu=False,
    )
    cfg_defaults.update(overrides.get("config", {}))
    cfg = Config(**cfg_defaults)
    return args, resources, cfg


def _rss_mb() -> float:
    """Return current RSS in MB (not peak). Falls back to maxrss if /proc unavailable."""
    try:
        with open("/proc/self/statm", "r") as f:
            pages = int(f.read().split()[1])  # resident pages
        import os
        return pages * os.sysconf("SC_PAGE_SIZE") / 1e6
    except Exception:
        pass
    import resource, sys
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return ru / 1e6
    return ru / 1024


def _process_mem_mb() -> Dict[str, float]:
    out = {"rss": _rss_mb()}
    if psutil is None:
        return out
    global _PSUTIL_PROCESS
    try:
        if _PSUTIL_PROCESS is None:
            _PSUTIL_PROCESS = psutil.Process(os.getpid())
        mi = _PSUTIL_PROCESS.memory_full_info()
        if hasattr(mi, "uss"):
            out["uss"] = float(mi.uss) / 1e6
        if hasattr(mi, "pss"):
            out["pss"] = float(mi.pss) / 1e6
    except Exception:
        pass
    return out


def _summarize_aug_structures(aug, *, top_n: int = 8, deep: bool = False) -> str:
    try:
        pairs = []
        attrs = getattr(aug, "__dict__", {})
        for name, val in attrs.items():
            if not isinstance(val, (dict, list, set, tuple)):
                continue
            try:
                ln = len(val)
            except Exception:
                ln = -1
            try:
                if deep and hasattr(aug, "_estimate_object_size_bytes"):
                    sz = int(aug._estimate_object_size_bytes(val, max_nodes=2_000_000, max_items_per_container=100_000))
                else:
                    import sys

                    sz = int(sys.getsizeof(val))
            except Exception:
                sz = 0
            pairs.append((sz, name, ln))
        pairs.sort(reverse=True)
        return ",".join(f"{name}:{ln}:{sz / 1e6:.1f}MB" for sz, name, ln in pairs[: max(0, int(top_n))])
    except Exception:
        return ""


def _memory_stats(aug, *, rows_done: int, top_structs: int = 8, deep_structs: bool = False) -> str:
    global _MEM_SNAPSHOT_PREV
    parts = []
    mem = _process_mem_mb()
    parts.append(f"rss={mem.get('rss', 0.0):.0f}MB")
    if "uss" in mem:
        parts.append(f"uss={mem['uss']:.0f}MB")
    if "pss" in mem:
        parts.append(f"pss={mem['pss']:.0f}MB")

    if tracemalloc.is_tracing():
        cur, peak = tracemalloc.get_traced_memory()
        parts.append(f"py_cur={cur / 1e6:.1f}MB")
        parts.append(f"py_peak={peak / 1e6:.1f}MB")
        prev_rows = _MEM_SNAPSHOT_PREV.get("rows")
        prev_cur = _MEM_SNAPSHOT_PREV.get("py_cur")
        if prev_rows is not None and prev_cur is not None and rows_done > int(prev_rows):
            d_cur = cur - int(prev_cur)
            parts.append(f"py_dmb={d_cur / 1e6:.1f}")
            parts.append(f"py_dkb_per_row={(d_cur / max(1, rows_done - int(prev_rows))) / 1e3:.1f}")
        try:
            cur_snap = tracemalloc.take_snapshot()
            prev_snap = _MEM_SNAPSHOT_PREV.get("trace")
            if prev_snap is not None:
                diffs = cur_snap.compare_to(prev_snap, "filename")
                pos = [d for d in diffs if d.size_diff > 0]
                if pos:
                    top = pos[:3]
                    parts.append(
                        "py_top="
                        + ",".join(
                            f"{Path(str(d.traceback[0].filename)).name}:{d.size_diff / 1e6:.1f}MB"
                            for d in top
                            if d.traceback
                        )
                    )
            _MEM_SNAPSHOT_PREV = {"rows": int(rows_done), "py_cur": int(cur), "trace": cur_snap}
        except Exception:
            _MEM_SNAPSHOT_PREV = {"rows": int(rows_done), "py_cur": int(cur), "trace": None}

    try:
        g0, g1, g2 = gc.get_count()
        parts.append(f"gc={g0}/{g1}/{g2}")
    except Exception:
        pass

    try:
        vocab = getattr(aug, "nlp", None).vocab if getattr(aug, "nlp", None) is not None else None
        if vocab is not None:
            parts.append(f"vocab_str={len(vocab.strings)}")
            parts.append(f"vocab_lex={len(vocab)}")
    except Exception:
        pass

    top = _summarize_aug_structures(aug, top_n=top_structs, deep=deep_structs)
    if top:
        parts.append(f"obj_top={top}")
    return " ".join(parts)


def _cache_stats(aug, *, rows_done: int | None = None, update_snapshot: bool = True) -> str:
    try:
        if hasattr(aug, "cache_diagnostics_snapshot"):
            snap = aug.cache_diagnostics_snapshot()
            c = snap.get("caches", {})
            total_bytes = int(snap.get("total_bytes", 0))
            ordered = ["v", "n", "a", "i", "b", "l", "p"]
            entries = "/".join(f"{k}{int(c.get(k, {}).get('entries', 0))}" for k in ordered)
            bytes_est = "/".join(f"{k}{int(c.get(k, {}).get('bytes_est', 0))}" for k in ordered)
            bpe = "/".join(f"{k}{c.get(k, {}).get('bytes_per_entry', 0.0):.1f}" for k in ordered)
            hit = "/".join(f"{k}{c.get(k, {}).get('hit_rate', 0.0):.3f}" for k in ordered)
            ops = "/".join(f"{k}{int(c.get(k, {}).get('ops', 0))}" for k in ordered)
            insert_avg = "/".join(f"{k}{c.get(k, {}).get('insert_delta_avg_bytes', 0.0):.1f}" for k in ordered)
            insert_ops = "/".join(f"{k}{int(c.get(k, {}).get('insertions', 0))}" for k in ordered)
            parts = [
                f"caches={entries}",
                f"cache_bytes={bytes_est}",
                f"cache_bpe={bpe}",
                f"cache_hit={hit}",
                f"cache_ops={ops}",
                f"cache_insert_bpi={insert_avg}",
                f"cache_insert_ops={insert_ops}",
                f"cache_total_mb={total_bytes / 1e6:.1f}",
            ]
            global _CACHE_SNAPSHOT_PREV
            prev_rows = _CACHE_SNAPSHOT_PREV.get("rows")
            prev_total = _CACHE_SNAPSHOT_PREV.get("total_bytes")
            if rows_done is not None and prev_rows is not None and prev_total is not None:
                d_rows = int(rows_done) - int(prev_rows)
                d_bytes = int(total_bytes) - int(prev_total)
                if d_rows > 0:
                    parts.append(f"cache_dbytes_mb={d_bytes / 1e6:.1f}")
                    parts.append(f"cache_dbytes_per_row_kb={(d_bytes / d_rows) / 1e3:.1f}")
            if update_snapshot and rows_done is not None:
                _CACHE_SNAPSHOT_PREV = {"rows": int(rows_done), "total_bytes": int(total_bytes)}
            return " ".join(parts)

        vc = len(getattr(aug, "_verb_candidates_cache", {}))
        nc = len(getattr(aug, "_noun_candidates_cache", {}))
        ac = len(getattr(aug, "_adj_candidates_cache", {}))
        ic = len(getattr(aug, "_inflect_cache", {}))
        bc = len(getattr(aug, "_ctx_bucket_level_stats_cache", {}))
        pc = len(getattr(aug, "_pos_cache", {}))
        return f"caches=v{vc}/n{nc}/a{ac}/i{ic}/b{bc}/p{pc}"
    except Exception:
        return "caches=?"


def _format_metrics(
    *,
    start_t: float,
    rows_done: int,
    start_rows: int = 0,
    augment_seconds: float = 0.0,
    aug=None,
    doc_len: int = 0,
    include_cache_stats: bool = False,
) -> str:
    elapsed = max(1e-9, time.time() - start_t)
    session_rows = max(0, int(rows_done) - int(start_rows))
    parts = [
        f"elapsed={elapsed:.1f}s rows/s={session_rows / elapsed:.2f}",
        f"augment={augment_seconds:.1f}s ({100.0 * augment_seconds / elapsed:.1f}%)",
        f"rss={_rss_mb():.0f}MB",
    ]
    if include_cache_stats and aug is not None:
        parts.append(_cache_stats(aug, rows_done=rows_done, update_snapshot=True))
    if doc_len:
        parts.append(f"last_doc_len={doc_len}")
    return " ".join(parts)


def _load_resume_state(state_path: str, args, *, default_index: int = 0):
    if not args.resume or not os.path.exists(state_path):
        return default_index, 0, False
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    return validate_resume_state(
        state,
        input_path=args.input,
        fmt=DOCBIN_FORMAT,
        text_field=args.text_field,
        out_path=args.out,
        output_fmt=args.output_fmt,
    )


def _write_state(state_path: str, args, *, next_record_index: int, next_part_index: int, completed: bool) -> None:
    state = make_resume_state(
        input_path=args.input,
        fmt=DOCBIN_FORMAT,
        text_field=args.text_field,
        out_path=args.out,
        output_fmt=args.output_fmt,
        next_record_index=next_record_index,
        next_part_index=next_part_index,
        completed=completed,
    )
    atomic_json_dump(state, state_path)


def _flush_pending(
    pending_rows: List[Dict[str, object]],
    writer: ChunkedOutputWriter,
    state_path: str,
    args,
    *,
    next_record_index: int,
    next_part_index: int,
) -> int:
    if not pending_rows:
        return next_part_index
    writer.write_chunk(pending_rows, part_index=next_part_index)
    _write_state(
        state_path,
        args,
        next_record_index=next_record_index,
        next_part_index=next_part_index + 1,
        completed=False,
    )
    pending_rows.clear()
    return next_part_index + 1


def run_augmentation_from_docbin(args, aug) -> int:
    global _CACHE_SNAPSHOT_PREV
    global _MEM_SNAPSHOT_PREV
    _CACHE_SNAPSHOT_PREV = {"rows": None, "total_bytes": None}
    _MEM_SNAPSHOT_PREV = {"rows": None, "py_cur": None, "trace": None}
    if (
        bool(getattr(args, "mem_diag", False))
        and bool(getattr(args, "mem_diag_tracemalloc", False))
        and not tracemalloc.is_tracing()
    ):
        tracemalloc.start(25)
    out_path = str(Path(args.out))
    state_path = out_path + ".state.json"

    start_row, end_row = resolve_row_range(
        args,
        count_rows=lambda: int(load_docbin_meta(args.input)["row_count"]),
    )

    writer = ChunkedOutputWriter(
        input_path=args.input,
        fmt=DOCBIN_FORMAT,
        text_field=args.text_field,
        out_path=out_path,
        output_fmt=args.output_fmt,
    )

    if not args.resume:
        if os.path.exists(state_path):
            os.remove(state_path)
        writer.reset()

    next_record_index, next_part_index, completed = _load_resume_state(
        state_path, args, default_index=start_row
    )
    if next_record_index < start_row or (end_row and next_record_index > end_row):
        raise ValueError(
            f"Resume state {state_path} is at record_index={next_record_index}, outside this run's "
            f"row range {format_row_range(start_row, end_row)}. Give each row range its own output "
            f"path, or delete that state file (or set resume=false) to restart this range."
        )
    writer.prune_parts(next_part_index)
    if completed and os.path.exists(out_path):
        print(f"[sambal.augment_from_docbin] already complete out={out_path}", flush=True)
        return 0
    if completed and not os.path.exists(out_path):
        print(f"[sambal.augment_from_docbin] rebuilding completed output from parts out={out_path}", flush=True)
        writer.finalize()
        return 0

    meta = load_docbin_meta(args.input)
    print(
        f"[sambal.augment_from_docbin] input={args.input} rows={meta['row_count']} output_fmt={args.output_fmt} "
        f"out={out_path} row_range={format_row_range(start_row, end_row)}",
        flush=True,
    )
    print(
        f"[sambal.augment_from_docbin] starting at record_index={next_record_index} next_part_index={next_part_index}",
        flush=True,
    )
    if bool(getattr(args, "cache_diag", False)):
        print(
            f"[sambal.augment_from_docbin] cache_diag rows={next_record_index} parts={next_part_index} "
            f"{_format_metrics(start_t=time.time(), rows_done=next_record_index, start_rows=next_record_index, augment_seconds=0.0, aug=aug, doc_len=0, include_cache_stats=True)}",
            flush=True,
        )
    if bool(getattr(args, "mem_diag", False)):
        print(
            f"[sambal.augment_from_docbin] mem_diag rows={next_record_index} parts={next_part_index} "
            f"{_memory_stats(aug, rows_done=next_record_index, top_structs=int(getattr(args, 'mem_diag_top_structs', 8) or 8), deep_structs=bool(getattr(args, 'mem_diag_deep', False)))}",
            flush=True,
        )

    start_t = time.time()
    last_flush_t = time.time()
    last_report_t = time.time()
    rows_done = next_record_index
    pending_rows: List[Dict[str, object]] = []
    stopped_early = False
    augment_seconds = 0.0
    last_doc_len = 0

    docs = iter_docbin_docs(args.input, aug.nlp.vocab)
    for record_index, doc in enumerate(docs):
        if record_index < next_record_index:
            continue
        if end_row and record_index >= end_row:
            break
        if args.max_rows and (rows_done - start_row) >= args.max_rows:
            stopped_early = True
            break

        last_doc_len = len(doc)
        if bool(getattr(args, "ablate_no_candidate_cache", False)):
            for attr in ("_verb_candidates_cache", "_noun_candidates_cache", "_adj_candidates_cache"):
                obj = getattr(aug, attr, None)
                if isinstance(obj, dict):
                    obj.clear()
        if bool(getattr(args, "ablate_no_ctx_level_cache", False)):
            obj = getattr(aug, "_ctx_bucket_level_stats_cache", None)
            if isinstance(obj, dict):
                obj.clear()
        augment_t0 = time.perf_counter()
        if bool(getattr(args, "passthrough_text", False)):
            augmented_text = doc.text
        else:
            augmented_text = aug.augment_parsed_doc(doc)
        augment_seconds += time.perf_counter() - augment_t0

        pending_rows.append({args.text_field: augmented_text})
        rows_done = record_index + 1

        now = time.time()
        cache_diag_every_rows = int(getattr(args, "cache_diag_every_rows", 0) or 0)
        if bool(getattr(args, "cache_diag", False)) and cache_diag_every_rows and (rows_done % cache_diag_every_rows == 0):
            print(
                f"[sambal.augment_from_docbin] cache_diag rows={rows_done} parts={next_part_index} "
                f"{_format_metrics(start_t=start_t, rows_done=rows_done, start_rows=next_record_index, augment_seconds=augment_seconds, aug=aug, doc_len=last_doc_len, include_cache_stats=True)}",
                flush=True,
            )
            if bool(getattr(args, "mem_diag", False)):
                print(
                    f"[sambal.augment_from_docbin] mem_diag rows={rows_done} parts={next_part_index} "
                    f"{_memory_stats(aug, rows_done=rows_done, top_structs=int(getattr(args, 'mem_diag_top_structs', 8) or 8), deep_structs=bool(getattr(args, 'mem_diag_deep', False)))}",
                    flush=True,
                )
        if (rows_done % args.flush_every_rows == 0) or ((now - last_flush_t) >= args.flush_every_secs):
            next_part_index = _flush_pending(
                pending_rows,
                writer,
                state_path,
                args,
                next_record_index=rows_done,
                next_part_index=next_part_index,
            )
            print(
                f"[sambal.augment_from_docbin] flush rows={rows_done} parts={next_part_index} "
                f"{_format_metrics(start_t=start_t, rows_done=rows_done, start_rows=next_record_index, augment_seconds=augment_seconds, aug=aug, doc_len=last_doc_len, include_cache_stats=bool(getattr(args, 'cache_diag', False)))}",
                flush=True,
            )
            last_flush_t = now

        if (now - last_report_t) >= 30:
            print(
                f"[sambal.augment_from_docbin] progress rows={rows_done} parts={next_part_index} "
                f"{_format_metrics(start_t=start_t, rows_done=rows_done, start_rows=next_record_index, augment_seconds=augment_seconds, aug=aug, doc_len=last_doc_len, include_cache_stats=bool(getattr(args, 'cache_diag', False)))}",
                flush=True,
            )
            last_report_t = now

    next_part_index = _flush_pending(
        pending_rows,
        writer,
        state_path,
        args,
        next_record_index=rows_done,
        next_part_index=next_part_index,
    )
    if stopped_early:
        print(
            f"[sambal.augment_from_docbin] stopped early rows={rows_done} parts={next_part_index} "
            f"{_format_metrics(start_t=start_t, rows_done=rows_done, start_rows=next_record_index, augment_seconds=augment_seconds, aug=aug, doc_len=last_doc_len, include_cache_stats=bool(getattr(args, 'cache_diag', False)))}",
            flush=True,
        )
        return 0
    print(
        f"[sambal.augment_from_docbin] finalizing rows={rows_done}/{meta['row_count']} parts={next_part_index} "
        f"{_format_metrics(start_t=start_t, rows_done=rows_done, start_rows=next_record_index, augment_seconds=augment_seconds, aug=aug, doc_len=last_doc_len, include_cache_stats=bool(getattr(args, 'cache_diag', False)))}",
        flush=True,
    )
    writer.finalize()
    _write_state(
        state_path,
        args,
        next_record_index=rows_done,
        next_part_index=next_part_index,
        completed=True,
    )
    print(
        f"[sambal.augment_from_docbin] done rows={rows_done} parts={next_part_index} out={out_path} "
        f"{_format_metrics(start_t=start_t, rows_done=rows_done, start_rows=next_record_index, augment_seconds=augment_seconds, aug=aug, doc_len=last_doc_len, include_cache_stats=bool(getattr(args, 'cache_diag', False)))}",
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to JSON config overrides")
    add_row_range_arguments(parser)
    cli_args = parser.parse_args()

    args, resources, cfg = build_runtime(cli_args.config)
    apply_row_range_arguments(args, cli_args)
    print(f"[sambal.augment_from_docbin] pre-import rss={_rss_mb():.0f}MB", flush=True)
    augmenter_mod = importlib.import_module(args.augmenter_module)
    Augmenter = getattr(augmenter_mod, "Augmenter")

    print(f"[sambal.augment_from_docbin] post-import rss={_rss_mb():.0f}MB", flush=True)
    aug = Augmenter(resources, cfg)
    print(f"[sambal.augment_from_docbin] post-init rss={_rss_mb():.0f}MB", flush=True)
    return run_augmentation_from_docbin(args, aug)


if __name__ == "__main__":
    raise SystemExit(main())
