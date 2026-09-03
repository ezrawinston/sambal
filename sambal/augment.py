#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Iterable, Iterator, List, Tuple

from sambal.aug_data_utils import (
    JSONL_FORMAT,
    PARQUET_FORMAT,
    ChunkedOutputWriter,
    atomic_json_dump,
    iter_input_records,
    make_resume_state,
    validate_resume_state,
)
from sambal.config import Config, ResourcePaths
from sambal.data_utils import LONG_DOC_WORD_THRESHOLD, iter_batched_parsed_text_records

if TYPE_CHECKING:  # pragma: no cover
    from sambal.engine import Augmenter


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
    args.fmt = "parquet"
    args.text_field = "text"
    args.out = "relexed_shard.parquet"
    args.flush_every_rows = 1000
    args.flush_every_secs = 300
    args.max_rows = 0
    args.resume = True
    args.augmenter_module = "sambal.engine"
    args.start_row = 0
    args.end_row = 0
    args.shard = 0
    args.num_shards = 1

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
        require_gpu=True,
    )
    cfg_defaults.update(overrides.get("config", {}))
    cfg = Config(**cfg_defaults)
    return args, resources, cfg


def shard_bounds(n_rows: int, shard: int, num_shards: int) -> Tuple[int, int]:
    """Half-open ``[start, end)`` input-row range for one shard of a run.

    The split is contiguous and ceil-sized: every shard takes
    ``ceil(n_rows / num_shards)`` rows except the last non-empty one, which
    takes the remainder. The ranges are therefore disjoint and their union is
    exactly ``range(n_rows)``, so concatenating the shard outputs in shard
    order reproduces the whole-input run's row order. With more shards than
    rows the trailing shards get empty ranges, and ``shard_bounds(n, 0, 1)``
    is ``(0, n)`` — the whole input.
    """
    if not isinstance(n_rows, int) or n_rows < 0:
        raise ValueError(f"n_rows must be a non-negative int, got {n_rows!r}")
    if not isinstance(num_shards, int) or num_shards < 1:
        raise ValueError(f"num_shards must be a positive int, got {num_shards!r}")
    if not isinstance(shard, int) or not 0 <= shard < num_shards:
        raise ValueError(
            f"shard must satisfy 0 <= shard < num_shards ({num_shards}), got {shard!r}"
        )
    per_shard = -(-n_rows // num_shards)  # ceil division
    start = min(shard * per_shard, n_rows)
    end = min(start + per_shard, n_rows)
    return start, end


def count_input_records(path: str, *, fmt: str, text_field: str = "text") -> int:
    """Number of records ``iter_input_records`` yields for this input."""
    if fmt == PARQUET_FORMAT:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    if fmt != JSONL_FORMAT:
        raise ValueError(f"Cannot count rows for fmt={fmt!r}; expected jsonl or parquet")
    return sum(1 for _ in iter_input_records(path, fmt=fmt, text_field=text_field))


def resolve_row_range(args, *, count_rows: Callable[[], int]) -> Tuple[int, int]:
    """Resolve the ``[start_row, end_row)`` input-row range for this run.

    ``start_row``/``end_row`` are the primitives (``end_row=0`` means "through
    the last row"); ``shard``/``num_shards`` is sugar for the contiguous
    ceil-split range of :func:`shard_bounds`, and is the only case that needs
    ``count_rows`` (a pass over the input). Slicing selects which input rows a
    run processes and nothing else — in particular the sampler seed keeps
    coming from the config, unchanged.

    The default (``shard=0``, ``num_shards=1``, no explicit rows) is the whole
    input, and takes the same code path as before slicing existed: no row
    count, no truncation.
    """
    start_row = int(getattr(args, "start_row", 0) or 0)
    end_row = int(getattr(args, "end_row", 0) or 0)
    shard = int(getattr(args, "shard", 0) or 0)
    num_shards = int(getattr(args, "num_shards", 1) or 1)

    if shard or num_shards != 1:
        if start_row or end_row:
            raise ValueError(
                "Row range is over-specified: pass either shard/num_shards or "
                f"start_row/end_row, not both (got shard={shard} num_shards={num_shards} "
                f"start_row={start_row} end_row={end_row})"
            )
        return shard_bounds(count_rows(), shard, num_shards)

    if start_row < 0 or end_row < 0:
        raise ValueError(f"Row range must be non-negative, got start_row={start_row} end_row={end_row}")
    if end_row and end_row < start_row:
        raise ValueError(f"Empty or inverted row range: start_row={start_row} end_row={end_row}")
    return start_row, end_row


def format_row_range(start_row: int, end_row: int) -> str:
    end_label = str(end_row) if end_row else "end"
    return f"[{start_row},{end_label})"


def iter_until_row(records: Iterable, end_row: int) -> Iterator:
    """Yield records until (excluding) absolute input-row index ``end_row``.

    ``end_row=0`` means "no end". Records only need a ``record_index``
    attribute, so this truncates both the raw-row and the parsed-doc stream at
    the same input row.
    """
    if not end_row:
        yield from records
        return
    for record in records:
        if record.record_index >= end_row:
            return
        yield record


def _format_metrics(
    *,
    start_t: float,
    rows_done: int,
    start_rows: int = 0,
    parse_seconds: float = 0.0,
    augment_seconds: float = 0.0,
) -> str:
    elapsed = max(1e-9, time.time() - start_t)
    session_rows = max(0, int(rows_done) - int(start_rows))
    return (
        f"elapsed={elapsed:.1f}s rows/s={session_rows / elapsed:.2f} "
        f"parse={parse_seconds:.1f}s ({100.0 * parse_seconds / elapsed:.1f}%) "
        f"augment={augment_seconds:.1f}s ({100.0 * augment_seconds / elapsed:.1f}%)"
    )


def _load_resume_state(state_path: str, args, *, default_index: int = 0) -> Tuple[int, int, bool]:
    if not args.resume or not os.path.exists(state_path):
        return default_index, 0, False

    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    return validate_resume_state(
        state,
        input_path=args.input,
        fmt=args.fmt,
        text_field=args.text_field,
        out_path=args.out,
        output_fmt=args.fmt,
    )


def _write_state(state_path: str, args, *, next_record_index: int, next_part_index: int, completed: bool) -> None:
    state = make_resume_state(
        input_path=args.input,
        fmt=args.fmt,
        text_field=args.text_field,
        out_path=args.out,
        output_fmt=args.fmt,
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


def run_augmentation(args, aug) -> int:
    out_path = str(Path(args.out))
    state_path = out_path + ".state.json"

    start_row, end_row = resolve_row_range(
        args,
        count_rows=lambda: count_input_records(args.input, fmt=args.fmt, text_field=args.text_field),
    )

    writer = ChunkedOutputWriter(
        input_path=args.input,
        fmt=args.fmt,
        text_field=args.text_field,
        out_path=out_path,
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
        print(f"[sambal.augment] already complete out={out_path}", flush=True)
        return 0
    if completed and not os.path.exists(out_path):
        print(f"[sambal.augment] rebuilding completed output from parts out={out_path}", flush=True)
        writer.finalize()
        return 0

    print(
        f"[sambal.augment] input={args.input} fmt={args.fmt} out={out_path} "
        f"row_range={format_row_range(start_row, end_row)}",
        flush=True,
    )
    print(
        f"[sambal.augment] starting at record_index={next_record_index} next_part_index={next_part_index}",
        flush=True,
    )

    start_t = time.time()
    last_flush_t = time.time()
    last_report_t = time.time()
    rows_done = next_record_index
    pending_rows: List[Dict[str, object]] = []
    stopped_early = False
    parse_seconds = 0.0
    augment_seconds = 0.0

    # Both streams are truncated at the same absolute input-row index, so a
    # sliced run sees exactly the rows a whole-input run sees at those indices.
    # The parser batches in windows, so the window straddling end_row is parsed
    # in full before the consumer stops: bounded waste, no effect on output.
    input_rows = iter_until_row(
        iter_input_records(
            args.input,
            fmt=args.fmt,
            text_field=args.text_field,
            start_index=next_record_index,
        ),
        end_row,
    )
    parsed_records = iter_until_row(
        iter_batched_parsed_text_records(
            aug.nlp,
            args.input,
            fmt=args.fmt,
            jsonl_field=args.text_field,
            parquet_field=args.text_field,
            start_index=next_record_index,
            single_parse_predicate=lambda wc: wc > LONG_DOC_WORD_THRESHOLD,
        ),
        end_row,
    )

    parsed_iter = iter(parsed_records)
    for record in input_rows:
        if args.max_rows and (rows_done - start_row) >= args.max_rows:
            stopped_early = True
            break

        parse_t0 = time.perf_counter()
        try:
            parsed_record = next(parsed_iter)
        except StopIteration:
            break
        parse_seconds += time.perf_counter() - parse_t0

        augment_t0 = time.perf_counter()
        augmented_text = aug.augment_parsed_doc(parsed_record.doc)
        augment_seconds += time.perf_counter() - augment_t0

        out_row = dict(record.row)
        out_row[args.text_field] = augmented_text
        pending_rows.append(out_row)
        rows_done = record.record_index + 1

        now = time.time()
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
                f"[sambal.augment] flush rows={rows_done} parts={next_part_index} out={out_path} "
                f"{_format_metrics(start_t=start_t, rows_done=rows_done, start_rows=next_record_index, parse_seconds=parse_seconds, augment_seconds=augment_seconds)}",
                flush=True,
            )
            last_flush_t = now

        if (now - last_report_t) >= 30:
            print(
                f"[sambal.augment] progress rows={rows_done} parts={next_part_index} "
                f"{_format_metrics(start_t=start_t, rows_done=rows_done, start_rows=next_record_index, parse_seconds=parse_seconds, augment_seconds=augment_seconds)}",
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
            f"[sambal.augment] stopped early rows={rows_done} parts={next_part_index} out={out_path} "
            f"{_format_metrics(start_t=start_t, rows_done=rows_done, start_rows=next_record_index, parse_seconds=parse_seconds, augment_seconds=augment_seconds)}",
            flush=True,
        )
        return 0
    print(
        f"[sambal.augment] finalizing rows={rows_done} parts={next_part_index} out={out_path} "
        f"{_format_metrics(start_t=start_t, rows_done=rows_done, start_rows=next_record_index, parse_seconds=parse_seconds, augment_seconds=augment_seconds)}",
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
        f"[sambal.augment] done rows={rows_done} parts={next_part_index} out={out_path} "
        f"{_format_metrics(start_t=start_t, rows_done=rows_done, start_rows=next_record_index, parse_seconds=parse_seconds, augment_seconds=augment_seconds)}",
        flush=True,
    )
    return 0


def add_row_range_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the input-row slicing flags shared by the augmentation drivers."""
    parser.add_argument(
        "--start-row", type=int, default=None,
        help="First input row to process (0-based, inclusive). Default 0.",
    )
    parser.add_argument(
        "--end-row", type=int, default=None,
        help="Stop before this input row (0 = run through the last row). Default 0.",
    )
    parser.add_argument(
        "--shard", type=int, default=None,
        help="Shard index; sugar for the contiguous ceil-split row range of this shard.",
    )
    parser.add_argument(
        "--num-shards", type=int, default=None,
        help="Number of shards the input is split into. Default 1 (whole input).",
    )


def apply_row_range_arguments(args, cli_args) -> None:
    """Let explicit slicing flags override the config's ``args`` block."""
    for name in ("start_row", "end_row", "shard", "num_shards"):
        value = getattr(cli_args, name, None)
        if value is not None:
            setattr(args, name, value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to JSON config overrides")
    add_row_range_arguments(parser)
    cli_args = parser.parse_args()

    args, resources, cfg = build_runtime(cli_args.config)
    apply_row_range_arguments(args, cli_args)
    augmenter_mod = importlib.import_module(args.augmenter_module)
    Augmenter = getattr(augmenter_mod, "Augmenter")
    aug = Augmenter(resources, cfg)
    return run_augmentation(args, aug)


if __name__ == "__main__":
    raise SystemExit(main())
