from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - optional at import time
    pq = None


JSONL_FORMAT = "jsonl"
PARQUET_FORMAT = "parquet"
SUPPORTED_FORMATS = {JSONL_FORMAT, PARQUET_FORMAT}

LONG_DOC_WORD_THRESHOLD = 1500
LONG_DOC_CHUNK_WORDS = 600
# Based on the observed shard distribution, 512 records is enough to find
# plenty of same-bucket neighbors for the 1k+ word tail without buffering
# as much text before parsing starts.
WINDOW_RECORDS = 512

_BATCH_SCHEDULE: Sequence[Tuple[int, int]] = (
    (128, 256),
    (256, 128),
    (512, 64),
    (1024, 32),
    (2048, 8),
)
LONG_DOC_BATCH_SIZE = 4


def chunk_doc_by_words(doc, max_words):
    from sambal.engine import chunk_doc_by_words as _chunk_doc_by_words

    return _chunk_doc_by_words(doc, max_words=max_words)


@dataclass(frozen=True)
class ParsedRecord:
    record_index: int
    docs: Tuple[Any, ...]
    word_count: int
    was_chunked: bool


@dataclass(frozen=True)
class ParsedTextRecord:
    record_index: int
    text: str
    doc: Any
    word_count: int
    was_single_parsed: bool


def whitespace_word_count(text: str) -> int:
    return len(text.split())


def batch_size_for_word_count(word_count: int) -> int:
    for max_words, batch_size in _BATCH_SCHEDULE:
        if word_count <= max_words:
            return batch_size
    return LONG_DOC_BATCH_SIZE


def _open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "rt", encoding="utf-8", errors="ignore")


def _iter_jsonl_records(path: str, jsonl_field: str) -> Iterator[str]:
    with _open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = obj.get(jsonl_field)
            if isinstance(text, str):
                text = text.strip()
                if text:
                    yield text


def _iter_parquet_records(path: str, parquet_field: str) -> Iterator[str]:
    if pq is None:
        raise ImportError("pyarrow is required for parquet input")
    parquet = pq.ParquetFile(path)
    schema_names = set(parquet.schema_arrow.names)
    if parquet_field not in schema_names:
        raise ValueError(
            f"Parquet column {parquet_field!r} not found in {path}. "
            f"Available columns: {sorted(schema_names)}"
        )
    for rg_idx in range(parquet.num_row_groups):
        table = parquet.read_row_group(rg_idx, columns=[parquet_field])
        column = table.column(parquet_field)
        for i in range(len(column)):
            text = column[i].as_py()
            if isinstance(text, str):
                text = text.strip()
                if text:
                    yield text


def iter_text_records(
    path: str,
    *,
    fmt: str,
    jsonl_field: str = "text",
    parquet_field: str = "text",
    start_index: int = 0,
) -> Iterator[Tuple[int, str]]:
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported fmt={fmt!r}; expected one of {sorted(SUPPORTED_FORMATS)}")

    if fmt == JSONL_FORMAT:
        text_iter = _iter_jsonl_records(path, jsonl_field)
    else:
        text_iter = _iter_parquet_records(path, parquet_field)

    for idx, text in enumerate(text_iter):
        if idx < start_index:
            continue
        yield idx, text


def _bucket_key(word_count: int) -> int:
    for max_words, _ in _BATCH_SCHEDULE:
        if word_count <= max_words:
            return max_words
    return _BATCH_SCHEDULE[-1][0]


def _process_window(
    nlp,
    window: Sequence[Tuple[int, str, int]],
    *,
    single_parse_predicate: Optional[Callable[[int], bool]] = None,
) -> List[ParsedTextRecord]:
    buckets: Dict[int, List[Tuple[int, str, int]]] = {}
    parsed_by_index: Dict[int, ParsedTextRecord] = {}
    max_chars = max((len(text) for _, text, _ in window), default=0)
    if hasattr(nlp, "max_length") and max_chars > getattr(nlp, "max_length", 0):
        # spaCy enforces max_length in characters, independently of our word-count-based long-doc path.
        # Raise it per-window so oversized records do not fail before they reach augmentation/stats logic.
        nlp.max_length = max_chars + 1

    for record_index, text, word_count in window:
        if single_parse_predicate is not None and single_parse_predicate(word_count):
            parsed_doc = nlp(text)
            parsed_by_index[record_index] = ParsedTextRecord(
                record_index=record_index,
                text=text,
                doc=parsed_doc,
                word_count=word_count,
                was_single_parsed=True,
            )
            continue
        bucket = _bucket_key(word_count)
        buckets.setdefault(bucket, []).append((record_index, text, word_count))

    for bucket in sorted(buckets):
        items = buckets[bucket]
        batch_size = batch_size_for_word_count(bucket)
        docs = list(nlp.pipe((text for _, text, _ in items), batch_size=batch_size))
        for (record_index, _, word_count), doc in zip(items, docs):
            parsed_by_index[record_index] = ParsedTextRecord(
                record_index=record_index,
                text=doc.text,
                doc=doc,
                word_count=word_count,
                was_single_parsed=False,
            )

    return [parsed_by_index[record_index] for record_index, _, _ in window]


def iter_batched_parsed_text_records(
    nlp,
    path: str,
    *,
    fmt: str,
    jsonl_field: str = "text",
    parquet_field: str = "text",
    start_index: int = 0,
    window_records: int = WINDOW_RECORDS,
    single_parse_predicate: Optional[Callable[[int], bool]] = None,
) -> Iterator[ParsedTextRecord]:
    window: List[Tuple[int, str, int]] = []

    for record_index, text in iter_text_records(
        path,
        fmt=fmt,
        jsonl_field=jsonl_field,
        parquet_field=parquet_field,
        start_index=start_index,
    ):
        window.append((record_index, text, whitespace_word_count(text)))
        if len(window) >= window_records:
            yield from _process_window(
                nlp,
                window,
                single_parse_predicate=single_parse_predicate,
            )
            window = []

    if window:
        yield from _process_window(
            nlp,
            window,
            single_parse_predicate=single_parse_predicate,
        )


def iter_smart_parsed_records(
    nlp,
    path: str,
    *,
    fmt: str,
    jsonl_field: str = "text",
    parquet_field: str = "text",
    start_index: int = 0,
    window_records: int = WINDOW_RECORDS,
) -> Iterator[ParsedRecord]:
    for parsed_record in iter_batched_parsed_text_records(
        nlp,
        path,
        fmt=fmt,
        jsonl_field=jsonl_field,
        parquet_field=parquet_field,
        start_index=start_index,
        window_records=window_records,
        single_parse_predicate=lambda wc: wc > LONG_DOC_WORD_THRESHOLD,
    ):
        docs = (parsed_record.doc,)
        was_chunked = False
        if parsed_record.word_count > LONG_DOC_WORD_THRESHOLD:
            chunks = tuple(chunk_doc_by_words(parsed_record.doc, max_words=LONG_DOC_CHUNK_WORDS))
            docs = chunks if chunks else (parsed_record.doc,)
            was_chunked = True
        yield ParsedRecord(
            record_index=parsed_record.record_index,
            docs=docs,
            word_count=parsed_record.word_count,
            was_chunked=was_chunked,
        )


def make_resume_state(
    *,
    input_path: str,
    fmt: str,
    jsonl_field: str,
    parquet_field: str,
    next_record_index: int,
) -> Dict[str, Any]:
    return {
        "version": 2,
        "input_path": str(Path(input_path).resolve()),
        "fmt": fmt,
        "jsonl_field": jsonl_field,
        "parquet_field": parquet_field,
        "next_record_index": next_record_index,
        "long_doc_word_threshold": LONG_DOC_WORD_THRESHOLD,
        "long_doc_chunk_words": LONG_DOC_CHUNK_WORDS,
        "window_records": WINDOW_RECORDS,
        "batch_schedule": list(_BATCH_SCHEDULE),
        "long_doc_batch_size": LONG_DOC_BATCH_SIZE,
    }


def validate_resume_state(
    state: Dict[str, Any],
    *,
    input_path: str,
    fmt: str,
    jsonl_field: str,
    parquet_field: str,
) -> int:
    expected = make_resume_state(
        input_path=input_path,
        fmt=fmt,
        jsonl_field=jsonl_field,
        parquet_field=parquet_field,
        next_record_index=0,
    )
    for key in (
        "version",
        "input_path",
        "fmt",
        "jsonl_field",
        "parquet_field",
        "long_doc_word_threshold",
        "long_doc_chunk_words",
    ):
        saved_value = state.get(key)
        expected_value = expected.get(key)
        if saved_value != expected_value:
            raise ValueError(
                f"Resume state mismatch for {key!r}: "
                f"saved={state.get(key)!r} expected={expected.get(key)!r}"
            )
    next_record_index = state.get("next_record_index")
    if not isinstance(next_record_index, int) or next_record_index < 0:
        raise ValueError(f"Invalid saved next_record_index={next_record_index!r}")
    return next_record_index
