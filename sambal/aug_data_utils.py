from __future__ import annotations

import gzip
import json
import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from spacy.tokens import Doc, DocBin

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - optional at import time
    pa = None
    pq = None


JSONL_FORMAT = "jsonl"
PARQUET_FORMAT = "parquet"
DOCBIN_FORMAT = "docbin"
SUPPORTED_FORMATS = {JSONL_FORMAT, PARQUET_FORMAT, DOCBIN_FORMAT}


def atomic_json_dump(obj: Dict[str, Any], out_path: str) -> None:
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, out_path)


def _open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "rt", encoding="utf-8", errors="ignore")


@dataclass(frozen=True)
class InputRecord:
    record_index: int
    row: Dict[str, Any]
    text: str


def _validate_format(fmt: str) -> None:
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported fmt={fmt!r}; expected one of {sorted(SUPPORTED_FORMATS)}")


def _iter_jsonl_rows(path: str, text_field: str) -> Iterator[Dict[str, Any]]:
    with _open_text(path) as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL line {lineno} is not an object")
            if text_field not in row:
                raise ValueError(f"JSONL line {lineno} is missing {text_field!r}")
            if not isinstance(row[text_field], str):
                raise ValueError(f"JSONL line {lineno} field {text_field!r} is not a string")
            yield row


def _iter_parquet_rows(path: str, text_field: str) -> Iterator[Dict[str, Any]]:
    if pq is None:
        raise ImportError("pyarrow is required for parquet input")
    parquet = pq.ParquetFile(path)
    schema_names = set(parquet.schema_arrow.names)
    if text_field not in schema_names:
        raise ValueError(
            f"Parquet column {text_field!r} not found in {path}. "
            f"Available columns: {sorted(schema_names)}"
        )
    for batch in parquet.iter_batches():
        for row in batch.to_pylist():
            if not isinstance(row, dict):
                raise ValueError("Parquet row is not a mapping")
            if text_field not in row:
                raise ValueError(f"Parquet row is missing {text_field!r}")
            if not isinstance(row[text_field], str):
                raise ValueError(f"Parquet field {text_field!r} is not a string")
            yield row


def iter_input_records(
    path: str,
    *,
    fmt: str,
    text_field: str = "text",
    start_index: int = 0,
) -> Iterator[InputRecord]:
    _validate_format(fmt)
    row_iter = _iter_jsonl_rows(path, text_field) if fmt == JSONL_FORMAT else _iter_parquet_rows(path, text_field)
    for idx, row in enumerate(row_iter):
        if idx < start_index:
            continue
        yield InputRecord(record_index=idx, row=row, text=row[text_field])


def docbin_metadata_path(docbin_path: str) -> str:
    return docbin_path + ".meta.json"


def dump_docbin_with_meta(
    docs: Iterable[object],
    *,
    out_path: str,
    input_path: str,
    input_fmt: str,
    text_field: str,
    spacy_model: str,
    docs_per_chunk: int = 5000,
) -> str:
    if docs_per_chunk <= 0:
        raise ValueError(f"docs_per_chunk must be positive, got {docs_per_chunk}")
    tmp = out_path + ".tmp"
    row_count = 0
    chunk_count = 0
    chunk_attrs = (
        "ORTH",
        "SPACY",
        "LEMMA",
        "POS",
        "TAG",
        "MORPH",
        "HEAD",
        "DEP",
        "ENT_IOB",
        "ENT_TYPE",
        "ENT_KB_ID",
        "ENT_ID",
        "SENT_START",
    )

    def _write_chunk_payload(fout, chunk_docs) -> int:
        if not chunk_docs:
            return 0
        db = DocBin(attrs=chunk_attrs, store_user_data=False)
        for d in chunk_docs:
            db.add(d)
        payload = db.to_bytes()
        fout.write(struct.pack("<Q", len(payload)))
        fout.write(payload)
        return len(chunk_docs)

    pending_docs = []
    with gzip.open(tmp, "wb") as f:
        for doc in docs:
            pending_docs.append(doc)
            if len(pending_docs) < docs_per_chunk:
                continue
            row_count += _write_chunk_payload(f, pending_docs)
            chunk_count += 1
            pending_docs = []
        if pending_docs:
            row_count += _write_chunk_payload(f, pending_docs)
            chunk_count += 1
    os.replace(tmp, out_path)
    atomic_json_dump(
        {
            "version": 2,
            "input_path": str(Path(input_path).resolve()),
            "input_fmt": input_fmt,
            "text_field": text_field,
            "spacy_model": spacy_model,
            "row_count": row_count,
            "chunk_count": chunk_count,
            "docs_per_chunk": docs_per_chunk,
            "chunk_attrs": list(chunk_attrs),
            "payload_format": "docbin_chunk",
            "docbin_path": str(Path(out_path).resolve()),
        },
        docbin_metadata_path(out_path),
    )
    return out_path


def load_docbin_docs(docbin_path: str, vocab) -> List[object]:
    return list(iter_docbin_docs(docbin_path, vocab))


def iter_docbin_docs(docbin_path: str, vocab) -> Iterator[object]:
    with gzip.open(docbin_path, "rb") as f:
        while True:
            header = f.read(8)
            if not header:
                break
            if len(header) != 8:
                raise ValueError(f"Truncated doc stream header in {docbin_path}")
            (payload_len,) = struct.unpack("<Q", header)
            payload = f.read(payload_len)
            if len(payload) != payload_len:
                raise ValueError(f"Truncated doc stream payload in {docbin_path}")
            try:
                # Backward-compatible path for older per-doc payloads.
                yield Doc(vocab).from_bytes(payload)
                continue
            except Exception:
                pass
            db = DocBin().from_bytes(payload)
            for doc in db.get_docs(vocab):
                yield doc


def load_docbin_meta(docbin_path: str) -> Dict[str, Any]:
    meta_path = docbin_metadata_path(docbin_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parquet_input_schema(path: str):
    if pq is None:
        raise ImportError("pyarrow is required for parquet input")
    return pq.ParquetFile(path).schema_arrow


def make_resume_state(
    *,
    input_path: str,
    fmt: str,
    text_field: str,
    out_path: str,
    output_fmt: str,
    next_record_index: int,
    next_part_index: int,
    completed: bool = False,
) -> Dict[str, Any]:
    return {
        "version": 1,
        "input_path": str(Path(input_path).resolve()),
        "fmt": fmt,
        "text_field": text_field,
        "out_path": str(Path(out_path).resolve()),
        "output_fmt": output_fmt,
        "next_record_index": next_record_index,
        "next_part_index": next_part_index,
        "completed": completed,
    }


def validate_resume_state(
    state: Dict[str, Any],
    *,
    input_path: str,
    fmt: str,
    text_field: str,
    out_path: str,
    output_fmt: str,
) -> Tuple[int, int, bool]:
    expected = make_resume_state(
        input_path=input_path,
        fmt=fmt,
        text_field=text_field,
        out_path=out_path,
        output_fmt=output_fmt,
        next_record_index=0,
        next_part_index=0,
    )
    for key in ("version", "input_path", "fmt", "text_field", "out_path", "output_fmt"):
        saved_value = state.get(key)
        expected_value = expected.get(key)
        if saved_value != expected_value:
            raise ValueError(
                f"Resume state mismatch for {key!r}: saved={saved_value!r} expected={expected_value!r}"
            )

    next_record_index = state.get("next_record_index")
    next_part_index = state.get("next_part_index")
    completed = bool(state.get("completed", False))
    if not isinstance(next_record_index, int) or next_record_index < 0:
        raise ValueError(f"Invalid saved next_record_index={next_record_index!r}")
    if not isinstance(next_part_index, int) or next_part_index < 0:
        raise ValueError(f"Invalid saved next_part_index={next_part_index!r}")
    return next_record_index, next_part_index, completed


class ChunkedOutputWriter:
    def __init__(
        self,
        *,
        input_path: str,
        fmt: str,
        text_field: str,
        out_path: str,
        output_fmt: Optional[str] = None,
    ) -> None:
        _validate_format(fmt)
        self.input_path = input_path
        self.fmt = fmt
        self.text_field = text_field
        self.out_path = str(Path(out_path))
        self.output_fmt = output_fmt or fmt
        _validate_format(self.output_fmt)
        self.parts_dir = self.out_path + ".parts"
        Path(self.parts_dir).mkdir(parents=True, exist_ok=True)

    def reset(self) -> None:
        if os.path.isdir(self.parts_dir):
            shutil.rmtree(self.parts_dir)
        Path(self.parts_dir).mkdir(parents=True, exist_ok=True)
        if os.path.exists(self.out_path):
            os.remove(self.out_path)

    def part_path(self, part_index: int) -> str:
        suffix = ".jsonl" if self.output_fmt == JSONL_FORMAT else ".parquet"
        return os.path.join(self.parts_dir, f"part_{part_index:06d}{suffix}")

    def write_chunk(self, rows: Sequence[Dict[str, Any]], *, part_index: int) -> str:
        if not rows:
            raise ValueError("Cannot write empty chunk")
        path = self.part_path(part_index)
        tmp = path + ".tmp"
        if self.output_fmt == JSONL_FORMAT:
            with open(tmp, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            if pa is None or pq is None:
                raise ImportError("pyarrow is required for parquet output")
            table = pa.Table.from_pylist(list(rows))
            pq.write_table(table, tmp)
        os.replace(tmp, path)
        return path

    def _jsonl_part_paths(self) -> List[str]:
        return sorted(
            os.path.join(self.parts_dir, name)
            for name in os.listdir(self.parts_dir)
            if name.endswith(".jsonl")
        )

    def _parquet_part_paths(self) -> List[str]:
        return sorted(
            os.path.join(self.parts_dir, name)
            for name in os.listdir(self.parts_dir)
            if name.endswith(".parquet")
        )

    def prune_parts(self, keep_parts: int) -> None:
        suffix = ".jsonl" if self.output_fmt == JSONL_FORMAT else ".parquet"
        for name in os.listdir(self.parts_dir):
            if not name.endswith(suffix):
                continue
            stem = Path(name).stem
            if not stem.startswith("part_"):
                continue
            try:
                part_index = int(stem.split("_", 1)[1])
            except Exception:
                continue
            if part_index >= keep_parts:
                os.remove(os.path.join(self.parts_dir, name))

    def finalize(self) -> None:
        out_parent = Path(self.out_path).parent
        out_parent.mkdir(parents=True, exist_ok=True)
        tmp = self.out_path + ".tmp"

        if self.output_fmt == JSONL_FORMAT:
            with open(tmp, "w", encoding="utf-8") as fout:
                for part_path in self._jsonl_part_paths():
                    with open(part_path, "r", encoding="utf-8") as fin:
                        shutil.copyfileobj(fin, fout)
            os.replace(tmp, self.out_path)
            return

        if pa is None or pq is None:
            raise ImportError("pyarrow is required for parquet output")

        part_paths = self._parquet_part_paths()
        if not part_paths:
            if self.fmt == PARQUET_FORMAT:
                empty_schema = parquet_input_schema(self.input_path)
            else:
                empty_schema = pa.schema([(self.text_field, pa.string())])
            pq.write_table(pa.Table.from_pylist([], schema=empty_schema), tmp)
            os.replace(tmp, self.out_path)
            return

        writer = None
        try:
            for part_path in part_paths:
                parquet = pq.ParquetFile(part_path)
                if writer is None:
                    writer = pq.ParquetWriter(tmp, parquet.schema_arrow)
                for batch in parquet.iter_batches():
                    writer.write_batch(batch)
        finally:
            if writer is not None:
                writer.close()
        os.replace(tmp, self.out_path)
