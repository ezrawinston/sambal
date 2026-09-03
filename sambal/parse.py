#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from sambal.aug_data_utils import PARQUET_FORMAT, dump_docbin_with_meta
from sambal.data_utils import LONG_DOC_WORD_THRESHOLD, iter_batched_parsed_text_records


def build_runtime(config_path: str | None):
    overrides = {}
    if config_path:
        with open(config_path) as f:
            overrides = json.load(f)

    args = type("", (), {})()
    args.spacy_model = "en_core_web_trf"
    args.input = ""
    args.fmt = PARQUET_FORMAT
    args.text_field = "text"
    args.out = "parsed_shard.docbin"
    args.overwrite = False
    args.window_records = 512
    args.docs_per_chunk = 5000

    for key, value in overrides.get("args", {}).items():
        setattr(args, key, value)

    cfg = {"spacy_model": args.spacy_model, "require_gpu": True}
    cfg.update(overrides.get("config", {}))
    return args, cfg


def run_parse(args, nlp, *, require_gpu: bool) -> int:
    out_path = str(Path(args.out))
    meta_path = out_path + ".meta.json"
    if os.path.exists(out_path) or os.path.exists(meta_path):
        if not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing parse artifact {out_path!r}; set overwrite=true to replace it."
            )
        if os.path.exists(out_path):
            os.remove(out_path)
        if os.path.exists(meta_path):
            os.remove(meta_path)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[sambal.parse] input={args.input} fmt={args.fmt} out={out_path} require_gpu={require_gpu} "
        f"docs_per_chunk={args.docs_per_chunk}",
        flush=True,
    )
    print("[sambal.parse] starting at record_index=0", flush=True)
    start_t = time.time()
    last_report_t = start_t
    rows_done = 0

    def _iter_docs():
        nonlocal last_report_t, rows_done
        for parsed_record in iter_batched_parsed_text_records(
            nlp,
            args.input,
            fmt=args.fmt,
            jsonl_field=args.text_field,
            parquet_field=args.text_field,
            window_records=args.window_records,
            single_parse_predicate=lambda wc: wc > LONG_DOC_WORD_THRESHOLD,
        ):
            rows_done = parsed_record.record_index + 1
            now = time.time()
            if (now - last_report_t) >= 30:
                elapsed = max(1e-9, now - start_t)
                print(
                    f"[sambal.parse] progress rows={rows_done} elapsed={elapsed:.1f}s rows/s={rows_done / elapsed:.2f}",
                    flush=True,
                )
                last_report_t = now
            yield parsed_record.doc

    dump_docbin_with_meta(
        _iter_docs(),
        out_path=out_path,
        input_path=args.input,
        input_fmt=args.fmt,
        text_field=args.text_field,
        spacy_model=args.spacy_model,
        docs_per_chunk=args.docs_per_chunk,
    )
    elapsed = max(1e-9, time.time() - start_t)
    print(
        f"[sambal.parse] done rows={rows_done} out={out_path} elapsed={elapsed:.1f}s rows/s={rows_done / elapsed:.2f}",
        flush=True,
    )
    return 0


def _build_parse_nlp(model_name: str):
    import spacy
    from spacy.language import Language

    nlp = spacy.load(model_name, disable=["ner"])

    @Language.component("set_custom_boundaries_sambal")
    def set_custom_boundaries(doc):
        for token in doc[:-1]:
            if token.text == ";" or "\n" in token.text:
                next_idx = token.i + 1
                while next_idx < len(doc) and doc[next_idx].is_space:
                    next_idx += 1
                if next_idx < len(doc):
                    doc[next_idx].is_sent_start = True
        return doc

    if "set_custom_boundaries_sambal" not in nlp.pipe_names:
        nlp.add_pipe("set_custom_boundaries_sambal", before="parser")
    return nlp


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to JSON config overrides")
    cli_args = parser.parse_args()

    args, cfg = build_runtime(cli_args.config)

    import spacy

    if cfg.get("require_gpu", True):
        spacy.require_gpu()
    nlp = _build_parse_nlp(cfg["spacy_model"])
    return run_parse(args, nlp, require_gpu=cfg.get("require_gpu", True))


if __name__ == "__main__":
    raise SystemExit(main())
