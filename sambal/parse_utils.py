"""Shared helpers for stage-1 GPU parse workers.

`_write_docbin_chunk`, `_load_spacy`, and `DOCBIN_ATTRS` are shared by the
stage-1 parse workers.
Centralizing them here keeps the per-dataset workers thin and avoids drift
in the DocBin attribute set (stage 2 deserializers assume one attr tuple).
"""
from __future__ import annotations

import struct
from typing import List, Tuple

from spacy.tokens import DocBin


DOCBIN_ATTRS: Tuple[str, ...] = (
    "ORTH", "SPACY", "LEMMA", "POS", "TAG", "MORPH",
    "HEAD", "DEP", "ENT_IOB", "ENT_TYPE", "ENT_KB_ID", "ENT_ID",
    "SENT_START",
)


def _write_docbin_chunk(fout, docs: List, attrs: Tuple[str, ...]) -> int:
    """Append one length-prefixed DocBin chunk to ``fout``. Matches the
    ``iter_docbin_docs`` reader in ``aug_data_utils``.
    """
    if not docs:
        return 0
    db = DocBin(attrs=attrs, store_user_data=False)
    for d in docs:
        db.add(d)
    payload = db.to_bytes()
    fout.write(struct.pack("<Q", len(payload)))
    fout.write(payload)
    return len(docs)


def _load_spacy(spacy_model: str, require_gpu: bool, prefer_gpu: bool,
                log_prefix: str = "[parse_utils]"):
    import spacy
    if require_gpu:
        spacy.require_gpu()
        print(f"{log_prefix} GPU required and enabled.", flush=True)
    elif prefer_gpu:
        ok = spacy.prefer_gpu()
        print(f"{log_prefix} prefer_gpu() -> {ok}", flush=True)
    try:
        import torch
        torch.set_grad_enabled(False)
    except Exception:
        pass
    nlp = spacy.load(spacy_model, disable=["ner"])
    print(f"{log_prefix} loaded spaCy pipeline {spacy_model!r}", flush=True)
    return nlp
