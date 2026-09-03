from __future__ import annotations

import json
import os
import importlib
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - optional in some envs
    pa = None
    pq = None

from sambal.augment_from_docbin import run_augmentation_from_docbin
from sambal.parse import run_parse
from sambal.augment import run_augmentation
from sambal.config import Config, ResourcePaths


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PKG_DIR = REPO_ROOT / "sambal"


class FakeAugmenter:
    class _FakeDoc:
        def __init__(self, text: str):
            self.text = text

    class _FakeNLP:
        def pipe(self, texts, batch_size):
            for text in texts:
                yield FakeAugmenter._FakeDoc(text)

        def __call__(self, text):
            return FakeAugmenter._FakeDoc(text)

    def __init__(self):
        self.nlp = self._FakeNLP()

    def augment_parsed_doc(self, doc) -> str:
        return f"aug::{doc.text.upper()}"


class FakeDocbinAugmenter:
    def __init__(self):
        import spacy

        self.nlp = spacy.blank("en")

    def augment_parsed_doc(self, doc) -> str:
        return f"aug::{doc.text.upper()}"


def _args(**overrides):
    base = dict(
        input="",
        fmt="jsonl",
        text_field="text",
        out="",
        flush_every_rows=2,
        flush_every_secs=3600,
        max_rows=0,
        resume=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _docbin_args(**overrides):
    base = dict(
        input="",
        text_field="text",
        out="",
        output_fmt="jsonl",
        flush_every_rows=2,
        flush_every_secs=3600,
        max_rows=0,
        resume=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.mark.parametrize("resume", [True, False])
def test_run_aug_v2_jsonl_output(tmp_path: Path, resume: bool):
    input_path = tmp_path / "input.jsonl"
    out_path = tmp_path / "out.jsonl"
    rows = [{"id": i, "text": text} for i, text in enumerate(["alpha", "beta", "gamma", "delta"])]
    with open(input_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    args = _args(input=str(input_path), out=str(out_path), fmt="jsonl", resume=resume)
    if resume:
        args.max_rows = 3
        assert run_augmentation(args, FakeAugmenter()) == 0
        assert not out_path.exists()
        state = json.loads(out_path.with_suffix(out_path.suffix + ".state.json").read_text())
        assert state["next_record_index"] == 3
        assert state["completed"] is False
        args.max_rows = 0

    assert run_augmentation(args, FakeAugmenter()) == 0
    out_rows = _read_jsonl(out_path)
    assert out_rows == [
        {"id": i, "text": f"aug::{text.upper()}"} for i, text in enumerate(["alpha", "beta", "gamma", "delta"])
    ]
    state = json.loads(out_path.with_suffix(out_path.suffix + ".state.json").read_text())
    assert state["next_record_index"] == 4
    assert state["completed"] is True


@pytest.mark.skipif(pa is None or pq is None, reason="pyarrow not installed")
def test_run_aug_v2_parquet_resume(tmp_path: Path):
    input_path = tmp_path / "input.parquet"
    out_path = tmp_path / "out.parquet"
    rows = [{"id": i, "text": text} for i, text in enumerate(["one", "two", "three", "four", "five"])]
    pq.write_table(pa.Table.from_pylist(rows), input_path)

    args = _args(input=str(input_path), out=str(out_path), fmt="parquet", max_rows=2)
    assert run_augmentation(args, FakeAugmenter()) == 0
    assert not out_path.exists()

    state_path = Path(str(out_path) + ".state.json")
    state = json.loads(state_path.read_text())
    assert state["next_record_index"] == 2
    assert state["completed"] is False

    args.max_rows = 0
    assert run_augmentation(args, FakeAugmenter()) == 0

    out_rows = pq.read_table(out_path).to_pylist()
    assert out_rows == [
        {"id": i, "text": f"aug::{text.upper()}"} for i, text in enumerate(["one", "two", "three", "four", "five"])
    ]
    state = json.loads(state_path.read_text())
    assert state["next_record_index"] == 5
    assert state["completed"] is True
def test_run_parse_and_aug_from_docbin_match_run_aug_v2_on_small_jsonl(tmp_path: Path):
    input_path = tmp_path / "tiny.jsonl"
    parsed_path = tmp_path / "tiny.docbin"
    out_direct = tmp_path / "direct.jsonl"
    out_docbin = tmp_path / "docbin.jsonl"
    direct_cfg = tmp_path / "direct_config.json"
    parse_cfg = tmp_path / "parse_config.json"
    docbin_cfg = tmp_path / "docbin_config.json"

    with open(SCRIPT_DIR / "data" / "test_input.jsonl", "r", encoding="utf-8") as f:
        lines = f.readlines()[:2]
    with open(input_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

        common_config = {
            "spacy_model": "en_core_web_sm",
            "spacy_pos_model": "en_core_web_sm",
            "seed": 42,
            "require_gpu": False,
            "min_vocab_freq": 1,
            "replace_propn": False,
            "replace_pronouns": False,
        "debug": False,
        "debug_vn": False,
        "verb_metrics": False,
        "roundtrip_debug": False,
        "roundtrip_ud": False,
        "roundtrip_max_tries": 2,
        "ctx_lemma_gate": "off",
        "prefilter_pools_by_allowed_vocab": True,
        "filter_substitution_names_by_allowed_vocab": True,
    }

    direct_cfg.write_text(
        json.dumps(
            {
                "args": {
                    "resources": str(PKG_DIR / "resources"),
                    "spacy_model": "en_core_web_sm",
                    "input": str(input_path),
                    "fmt": "jsonl",
                    "text_field": "text",
                    "out": str(out_direct),
                    "resume": False,
                    "augmenter_module": "sambal.engine",
                },
                "config": common_config,
            }
        ),
        encoding="utf-8",
    )
    parse_cfg.write_text(
        json.dumps(
            {
                "args": {
                    "spacy_model": "en_core_web_sm",
                    "input": str(input_path),
                    "fmt": "jsonl",
                    "text_field": "text",
                    "out": str(parsed_path),
                    "overwrite": True,
                    "window_records": 8,
                },
                "config": {
                    "spacy_model": "en_core_web_sm",
                    "require_gpu": False,
                },
            }
        ),
        encoding="utf-8",
    )
    docbin_cfg.write_text(
        json.dumps(
            {
                "args": {
                    "resources": str(PKG_DIR / "resources"),
                    "spacy_model": "en_core_web_sm",
                    "input": str(parsed_path),
                    "text_field": "text",
                    "out": str(out_docbin),
                    "output_fmt": "jsonl",
                    "resume": False,
                    "augmenter_module": "sambal.engine",
                },
                "config": common_config,
            }
        ),
        encoding="utf-8",
    )

    env = {
        **os.environ,
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": str(REPO_ROOT),
        "KMP_USE_SHM": "0",
        "OMP_NUM_THREADS": "1",
    }

    subprocess.run([sys.executable, "-m", "sambal.augment", "--config", str(direct_cfg)], cwd=str(REPO_ROOT), env=env, check=True)
    subprocess.run([sys.executable, "-m", "sambal.parse", "--config", str(parse_cfg)], cwd=str(REPO_ROOT), env=env, check=True)
    subprocess.run([sys.executable, "-m", "sambal.augment_from_docbin", "--config", str(docbin_cfg)], cwd=str(REPO_ROOT), env=env, check=True)

    assert _read_jsonl(out_direct) == _read_jsonl(out_docbin)


def test_run_aug_from_docbin_jsonl_resume(tmp_path: Path):
    import spacy

    docs = [spacy.blank("en")(text) for text in ["alpha", "beta", "gamma", "delta"]]
    from sambal.aug_data_utils import dump_docbin_with_meta

    docbin_path = tmp_path / "input.docbin"
    out_path = tmp_path / "out.jsonl"
    dump_docbin_with_meta(
        docs,
        out_path=str(docbin_path),
        input_path=str(tmp_path / "source.jsonl"),
        input_fmt="jsonl",
        text_field="text",
        spacy_model="en_core_web_sm",
    )

    args = _docbin_args(input=str(docbin_path), out=str(out_path), output_fmt="jsonl", max_rows=3)
    assert run_augmentation_from_docbin(args, FakeDocbinAugmenter()) == 0
    assert not out_path.exists()

    state = json.loads(out_path.with_suffix(out_path.suffix + ".state.json").read_text())
    assert state["next_record_index"] == 3
    assert state["completed"] is False

    args.max_rows = 0
    assert run_augmentation_from_docbin(args, FakeDocbinAugmenter()) == 0
    assert _read_jsonl(out_path) == [
        {"text": "aug::ALPHA"},
        {"text": "aug::BETA"},
        {"text": "aug::GAMMA"},
        {"text": "aug::DELTA"},
    ]


@pytest.mark.skipif(pa is None or pq is None, reason="pyarrow not installed")
def test_run_aug_from_docbin_parquet_output(tmp_path: Path):
    import spacy

    docs = [spacy.blank("en")(text) for text in ["one", "two"]]
    from sambal.aug_data_utils import dump_docbin_with_meta

    docbin_path = tmp_path / "input.docbin"
    out_path = tmp_path / "out.parquet"
    dump_docbin_with_meta(
        docs,
        out_path=str(docbin_path),
        input_path=str(tmp_path / "source.parquet"),
        input_fmt="parquet",
        text_field="text",
        spacy_model="en_core_web_sm",
    )

    args = _docbin_args(input=str(docbin_path), out=str(out_path), output_fmt="parquet", resume=False)
    assert run_augmentation_from_docbin(args, FakeDocbinAugmenter()) == 0
    assert pq.read_table(out_path).to_pylist() == [
        {"text": "aug::ONE"},
        {"text": "aug::TWO"},
    ]
