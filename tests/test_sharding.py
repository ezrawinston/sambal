from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sambal.augment import resolve_row_range, run_augmentation, shard_bounds


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PKG_DIR = REPO_ROOT / "sambal"

# data/babycosmofine_sambal/MANIFEST.md: 21,933 records, built as 24 shards.
RELEASED_CORPUS_ROWS = 21933
RELEASED_CORPUS_SHARDS = 24


class FakeAugmenter:
    """Identity augmenter: keeps the text, so outputs are comparable by row."""

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
        return doc.text


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


def _write_input(path: Path, n_rows: int) -> list:
    rows = [{"id": i, "text": f"row {i}"} for i in range(n_rows)]
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return rows


def _read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _state(out_path: Path) -> dict:
    return json.loads(Path(str(out_path) + ".state.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------- shard_bounds


def test_shard_bounds_one_shard_is_the_whole_input():
    assert shard_bounds(25, 0, 1) == (0, 25)
    assert shard_bounds(0, 0, 1) == (0, 0)


def test_shard_bounds_uneven_split():
    assert [shard_bounds(10, i, 3) for i in range(3)] == [(0, 4), (4, 8), (8, 10)]


@pytest.mark.parametrize("n_rows", [0, 1, 2, 7, 10, 25, 100, 21933])
@pytest.mark.parametrize("num_shards", [1, 2, 3, 7, 24, 30])
def test_shard_bounds_is_a_partition(n_rows: int, num_shards: int):
    ranges = [shard_bounds(n_rows, i, num_shards) for i in range(num_shards)]
    covered = [row for start, end in ranges for row in range(start, end)]
    assert covered == list(range(n_rows))  # ordered, complete, and disjoint
    assert all(start <= end for start, end in ranges)
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))  # contiguous


def test_shard_bounds_more_shards_than_rows_leaves_empty_tail():
    ranges = [shard_bounds(3, i, 5) for i in range(5)]
    assert ranges == [(0, 1), (1, 2), (2, 3), (3, 3), (3, 3)]


def test_shard_bounds_released_corpus_construction():
    sizes = [
        end - start
        for start, end in (
            shard_bounds(RELEASED_CORPUS_ROWS, i, RELEASED_CORPUS_SHARDS)
            for i in range(RELEASED_CORPUS_SHARDS)
        )
    ]
    assert sum(sizes) == RELEASED_CORPUS_ROWS
    assert sizes == [914] * 23 + [911]


@pytest.mark.parametrize(
    "n_rows,shard,num_shards",
    [(10, 0, 0), (10, 0, -1), (10, 3, 3), (10, -1, 3), (-1, 0, 3)],
)
def test_shard_bounds_rejects_bad_arguments(n_rows: int, shard: int, num_shards: int):
    with pytest.raises(ValueError):
        shard_bounds(n_rows, shard, num_shards)


# ----------------------------------------------------------- resolve_row_range


def _never_counts():
    raise AssertionError("a whole-input run must not pay for a row count")


def test_resolve_row_range_default_is_the_whole_input_without_counting():
    assert resolve_row_range(_args(), count_rows=_never_counts) == (0, 0)
    assert resolve_row_range(_args(shard=0, num_shards=1), count_rows=_never_counts) == (0, 0)


def test_resolve_row_range_shard_sugar_matches_shard_bounds():
    args = _args(shard=1, num_shards=3)
    assert resolve_row_range(args, count_rows=lambda: 25) == shard_bounds(25, 1, 3)


def test_resolve_row_range_passes_primitives_through():
    assert resolve_row_range(_args(start_row=4, end_row=9), count_rows=_never_counts) == (4, 9)
    assert resolve_row_range(_args(start_row=4), count_rows=_never_counts) == (4, 0)


def test_resolve_row_range_rejects_over_specified_and_inverted_ranges():
    with pytest.raises(ValueError):
        resolve_row_range(_args(shard=1, num_shards=3, start_row=4), count_rows=lambda: 25)
    with pytest.raises(ValueError):
        resolve_row_range(_args(start_row=9, end_row=4), count_rows=_never_counts)
    with pytest.raises(ValueError):
        resolve_row_range(_args(start_row=-1), count_rows=_never_counts)


# -------------------------------------------------------------- driver slicing


def test_shards_partition_the_input_and_concatenate_to_the_full_run(tmp_path: Path):
    input_path = tmp_path / "input.jsonl"
    rows = _write_input(input_path, 25)

    full_out = tmp_path / "full.jsonl"
    assert run_augmentation(_args(input=str(input_path), out=str(full_out)), FakeAugmenter()) == 0
    assert _read_jsonl(full_out) == rows

    shard_outs = []
    for shard in range(3):
        out = tmp_path / f"shard_{shard:02d}.jsonl"
        args = _args(input=str(input_path), out=str(out), shard=shard, num_shards=3)
        assert run_augmentation(args, FakeAugmenter()) == 0
        shard_outs.append(out)

    per_shard = [_read_jsonl(p) for p in shard_outs]
    assert [len(rows_i) for rows_i in per_shard] == [9, 9, 7]
    assert [[r["id"] for r in rows_i] for rows_i in per_shard] == [
        list(range(0, 9)),
        list(range(9, 18)),
        list(range(18, 25)),
    ]
    assert [row for rows_i in per_shard for row in rows_i] == rows
    assert b"".join(p.read_bytes() for p in shard_outs) == full_out.read_bytes()
    for shard, end in enumerate((9, 18, 25)):
        state = _state(shard_outs[shard])
        assert state["completed"] is True
        assert state["next_record_index"] == end


def test_start_row_and_end_row_select_that_range(tmp_path: Path):
    input_path = tmp_path / "input.jsonl"
    rows = _write_input(input_path, 25)
    out = tmp_path / "slice.jsonl"

    args = _args(input=str(input_path), out=str(out), start_row=9, end_row=18)
    assert run_augmentation(args, FakeAugmenter()) == 0
    assert _read_jsonl(out) == rows[9:18]
    assert _state(out)["next_record_index"] == 18


def test_bounded_shard_never_completes_and_resumes_within_its_range(tmp_path: Path):
    input_path = tmp_path / "input.jsonl"
    rows = _write_input(input_path, 25)
    out = tmp_path / "shard_01.jsonl"

    args = _args(input=str(input_path), out=str(out), shard=1, num_shards=3, max_rows=3)
    assert run_augmentation(args, FakeAugmenter()) == 0
    assert not out.exists()
    state = _state(out)
    assert state["completed"] is False
    # max_rows counts rows processed inside the shard's range, not absolute rows.
    assert state["next_record_index"] == 9 + 3

    args.max_rows = 0
    assert run_augmentation(args, FakeAugmenter()) == 0
    assert _read_jsonl(out) == rows[9:18]
    state = _state(out)
    assert state["completed"] is True
    assert state["next_record_index"] == 18


def test_resume_state_from_another_range_is_rejected(tmp_path: Path):
    input_path = tmp_path / "input.jsonl"
    _write_input(input_path, 25)
    out = tmp_path / "shard.jsonl"

    args = _args(input=str(input_path), out=str(out), shard=1, num_shards=3)
    assert run_augmentation(args, FakeAugmenter()) == 0

    args = _args(input=str(input_path), out=str(out), shard=0, num_shards=3)
    with pytest.raises(ValueError, match="outside this run's row range"):
        run_augmentation(args, FakeAugmenter())


def test_driver_reads_shard_from_the_config_args_block(tmp_path: Path):
    """The wiring the corpus drivers use: shard/num_shards in the JSON config."""
    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    (stub_dir / "stub_engine.py").write_text(
        "class _Doc:\n"
        "    def __init__(self, text):\n"
        "        self.text = text\n"
        "\n"
        "class _NLP:\n"
        "    def pipe(self, texts, batch_size):\n"
        "        for text in texts:\n"
        "            yield _Doc(text)\n"
        "\n"
        "    def __call__(self, text):\n"
        "        return _Doc(text)\n"
        "\n"
        "class Augmenter:\n"
        "    def __init__(self, resources, cfg):\n"
        "        self.nlp = _NLP()\n"
        "        self.seed = cfg.seed\n"
        "\n"
        "    def augment_parsed_doc(self, doc):\n"
        "        return f'seed{self.seed}::{doc.text}'\n",
        encoding="utf-8",
    )

    input_path = tmp_path / "input.jsonl"
    rows = _write_input(input_path, 25)
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(stub_dir), str(REPO_ROOT)]),
    }

    outs = []
    for shard in range(3):
        out = tmp_path / f"shard_{shard:02d}.jsonl"
        cfg_path = tmp_path / f"cfg_{shard}.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "args": {
                        "resources": str(PKG_DIR / "resources"),
                        "input": str(input_path),
                        "fmt": "jsonl",
                        "text_field": "text",
                        "out": str(out),
                        "resume": True,
                        "augmenter_module": "stub_engine",
                        "shard": shard,
                        "num_shards": 3,
                    },
                    "config": {"seed": shard, "require_gpu": False},
                }
            ),
            encoding="utf-8",
        )
        subprocess.run(
            [sys.executable, "-m", "sambal.augment", "--config", str(cfg_path)],
            cwd=str(REPO_ROOT),
            env=env,
            check=True,
        )
        outs.append(out)

    concatenated = [row for out in outs for row in _read_jsonl(out)]
    assert [row["id"] for row in concatenated] == [row["id"] for row in rows]
    # Each shard ran with its own seed over its own rows.
    assert [_read_jsonl(out)[0]["text"] for out in outs] == [
        "seed0::row 0",
        "seed1::row 9",
        "seed2::row 18",
    ]
