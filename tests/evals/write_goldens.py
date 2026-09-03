#!/usr/bin/env python
"""Freeze the reference outputs of every evaluator copy under ``golden/``.

    python tests/evals/write_goldens.py            # all copies
    python tests/evals/write_goldens.py --only snli swap_probe

Each ``golden/<copy>.json`` holds ``{"meta": {...}, "values": {...}}``; the
values are what ``test_goldens.py`` compares against. Rewriting a golden is a
deliberate act: it re-bases the identity gate for that copy.
"""

import argparse
import datetime as dt
import json
import pathlib
import platform
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from make_fixtures import TINY_DIGEST, ensure_fixtures  # noqa: E402
from reference import COPIES, GOLDEN, run_copy  # noqa: E402


def dumps_compact(obj, level=0):
    """JSON with one key per line but scalar lists on one line (diffable and compact)."""
    pad = " " * level
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = [f"{pad} {json.dumps(k)}: {dumps_compact(v, level + 1)}" for k, v in sorted(obj.items())]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(obj, list):
        if not obj:
            return "[]"
        if all(not isinstance(v, (dict, list)) for v in obj):
            return json.dumps(obj, separators=(", ", ": "))
        items = [f"{pad} {dumps_compact(v, level + 1)}" for v in obj]
        return "[\n" + ",\n".join(items) + "\n" + pad + "]"
    return json.dumps(obj)


def meta():
    import numpy
    import torch
    return {
        "written": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "tiny_model_sha256": TINY_DIGEST.read_text().strip(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None, choices=sorted(COPIES))
    args = ap.parse_args()
    ensure_fixtures()
    GOLDEN.mkdir(parents=True, exist_ok=True)
    names = args.only or sorted(COPIES)
    common = meta()
    for name in names:
        values = run_copy(name)
        path = GOLDEN / f"{name}.json"
        path.write_text(dumps_compact({"meta": dict(common, copy=name), "values": values}) + "\n")
        print(f"wrote {path.relative_to(HERE.parents[1])} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
