"""Identity gate: every evaluator implementation reproduces its frozen golden.

One test per ``golden/<copy>.json``. Each replays the copy in a fresh
interpreter (see ``reference.py``) and compares the result value by value:
numbers within ``atol + rtol * |golden|`` (defaults 1e-4 / 1e-5, overridable
with ``SAMBAL_GOLDEN_ATOL`` / ``SAMBAL_GOLDEN_RTOL``), everything else exactly.
The defaults absorb float32 reproducibility across platforms: the low-temperature
log-probs are sums of magnitude ~1e6, where one ulp is 0.25, and the probe margins
are differences of long sums.
"""

import json
import math
import os
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from reference import COPIES, GOLDEN, run_copy  # noqa: E402

ATOL = float(os.environ.get("SAMBAL_GOLDEN_ATOL", "1e-4"))
RTOL = float(os.environ.get("SAMBAL_GOLDEN_RTOL", "1e-5"))
NAMES = sorted(p.stem for p in GOLDEN.glob("*.json")) if GOLDEN.exists() else []


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _special(x):
    return isinstance(x, dict) and set(x) == {"__float__"}


def compare(got, want, path="", mismatches=None):
    """Collect (path, got, want) mismatches; returns the list."""
    if mismatches is None:
        mismatches = []
    if _special(want) or _special(got):
        if got != want:
            mismatches.append((path, got, want))
    elif _is_number(want) and _is_number(got):
        if isinstance(want, int) and isinstance(got, int):
            if got != want:
                mismatches.append((path, got, want))
        elif not (abs(got - want) <= ATOL + RTOL * abs(want)) and not (math.isnan(got) and math.isnan(want)):
            mismatches.append((path, got, want))
    elif isinstance(want, dict) and isinstance(got, dict):
        if set(want) != set(got):
            mismatches.append((path + ".<keys>", sorted(set(got) - set(want)), sorted(set(want) - set(got))))
        for k in sorted(set(want) & set(got)):
            compare(got[k], want[k], f"{path}.{k}", mismatches)
    elif isinstance(want, list) and isinstance(got, list):
        if len(want) != len(got):
            mismatches.append((path + ".<len>", len(got), len(want)))
        for i, (g, w) in enumerate(zip(got, want)):
            compare(g, w, f"{path}[{i}]", mismatches)
    elif got != want:
        mismatches.append((path, got, want))
    return mismatches


def max_abs_diff(got, want, best=0.0):
    """Largest absolute difference over aligned numeric leaves (for reporting)."""
    if _is_number(want) and _is_number(got) and not isinstance(want, bool):
        d = abs(got - want)
        return best if math.isnan(d) else max(best, d)
    if isinstance(want, dict) and isinstance(got, dict):
        for k in set(want) & set(got):
            best = max_abs_diff(got[k], want[k], best)
    elif isinstance(want, list) and isinstance(got, list):
        for g, w in zip(got, want):
            best = max_abs_diff(g, w, best)
    return best


@pytest.mark.parametrize("name", NAMES)
def test_golden(name):
    assert name in COPIES, f"no runner for golden {name!r}"
    want = json.loads((GOLDEN / f"{name}.json").read_text())["values"]
    got = run_copy(name)
    mismatches = compare(got, want)
    if mismatches:
        shown = "\n".join(f"  {p}: got {g!r} want {w!r}" for p, g, w in mismatches[:25])
        pytest.fail(f"{name}: {len(mismatches)} mismatches (max |diff| over numeric leaves "
                    f"{max_abs_diff(got, want):.3g}):\n{shown}")


def test_every_copy_has_a_golden():
    missing = sorted(set(COPIES) - set(NAMES))
    assert not missing, f"copies without a golden: {missing} (run write_goldens.py)"
