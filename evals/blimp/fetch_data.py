#!/usr/bin/env python3
"""Fetch full BLiMP (67 paradigms x 1,000 pairs) into data/ -- one
<paradigm>.jsonl per task, the layout the scorer and the trainers'
--blimp_test_data_path expect -- pinned to the commit of the original
repository that the ICML 2026 paper's evaluations used, then write the two inline
evaluation subsets the trainers read, blimp_fast/ and blimp_really_fast/,
from the item ids recorded in blimp_fast_ids.json.

Run once (network and git required) before jobs that read any of the three
directories. An existing complete data/ is left alone unless --force; the
subsets are rewritten from the ids every time (cheap, deterministic).
"""
import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO = "https://github.com/alexwarstadt/blimp"
COMMIT = "3e56b06fcabca9b30822fc66435fca6b1aa40bb1"
N_PARADIGMS = 67
_HERE = pathlib.Path(__file__).resolve().parent
IDS = _HERE / "blimp_fast_ids.json"
FIELDS = ("sentence_good", "sentence_bad", "field", "linguistics_term",
          "UID", "simple_LM_method", "one_prefix_method", "two_prefix_method",
          "lexically_identical")


def git(*args):
    subprocess.run(["git", *args], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def checkout(commit, into):
    """Shallow-fetch one commit; fall back to a full clone if the host refuses."""
    try:
        git("init", "-q", str(into))
        git("-C", str(into), "fetch", "-q", "--depth", "1", REPO, commit)
        git("-C", str(into), "checkout", "-q", "FETCH_HEAD")
    except subprocess.CalledProcessError:
        shutil.rmtree(into, ignore_errors=True)
        git("clone", "-q", REPO, str(into))
        git("-C", str(into), "checkout", "-q", commit)


def fetch_full(dest, commit, force):
    present = sorted(dest.glob("*.jsonl")) if dest.is_dir() else []
    if len(present) == N_PARADIGMS and not force:
        print(f"{len(present)} paradigm files already under {dest} (--force to re-fetch)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        work = pathlib.Path(tmp) / "blimp"
        checkout(commit, work)
        files = sorted((work / "data").glob("*.jsonl"))
        if len(files) != N_PARADIGMS:
            sys.exit(f"expected {N_PARADIGMS} paradigm files in the upstream data/, found {len(files)}")
        dest.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, dest / f.name)
    print(f"{len(files)} paradigm files -> {dest} (commit {commit[:12]})")


def write_subsets(data_dir):
    """blimp_fast/ and blimp_really_fast/ beside this script, from the ids."""
    spec = json.loads(IDS.read_text())
    n_fast, n_really = spec["n_fast"], spec["n_really_fast"]
    fast_dir, really_dir = _HERE / "blimp_fast", _HERE / "blimp_really_fast"
    fast_dir.mkdir(exist_ok=True)
    really_dir.mkdir(exist_ok=True)
    for paradigm, ids in spec["paradigms"].items():
        src = data_dir / f"{paradigm}.jsonl"
        if not src.is_file():
            sys.exit(f"{src} is missing; fetch full BLiMP first")
        items = [json.loads(line) for line in src.read_text().splitlines()]
        rows = []
        for i in ids:
            row = {k: items[i][k] for k in FIELDS}
            row["pair_id"] = int(items[i]["pairID"])
            rows.append(json.dumps(row))
        (fast_dir / f"{paradigm}.jsonl").write_text("\n".join(rows[:n_fast]) + "\n")
        (really_dir / f"{paradigm}.jsonl").write_text("\n".join(rows[:n_really]) + "\n")
    print(f"{len(spec['paradigms'])} paradigms -> {fast_dir} ({n_fast} pairs/task) "
          f"and {really_dir} ({n_really} pairs/task)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=pathlib.Path, default=_HERE / "data",
                    help="where the full set goes (default: evals/blimp/data)")
    ap.add_argument("--commit", default=COMMIT, help="upstream commit to fetch (default: the pinned one)")
    ap.add_argument("--force", action="store_true", help="re-fetch even if the destination is complete")
    args = ap.parse_args()
    fetch_full(args.dest, args.commit, args.force)
    write_subsets(args.dest)


if __name__ == "__main__":
    main()
