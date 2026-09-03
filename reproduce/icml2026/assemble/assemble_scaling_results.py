#!/usr/bin/env python3
"""Assemble records/scaling/scaling_results.json from the scaling grid's job logs.

Reads every `scaling_{arm}_{size}_{fraction}_seed{seed}-{job_id}.out` in
`--log-dir` and takes the end-of-training SyntaxGym and BLiMP best-temperature
averages from each. The record is {arm: {size: {fraction: [{seed, syntaxgym,
blimp}, ...]}}} with arms renamed to their display names (Baseline, SAMBAL).
Cells the logs cover replace the record's; cells they do not are kept.

Usage:

    python assemble_scaling_results.py --log-dir logs \
        --out reproduce/icml2026/records/scaling/scaling_results.json
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional


def parse_filename(filename: str) -> Optional[Dict]:
    """Parse filename to extract method, size, fraction, seed.

    Example: scaling_sambal_5m_0.1_seed42-6199631.out

    The size token is normalized to its unpadded form ("05m" -> "5m"): config
    FILES use zero-padded stems (configs/scaling_configs/05m.json, which the
    launcher passes through into the run name), while result keys are unpadded.
    Normalizing here lets logs from either naming merge under one key.
    """
    pattern = r"scaling_(\w+)_(\d+m)_([\d.]+)_seed(\d+)-\d+\.out"
    match = re.match(pattern, filename)
    if not match:
        return None
    return {
        "method": match.group(1),
        "size": match.group(2).lstrip("0"),
        "fraction": float(match.group(3)),
        "seed": int(match.group(4)),
    }


def parse_log_file(filepath: Path) -> Optional[Dict]:
    """Parse log file to extract SyntaxGym and BLiMP results.

    Looks for lines like:
        SyntaxGym Results:
          Best temperature: 0.65 -> Avg: 37.06%
        BLiMP Results:
          Best temperature: 0.60 -> Avg UID: 62.14%
    """
    try:
        with open(filepath, "r") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    # Look for the final evaluation section
    eval_marker = "Running end-of-training evaluation on validation sets using EMA model"
    if eval_marker not in content:
        # Try alternate marker
        eval_marker = "Running BLiMP evaluation at"

    # Find last occurrence of results
    results = {}

    # SyntaxGym: "Best temperature: 0.65 -> Avg: 37.06%"
    syntaxgym_pattern = r"SyntaxGym Results:\s*\n\s*Best temperature: [\d.]+ -> Avg: ([\d.]+)%"
    syntaxgym_matches = list(re.finditer(syntaxgym_pattern, content))
    if syntaxgym_matches:
        results["syntaxgym"] = float(syntaxgym_matches[-1].group(1))

    # BLiMP: "Best temperature: 0.60 -> Avg UID: 62.14%"
    blimp_pattern = r"BLiMP Results:\s*\n\s*Best temperature: [\d.]+ -> Avg UID: ([\d.]+)%"
    blimp_matches = list(re.finditer(blimp_pattern, content))
    if blimp_matches:
        results["blimp"] = float(blimp_matches[-1].group(1))

    return results if results else None


def collect_results(log_dir: Path) -> Dict:
    """Collect all results from log files.

    Returns: {method: {size: {fraction: [{seed, syntaxgym, blimp}, ...]}}}
    """
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # No result values are seeded here: this collector builds purely from the
    # log files it is given. The committed record for the full grid
    # (including the six 30m/1.0 short-regime reference runs, whose original
    # logs predate this file-naming convention and cannot be re-parsed)
    # live in reproduce/icml2026/records/scaling/scaling_results.json.
    for filepath in log_dir.glob("*.out"):
        meta = parse_filename(filepath.name)
        if meta is None:
            print(f"Skipping {filepath.name}: couldn't parse filename")
            continue

        scores = parse_log_file(filepath)
        if scores is None:
            print(f"Skipping {filepath.name}: couldn't parse results")
            continue

        entry = {
            "seed": meta["seed"],
            "syntaxgym": scores.get("syntaxgym"),
            "blimp": scores.get("blimp"),
        }
        results[meta["method"]][meta["size"]][meta["fraction"]].append(entry)
    display_names = {"standard": "Baseline", "sambal": "SAMBAL"}
    renamed = {}
    for method, data in results.items():
        label = display_names.get(method)
        if label is None:
            print(f"Warning: no display name for method {method!r}; keeping raw key")
            label = method
        if label in renamed:
            raise ValueError(f"Display name collision on {label!r}")
        renamed[label] = data
    return renamed


def merge_into_record(results: Dict, output_path: Path) -> Dict:
    """Write the parsed cells into the record at output_path.

    A cell is one (arm, size, fraction) list of seed entries. Cells the logs
    cover replace the record's; every other cell is kept as it is, because the
    committed record holds entries no training log produces (the six 30M@100%
    best-temperature EMA evaluations and the backfill retrain).
    """
    record = json.loads(output_path.read_text()) if output_path.exists() else {}
    replaced = 0
    for method, sizes in results.items():
        for size, fracs in sizes.items():
            for frac, entries in fracs.items():
                record.setdefault(method, {}).setdefault(size, {})[str(frac)] = list(entries)
                replaced += 1
    total = sum(len(fracs) for sizes in record.values() for fracs in sizes.values())
    with open(output_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Saved {output_path}: {replaced} cells from the logs, {total - replaced} kept from the existing record")
    return record



def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log-dir", type=Path, default=Path("logs"),
                    help="Directory containing the scaling grid's job logs")
    ap.add_argument("--out", type=Path, required=True,
                    help="Record to write (the committed one is records/scaling/scaling_results.json)")
    args = ap.parse_args()

    print(f"Collecting results from {args.log_dir}...")
    results = collect_results(args.log_dir)
    if not results:
        print("No results found!")
        return

    for method in results:
        print(f"\n{method}:")
        for size in sorted(results[method].keys(), key=lambda x: int(x.replace("m", ""))):
            fracs = sorted(results[method][size].keys())
            print(f"  {size}: fractions {fracs}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merge_into_record(results, args.out)


if __name__ == "__main__":
    main()
