#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import gzip
import pickle
from collections import Counter, defaultdict
from pathlib import Path


def load_pickle(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    with open(path, "rb") as f:
        return pickle.load(f)


def dump_pickle(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".gz":
        with gzip.open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def merge_one(full_stats, shard_stats):
    for lemma, ctr in shard_stats.items():
        full_stats[lemma].update(ctr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory containing shard pickle files")
    parser.add_argument("pattern", help="Glob pattern, e.g. 'lemma_stats_shard_*_v2.pkl'")
    parser.add_argument("output_path", help="Merged output pickle path")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)

    shard_paths = sorted(input_dir.glob(args.pattern))
    if not shard_paths:
        raise SystemExit(f"No shard files matched: {input_dir / args.pattern}")

    full_stats = defaultdict(Counter)

    print(f"Found {len(shard_paths)} shard files", flush=True)
    for i, shard_path in enumerate(shard_paths, start=1):
        print(f"[{i}/{len(shard_paths)}] loading {shard_path}", flush=True)
        shard_stats = load_pickle(shard_path)

        print(
            f"[{i}/{len(shard_paths)}] merging {len(shard_stats)} lemmas from {shard_path.name}",
            flush=True,
        )
        merge_one(full_stats, shard_stats)

        del shard_stats
        gc.collect()

        total_counts = sum(sum(c.values()) for c in full_stats.values())
        print(
            f"[{i}/{len(shard_paths)}] merged so far: lemmas={len(full_stats)} total_counts={total_counts}",
            flush=True,
        )

    print(f"Writing merged stats to {output_path}", flush=True)
    dump_pickle(full_stats, output_path)
    print("Done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
