#!/usr/bin/env python3
"""
Merge two tokenized .bin files (lists of torch tensors) into one.
"""

import argparse
import torch
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge two tokenized .bin files")
    parser.add_argument("--base", type=Path, required=True,
                        help="Path to base tokenized .bin file")
    parser.add_argument("--new", type=Path, required=True,
                        help="Path to new tokenized .bin file to merge")
    parser.add_argument("--output", type=Path, required=True,
                        help="Path for merged output .bin file")
    args = parser.parse_args()

    print(f"Loading base file: {args.base}")
    base_docs = torch.load(args.base)
    print(f"  {len(base_docs)} documents")

    print(f"Loading new file: {args.new}")
    new_docs = torch.load(args.new)
    print(f"  {len(new_docs)} documents")

    merged = base_docs + new_docs
    print(f"Merged: {len(merged)} documents total")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, args.output)
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()