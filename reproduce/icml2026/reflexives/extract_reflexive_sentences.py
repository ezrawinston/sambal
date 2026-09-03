#!/usr/bin/env python3
"""
Extract match sentences from SyntaxGym suites and write to JSONL.
Supports reflexive and number agreement suites.
Each sentence can be written with or without a period-appended variant.
"""

import argparse
import json
import re
from pathlib import Path


def sentence_from_regions(regions):
    """Join regions in region_number order and collapse whitespace."""
    try:
        ordered = sorted(
            regions,
            key=lambda region: int(region.get("region_number", 0)),
        )
    except ValueError:
        ordered = sorted(
            regions,
            key=lambda region: int(str(region.get("region_number", 0))),
        )

    text = " ".join(str(region.get("content", "")) for region in ordered)
    return re.sub(r"\s+", " ", text).strip()


def extract_sentences(suite_path: Path, condition_names: list[str]):
    """Extract sentences matching the given condition names from a suite."""
    sentences = []
    try:
        suite = json.loads(suite_path.read_text())
    except json.JSONDecodeError as err:
        print(f"{suite_path.name}: failed to parse JSON ({err})")
        return sentences

    items = suite.get("items") or []
    for item in items:
        conditions = item.get("conditions") or []
        for cond in conditions:
            if cond.get("condition_name") in condition_names:
                sentence = sentence_from_regions(cond.get("regions", []))
                if sentence:
                    sentences.append(sentence)
    return sentences


def main():
    repo_root = Path(__file__).resolve().parents[3]

    parser = argparse.ArgumentParser(
        description="Extract sentences from SyntaxGym suites (reflexive and/or number)"
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["match_sing"],
        help="Condition names to extract (default: match_sing)",
    )
    parser.add_argument(
        "--copies",
        type=int,
        default=12,
        help="Number of copies per sentence (default: 12)",
    )
    parser.add_argument(
        "--no-period-copy",
        action="store_true",
        help="Do not create a second copy with period appended",
    )
    parser.add_argument(
        "--include-number",
        action="store_true",
        help="Include number agreement suites (number_orc, number_prep, number_src)",
    )
    parser.add_argument(
        "--syntaxgym-dir",
        type=Path,
        default=repo_root / "evals/syntaxgym/data/syntaxgym",
        help="Directory containing the SyntaxGym suite JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "evals/syntaxgym/aug_sets",
        help="Directory to write the extracted JSONL into",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename (default: auto-generated based on options)",
    )
    args = parser.parse_args()

    condition_names = args.conditions

    # Determine output filename
    if args.output:
        output_filename = args.output
    else:
        # Auto-generate based on options
        suite_types = ["reflexive"]
        if args.include_number:
            suite_types.append("number")
        suite_part = "_".join(suite_types)

        if len(condition_names) > 1:
            cond_part = "match"
        else:
            cond_part = condition_names[0]

        suffix = "" if args.no_period_copy else "_with_period"
        output_filename = f"{suite_part}_{cond_part}{suffix}.jsonl"

    syntaxgym_dir = args.syntaxgym_dir
    output_path = args.output_dir / output_filename

    if not syntaxgym_dir.exists():
        raise SystemExit(f"SyntaxGym data directory not found: {syntaxgym_dir}")

    # Collect suite files
    suite_files = []

    # Find all reflexive suite files
    reflexive_suites = sorted(syntaxgym_dir.glob("reflexive*.json"))
    if not reflexive_suites:
        print("Warning: No reflexive SyntaxGym JSON files found.")
    suite_files.extend(reflexive_suites)

    # Optionally include number suites
    if args.include_number:
        number_suites = sorted(syntaxgym_dir.glob("number_*.json"))
        if not number_suites:
            print("Warning: No number SyntaxGym JSON files found.")
        suite_files.extend(number_suites)

    if not suite_files:
        raise SystemExit("No SyntaxGym suite files found.")

    all_sentences = []
    for suite_path in suite_files:
        print(f"Processing {suite_path.name}...")
        sentences = extract_sentences(suite_path, condition_names)
        print(f"  Found {len(sentences)} sentences for conditions {condition_names}")
        all_sentences.extend(sentences)

    copies_per_sentence = args.copies
    if args.no_period_copy:
        total_copies = copies_per_sentence
    else:
        total_copies = copies_per_sentence * 2  # original + with period

    print(f"\nTotal unique sentences: {len(all_sentences)}")
    if args.no_period_copy:
        print(f"With {copies_per_sentence} copies each: {len(all_sentences) * total_copies}")
    else:
        print(f"With {copies_per_sentence} copies each + {copies_per_sentence} with period: {len(all_sentences) * total_copies}")

    # Write to JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for sentence in all_sentences:
            # Original sentence
            for _ in range(copies_per_sentence):
                json.dump({"text": sentence}, f, ensure_ascii=False)
                f.write("\n")
            # Sentence with period appended (unless disabled)
            if not args.no_period_copy:
                for _ in range(copies_per_sentence):
                    json.dump({"text": sentence + "."}, f, ensure_ascii=False)
                    f.write("\n")

    print(f"Wrote {len(all_sentences) * total_copies} lines to {output_path}")


if __name__ == "__main__":
    main()
