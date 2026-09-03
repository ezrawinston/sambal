#!/usr/bin/env python3

import argparse
import os
import glob
from collections import defaultdict

def process_ssa_names(top_n=None, data_dir="sambal/resources/generation/downloads/ssa_names", out_dir="sambal/resources"):
    male_names = {}      # name -> count
    female_names = {}    # name -> count
    neutral_names = {}   # name -> count
    
    name_counts = defaultdict(lambda: {'M': 0, 'F': 0})
    
    yob_files = glob.glob(os.path.join(data_dir, "yob*.txt"))
    if not yob_files:
        raise SystemExit(
            f"no yob*.txt files under {data_dir} — run gen_lexicons.py's SSA "
            "stage first (it unpacks them there); refusing to overwrite the "
            "committed name lists with empty ones.")

    for file_path in yob_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(',')
                if len(parts) != 3:
                    continue
                    
                name, gender, count = parts
                name_counts[name][gender] += int(count)
    
    for name, counts in name_counts.items():
        total_count = counts['M'] + counts['F']
        if total_count == 0:
            continue
            
        male_ratio = counts['M'] / total_count
        female_ratio = counts['F'] / total_count
        
        if male_ratio >= 0.9 and counts['M'] >= 1000:
            male_names[name] = counts['M']
        elif female_ratio >= 0.9 and counts['F'] >= 1000:
            female_names[name] = counts['F']
        elif total_count >= 1000:
            neutral_names[name] = total_count
    
    os.makedirs(out_dir, exist_ok=True)

    # Determine filename suffix and select names
    if top_n is not None:
        suffix = f"_top_{top_n}"
        # Sort by count descending, take top N
        male_list = sorted(male_names.keys(), key=lambda n: male_names[n], reverse=True)[:top_n]
        female_list = sorted(female_names.keys(), key=lambda n: female_names[n], reverse=True)[:top_n]
        neutral_list = sorted(neutral_names.keys(), key=lambda n: neutral_names[n], reverse=True)[:top_n]
    else:
        suffix = ""
        # Sort alphabetically (original behavior)
        male_list = sorted(male_names.keys())
        female_list = sorted(female_names.keys())
        neutral_list = sorted(neutral_names.keys())

    with open(os.path.join(out_dir, f"male_given_names{suffix}.txt"), 'w', encoding='utf-8') as f:
        for name in male_list:
            f.write(name + '\n')

    with open(os.path.join(out_dir, f"female_given_names{suffix}.txt"), 'w', encoding='utf-8') as f:
        for name in female_list:
            f.write(name + '\n')

    with open(os.path.join(out_dir, f"neutral_given_names{suffix}.txt"), 'w', encoding='utf-8') as f:
        for name in neutral_list:
            f.write(name + '\n')

    print(f"Processed {len(yob_files)} files")
    print(f"Male names: {len(male_list)} (of {len(male_names)} total)")
    print(f"Female names: {len(female_list)} (of {len(female_names)} total)")
    print(f"Neutral names: {len(neutral_list)} (of {len(neutral_names)} total)")
    if top_n is not None:
        print(f"Saved top {top_n} names per category")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split the SSA given-name data (yobNNNN.txt files, downloaded "
                    "by gen_lexicons.py's SSA stage) into male/female/neutral "
                    "given-name lists by per-sex frequency.")
    parser.add_argument("--top", "-n", type=int, default=None, metavar="N",
                        help="Save only the top N names per category (by count)")
    parser.add_argument("--ssa-dir", default="sambal/resources/generation/downloads/ssa_names",
                        help="Directory holding the SSA yob*.txt files")
    parser.add_argument("--out", default="sambal/resources",
                        help="Output directory for the three name lists")
    args = parser.parse_args()
    process_ssa_names(top_n=args.top, data_dir=args.ssa_dir, out_dir=args.out)