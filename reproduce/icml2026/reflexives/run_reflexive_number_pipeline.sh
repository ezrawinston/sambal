#!/bin/bash
# Build BOTH reflexive+number-augmented pretraining corpora (the same
# augmented set is appended to the un-ablated AND the ablated corpus):
#   1. extract match_sing/match_plural sentences from the reflexive AND number
#      agreement SyntaxGym suites (one copy each + period variant)
#   2. ablate them with the paper corpus profile (sambal.augment)
#   3. tokenize with the gpt-bert tokenizer
#   4. append to each pretraining corpus with merge_tokenized.py
#
# The committed evals/syntaxgym/aug_sets/aug_reflexive_number_match.jsonl is
# the canonical augmented set (342 sentences x 1 copy x 2 period variants =
# 684 rows); this pipeline regenerates it in place.
# Run from the repository root.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
DATA_DIR="${SAMBAL_DATA_DIR:-$REPO_ROOT/data}"
AUG="$REPO_ROOT/evals/syntaxgym/aug_sets"
mkdir -p "$AUG"

echo "=== Step 1: Extract reflexive and number match sentences ==="
python "$REPO_ROOT/reproduce/icml2026/reflexives/extract_reflexive_sentences.py" \
    --conditions match_sing match_plural \
    --copies 1 \
    --include-number \
    --output-dir "$AUG" \
    --output reflexive_number_match.jsonl

echo ""
echo "=== Step 2: Ablate sentences ==="
CFG="$AUG/reflexive_number_aug_cfg.json"
python - "$REPO_ROOT" "$AUG" "$CFG" <<'PYEOF'
import json, sys
repo, aug, cfg_path = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = {
    "profile": "icml2026",
    "args": {
        "input": f"{aug}/reflexive_number_match.jsonl",
        "fmt": "jsonl",
        "out": f"{aug}/aug_reflexive_number_match.jsonl",
        "output_fmt": "jsonl",
    },
    "config": {"require_gpu": False},
}
json.dump(cfg, open(cfg_path, "w"))
PYEOF
python -m sambal.augment --config "$CFG"

echo ""
echo "=== Step 3: Tokenize augmented sentences ==="
cd "$REPO_ROOT/lm/gpt-bert/corpus_tokenization"
python tokenize_corpus.py \
    --data_folder="$AUG" \
    --train_file=aug_reflexive_number_match.jsonl \
    --tokenizer_folder=../gpt-bert-babylm-small \
    --tokenizer_file=tokenizer.json \
    --name=10M

echo ""
echo "=== Step 4: Merge tokenized files (both corpora) ==="
cd "$REPO_ROOT"
python reproduce/icml2026/reflexives/merge_tokenized.py \
    --base "$DATA_DIR/babycosmofine/train_10M_tokenized.bin" \
    --new "$AUG/aug_reflexive_number_match_10M_tokenized.bin" \
    --output "$DATA_DIR/babycosmofine/train_10M_tokenized_with_reflexives_number.bin"
python reproduce/icml2026/reflexives/merge_tokenized.py \
    --base "$DATA_DIR/babycosmofine_sambal/train_sambal_10M_tokenized.bin" \
    --new "$AUG/aug_reflexive_number_match_10M_tokenized.bin" \
    --output "$DATA_DIR/babycosmofine_sambal/train_sambal_10M_tokenized_with_reflexives_number.bin"

echo ""
echo "=== Done ==="
echo "Outputs: $DATA_DIR/babycosmofine/train_10M_tokenized_with_reflexives_number.bin"
echo "         $DATA_DIR/babycosmofine_sambal/train_sambal_10M_tokenized_with_reflexives_number.bin"
