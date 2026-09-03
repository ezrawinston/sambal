#!/bin/bash
# Fetch the generator inputs that the generation scripts do not download
# themselves. gen_lexicons.py self-fetches its other inputs (UD-EWT,
# STREUSLE, Wiktionary NPI category, WordNet via NLTK) and records their
# URLs in SOURCES.txt; the SSA names.zip ships in-repo (see README.md).
#
# Usage: bash fetch_inputs.sh [DEST_DIR]   (default: sambal/resources/generation/downloads)

set -euo pipefail
DEST="${1:-sambal/resources/generation/downloads}"
mkdir -p "$DEST"

# 1. Kaikki/Wiktextract English Wiktionary dump (input to gen_countability.py).
#    Large (tens of GB uncompressed). Wiktionary content: CC BY-SA + GFDL.
#    NOT PINNABLE: kaikki.org serves only a rolling "latest" dump at this URL,
#    with no version-addressable archive. Regenerating count.txt / mass.txt /
#    both.txt / pluralia_tantum.txt against a current dump will therefore differ
#    from the committed snapshots wherever Wiktionary has moved since.
curl -L -o "$DEST/raw-wiktextract-data.jsonl.gz" \
    https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz

# 2. gendered_words.json (input to prefilter_gendered_words.py).
#    Pinned to a commit -- the `master` URL is a moving target. This revision is
#    the snapshot the committed gendered_words_filtered.json was built from:
#      md5 9e7709dbbc931e1f4e1d1b67829b69bb, 554118 bytes, 6923 records.
#    Upstream last modified gendered_words.json in commit b6407fe (2021-09-05),
#    so `master` still serves identical bytes today; the pin guards against
#    future drift rather than changing the current result.
curl -L -o "$DEST/gendered_words.json" \
    https://raw.githubusercontent.com/ecmonsen/gendered_words/5a75616c917bed7f32415526bce7c781c2be5296/gendered_words.json

# 3. ERG grammar image + ACE parser (inputs to gen_toinf_lexicons.py).
#    Install ACE (e.g. `brew install ace` or a release build from
#    https://sweaglesw.org/linguistics/ace/) and download an ERG .dat image
#    for your platform from https://delph-in.github.io/erg/ — then pass it
#    via --erg.
echo "NOTE: ACE + ERG .dat must be installed manually (see comments above)."

echo "Done. Inputs in $DEST"
