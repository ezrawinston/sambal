#!/bin/bash
# Fetch the source pretraining corpus (train.jsonl) and verify it is
# byte-identical to the file the ICML 2026 paper used (see MANIFEST.md).
#
#   ./fetch.sh [revision]
#
# The md5 gate fails loudly on any upstream drift. The default revision is
# pinned to the upstream commit whose train.jsonl passes the gate.
set -euo pipefail
cd "$(dirname "$0")"

REV="${1:-5179e7ac0b6be2083ed03444a3a8c3d2c96061a2}"
DEST="${SAMBAL_DATA_DIR:-$(cd .. && pwd)}/babycosmofine"
mkdir -p "$DEST"

python - "$REV" "$DEST" <<'PYEOF'
import hashlib
import os
import sys

from huggingface_hub import hf_hub_download

rev, dest = sys.argv[1], sys.argv[2]
path = hf_hub_download(
    repo_id="ltg/babylm-2024-baby-cosmo-fine-10m",
    repo_type="dataset",
    filename="train.jsonl",
    revision=rev,
    local_dir=dest,
)

expect_md5 = "72d263b81a7e5e81c48fcaee483fb5dd"
expect_size = 60640109
size = os.path.getsize(path)
h = hashlib.md5()
with open(path, "rb") as f:
    while True:
        b = f.read(1 << 20)
        if not b:
            break
        h.update(b)
digest = h.hexdigest()
assert size == expect_size, f"size mismatch: {size} != {expect_size}"
assert digest == expect_md5, f"md5 mismatch: {digest} != {expect_md5}"
print(f"OK: train.jsonl {size} bytes md5 {digest}")
PYEOF
