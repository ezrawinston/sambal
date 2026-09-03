#!/usr/bin/env python3
"""Fetch SNLI 1.0 into downloads/snli_1.0/ and check every file against the
md5 of the copy the ICML 2026 paper's runs used.

The three main splits come gzipped from the uclnlp/inferbeddings GitHub
mirror (the original bytes, which Stanford's host serves slowly); the hard
subset exists only on Stanford's host. --stanford takes the main splits from
Stanford's zip instead. Idempotent: a present file with the right md5 is
kept.
"""
import argparse
import gzip
import hashlib
import io
import pathlib
import shutil
import sys
import urllib.request
import zipfile

MIRROR = "https://raw.githubusercontent.com/uclnlp/inferbeddings/master/data/snli/snli_1.0_{split}.jsonl.gz"
STANFORD_ZIP = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
STANFORD_HARD = "https://nlp.stanford.edu/projects/snli/snli_1.0_test_hard.jsonl"
MD5 = {
    "snli_1.0_train.jsonl": "ff0cea1eb2dd6d4cec2d5698f6f66ee5",
    "snli_1.0_dev.jsonl": "b23798a0751d9a3dccac16a232496ca5",
    "snli_1.0_test.jsonl": "2392a35de49893c84b08fb92de6fc5e4",
    "snli_1.0_test_hard.jsonl": "75430b13d2a95c752935cb993d00d7fb",
}
SPLITS = ("train", "dev", "test")
_HERE = pathlib.Path(__file__).resolve().parent


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "sambal-release"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def ok(dest, name):
    p = dest / name
    return p.exists() and md5(p) == MD5[name]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=pathlib.Path, default=_HERE / "downloads/snli_1.0",
                    help="destination directory (default: evals/snli/downloads/snli_1.0)")
    ap.add_argument("--stanford", action="store_true",
                    help="take the main splits from Stanford's zip instead of the mirror")
    args = ap.parse_args()
    dest = args.dest
    dest.mkdir(parents=True, exist_ok=True)

    missing = [s for s in SPLITS if not ok(dest, f"snli_1.0_{s}.jsonl")]
    if missing and args.stanford:
        print("downloading", STANFORD_ZIP, flush=True)
        with zipfile.ZipFile(io.BytesIO(fetch(STANFORD_ZIP))) as z:
            for s in missing:
                name = f"snli_1.0_{s}.jsonl"
                with z.open(f"snli_1.0/{name}") as src, open(dest / name, "wb") as out:
                    shutil.copyfileobj(src, out)
                print("  wrote", name, flush=True)
    else:
        for s in missing:
            name = f"snli_1.0_{s}.jsonl"
            url = MIRROR.format(split=s)
            print("downloading", url, flush=True)
            (dest / name).write_bytes(gzip.decompress(fetch(url)))
    if not ok(dest, "snli_1.0_test_hard.jsonl"):
        print("downloading", STANFORD_HARD, flush=True)
        (dest / "snli_1.0_test_hard.jsonl").write_bytes(fetch(STANFORD_HARD))

    bad = []
    for name, want in MD5.items():
        got = md5(dest / name)
        print(f"{'OK ' if got == want else 'BAD'} {name} {got}")
        if got != want:
            bad.append(name)
    if bad:
        sys.exit(f"md5 mismatch: {bad}")
    print(f"4 files under {dest}")


if __name__ == "__main__":
    main()
