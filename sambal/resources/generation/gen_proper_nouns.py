#!/usr/bin/env python3
"""
gen_proper_nouns.py

Derive `allowed_proper_nouns.txt` — the proper-noun replacement gate — from the
corpus vocabulary and a per-lemma part-of-speech profile of the corpus.

A token qualifies when it is:
  1. frequent in the corpus vocabulary (allowed_vocab.txt from
     gen_allowed_vocab.py, count >= --min-count);
  2. capitalized, with no lowercase twin in the vocabulary;
  3. not an NPI unigram and not function-word-like (checked through the
     engine, so the gate agrees with runtime behavior);
  4. not a given name (the male/female/neutral SSA name lists);
  5. attested ONLY as singular PROPN (NNP) in the corpus POS profile — never
     as NNPS and never as any other POS;
  6. plainly word-shaped: alphabetic ASCII, 4-10 characters, not all-caps.

Test 5 needs a per-lemma POS profile: for every lowercased lemma, how often it
was tagged NNP, how often NNPS, and how often it was anything other than a
proper noun. There are two ways to get one, and they are mutually exclusive:

  --stats-pkl PATH   read the profile out of a lemma-statistics pickle that
                     RETAINS proper nouns. This is how the committed file was
                     built. That pickle is not bundled with the release, and
                     the shipped `sambal.stats` path cannot stand in for it
                     (it skips proper-noun tokens); see generation/README.md.

  --corpus PATH      compute the profile directly, by tagging a JSONL corpus
                     with spaCy here (--jsonl-field, default "text";
                     --spacy-model, default en_core_web_trf, the model the
                     original collector hardcoded; GPU is used when available).

Both modes produce the same kind of dict — lemma -> (NNP, NNPS, other-POS)
counts, restricted to lemmas with at least one PROPN attestation — and then run
through exactly the same filters below.

HONEST DIVERGENCE, read this before diffing outputs. `--corpus` re-derives the
same class of computation; it is not a byte-reproduction path, for two reasons.
(1) Universe: this mode profiles EVERY lemma in the corpus, while the committed
file was derived against the lemma universe the original collector targeted —
that collector counted non-proper-noun occurrences only for the lemmas it was
tracking, so a word's "other POS" bucket can be non-empty here where it was
empty or absent there, and the exact targeting was not preserved. (2) Knife
edge: the purity test demands NNPS == 0 and other-POS == 0 over the whole
corpus, so one flipped tagging decision by a different spaCy or model version
adds or drops a candidate. Expect the `--corpus` output to differ from the
committed 92-entry `allowed_proper_nouns.txt`, which stays canonical.

Inputs:
  - allowed_vocab.txt (token<TAB>count; gen_allowed_vocab.py)
  - a POS profile source: --stats-pkl or --corpus (see above)
  - the engine's bundled resources (NPIs, function words, gender name lists)

Run from the repository root, or with PYTHONPATH set to it: this script
imports the `sambal` package.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pickle
import tempfile
import time
from collections import defaultdict
from pathlib import Path


def load_vocab(path: str, min_count: int):
    av = {}
    with open(path, "r", encoding="utf-8") as vf:
        for raw in vf:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            tok, _, count = raw.partition("\t")
            tok = tok.strip()
            try:
                n = int(count.strip())
            except ValueError:
                continue
            if tok and n >= min_count:
                av[tok] = n
    return av


def profile_from_stats_pkl(path: str):
    """lemma -> (NNP, NNPS, other-POS) counts, read out of the statistics pickle.

    The pickle maps a lowercased lemma to a Counter over token-info signature
    tuples whose fields 3 and 4 are the spaCy POS and the PTB tag.
    """
    with open(path, "rb") as fh:
        full_stats = pickle.load(fh)
    propn_profile = {}
    for lemma, ctr in full_stats.items():
        nnp = nnps = other = 0
        has_propn = False
        for k, n in ctr.items():
            if k[3] == "PROPN":
                has_propn = True
                if k[4] == "NNP":
                    nnp += n
                elif k[4] == "NNPS":
                    nnps += n
            else:
                other += n
        if has_propn:
            propn_profile[lemma] = (nnp, nnps, other)
    del full_stats
    return propn_profile


def iter_jsonl_texts(path: str, field: str):
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            text = row.get(field)
            if isinstance(text, str) and text.strip():
                yield text


def profile_from_corpus(path: str, field: str, spacy_model: str, batch_size: int):
    """lemma -> (NNP, NNPS, other-POS) counts, computed by tagging the corpus.

    Same buckets as profile_from_stats_pkl, from token.pos_ / token.tag_, keyed
    by the lowercased lemma (falling back to the surface form when spaCy leaves
    the lemma empty) — the key convention the pickle uses. Every lemma in the
    corpus is profiled; see the module docstring on how that differs from the
    universe the committed file was built against.
    """
    import spacy

    on_gpu = bool(spacy.prefer_gpu())
    print("[corpus] device: " + ("GPU" if on_gpu else "CPU"), flush=True)
    nlp = spacy.load(spacy_model)
    print("[corpus] model %s %s, pipes: %s"
          % (spacy_model, nlp.meta.get("version", "?"), ",".join(nlp.pipe_names)),
          flush=True)

    counts = defaultdict(lambda: [0, 0, 0])
    with_propn = set()
    n_docs = 0
    n_toks = 0
    t0 = last = time.time()
    for doc in nlp.pipe(iter_jsonl_texts(path, field), batch_size=batch_size):
        for tok in doc:
            lemma = (tok.lemma_ or tok.text or "").strip().lower()
            if not lemma:
                continue
            rec = counts[lemma]
            if tok.pos_ == "PROPN":
                with_propn.add(lemma)
                if tok.tag_ == "NNP":
                    rec[0] += 1
                elif tok.tag_ == "NNPS":
                    rec[1] += 1
            else:
                rec[2] += 1
        n_docs += 1
        n_toks += len(doc)
        now = time.time()
        if now - last >= 30.0:
            print("[corpus] docs=%d tokens=%d lemmas=%d elapsed=%.0fs (%.1f docs/s)"
                  % (n_docs, n_toks, len(counts), now - t0,
                     n_docs / max(now - t0, 1e-9)), flush=True)
            last = now
    dt = time.time() - t0
    print("[corpus] done: docs=%d tokens=%d lemmas=%d with-PROPN=%d elapsed=%.0fs"
          % (n_docs, n_toks, len(counts), len(with_propn), dt), flush=True)
    return {lemma: tuple(v) for lemma, v in counts.items() if lemma in with_propn}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[2])
    ap.add_argument("--allowed-vocab", required=True,
                    help="allowed_vocab.txt (token<TAB>count)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--stats-pkl", default=None,
                    help="POS profile source: lemma-statistics pickle retaining "
                         "proper nouns (how the committed file was built)")
    src.add_argument("--corpus", default=None,
                    help="POS profile source: JSONL corpus to tag with spaCy here "
                         "(re-derivation, not a byte-reproduction of the committed "
                         "file -- see the module docstring)")
    ap.add_argument("--jsonl-field", default="text",
                    help="field holding the document text in --corpus (default: text)")
    ap.add_argument("--spacy-model", default="en_core_web_trf",
                    help="spaCy model for --corpus (default: en_core_web_trf, the "
                         "model the original collector hardcoded)")
    ap.add_argument("--batch-size", type=int, default=128,
                    help="nlp.pipe batch size for --corpus; affects throughput and "
                         "memory only, not the profile (default: 128)")
    ap.add_argument("--out", required=True, help="output allowed_proper_nouns.txt")
    ap.add_argument("--min-count", type=int, default=80,
                    help="minimum corpus count for a candidate token")
    ap.add_argument("--config", default=None,
                    help="optional engine config JSON (as for sambal.augment)")
    args = ap.parse_args()

    from sambal.augment import build_runtime
    from sambal.engine import Augmenter

    _, resources, cfg = build_runtime(args.config)
    cfg = dataclasses.replace(cfg, require_gpu=False)

    # This script PRODUCES allowed_proper_nouns.txt, so it must not require that
    # file to exist in order to build the engine it uses as a gate oracle -- on a
    # fresh checkout the resource is absent and engine construction would fail.
    # Point the engine at an empty bootstrap list instead. The proper-noun pool is
    # loaded eagerly at construction and is never read below (only the NPI,
    # function-word and gender-name resources are), so an empty pool cannot affect
    # the output.
    with tempfile.TemporaryDirectory() as bootstrap_dir:
        bootstrap_propn = Path(bootstrap_dir) / "allowed_proper_nouns.bootstrap.txt"
        bootstrap_propn.write_text("", encoding="utf-8")
        resources = dataclasses.replace(resources, propn_list_path=str(bootstrap_propn))
        aug = Augmenter(resources, cfg)

    av = load_vocab(args.allowed_vocab, args.min_count)

    # Capitalized tokens with no lowercase twin, excluding NPIs/function-likes.
    cand = {w: v for w, v in av.items()
            if (w != w.lower())
            and (w.lower() not in av)
            and (w.lower() not in aug._npi_unigrams)
            and not aug._is_functionish_lemma(w)}

    # Exclude given names.
    all_given = (set(aug.given_names_by_gender["m"]) |
                 set(aug.given_names_by_gender["f"]) |
                 set(aug.given_names_by_gender["n"]))
    cand = {w: v for w, v in cand.items() if w not in all_given}

    # Corpus POS profile: keep lemmas attested only as singular PROPN.
    if args.stats_pkl:
        propn_profile = profile_from_stats_pkl(args.stats_pkl)
    else:
        propn_profile = profile_from_corpus(
            args.corpus, args.jsonl_field, args.spacy_model, args.batch_size)
    print(f"POS profile: {len(propn_profile)} lemmas with PROPN attestation")
    cand = {w: v for w, v in cand.items()
            if w.lower() in propn_profile
            and propn_profile[w.lower()][1] == 0
            and propn_profile[w.lower()][2] == 0}

    # No lowercase twin at any count, and plain word shape.
    av_all = set(load_vocab(args.allowed_vocab, min_count=0))
    cand = {w: v for w, v in cand.items() if w.lower() not in av_all}
    cand = {w: v for w, v in cand.items()
            if w.isalpha() and 3 < len(w) < 11 and w.isascii() and not w.isupper()}

    with open(args.out, "w", encoding="utf-8") as fout:
        fout.write("\n".join(cand.keys()))
    print(f"wrote {len(cand)} proper nouns -> {args.out}")


if __name__ == "__main__":
    main()
