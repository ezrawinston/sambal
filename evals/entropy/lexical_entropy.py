"""Lexical entropy of a corpus: unigram entropy + block-entropy h(n) curves.

Definitions
-----------
Word tokens are the units (see ``tokenize``); "lexical" means we measure the
distribution over word types, not BPE pieces or characters.

    H(n)  = entropy (bits) of the empirical distribution over word n-grams
    h(n) := H(n) - H(n-1)          with H(0) := 0, so h(1) = H(1)

h(n) is the block-entropy-difference estimate of the conditional entropy of the
next word given the previous n-1 words. It is the standard "entropy of order n"
of a text (Shannon 1951; Ebeling & Nicolis 1994).

Estimators (all reported, per n)
--------------------------------
plugin       maximum-likelihood / naive:  H = -sum p_i log2 p_i on observed counts.
             Downward-biased; the bias grows with the number of types relative
             to the sample size, so plug-in h(n) is driven to 0 once nearly
             every n-gram is unique. This is the literal H(n)-H(n-1) asked for.
miller_madow plugin + (K-1)/(2 N ln 2). A first-order bias correction; it is
             known to be inadequate once K ~ N, and is reported mostly to show
             the size of the leading bias term.
chao_shen    Chao & Shen (2003) coverage-adjusted Horvitz-Thompson estimator.
             Much better behaved on heavy-tailed word distributions, but it too
             collapses when the sample carries no repeat information.

Because every estimator of a high-order block entropy is sample-size dependent,
the CLI measures each metric at a ladder of token budgets (``--sweep-tokens``)
and on two disjoint halves of the stream. Two rules follow from that output:

  * compare corpora only at a COMMON token budget, and
  * treat an h(n) value as meaningful only where it has stopped moving with the
    budget, and where the between-halves spread is small relative to the
    between-corpora difference. ``saturation`` (= types/tokens for that n) says
    how close to the all-unique degenerate regime the estimate sits.

n-grams are counted by a 64-bit rolling hash + sort (``np.unique``), so memory
is O(tokens) rather than O(distinct n-grams as Python tuples). Expected hash
collisions for m distinct n-grams are ~m^2 / 2^65, i.e. ~1e-5 at m = 8e6 --
far below every other source of error here.

Usage
-----
    python lexical_entropy.py --name babycosmofine \
        --paths 'data/babycosmofine/train.jsonl' \
        --format jsonl --max-tokens 8000000 --max-n 8 \
        --out babycosmofine_entropy.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import time

import numpy as np

LN2 = math.log(2.0)

# Word tokens: unicode letter runs (with internal apostrophes) OR digit runs.
# Punctuation is dropped; it is not lexical material.
WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*|\d+", re.UNICODE)

# Odd 64-bit multiplier (FNV-style); any odd constant with good bit mixing works.
HASH_MULT = np.uint64(1099511628211)


# --------------------------------------------------------------------------
# corpus readers
# --------------------------------------------------------------------------

def _iter_parquet(path, text_key):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg, columns=[text_key])
        for doc in table.column(text_key).to_pylist():
            if doc:
                yield doc


def _iter_jsonl(path, text_key):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                yield line
                continue
            if isinstance(obj, dict):
                doc = obj.get(text_key)
                if doc:
                    yield doc
            elif isinstance(obj, str):
                yield obj


def _iter_text(path, text_key):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line:
                yield line


_READERS = {"parquet": _iter_parquet, "jsonl": _iter_jsonl, "text": _iter_text}


def detect_format(path):
    low = path.lower()
    if low.endswith(".parquet"):
        return "parquet"
    if low.endswith((".jsonl", ".json", ".ndjson")):
        return "jsonl"
    return "text"


def iter_docs(paths, fmt="auto", text_key="text", round_robin=True):
    """Yield documents from ``paths``, round-robining across files by default.

    Round-robin keeps a small token budget from being a slice of one shard's
    local document ordering.
    """
    iters = []
    for p in paths:
        f = detect_format(p) if fmt == "auto" else fmt
        iters.append(_READERS[f](p, text_key))
    if not round_robin or len(iters) == 1:
        for it in iters:
            yield from it
        return
    alive = list(iters)
    while alive:
        nxt = []
        for it in alive:
            try:
                yield next(it)
            except StopIteration:
                continue
            nxt.append(it)
        alive = nxt


def tokenize(text, lowercase=True):
    if lowercase:
        text = text.lower()
    return WORD_RE.findall(text)


def build_token_ids(paths, max_tokens, fmt="auto", text_key="text",
                    lowercase=True, round_robin=True, progress_every=2_000_000,
                    log=print):
    """Stream documents until ``max_tokens`` word tokens are collected.

    Returns (ids uint32 array, vocab list, stats dict). Documents are NOT
    concatenated across a boundary marker: n-grams spanning a document
    boundary are excluded (see ``doc_starts``) so the measurement is of
    within-document word sequences.
    """
    vocab = {}
    ids_chunks = []
    doc_lengths = []
    n_tokens = 0
    n_docs = 0
    n_chars = 0
    t0 = time.time()
    for doc in iter_docs(paths, fmt=fmt, text_key=text_key, round_robin=round_robin):
        toks = tokenize(doc, lowercase=lowercase)
        if not toks:
            continue
        if n_tokens + len(toks) > max_tokens:
            toks = toks[: max_tokens - n_tokens]
            if not toks:
                break
        arr = np.empty(len(toks), dtype=np.uint32)
        for i, t in enumerate(toks):
            tid = vocab.get(t)
            if tid is None:
                tid = len(vocab)
                vocab[t] = tid
            arr[i] = tid
        ids_chunks.append(arr)
        doc_lengths.append(len(toks))
        n_tokens += len(toks)
        n_docs += 1
        n_chars += len(doc)
        if progress_every and n_tokens % progress_every < len(toks):
            log(f"    [read] {n_tokens:,} tokens / {n_docs:,} docs "
                f"({time.time() - t0:.0f}s)")
        if n_tokens >= max_tokens:
            break
    ids = np.concatenate(ids_chunks) if ids_chunks else np.empty(0, np.uint32)
    doc_starts = np.concatenate([[0], np.cumsum(doc_lengths)[:-1]]).astype(np.int64) \
        if doc_lengths else np.zeros(0, np.int64)
    stats = {
        "tokens_read": int(ids.size),
        "docs_read": n_docs,
        "chars_read": n_chars,
        "chars_per_token": (n_chars / ids.size) if ids.size else float("nan"),
        "mean_tokens_per_doc": (ids.size / n_docs) if n_docs else float("nan"),
        "read_seconds": time.time() - t0,
        "exhausted_input": n_tokens < max_tokens,
    }
    return ids, list(vocab), doc_starts, np.asarray(doc_lengths, dtype=np.int64), stats


# --------------------------------------------------------------------------
# entropy estimators
# --------------------------------------------------------------------------

def entropy_estimates(counts, n_samples):
    """Entropy (bits) of a distribution observed as integer ``counts``."""
    counts = counts[counts > 0].astype(np.float64)
    n = float(n_samples)
    k = counts.size
    p = counts / n
    plugin = float(-np.sum(p * np.log2(p)))

    miller_madow = plugin + (k - 1.0) / (2.0 * n * LN2)

    f1 = float(np.count_nonzero(counts == 1))
    f1_adj = f1 if f1 < n else n - 1.0          # guard: coverage would be 0
    coverage = 1.0 - f1_adj / n
    if coverage <= 0.0:
        chao_shen = float("nan")
    else:
        pa = coverage * p
        denom = 1.0 - np.power(1.0 - pa, n)
        good = denom > 0
        chao_shen = float(-np.sum(pa[good] * np.log2(pa[good]) / denom[good]))

    return {
        "plugin": plugin,
        "miller_madow": float(miller_madow),
        "chao_shen": chao_shen,
        "types": int(k),
        "samples": int(n_samples),
        "singletons": int(f1),
        "singleton_frac": f1 / n if n else float("nan"),
        "coverage": float(coverage),
        "saturation": k / n if n else float("nan"),
        "max_entropy_bits": math.log2(n) if n > 0 else float("nan"),
    }


def _ngram_hashes(ids_u64, prev_hash, n):
    """Rolling 64-bit hash of every n-gram: h_n[i] = h_{n-1}[i]*P + ids[i+n-1]."""
    m = ids_u64.size - n + 1
    if m <= 0:
        return np.empty(0, dtype=np.uint64)
    with np.errstate(over="ignore"):
        return prev_hash[:m] * HASH_MULT + ids_u64[n - 1: n - 1 + m]


def _valid_ngram_mask(doc_lengths, n, total):
    """True for n-gram start positions that lie fully inside one document."""
    if n == 1:
        return None
    mask = np.zeros(total, dtype=bool)
    offset = 0
    for L in doc_lengths:
        if L >= n:
            mask[offset: offset + L - n + 1] = True
        offset += L
    return mask[: total - n + 1] if total >= n else mask[:0]


def block_entropies(ids, doc_lengths, max_n, log=print):
    """H(n) for n = 1..max_n on one token stream (documents kept separate)."""
    t = ids.size
    ids_u64 = ids.astype(np.uint64)
    out = {}

    counts = np.bincount(ids)
    out[1] = entropy_estimates(counts[counts > 0], t)

    prev_hash = ids_u64
    for n in range(2, max_n + 1):
        h = _ngram_hashes(ids_u64, prev_hash, n)
        prev_hash = h
        if h.size == 0:
            break
        mask = _valid_ngram_mask(doc_lengths, n, t)
        hv = h if mask is None else h[mask]
        if hv.size == 0:
            break
        _, c = np.unique(hv, return_counts=True)
        out[n] = entropy_estimates(c, hv.size)
        log(f"    [H] n={n} types={out[n]['types']:,} samples={hv.size:,} "
            f"H_plugin={out[n]['plugin']:.4f}")
    return out


def h_curve(block):
    """h(n) = H(n) - H(n-1) per estimator, with H(0)=0."""
    curve = {}
    for n in sorted(block):
        prev = block.get(n - 1)
        row = {"n": n}
        for est in ("plugin", "miller_madow", "chao_shen"):
            hn = block[n][est]
            hprev = 0.0 if prev is None else prev[est]
            row[est] = hn - hprev
        row["H_plugin"] = block[n]["plugin"]
        row["H_chao_shen"] = block[n]["chao_shen"]
        row["saturation"] = block[n]["saturation"]
        row["types"] = block[n]["types"]
        row["samples"] = block[n]["samples"]
        curve[n] = row
    return curve


# --------------------------------------------------------------------------
# analysis over budgets
# --------------------------------------------------------------------------

def analyze(ids, doc_lengths, max_n, budgets, log=print):
    """Measure at each token budget, plus on two disjoint halves of the stream."""
    total = ids.size

    def _slice(lo, hi):
        """Token slice [lo, hi) with the doc-length structure restricted to it."""
        sub = ids[lo:hi]
        edges = np.concatenate([[0], np.cumsum(doc_lengths)])
        lens = []
        for i in range(len(doc_lengths)):
            a, b = edges[i], edges[i + 1]
            s, e = max(a, lo), min(b, hi)
            if e > s:
                lens.append(e - s)
            if a >= hi:
                break
        return sub, np.asarray(lens, dtype=np.int64)

    results = {"budgets": {}, "halves": {}}
    for b in budgets:
        if b > total:
            continue
        sub, lens = _slice(0, b)
        log(f"  [budget] {b:,} tokens")
        results["budgets"][str(b)] = h_curve(block_entropies(sub, lens, max_n, log=log))

    half = total // 2
    if half > 1000:
        for name, (lo, hi) in (("first", (0, half)), ("second", (half, 2 * half))):
            sub, lens = _slice(lo, hi)
            log(f"  [half:{name}] {sub.size:,} tokens")
            results["halves"][name] = h_curve(block_entropies(sub, lens, max_n, log=log))
        results["half_tokens"] = half
    return results


def measure_corpus(name, paths, max_tokens, max_n, budgets, fmt="auto",
                   text_key="text", lowercase=True, round_robin=True, log=print):
    log(f"[corpus] {name}: {len(paths)} file(s), budget {max_tokens:,} tokens")
    ids, vocab, _starts, doc_lengths, read_stats = build_token_ids(
        paths, max_tokens, fmt=fmt, text_key=text_key,
        lowercase=lowercase, round_robin=round_robin, log=log)
    log(f"  read {read_stats['tokens_read']:,} tokens, "
        f"{len(vocab):,} types, {read_stats['read_seconds']:.0f}s")
    budgets = [b for b in budgets if b <= ids.size] or [ids.size]
    res = analyze(ids, doc_lengths, max_n, budgets, log=log)
    return {
        "name": name,
        "paths": paths,
        "config": {
            "max_tokens": max_tokens,
            "max_n": max_n,
            "budgets": budgets,
            "lowercase": lowercase,
            "round_robin": round_robin,
            "token_regex": WORD_RE.pattern,
            "text_key": text_key,
        },
        "read_stats": read_stats,
        "vocab_size_full_sample": len(vocab),
        "results": res,
    }


def format_table(measurement, budget=None):
    res = measurement["results"]["budgets"]
    if budget is None:
        budget = max(int(b) for b in res)
    curve = res[str(budget)]
    lines = [f"{measurement['name']}  @ {int(budget):,} word tokens",
             f"  {'n':>2}  {'H(n)':>9}  {'h(n)':>8}  {'h_CS(n)':>8}  "
             f"{'types':>12}  {'sat':>6}"]
    for n in sorted(curve):
        r = curve[n]
        lines.append(f"  {n:>2}  {r['H_plugin']:9.4f}  {r['plugin']:8.4f}  "
                     f"{r['chao_shen']:8.4f}  {r['types']:12,}  {r['saturation']:6.3f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--paths", nargs="+", required=True,
                    help="files or globs")
    ap.add_argument("--format", default="auto",
                    choices=["auto", "parquet", "jsonl", "text"])
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--max-tokens", type=int, default=8_000_000)
    ap.add_argument("--max-n", type=int, default=8)
    ap.add_argument("--sweep-tokens", default="",
                    help="comma-separated token budgets; default = halving ladder")
    ap.add_argument("--lowercase", type=int, default=1)
    ap.add_argument("--no-round-robin", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    paths = []
    for p in args.paths:
        hits = sorted(glob.glob(p))
        paths.extend(hits if hits else [p])
    if not paths:
        sys.exit("no input files matched")

    if args.sweep_tokens:
        budgets = sorted(int(x) for x in args.sweep_tokens.split(","))
    else:
        budgets, b = [], args.max_tokens
        while b >= 250_000:
            budgets.append(b)
            b //= 2
        budgets = sorted(budgets)

    m = measure_corpus(args.name, paths, args.max_tokens, args.max_n, budgets,
                       fmt=args.format, text_key=args.text_key,
                       lowercase=bool(args.lowercase),
                       round_robin=not args.no_round_robin)
    print()
    print(format_table(m))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(m, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
