#!/usr/bin/env python
"""Per-layer InfoNCE alignment of hidden states with UPOS tags and with lexical identity.

Every word of a CoNLL-U treebank becomes one anchor: its sentence is encoded
without special tokens, the hidden state after each requested block is pooled
over the word's subwords, and a multi-positive InfoNCE loss is averaged over
all anchors, once with UPOS classes as the positives and once with lexical
types. Scores are reported negated, so higher is better.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from evals.backends import LanguageModel  # noqa: E402
from evals.common import add_backend_arguments, backend_from_args  # noqa: E402
from evals.infonce.common import infonce_multi_positive, pool_wordpieces_mean  # noqa: E402
from evals.infonce.data_ud import load_conllu  # noqa: E402

FUNCTION_UPOS = {"DET", "ADP", "AUX", "PART", "PRON", "CCONJ", "SCONJ", "PUNCT", "INTJ", "X", "SYM"}


def build_groups_from_offsets(enc_offsets: Sequence[Tuple[int, int]],
                              words: Sequence[str]) -> Tuple[List[List[int]], List[int]]:
    """Map UD words (joined by single spaces) to the token indices overlapping them.

    Returns the per-word token index lists and the indices of the words that
    got at least one token.
    """
    spans = []
    pos = 0
    for w in words:
        spans.append((pos, pos + len(w)))
        pos += len(w) + 1

    T_sub = len(enc_offsets)
    groups: List[List[int]] = []
    keep_word_idx: List[int] = []
    j = 0
    for i, (ws, we) in enumerate(spans):
        while j < T_sub and enc_offsets[j][1] <= ws:
            j += 1
        jj = j
        cur: List[int] = []
        while jj < T_sub and not (enc_offsets[jj][0] >= we):
            if not (enc_offsets[jj][1] <= ws or enc_offsets[jj][0] >= we):
                cur.append(jj)
            jj += 1
        if not cur and j < T_sub:
            cur = [j]
        if cur:
            groups.append(cur)
            keep_word_idx.append(i)
    return groups, keep_word_idx


def collect_word_features(backend: LanguageModel, conllu_path: str, layers: Sequence[int],
                          pool: str, max_tokens: int):
    """Per requested layer, the pooled word representations, plus each word's UPOS tag and type."""
    feats_per_layer: Dict[int, List[torch.Tensor]] = {L: [] for L in layers}
    upos_vals: List[str] = []
    lex_vals: List[str] = []
    total_words = 0

    for sent in load_conllu(conllu_path):
        text = " ".join(sent.words)
        ids, offsets = backend.encode_with_offsets(text)
        if len(ids) == 0:
            continue
        hiddens = backend.hidden_states([ids], list(layers), add_special_tokens=False)[0]  # [n_sel, T, D]

        groups, keep_word_idx = build_groups_from_offsets(offsets, sent.words)
        if len(groups) < 2:
            continue

        for position, L in enumerate(layers):
            H_full = hiddens[position]  # [T, D]
            if pool == "first":
                idx = torch.tensor([g[0] for g in groups], dtype=torch.long)
                X = H_full.index_select(0, idx)
            else:
                X = pool_wordpieces_mean(H_full, groups)
            feats_per_layer[L].append(X)

        upos_vals.extend(sent.upos[i] for i in keep_word_idx)
        lex_vals.extend(sent.words[i].lower() for i in keep_word_idx)

        total_words += len(keep_word_idx)
        if total_words >= max_tokens:
            break

    stacked = {L: (torch.cat(v, dim=0) if v else torch.empty(0)) for L, v in feats_per_layer.items()}
    return stacked, upos_vals, lex_vals


def evaluate_infonce(backend: LanguageModel, conllu_path: str, layers: Sequence[int] = (-1,),
                     pool: str = "mean", tau: float = 0.1, max_tokens: int = 20000,
                     min_lex_freq: int = 2, batch_size: int = 2048,
                     lex_exclude_func: bool = False) -> dict:
    """Score every requested layer against UPOS classes and against lexical types."""
    layers = list(layers)
    feats, upos_vals, lex_vals = collect_word_features(backend, conllu_path, layers, pool, max_tokens)

    upos_vocab = sorted(set(upos_vals))
    upos2i = {t: i for i, t in enumerate(upos_vocab)}
    upos_ids = torch.tensor([upos2i[u] for u in upos_vals], dtype=torch.long)

    counts = Counter(lex_vals)
    lex_vocab = sorted({w for w, f in counts.items() if f >= min_lex_freq})
    lex2i = {w: i for i, w in enumerate(lex_vocab)}
    lex_ids = torch.tensor([lex2i.get(w, -1) for w in lex_vals], dtype=torch.long)

    rows: List[dict] = []
    for L in layers:
        X = feats[L]
        if X.numel() == 0:
            rows.append({"layer": L, "infonce_upos": float("nan"), "infonce_lex": float("nan"), "n_tokens": 0})
            continue

        N = X.size(0)
        Xd = X.to(backend.device)
        upos_loss = infonce_multi_positive(Xd, upos_ids[:N].to(backend.device), tau=tau, batch_size=batch_size)

        valid_mask = lex_ids[:N] >= 0
        if lex_exclude_func:
            func_mask = torch.tensor([u in FUNCTION_UPOS for u in upos_vals[:N]], dtype=torch.bool)
            valid_mask = valid_mask & (~func_mask)
        if valid_mask.any():
            lex_loss = infonce_multi_positive(Xd[valid_mask.to(backend.device)],
                                              lex_ids[:N][valid_mask].to(backend.device),
                                              tau=tau, batch_size=batch_size)
        else:
            lex_loss = float("nan")

        rows.append({"layer": L, "infonce_upos": -float(upos_loss), "infonce_lex": -float(lex_loss),
                     "n_tokens": int(N)})

    summary: Dict[str, float] = {}
    for row in rows:
        summary[f"infonce/upos/{row['layer']}"] = row["infonce_upos"]
        summary[f"infonce/lex/{row['layer']}"] = row["infonce_lex"]
        summary[f"infonce/n_tokens/{row['layer']}"] = row["n_tokens"]

    protocol = {"layers": layers, "pool": pool, "tau": float(tau), "max_tokens": max_tokens,
                "min_lex_freq": min_lex_freq, "lex_exclude_func": bool(lex_exclude_func)}
    return {"summary": summary, "rows": rows, "protocol": protocol}


def record_from_result(result: dict) -> dict:
    """The committed per-arm record: the protocol that produced it and one value per layer."""
    return {
        "layers": result["protocol"]["layers"],
        "tau": result["protocol"]["tau"],
        "pool": result["protocol"]["pool"],
        "infonce_upos": [row["infonce_upos"] for row in result["rows"]],
        "infonce_lex": [row["infonce_lex"] for row in result["rows"]],
        "N_tokens": [row["n_tokens"] for row in result["rows"]],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_backend_arguments(ap)
    ap.add_argument("--ud_conllu", required=True, help="CoNLL-U treebank the anchors come from")
    ap.add_argument("--pool", choices=["first", "mean"], default="mean",
                    help="pool a word's subword states, or take the first")
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--layers", type=int, nargs="+", default=[-1], help="layer indices; -1 is the last")
    ap.add_argument("--max_tokens", type=int, default=20000, help="stop after this many words")
    ap.add_argument("--min_lex_freq", type=int, default=2,
                    help="lexical types rarer than this are dropped from the lexical anchors")
    ap.add_argument("--batch_size", type=int, default=2048, help="anchors per InfoNCE block")
    ap.add_argument("--lex_exclude_func", action="store_true",
                    help="exclude function-word tokens (by UPOS) from Lex InfoNCE.")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_png", default=None, help="optional per-layer plot of the two curves")
    args = ap.parse_args()

    backend = backend_from_args(args)
    result = evaluate_infonce(backend, args.ud_conllu, layers=args.layers, pool=args.pool, tau=args.tau,
                              max_tokens=args.max_tokens, min_lex_freq=args.min_lex_freq,
                              batch_size=args.batch_size, lex_exclude_func=args.lex_exclude_func)
    record = record_from_result(result)

    out_json = pathlib.Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(record, indent=2))
    if not args.out_png:
        return

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(record["layers"], record["infonce_upos"], marker="o", label="UPOS (−InfoNCE)")
    plt.plot(record["layers"], record["infonce_lex"], marker="o", label="Lex (−InfoNCE)")
    plt.xlabel("Layer index (−1=last)")
    plt.ylabel("Score (higher=better)")
    plt.title(f"Per-layer InfoNCE (pool={args.pool}, tau={args.tau})")
    plt.legend()
    plt.tight_layout()
    out_png = pathlib.Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


if __name__ == "__main__":
    main()
