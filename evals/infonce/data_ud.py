
"""
Simple UD CoNLL-U loader (English EWT or similar).
Parses tokens, UPOS, HEAD, DEPREL, sentence boundaries.
"""
from typing import List, Dict, Tuple, Iterable
from dataclasses import dataclass

import re

@dataclass
class UDSent:
    words: List[str]
    upos: List[str]
    head: List[int]      # 1-based head indices; 0 = root
    deprel: List[str]

def load_conllu(path: str) -> List[UDSent]:
    sents = []
    words, upos, head, deprel = [], [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if words:
                    sents.append(UDSent(words, upos, head, deprel))
                    words, upos, head, deprel = [], [], [], []
                continue
            if line.startswith("#"):  # comment
                continue
            cols = line.split("\t")
            if "-" in cols[0] or "." in cols[0]:
                # skip multi-word tokens and empty nodes
                continue
            idx = int(cols[0])
            w = cols[1]
            u = cols[3]
            h = int(cols[6])
            d = cols[7]
            words.append(w)
            upos.append(u)
            head.append(h)
            deprel.append(d)
    if words:
        sents.append(UDSent(words, upos, head, deprel))
    return sents

def find_subject_verb_indices(sent: UDSent) -> List[Tuple[int,int]]:
    """
    Returns (subj_idx, verb_idx) pairs where verb_idx is head of subject.
    Indices are 0-based for Python.
    """
    out = []
    for i, rel in enumerate(sent.deprel):
        if rel.startswith("nsubj"):
            v_head = sent.head[i]
            if v_head > 0:
                out.append((i, v_head - 1))
    return out

def find_attractors(sent: UDSent, subj_idx: int, verb_idx: int) -> List[int]:
    """
    Simple heuristic: other NOUN/PROPN tokens between subject and verb.
    Returns 0-based indices.
    """
    lo = min(subj_idx, verb_idx)
    hi = max(subj_idx, verb_idx)
    attr = []
    for i in range(lo+1, hi):
        if sent.upos[i] in ("NOUN","PROPN"):
            attr.append(i)
    return attr
