"""Tokenization + swap-only checking utilities.

We want conflict pairs to differ *only* by swapping two tokens. This reduces confounds
from unigram frequency and keeps the manipulation surgical.

For our templates, a simple regex tokenizer is sufficient.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

_TOK_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def simple_tokenize(s: str) -> List[str]:
    return _TOK_RE.findall(s)

def is_swap_only(a: str, b: str) -> Tuple[bool, Optional[Tuple[int,int]], Optional[Tuple[str,str]]]:
    """Return whether b can be obtained from a by swapping exactly two tokens.

    Returns:
      ok, (i,j) indices swapped, (tok_i, tok_j) swapped tokens
    """
    ta = simple_tokenize(a)
    tb = simple_tokenize(b)
    if len(ta) != len(tb):
        return False, None, None
    diffs = [i for i,(x,y) in enumerate(zip(ta,tb)) if x!=y]
    if len(diffs) != 2:
        return False, None, None
    i,j = diffs
    if ta[i] == tb[j] and ta[j] == tb[i] and all(k not in diffs or True for k in diffs):
        # Ensure everything else identical
        for k in range(len(ta)):
            if k==i or k==j:
                continue
            if ta[k] != tb[k]:
                return False, None, None
        return True, (i,j), (ta[i], ta[j])
    return False, None, None
