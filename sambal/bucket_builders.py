"""
Bucket builders for context-based lemma statistics.

These functions define the exact tuple structure expected by _ctx_backoff_chain.
Both _ctx_bucket_for_token (live Token) and _bucket_from_tokinfo (stored stats)
MUST use these to ensure the invariant:
  _ctx_bucket_for_token(tok) == _bucket_from_tokinfo(tokeninfo_key(tok))
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def _build_verb_bucket(
    vtag: str,
    dep: str,
    head_pos: str,
    head_tag: str,
    head_has_obj: bool,
    frame_key: str,
    frame_has_prt: bool,
    preps_set: Tuple[str, ...],
    comp_kind: str,
    is_passive: bool,
    has_expl_there: bool,
    to_be_xcomp: bool,
    xcomp_has_to: bool,
    xcomp_has_for: bool,
    xcomp_is_ger: bool,
    verb_has_to_aux: bool,
    vtok_child_mask: int,
) -> Tuple[Any, ...]:
    """Build VERB bucket tuple (18 elements).

    Structure expected by _ctx_backoff_chain:
    (kind, vtag, dep, head_pos, head_tag, head_has_obj, frame_key, frame_has_prt,
     preps_set, comp_kind, is_passive, has_expl_there, to_be_xcomp, x_to, x_for,
     x_ger, verb_has_to_aux, child_mask)
    """
    return (
        "VERB",
        vtag,
        dep,
        head_pos,
        head_tag,
        bool(head_has_obj),
        frame_key,
        bool(frame_has_prt),
        preps_set,
        comp_kind,
        bool(is_passive),
        bool(has_expl_there),
        bool(to_be_xcomp),
        bool(xcomp_has_to),
        bool(xcomp_has_for),
        bool(xcomp_is_ger),
        bool(verb_has_to_aux),
        int(vtok_child_mask),
    )


def _build_noun_bucket(ptb_tag: str, is_bare: bool) -> Tuple[Any, ...]:
    """Build NOUN bucket tuple (3 elements).

    Structure: (kind, ptb_tag, is_bare)
    """
    return ("NOUN", ptb_tag, bool(is_bare))


def _build_adj_bucket(
    atag: str,
    dep: str,
    head_pos: str,
    head_tag: str,
    relpos: int,
    prev_tag: str,
    next_tag: str,
    morph_sig: Tuple[str, ...],
    adj_has_pp: bool,
    adj_has_xcomp: bool,
    adj_has_ccomp: bool,
    head_has_obj: bool,
    head_has_aux: bool,
) -> Tuple[Any, ...]:
    """Build ADJ bucket tuple (14 elements).

    Structure expected by _ctx_backoff_chain:
    (kind, tag, dep, head_pos, head_tag, relpos, prev_tag, next_tag, morph_sig,
     adj_has_pp, adj_has_xcomp, adj_has_ccomp, head_has_obj, head_has_aux)
    """
    return (
        "ADJ",
        atag,
        dep,
        head_pos,
        head_tag,
        int(relpos),
        prev_tag,
        next_tag,
        morph_sig,
        int(bool(adj_has_pp)),
        int(bool(adj_has_xcomp)),
        int(bool(adj_has_ccomp)),
        int(bool(head_has_obj)),
        int(bool(head_has_aux)),
    )


def _build_adv_bucket(
    adv_tag: str,
    dep: str,
    head_pos: str,
    head_tag: str,
    relpos: int,
    prev_b: str,
    next_b: str,
    prev_tag: str,
    next_tag: str,
    morph_sig: Tuple[str, ...],
) -> Tuple[Any, ...]:
    """Build ADV bucket tuple (11 elements).

    Structure expected by _ctx_backoff_chain:
    (kind, tag, dep, head_pos, head_tag, relpos, prev_b, next_b, prev_tag, next_tag, morph_sig)
    """
    return (
        "ADV",
        adv_tag,
        dep,
        head_pos,
        head_tag,
        int(relpos),
        prev_b,
        next_b,
        prev_tag,
        next_tag,
        morph_sig,
    )


def _build_adv_pcomp_bucket(
    adv_tag: str,
    head_prep: str,
    adp_dep: str,
    grandhead_pos: str,
    grandhead_tag: str,
    prev_tag: str,
    next_tag: str,
    morph_sig: Tuple[str, ...],
) -> Tuple[Any, ...]:
    """Build ADV_PCOMP bucket tuple (9 elements).

    Structure expected by _ctx_backoff_chain:
    (kind, tag, head_prep, adp_dep, grandhead_pos, grandhead_tag, prev_tag, next_tag, morph_sig)
    """
    return (
        "ADV_PCOMP",
        adv_tag,
        head_prep,
        adp_dep,
        grandhead_pos,
        grandhead_tag,
        prev_tag,
        next_tag,
        morph_sig,
    )


def bucket_from_tokinfo(tokinfo: tuple, *, schema: Dict[str, int]) -> Optional[Tuple[Any, ...]]:
    """Extract a bucket tuple from a stored tokeninfo tuple.

    Uses the shared bucket builders to ensure tuple structure matches
    _ctx_bucket_for_token (the invariant for correct stats lookup).
    """
    # Base fields
    pos = tokinfo[schema["pos"]]
    tag = tokinfo[schema["tag"]]
    dep = tokinfo[schema["dep"]]
    head_pos = tokinfo[schema["head_pos"]]
    head_tag = tokinfo[schema.get("head_tag", schema["tag"])]

    is_verb_like = bool(tokinfo[schema["is_verb_like"]])
    is_bare = bool(tokinfo[schema["is_bare_singular"]])

    ptb_noun_forced = tokinfo[schema["ptb_noun_forced"]]
    ptb_adj = tokinfo[schema["ptb_adj"]]
    ptb_adv = tokinfo[schema["ptb_adv"]]

    # Optional ctx bits appended by collector (tokinfo[-1])
    relpos = 0
    prev_b = next_b = "OTHER"
    adv_head_prep = ""
    adj_has_pp = adj_has_xcomp = adj_has_ccomp = 0
    verb_has_to_aux = 0
    head_has_obj = 0
    head_has_aux = 0
    vtok_child_mask = 0

    prev_tag = "OTHER"
    next_tag = "OTHER"
    morph_sig = ("", "", "", "")
    adp_dep = ""
    grandhead_pos = ""
    grandhead_tag = ""

    try:
        cb = tokinfo[-1]
        if isinstance(cb, tuple) and cb and cb[0] in {"ctx_v2", "ctx_v3"}:
            relpos = int(cb[1]) if len(cb) > 1 else 0
            prev_b = str(cb[2]) if len(cb) > 2 else "OTHER"
            next_b = str(cb[3]) if len(cb) > 3 else "OTHER"
            adv_head_prep = str(cb[4]) if len(cb) > 4 else ""
            adj_has_pp = int(cb[5]) if len(cb) > 5 else 0
            adj_has_xcomp = int(cb[6]) if len(cb) > 6 else 0
            adj_has_ccomp = int(cb[7]) if len(cb) > 7 else 0
            verb_has_to_aux = int(cb[8]) if len(cb) > 8 else 0
            head_has_obj = int(cb[9]) if len(cb) > 9 else 0
            head_has_aux = int(cb[10]) if len(cb) > 10 else 0
            vtok_child_mask = int(cb[11]) if len(cb) > 11 else 0

            # new fields (ctx_v3)
            prev_tag = str(cb[12]) if len(cb) > 12 else "OTHER"
            next_tag = str(cb[13]) if len(cb) > 13 else "OTHER"
            morph_sig = cb[14] if len(cb) > 14 else ("", "", "", "")
            adp_dep = str(cb[15]) if len(cb) > 15 else ""
            grandhead_pos = str(cb[16]) if len(cb) > 16 else ""
            grandhead_tag = str(cb[17]) if len(cb) > 17 else ""
    except Exception:
        pass

    # === VERB ===
    if is_verb_like or pos in {"VERB", "AUX"}:
        frame_key = tokinfo[schema["frame_key"]]
        frame_has_prt = bool(tokinfo[schema["frame_has_prt"]])
        preps_set = tuple(tokinfo[schema["preps_set"]] or ())
        comp_kind = tokinfo[schema["comp_kind"]] or ""
        is_passive = bool(tokinfo[schema["is_passive"]])
        has_expl_there = bool(tokinfo[schema["has_expl_there"]])
        to_be_xcomp = bool(tokinfo[schema["to_be_xcomp"]])
        xcomp_has_to = bool(tokinfo[schema["xcomp_has_to"]])
        xcomp_has_for = bool(tokinfo[schema["xcomp_has_for"]])
        xcomp_is_ger = bool(tokinfo[schema["xcomp_is_ger"]])

        # Mirror VN query relaxation used at runtime
        if frame_key == "intrans" and preps_set:
            comp_like = {"like", "as", "than"} & set(preps_set)
            if comp_like:
                preps_set = tuple(sorted(set(preps_set) - comp_like))

        verb_tag = tokinfo[schema.get("ptb_verb_forced", schema["tag"])] or tag

        return _build_verb_bucket(
            vtag=verb_tag,
            dep=dep,
            head_pos=head_pos,
            head_tag=head_tag,
            head_has_obj=head_has_obj,
            frame_key=frame_key,
            frame_has_prt=frame_has_prt,
            preps_set=preps_set,
            comp_kind=comp_kind,
            is_passive=is_passive,
            has_expl_there=has_expl_there,
            to_be_xcomp=to_be_xcomp,
            xcomp_has_to=xcomp_has_to,
            xcomp_has_for=xcomp_has_for,
            xcomp_is_ger=xcomp_is_ger,
            verb_has_to_aux=verb_has_to_aux,
            vtok_child_mask=vtok_child_mask,
        )

    # === NOUN ===
    if pos in {"NOUN", "PROPN"}:
        return _build_noun_bucket(ptb_noun_forced or tag, is_bare)

    # === ADJ ===
    if pos == "ADJ":
        return _build_adj_bucket(
            atag=ptb_adj or tag,
            dep=dep,
            head_pos=head_pos,
            head_tag=head_tag,
            relpos=relpos,
            prev_tag=prev_tag,
            next_tag=next_tag,
            morph_sig=morph_sig,
            adj_has_pp=adj_has_pp,
            adj_has_xcomp=adj_has_xcomp,
            adj_has_ccomp=adj_has_ccomp,
            head_has_obj=head_has_obj,
            head_has_aux=head_has_aux,
        )

    # === ADV ===
    if pos == "ADV":
        if dep == "pcomp" and head_pos == "ADP" and adv_head_prep:
            return _build_adv_pcomp_bucket(
                adv_tag=ptb_adv or tag,
                head_prep=adv_head_prep,
                adp_dep=adp_dep,
                grandhead_pos=grandhead_pos,
                grandhead_tag=grandhead_tag,
                prev_tag=prev_tag,
                next_tag=next_tag,
                morph_sig=morph_sig,
            )
        return _build_adv_bucket(
            adv_tag=ptb_adv or tag,
            dep=dep,
            head_pos=head_pos,
            head_tag=head_tag,
            relpos=relpos,
            prev_b=prev_b,
            next_b=next_b,
            prev_tag=prev_tag,
            next_tag=next_tag,
            morph_sig=morph_sig,
        )

    return None
