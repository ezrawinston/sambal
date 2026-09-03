"""
Verb frame analysis functions.

Encapsulates extraction of verb frames (intrans/trans/ditrans), complement kinds,
valency fixes for tough-adjectives and WH-gaps, and auxiliary chain detection.
Extracted from Augmenter to reduce class complexity.
"""
from __future__ import annotations
from typing import Optional, Tuple, Set, TYPE_CHECKING
from spacy.tokens import Token

from .token_utils import (
    is_particle_dep,
    is_auxpass_dep,
    is_nsubjpass_dep,
    is_obj_dep,
    is_infinitival_to_as_prep,
    iter_governed_preps,
    has_expl_there_child,
)

if TYPE_CHECKING:
    from .to_inf_handler import ToInfHandler


# =============================================================================
# WH/Interrogative Detection
# =============================================================================

WH_TAGS = {"WDT", "WP", "WP$", "WRB"}
WH_FORMS = {"who", "whom", "whose", "what", "which", "where", "when", "why", "how"}


def pron_type_parts(tok: Token) -> Set[str]:
    """Return PronType features as a flat set (splits comma-joined values)."""
    out: Set[str] = set()
    try:
        vals = tok.morph.get("PronType")
    except Exception:
        vals = []
    for v in vals:
        for p in str(v).split(","):
            p = p.strip()
            if p:
                out.add(p)
    return out


def is_interrogative_wh_token(tok: Token, doc=None) -> bool:
    """Heuristic: treat WH as interrogative only when PronType indicates Int.

    Used to avoid treating extracted WH objects (e.g., "What did you see?") as ordinary objects.
    """
    if tok.tag_ not in WH_TAGS:
        return False
    if doc is None:
        doc = tok.doc

    ptypes = pron_type_parts(tok)
    if ptypes:
        return "Int" in ptypes

    # Fallback: if morph is missing, only trust direct questions.
    try:
        return doc.text.strip().endswith("?")
    except Exception:
        return False


# =============================================================================
# Passive Clause Detection
# =============================================================================

def is_passive_clause(head: Token) -> bool:
    """Heuristic passive detection that works for spaCy and UD."""
    if "Pass" in set(head.morph.get("Voice")):
        return True
    if any(is_auxpass_dep(c.dep_) for c in head.children):
        return True
    if any(is_nsubjpass_dep(c.dep_) for c in head.children):
        return True
    return False


# =============================================================================
# Preposition Extraction
# =============================================================================

def extract_preps_set(tok: Token) -> Set[str]:
    """Collect governed prepositions for the verb head in this clause.

    Passive 'by' is filtered out as it's adjunct-like.
    """
    preps: Set[str] = set(p for p in iter_governed_preps(tok))
    # Passive 'by' is adjunct-like -> drop it
    if is_passive_clause(tok) and "by" in preps:
        preps.discard("by")
    return preps


# =============================================================================
# Frame Extraction
# =============================================================================

def extract_frame(tok: Token) -> Tuple[str, bool, Optional[str]]:
    """
    Extract a coarse frame signature for a VERB/AUX head.

    Returns (key, has_particle, required_preposition_or_None) where:
      - key in {"intrans", "trans", "ditrans", "trans_prt"}
      - has_particle flags phrasal verbs (prt/compound:prt)
      - required_preposition is one governed ADP lemma if any (passive 'by' ignored)
    """
    children = list(tok.children)

    # particle flag (spaCy: 'prt'; UD: 'compound:prt')
    has_prt = any(is_particle_dep(c.dep_) for c in children)

    # Used by the passive override below.
    is_passive = is_passive_clause(tok)

    # pick ONE governed preposition for logging (ignoring passive agentive "by")
    req_prep = None
    for p in iter_governed_preps(tok):
        if is_passive and p.lower() == "by":
            continue
        req_prep = p

    # Ignore "fronted" WH direct object when a finite clausal complement is present.
    clause_comp_children = [ch for ch in tok.children if ch.dep_ in {"ccomp", "csubj"}]
    has_clause_comp_core = bool(clause_comp_children)

    # Leftmost non-punct token in the tok-subtree (helps detect clause-initial WH movement).
    tok_left_i = min((t.i for t in tok.subtree if not t.is_punct), default=tok.i)

    def _ignore_as_extracted_wh_obj(c: Token) -> bool:
        if not has_clause_comp_core:
            return False
        if c.dep_ not in {"dobj", "obj"}:
            return False
        if c.i >= tok.i:
            return False
        # Only treat clause-initial WH as extraction.
        if c.i != tok_left_i:
            return False
        if is_interrogative_wh_token(c, doc=tok.doc):
            return True
        # Fallback: if tag looks WH and not marked as relative, treat as extracted.
        if c.tag_ in WH_TAGS and c.lower_ in WH_FORMS and c.dep_ != "det":
            if "Rel" in c.morph.get("PronType"):
                return False
            return True
        return False

    has_obj = any(
        is_obj_dep(c.dep_) and c.dep_ in {"obj", "dobj"} and not _ignore_as_extracted_wh_obj(c)
        for c in children
    )
    has_iobj = any(c.dep_ == "iobj" for c in children)

    # clausal complements (do not increase object count)
    has_clause_comp = any(c.dep_ in {"ccomp", "xcomp", "csubj", "advcl"} for c in children)

    # passive override
    if is_passive:
        return ("trans", has_prt, req_prep)

    # ditransitive: NP object + (iobj or governed PP recipient)
    if has_iobj or (has_obj and req_prep):
        return ("ditrans", has_prt, req_prep)

    # simple transitive or phrasal
    if has_obj or has_prt:
        if has_prt:
            return ("trans_prt", has_prt, req_prep)
        return ("trans", has_prt, req_prep)

    # clausal complement but no NP object -> intrans
    if has_clause_comp:
        return ("intrans", has_prt, req_prep)

    return ("intrans", has_prt, req_prep)


# =============================================================================
# Complement Kind Detection
# =============================================================================

def comp_kind_from_parse(v: Token) -> str:
    """Classify complement kind from parse tree structure.

    Order matters: classify CPs before NP object to avoid double counts in noisy parses.
    """
    # CP_THAT / CP_WH via ccomp
    for ch in v.children:
        if ch.dep_ == "ccomp":
            marks = {m.lemma_.lower() for m in ch.children if m.dep_ == "mark"}
            has_that = bool({"that", "if", "whether"} & marks)
            return "CP_THAT" if has_that else "CP_WH"

    # CP_TO / CP_FOR_TO / CP_ING via xcomp
    for ch in v.children:
        if ch.dep_ == "xcomp":
            has_to = any(t.lemma_.lower() == "to" and t.pos_ in {"PART", "SCONJ", "AUX"} for t in ch.subtree)
            has_for = any(t.lemma_.lower() == "for" and t.dep_ == "mark" for t in ch.children)
            if has_for and has_to:
                return "CP_FOR_TO"
            if has_to:
                return "CP_TO"
            if ch.tag_ == "VBG" or any(t.tag_ == "VBG" for t in ch.subtree):
                return "CP_ING"
            return "NONE"

    # DOUBLE_OBJ if iobj present
    if any(ch.dep_ == "iobj" for ch in v.children):
        return "DOUBLE_OBJ"

    # NP object
    if any(ch.dep_ in {"obj", "dobj"} for ch in v.children):
        return "NP_OBJ"

    return "NONE"


def parse_comp_kind(v: Token, key: str) -> str:
    """Map parse structure to complement kind, considering valency key.

    More comprehensive than comp_kind_from_parse, handles edge cases.
    """
    children = list(v.children)
    ccomps = [c for c in children if c.dep_ == "ccomp"]
    xcomps = [c for c in children if c.dep_ == "xcomp"]
    has_iobj = any(c.dep_ == "iobj" for c in children)
    has_obj = any(c.dep_ in {"obj", "dobj"} for c in children)

    # finite CP
    if ccomps:
        # spaCy sometimes attaches infinitival complements as `ccomp` (esp. ECM / control).
        # If a `ccomp` subtree contains infinitival *to*, treat as CP_TO / CP_FOR_TO.
        has_inf_to = any(
            (t.lemma_.lower() == "to" and t.pos_ in {"PART", "SCONJ", "AUX"})
            for c in ccomps for t in c.subtree
        )
        if has_inf_to:
            has_for = any(
                (m.dep_ == "mark" and m.lemma_.lower() == "for")
                for c in ccomps for m in c.children
            )
            return "CP_FOR_TO" if has_for else "CP_TO"

        # Stricter WH detection: require interrogative WH token or explicit complementizer.
        has_wh = False
        for c in ccomps:
            mark_lemmas = {m.lemma_.lower() for m in c.children if m.dep_ == "mark"}
            if {"whether", "if"} & mark_lemmas:
                has_wh = True
                break
            fronted_comp = c.i < v.i
            if any(is_interrogative_wh_token(t) and (fronted_comp or (t.i > v.i)) for t in c.subtree):
                has_wh = True
                break
        return "CP_WH" if has_wh else "CP_THAT"

    # non-finite CP / ECM
    if xcomps:
        has_for = any(m.dep_ == "mark" and m.lemma_.lower() == "for"
                      for c in xcomps for m in c.children)
        has_to = any(t.lemma_.lower() == "to" and t.pos_ in {"PART", "SCONJ", "AUX"}
                     for c in xcomps for t in c.subtree)
        is_ger = any((c.tag_ == "VBG") or any(t.tag_ == "VBG" for t in c.subtree) for c in xcomps)

        if has_for and has_to:
            return "CP_FOR_TO"
        if has_to:
            return "CP_TO"
        if is_ger:
            return "CP_ING"

        # bare xcomp with an NP object -> ECM/permissive
        if has_iobj:
            return "DOUBLE_OBJ"
        if has_obj or key in {"trans", "ditrans"}:
            return "NP_OBJ"
        return "NONE"

    # no CP observed -> fall back to valency
    if key == "ditrans" or has_iobj:
        return "DOUBLE_OBJ"
    if key == "trans":
        return "NP_OBJ"
    return "NONE"


# =============================================================================
# Auxiliary Chain Detection
# =============================================================================

def has_aux_chain(v: Token) -> bool:
    """True if v hosts an aux/auxpass (perfect/prog/periphrasis)."""
    return any(ch.dep_ in {"aux", "auxpass"} for ch in v.children)


def has_heavy_aux_chain(v: Token) -> bool:
    """
    Return True only for perfect/passive periphrasis on the lexical head:
      - PERFECT: have + VBN on the head
      - PASSIVE: auxpass present, or be/get + VBN with passive subject cues
    Modals, do-support, and simple progressive (be + VBG) are ignored.
    """
    auxes = [ch for ch in v.children if ch.dep_ in {"aux", "auxpass"}]
    if not auxes:
        return False

    lemmas = {a.lemma_.lower() for a in auxes}
    has_have = "have" in lemmas
    has_be = "be" in lemmas
    has_get = "get" in lemmas
    has_auxpass = any(is_auxpass_dep(a.dep_) for a in auxes)

    # head is a past participle (non-progressive)
    is_vbn = (v.tag_ == "VBN") or (
        v.morph.get("VerbForm") == ["Part"] and "Prog" not in v.morph.get("Aspect")
    )

    # perfect
    if has_have and is_vbn:
        return True

    # passive
    if has_auxpass:
        return True
    has_passive_subj = any(ch.dep_ == "nsubjpass" for ch in v.children)
    if is_vbn and (has_be or has_get) and has_passive_subj:
        return True

    return False


# =============================================================================
# Expletive and Structural Helpers
# =============================================================================

def is_to_be_xcomp_head(t: Token) -> bool:
    """Check if token is an embedded 'to be' xcomp head (prevents breaking expletive spines)."""
    if t.lemma_.lower() != "be" or t.pos_ not in {"AUX", "VERB"}:
        return False
    if t.dep_ != "xcomp":
        return False
    # require an infinitival marker 'to' under this xcomp
    return any(x.lemma_.lower() == "to" and x.pos_ in {"PART", "SCONJ", "AUX"} for x in t.subtree)


# =============================================================================
# Valency Fix Functions
# =============================================================================

def fix_wh_gap_valency(tok: Token, key: str, req_preps: Set[str], *, debug_fn=None) -> str:
    """Heuristic: in wh-object extraction contexts, treat the gap-site verb as transitive.

    We mainly want to catch cases like:
        "Which book did Mary say John bought ___ ?"
    where spaCy often fails to attach the extracted object to the deepest verb.

    Guardrails:
      - Only applies when verb has overt (non-wh) subject, no direct object, no governed PP.
      - Requires a wh-word in sentence AND do-support (aux 'do') to reduce false positives.
    """
    try:
        if key != "intrans":
            return key
        if req_preps:
            return key

        # If this verb itself selects a clausal complement, the wh-gap is almost always
        # *inside* that complement. Avoid forcing transitivity on the matrix verb.
        if any(ch.dep_ in {"ccomp", "xcomp", "csubj"} for ch in tok.children):
            if debug_fn:
                debug_fn(f"not enforcing trans on {tok}")
            return key

        if tok.pos_ != "VERB":
            return key

        has_obj = any(ch.dep_ in {"dobj", "obj"} for ch in tok.children)
        if has_obj:
            return key

        subj = next((ch for ch in tok.children if ch.dep_ == "nsubj"), None)
        if subj is None:
            return key

        # If the subject itself is wh (subject extraction), do NOT force transitivity.
        if subj.tag_.startswith("W") or subj.lower_ in {"what", "which", "who", "whom", "whose"}:
            return key

        sent = tok.sent
        has_wh = any(
            (t.tag_.startswith("W") or t.lower_ in {"what", "which", "who", "whom", "whose"})
            for t in sent
        )
        if not has_wh:
            return key

        has_do_aux = any((t.dep_ == "aux" and t.lemma_ == "do") for t in sent)
        if not has_do_aux:
            return key

        if debug_fn:
            debug_fn(f"[WH-GAP] Forcing transitivity at gap site: {tok.text} (key {key} -> trans)")

        return "trans"
    except Exception:
        return key


# =============================================================================
# Verb Child Mask
# =============================================================================

def verb_child_mask(vtok: Token) -> int:
    """Compute bitmask of verb child dependencies.

    Used for context-based bucket building to capture structural info about
    a verb's immediate children.

    Bit positions:
    - 0: has obj/dobj/iobj
    - 1: has ccomp
    - 2: has acomp/attr/oprd
    - 3: has advmod
    - 4: has intj

    Args:
        vtok: A verb Token.

    Returns:
        Integer bitmask encoding the presence of various child types.
    """
    deps = set()
    for ch in vtok.children:
        if ch.dep_ == "aux" or ch.pos_ == "PUNCT":
            continue
        deps.add(ch.dep_)
    has_obj = int(bool(deps & {"obj", "dobj", "iobj"}))
    has_ccomp = int("ccomp" in deps)
    has_pred = int(bool(deps & {"acomp", "attr", "oprd"}))
    has_adv = int("advmod" in deps)
    has_intj = int("intj" in deps)
    return (has_obj << 0) | (has_ccomp << 1) | (has_pred << 2) | (has_adv << 3) | (has_intj << 4)


class VerbFrameAnalyzer:
    """
    Encapsulates verb frame analysis with access to ToInfHandler for to-inf lexicons.

    This class provides methods that need access to the to-inf lexicons
    and configuration. Methods that don't need such access are available as
    standalone functions in this module.
    """

    def __init__(self, toinf: "ToInfHandler"):
        self.toinf = toinf

    def fix_tough_xcomp_valency(self, v: Token, key: str) -> str:
        """
        Fix valency for tough-to-INF adjective contexts.

        UD parses for tough-to-INF adjectives (e.g., "a book that was easy to read")
        typically contain no explicit object on the embedded infinitive. This makes
        truly transitive verbs like `read` look "intrans" to `extract_frame`.

        When we can confidently identify a *tough* adjective environment with an NP
        surface subject and no PP-gap, treat the embedded xcomp verb as TRANSITIVE.
        """
        # Import standalone detection functions from to_inf_handler
        from .to_inf_handler import xcomp_has_to, xcomp_has_prep_gap, adj_subject_license

        if key != "intrans":
            return key
        if v.dep_ != "xcomp":
            return key

        # If the embedded infinitive itself takes clausal complements (xcomp/ccomp),
        # don't override: this is usually not the simple tough-object-gap pattern.
        if any(ch.dep_ in {"ccomp", "xcomp"} for ch in v.children):
            return key

        # Only for to-INF clauses
        if not xcomp_has_to(v):
            return key

        # Only block override for *true* PP-gap/stranding cases
        try:
            if xcomp_has_prep_gap(v):
                return key
        except Exception:
            return key

        # Find the predicate ADJ that licenses this to-INF clause.
        # Walk UP the xcomp chain to handle nested infinitives.
        adj: Optional[Token] = None
        cur = v
        seen = {v.i}  # cycle guard
        while cur.head is not None and cur.head.i not in seen:
            seen.add(cur.head.i)
            head = cur.head
            if head.pos_ == "ADJ":
                adj = head
                break
            elif head.pos_ in {"AUX", "VERB"}:
                # Check if the AUX/VERB has an ADJ child (copular structure)
                adjs = [ch for ch in head.children if ch.pos_ == "ADJ" and ch.dep_ in {"acomp", "attr"}]
                if len(adjs) == 1:
                    adj = adjs[0]
                    break
            # Continue up the chain only if this was an xcomp link
            if cur.dep_ != "xcomp":
                break
            cur = head

        if adj is None:
            return key

        # Must be an NP-subject configuration (not expletive it/there).
        try:
            if adj_subject_license(adj) != "np":
                return key
        except Exception:
            return key

        adj_lem = adj.lemma_.lower()

        # Use ToInfHandler to check if this is a tough adjective
        if self.toinf.is_tough_adj(adj_lem):
            try:
                from . import dbg
                dbg(f"[VERB-CANDS] lemma='{v.lemma_}' tough-toinf object gap under ADJ='{adj.text}' -> treat as trans")
            except ImportError:
                pass
            return "trans"

        return key

    def analyze_frame(self, tok: Token) -> dict:
        """
        Complete frame analysis for a verb token.

        Returns a dict with:
          - frame_key_raw: raw frame from extract_frame
          - frame_key: after valency fixes
          - has_prt: particle flag
          - req_prep: required preposition
          - preps_set: all governed prepositions
          - comp_kind: complement kind
          - comp_kind_simple: simple complement kind
          - is_passive: passive clause flag
          - has_aux: has auxiliary chain
          - has_heavy_aux: has heavy auxiliary chain
          - has_expl_there: has expletive there
          - is_to_be_xcomp: is to-be xcomp head
        """
        k_raw, has_prt, req_prep = extract_frame(tok)
        preps_set = tuple(sorted(extract_preps_set(tok)))

        # Apply valency fixes
        key = k_raw
        key = self.fix_tough_xcomp_valency(tok, key)
        tough_forced_trans = (k_raw == "intrans" and key == "trans")

        # Drop comparative adjunct preps for intrans if tough-forced
        preps_set_final = preps_set
        if tough_forced_trans and preps_set:
            preps_set_final = ()
            req_prep = None

        key = fix_wh_gap_valency(tok, key, set(preps_set_final))

        # Drop comparative adjunct preps for intrans
        if key == "intrans" and preps_set_final:
            comp_like = {"like", "as", "than"} & set(preps_set_final)
            if comp_like:
                preps_set_final = tuple(sorted(set(preps_set_final) - comp_like))

        return {
            "frame_key_raw": k_raw,
            "frame_key": key,
            "has_prt": has_prt,
            "req_prep": req_prep,
            "preps_set": preps_set_final,
            "comp_kind": parse_comp_kind(tok, key or "intrans"),
            "comp_kind_simple": comp_kind_from_parse(tok),
            "is_passive": is_passive_clause(tok),
            "has_aux": has_aux_chain(tok),
            "has_heavy_aux": has_heavy_aux_chain(tok),
            "has_expl_there": has_expl_there_child(tok),
            "is_to_be_xcomp": is_to_be_xcomp_head(tok),
        }
