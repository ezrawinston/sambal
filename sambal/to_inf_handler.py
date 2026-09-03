"""
To-infinitive complement handling.

Encapsulates detection and filtering for to-infinitive constructions including:
- Subject/object control verbs
- Raising verbs
- ECM (Exceptional Case Marking) constructions
- Tough adjectives

This module owns all to-inf lexicon data and loading.
"""
from __future__ import annotations
import json
from collections import defaultdict
from typing import Optional, List, Set, Dict, Any, Tuple
from spacy.tokens import Token

from .token_utils import has_expl_there_child


# =============================================================================
# Core Detection Functions (no lexicon needed)
# =============================================================================

def xcomp_has_to(xc: Token) -> bool:
    """True iff xc's subtree contains the infinitival marker 'to'."""
    return any(
        t.lemma_.lower() == "to" and t.pos_ in {"PART", "SCONJ", "AUX"}
        for t in xc.subtree
    )


def xcomp_has_prep_gap(v: Token) -> bool:
    """
    True only for stranded prepositions/particles directly attached to the xcomp verb:
      - "easy to talk to"   (to has no pobj/pcomp)
      - "hard to sit on"    (on has no pobj/pcomp)
    False for adjunct PPs with objects:
      - "easy to read in bed"     (in -> pobj bed)
      - "easy to eat with a fork" (with -> pobj fork)
    """
    for p in v.children:
        if p.dep_ not in {"prep", "prt"}:
            continue
        if p.pos_ != "ADP":
            continue
        has_obj = any(c.dep_ in {"pobj", "obj", "pcomp"} for c in p.children)
        if not has_obj:
            return True
    return False


def is_passive_participle_head(v: Token) -> bool:
    """Detect the narrow "be + VBN + to ..." pattern.

    Keeps it conservative: require VBN head and a BE auxiliary.
    """
    if v.tag_ != "VBN":
        return False
    for ch in v.children:
        if ch.dep_ in {"auxpass", "aux"} and ch.lemma_.lower() == "be":
            return True
    return False


def adj_subject_license(adj: Token) -> str:
    """Return one of {np, it, there} based on the surface subject configuration.

    Conservative behavior:
      * Treat explicit UD `expl` subjects (`there`, `it`) as the controlling signal.
      * Fall back to `nsubj`/`nsubjpass` when no `expl` is available.
    """
    # predicate adjectives are usually 'acomp/attr' under a copular AUX/VERB
    host = adj.head if (adj.head is not None and adj.dep_ in {"acomp", "attr"}) else adj
    if host is None:
        return "np"

    # Expletives first (most important for safe swapping)
    for ch in host.children:
        if ch.dep_ == "expl":
            lem = ch.lemma_.lower()
            if lem == "there":
                return "there"
            if lem == "it":
                return "it"

    # Then ordinary subjects
    for ch in host.children:
        if ch.dep_ in {"nsubj", "nsubjpass"}:
            if ch.pos_ == "PRON" and ch.lemma_.lower() == "it":
                return "it"
            return "np"

    return "np"


# =============================================================================
# Tag Selection Functions
# =============================================================================

def pick_adj_tag(tags: Set[str]) -> Optional[str]:
    """Pick the most appropriate to-inf adjective tag."""
    for t in ("toinf:tough", "toinf:raising", "toinf:control"):
        if t in tags:
            return t
    return None


def pick_verb_tag(tags: Set[str], req_license: str) -> Optional[str]:
    """Pick the most appropriate to-inf verb tag based on required license."""
    # ECM licenses (embedded subject inside the to-inf complement).
    if req_license.startswith("ecm_"):
        return "toinf:ecm" if "toinf:ecm" in tags else None

    # Object-control style (matrix has an overt object).
    if req_license == "obj":
        if "toinf:purpose" in tags and "toinf:ocontrol" in tags:
            return "toinf:purpose"
        if "toinf:ocontrol" in tags:
            return "toinf:ocontrol"
        if "toinf:scontrol" in tags:
            return "toinf:scontrol"
        if "toinf:ecm" in tags:
            return "toinf:ecm"
        return None

    # Passive participle raising (be + VBN + to ...).
    if req_license == "subjpass":
        return "toinf:raising_passive" if "toinf:raising_passive" in tags else None

    # Normal subject raising/control.
    if req_license == "subj":
        if "toinf:scontrol" in tags:
            return "toinf:scontrol"
        if "toinf:raising" in tags:
            return "toinf:raising"
        return None

    # Matrix expletive 'there' raising.
    if req_license == "there":
        if "toinf:raising_passive" in tags:
            return "toinf:raising_passive"
        return "toinf:raising" if "toinf:raising" in tags else None

    return None


# =============================================================================
# Lexicon Loading
# =============================================================================

def _load_toinf_lexicon_jsonl(
    path: Optional[str], expected_pos: str
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]]]:
    """Load ERG-derived to-inf lexicon JSONL.

    The generator can emit multiple rows per lemma (multi-frame / multi-tag).
    For augmenter compatibility we merge duplicate lemmas by unioning:
      - tags
      - licenses
      - erg_types
    """
    if not path:
        return {}, {}

    lex: Dict[str, Dict[str, Any]] = {}

    def _norm_list(x: Any) -> List[str]:
        if not x:
            return []
        if isinstance(x, str):
            return [x]
        if isinstance(x, (list, tuple, set)):
            return [y for y in x if isinstance(y, str)]
        return []

    def _merge(prev: Dict[str, Any], cur: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(prev)
        out["tags"] = sorted(set(_norm_list(prev.get("tags")) + _norm_list(cur.get("tags"))))
        out["licenses"] = sorted(
            set(_norm_list(prev.get("licenses")) + _norm_list(cur.get("licenses")))
        )
        out["erg_types"] = sorted(
            set(_norm_list(prev.get("erg_types")) + _norm_list(cur.get("erg_types")))
        )
        if not out.get("primary_tag") and cur.get("primary_tag"):
            out["primary_tag"] = cur.get("primary_tag")
        return out

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)

                pos = obj.get("pos")
                if pos and str(pos).upper() != expected_pos.upper():
                    continue

                lemma = (obj.get("lemma") or "").strip().lower()
                if not lemma:
                    continue

                frames = obj.get("frames") or []
                frame_tags = [fr.get("tag") for fr in frames if fr.get("tag")]
                frame_licenses: List[str] = []
                for fr in frames:
                    frame_licenses.extend(fr.get("licenses") or [])

                entry: Dict[str, Any] = {
                    "lemma": lemma,
                    "tags": _norm_list(obj.get("tags")) + _norm_list(frame_tags),
                    "licenses": _norm_list(obj.get("licenses")) + _norm_list(frame_licenses),
                    "primary_tag": obj.get("primary_tag"),
                    "erg_types": _norm_list(obj.get("erg_types") or []),
                }

                if lemma in lex:
                    lex[lemma] = _merge(lex[lemma], entry)
                else:
                    entry["tags"] = sorted(set(entry["tags"]))
                    entry["licenses"] = sorted(set(entry["licenses"]))
                    entry["erg_types"] = sorted(set(entry["erg_types"]))
                    lex[lemma] = entry
    except Exception as e:
        # A declared lexicon must not degrade silently to an empty pool.
        raise RuntimeError(f"failed to load to-inf lexicon from {path}") from e

    # Build tag -> lemma index after merging for determinism.
    by_tag: Dict[str, List[str]] = defaultdict(list)
    for lem, ent in lex.items():
        for t in ent.get("tags") or []:
            by_tag[t].append(lem)

    for t in list(by_tag.keys()):
        by_tag[t] = sorted(set(by_tag[t]))

    return lex, dict(by_tag)


# =============================================================================
# ToInfHandler Class
# =============================================================================

class ToInfHandler:
    """
    Handles to-infinitive complement detection and candidate filtering.

    This class owns the to-inf lexicon data and provides all methods for
    analyzing to-inf constructions.
    """

    def __init__(
        self,
        adj_lexicon_path: Optional[str] = None,
        verb_lexicon_path: Optional[str] = None,
        skip_toinf_freeze: bool = False,
    ):
        self.skip_toinf_freeze = skip_toinf_freeze

        # Load lexicons
        self.adj_lex, self.adj_by_tag = _load_toinf_lexicon_jsonl(adj_lexicon_path, "ADJ")
        self.verb_lex, self.verb_by_tag = _load_toinf_lexicon_jsonl(verb_lexicon_path, "VERB")

        # Build adj tags index for quick lookup
        self.adj_tags: Dict[str, Set[str]] = {
            lem: set(ent.get("tags") or []) for lem, ent in self.adj_lex.items()
        }

    def has_lexicons(self) -> bool:
        """True if any lexicons are loaded."""
        return bool(self.adj_lex or self.verb_lex)

    def prefilter_by_allowed_vocab(self, allowed_vocab_hint_ok_fn) -> None:
        """Filter lexicons to only include lemmas that pass the allowed vocab check."""
        if self.adj_lex:
            self.adj_lex = {k: v for k, v in self.adj_lex.items() if allowed_vocab_hint_ok_fn(k)}
            # Rebuild by_tag index
            self.adj_by_tag = defaultdict(list)
            for lem, ent in self.adj_lex.items():
                for t in ent.get("tags") or []:
                    self.adj_by_tag[t].append(lem)
            for t in list(self.adj_by_tag.keys()):
                self.adj_by_tag[t] = sorted(set(self.adj_by_tag[t]))
            # Rebuild adj_tags
            self.adj_tags = {lem: set(ent.get("tags") or []) for lem, ent in self.adj_lex.items()}

        if self.verb_lex:
            self.verb_lex = {k: v for k, v in self.verb_lex.items() if allowed_vocab_hint_ok_fn(k)}
            # Rebuild by_tag index
            self.verb_by_tag = defaultdict(list)
            for lem, ent in self.verb_lex.items():
                for t in ent.get("tags") or []:
                    self.verb_by_tag[t].append(lem)
            for t in list(self.verb_by_tag.keys()):
                self.verb_by_tag[t] = sorted(set(self.verb_by_tag[t]))

    # =========================================================================
    # Adjective Analysis
    # =========================================================================

    def adj_xcomps(self, adj: Token) -> List[Token]:
        """Return xcomp complements (to-INF) associated with a predicate adjective."""
        xcs: List[Token] = []
        # xcomp directly under the adjective
        for xc in (ch for ch in adj.children if ch.dep_ == "xcomp"):
            if xcomp_has_to(xc):
                xcs.append(xc)

        # xcomp under the copular host (common: Kim is ADJ to VP)
        h = adj.head
        if h is not None and h.pos_ in {"AUX", "VERB"} and adj.dep_ in {"acomp", "attr"}:
            for xc in (ch for ch in h.children if ch.dep_ == "xcomp"):
                if xcomp_has_to(xc):
                    xcs.append(xc)
        return xcs

    def adj_required_licenses(self, adj: Token) -> Optional[Set[str]]:
        """If `adj` is in a to-INF frame, return the set of required license keys."""
        xcs = self.adj_xcomps(adj)
        if not xcs:
            return None
        req: Set[str] = {adj_subject_license(adj)}
        if any(xcomp_has_prep_gap(xc) for xc in xcs):
            req.add("gap_pp")
        return req

    def get_adj_tags(self, lemma: str) -> Set[str]:
        """Get the to-inf tags for an adjective lemma."""
        return self.adj_tags.get(lemma.lower(), set())

    def is_tough_adj(self, lemma: str) -> bool:
        """Check if an adjective lemma is a tough adjective."""
        tags = self.get_adj_tags(lemma)
        if "toinf:tough" in tags:
            return True
        # Minimal fallback for canonical tough adjectives
        if not tags and not self.skip_toinf_freeze:
            return lemma.lower() in {"easy", "hard", "tough", "difficult"}
        return False

    # =========================================================================
    # Verb Analysis
    # =========================================================================

    def verb_xcomps(self, v: Token) -> List[Token]:
        """Return to-INF xcomp/ccomp children of a verb.

        NOTE: spaCy often uses ccomp (not xcomp) for ECM/to-inf complements
        (e.g., "expect it to rain"). We treat both as potential to-inf complements.
        """
        return [
            ch
            for ch in v.children
            if ch.dep_ in {"xcomp", "ccomp"} and xcomp_has_to(ch)
        ]

    def required_verb_license(self, v: Token) -> Optional[str]:
        """Return the required license key for a to-inf selecting VERB head.

        Licenses:
          - 'there'      : matrix existential/expletive there as subject
          - 'subjpass'   : passive participle raising ("was supposed to ...")
          - 'ecm_it'     : embedded subject 'it' inside the to-inf complement
          - 'ecm_there'  : embedded subject 'there' inside the to-inf complement
          - 'obj'        : overt object under the matrix head (object-control style)
          - 'subj'       : default raising/control with normal subject
        """
        xcs = self.verb_xcomps(v)
        if not xcs:
            return None

        xc_lemmas = {(xc.lemma_ or "").lower() for xc in xcs}
        is_weather_xcomp = bool(xc_lemmas & {"rain", "snow", "hail"})

        # A) Matrix existential / expletive 'there' as subject of the head.
        if has_expl_there_child(v):
            return "there"

        # B) Embedded 'there' in the to-inf complement
        embedded_expl_there = any(
            t.lemma_.lower() == "there" and t.dep_ in {"expl", "nsubj"}
            for xc in xcs
            for t in xc.subtree
        )
        if embedded_expl_there:
            return "ecm_there"

        # C) Embedded 'it' in the to-inf complement
        embedded_it = any(
            t.lemma_.lower() == "it" and t.pos_ == "PRON" and t.dep_ in {"expl", "nsubj"}
            for xc in xcs
            for t in xc.subtree
        )
        if embedded_it:
            return "ecm_it_rain" if is_weather_xcomp else "ecm_it"

        # D) Object 'it' directly under the head
        has_obj_it = any(
            ch.dep_ in {"dobj", "obj"} and ch.lemma_.lower() == "it" for ch in v.children
        )
        if has_obj_it:
            return "ecm_it_rain" if is_weather_xcomp else "ecm_it"

        # E) Overt object under the head (object-control style).
        has_obj = any(
            ch.dep_ in {"dobj", "obj"} and ch.pos_ in {"NOUN", "PROPN", "PRON"} for ch in v.children
        )
        if has_obj:
            return "obj"

        # F) Passive participle raising head: "be + VBN + to ..."
        if is_passive_participle_head(v):
            return "subjpass"

        return "subj"

    # =========================================================================
    # Pool Filtering
    # =========================================================================

    def filter_pool(self, pool: List[str], required: Set[str], lex: Dict[str, Any]) -> List[str]:
        """Filter a pool to only lemmas that have all required licenses."""
        out: List[str] = []
        for lem in pool:
            ent = lex.get(lem)
            if not ent:
                continue
            lic = set(ent.get("licenses") or [])
            if required.issubset(lic):
                out.append(lem)
        return sorted(set(out))

    def adj_pool_candidates(self, lem: str, required: Set[str]) -> Optional[List[str]]:
        """Get candidate pool for a to-inf adjective."""
        lem = lem.lower()
        ent = self.adj_lex.get(lem)
        if not ent:
            return None
        lic = set(ent.get("licenses") or [])
        if required and not required.issubset(lic):
            return None
        tags = set(ent.get("tags") or [])
        tag = pick_adj_tag(tags)
        if not tag:
            return None
        pool = [x for x in (self.adj_by_tag.get(tag) or []) if x != lem]
        return self.filter_pool(pool, required, self.adj_lex)

    def verb_pool_candidates(self, tok: Token, req_license: str) -> Optional[List[str]]:
        """Get candidate pool for a to-inf verb."""
        lem = tok.lemma_.lower()
        ent = self.verb_lex.get(lem)
        if not ent:
            return None
        lic = set(ent.get("licenses") or [])
        if req_license not in lic:
            return None
        tags = set(ent.get("tags") or [])
        tag = pick_verb_tag(tags, req_license)
        if not tag:
            return None
        pool = [x for x in (self.verb_by_tag.get(tag) or []) if x != lem]
        return self.filter_pool(pool, {req_license}, self.verb_lex)
