"""
VerbNet adapter for verb frame analysis.

Provides access to NLTK VerbNet XML to obtain lemmas whose frames match
a coarse signature (intrans/trans/ditrans with prep and clausal cues).
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

from .wordnet_helper import _wordnet_helper


class VerbNetAdapter:
    """
    Adapter over NLTK VerbNet XML to obtain lemmas whose frames match a coarse signature.

    We normalize frames to a coarse set:
        key in {"intrans", "trans", "ditrans"}
    plus:
        required_prep in {ADP lemma or None}
        flags in {"-", "that", "to", "that+to"}  (clausal cues seen in SYNTAX)

    Index layout:
        self._index[(key, prep, flags)] -> set(lemma)

    Notes
    -----
    * We count SYNTAX//NP as argument NPs and treat the first NP as subject, so:
        obj_np_count = max(total_np - 1, 0)
        0 -> intrans, 1 -> trans, >=2 -> ditrans
      This is a coarse but robust heuristic across VN classes.
    * required_prep is taken from SYNTAX//PREP/@value (first value if multiple),
      OR expanded from SELRESTRS semantic classes (loc, dest, src, etc.).
    * flags detect LEX value "that" and "to" under SYNTAX (simple and robust).
    """

    # Mapping from VerbNet SELRESTR types to actual prepositions.
    # When a PREP element has no explicit @value but has SELRESTRS like
    # <SELRESTR type="loc" value="+"/>, we expand to these prep sets.
    # Based on VerbNet 3.x documentation and common usage patterns.
    SELRESTR_TO_PREPS: Dict[str, Set[str]] = {
        # Locative (static location)
        'loc': {'in', 'on', 'at', 'under', 'over', 'near', 'beside', 'behind',
                'between', 'among', 'inside', 'outside', 'beneath', 'above',
                'below', 'within', 'around', 'by', 'upon'},
        # Destination (goal of motion)
        'dest': {'to', 'into', 'onto', 'toward', 'towards'},
        'dest_dir': {'into', 'onto', 'to', 'toward', 'towards'},
        # Destination with containment/surface configuration
        'dest_conf': {'into', 'onto', 'in', 'on'},
        # Directional
        'dir': {'into', 'onto', 'to', 'toward', 'towards', 'through', 'across',
                'along', 'around', 'past', 'up', 'down'},
        # Source (origin of motion)
        'src': {'from', 'out', 'off', 'out_of', 'off_of'},
        # Path
        'path': {'through', 'across', 'along', 'around', 'over', 'past', 'via'},
        # General spatial (union of loc + directional)
        'spatial': {'in', 'on', 'at', 'to', 'from', 'into', 'onto', 'through',
                    'across', 'along', 'around', 'over', 'under', 'behind',
                    'beside', 'between', 'among', 'near', 'toward', 'towards',
                    'above', 'below', 'beneath', 'within', 'upon', 'by'},
    }

    def __init__(self):
        # Import here to avoid relying on outer module state
        import nltk
        try:
            from nltk.corpus import verbnet as vn  # type: ignore
            # trigger lazy load; download if missing
            _ = vn.classids()
        except LookupError:
            nltk.download("verbnet")
            from nltk.corpus import verbnet as vn  # type: ignore
            _ = vn.classids()

        self._vn = vn
        self._index: Dict[Tuple[str, str, FrozenSet[str], str], Set[str]] = {}

        self._build_index()

    @lru_cache(maxsize=50000)
    def explain_lemma(self, lemma: str):
        lemma = (lemma or "").lower()
        hits = []
        for (k, ck, prep_key, flag_key), lemmas in self._index.items():
            if lemma not in lemmas:
                continue
            # normalize preps
            if prep_key is None:
                prep_t = ()
            elif isinstance(prep_key, (set, frozenset)):
                prep_t = tuple(sorted(prep_key))
            elif isinstance(prep_key, (list, tuple)):
                prep_t = tuple(sorted(str(x) for x in prep_key))
            elif isinstance(prep_key, str):
                prep_t = (prep_key,) if prep_key else ()
            else:
                try:
                    prep_t = tuple(sorted(str(x) for x in prep_key))
                except Exception:
                    prep_t = (str(prep_key),)
            # normalize flags
            if not flag_key or flag_key == "-":
                flags_t = ()
            elif isinstance(flag_key, (set, frozenset)):
                flags_t = tuple(sorted(flag_key))
            elif isinstance(flag_key, (list, tuple)):
                flags_t = tuple(sorted(str(x) for x in flag_key))
            elif isinstance(flag_key, str):
                flags_t = tuple(sorted([p for p in flag_key.split("|") if p and p != "-"]))
            else:
                flags_t = (str(flag_key),)
            hits.append((k, prep_t, flags_t))
        hits.sort(key=lambda t: (t[0], len(t[1]), t[1], len(t[2]), t[2]))
        return hits

    # ---------------- XML helpers ----------------
    @staticmethod
    def _as_element(obj) -> ET.Element:
        """Normalize NLTK's vnclass return (Element or wrapper) to Element."""
        if isinstance(obj, ET.Element):
            return obj
        # Older NLTK may wrap; expose underlying root
        root = getattr(obj, "_root", None)
        return root if isinstance(root, ET.Element) else obj

    @staticmethod
    def _members_from_xml(vnclass_el: ET.Element) -> List[str]:
        return [
            (m.attrib.get("name") or "").strip().lower()
            for m in vnclass_el.findall(".//MEMBER")
            if m.attrib.get("name")
        ]

    @classmethod
    def _preps_from_frame_xml(cls, frame_el: ET.Element) -> List[str]:
        """
        Extract all PREP alternatives mentioned in SYNTAX.

        Handles two VerbNet encoding styles:
        1. Explicit lexical preps: <PREP value="with"/> or <PREP value="in on at"/>
        2. Semantic prep classes via SELRESTRS: <PREP><SELRESTRS><SELRESTR type="loc" value="+"/></SELRESTRS></PREP>

        For style 2, we expand the semantic class to actual prepositions using SELRESTR_TO_PREPS.

        VN uses either '|' or whitespace to enumerate options; multi-word items
        are usually written with underscores (e.g., 'off_of'). We split on '|' and
        whitespace, but we do NOT split underscores.
        """
        splitter = re.compile(r"[\/\s\|]+")
        preps: Set[str] = set()

        for prep in frame_el.findall(".//SYNTAX//PREP"):
            # 1) Check explicit @value attribute
            raw = (prep.attrib.get("value") or "").strip().lower()
            if raw:
                for piece in splitter.split(raw):
                    if piece and piece != "none":
                        preps.add(piece)  # 'off_of' stays one token
                continue  # explicit value found, skip SELRESTRS check

            # 2) No explicit value -- check SELRESTRS for semantic prep classes
            for selrestr in prep.findall(".//SELRESTRS/SELRESTR"):
                restr_type = (selrestr.attrib.get("type") or "").lower()
                restr_val = selrestr.attrib.get("value", "")
                # Only expand positive restrictions (value="+")
                if restr_val == "+" and restr_type in cls.SELRESTR_TO_PREPS:
                    preps.update(cls.SELRESTR_TO_PREPS[restr_type])

        return list(preps)

    @staticmethod
    def _obj_np_count(frame_el: ET.Element) -> int:
        """
        Count core object NPs in a VN frame.
        Logic:
          - Walk direct children of <SYNTAX> in order.
          - First NP before the VERB is the subject.
          - After VERB, count an NP as a core object only if we have NOT seen a PREP
            since the previous NP/VERB (so VERB PREP (ADV) NP is PP-internal and does not count).
        """
        syn = frame_el.find(".//SYNTAX")
        if syn is None:
            return 0

        children = list(syn)
        subject_seen = False
        after_verb = False
        in_pp_waiting_for_np = False
        count = 0

        for el in children:
            tag = (el.tag or "").upper()

            if tag == "VERB":
                after_verb = True
                in_pp_waiting_for_np = False
                continue

            if tag == "PREP":
                if after_verb:
                    in_pp_waiting_for_np = True
                continue

            if tag == "NP":
                if not subject_seen and not after_verb:
                    # subject NP
                    subject_seen = True
                    in_pp_waiting_for_np = False
                elif after_verb:
                    if in_pp_waiting_for_np:
                        # NP closes a PP -> not a core object
                        in_pp_waiting_for_np = False
                    else:
                        # VERB ... NP with no pending PREP -> core object
                        count += 1
                continue

            # Other elements (ADV/ADVP/ADJ/etc) do not reset in_pp_waiting_for_np

        return count

    @staticmethod
    def _flags_from_frame_xml(frame_el: ET.Element) -> str:
        """
        Detect clausal cues:
          'that' for declarative CP, 'q' for WH/if/whether, 'to' for infinitivals.
        Prefer SYNTAX//LEX; fall back to DESCRIPTION/@primary only if needed.
        """
        flags = set()

        # 1) SYNTAX // LEX (preferred)
        lex_vals = {(lex.attrib.get("value") or "").strip().lower()
                    for lex in frame_el.findall(".//SYNTAX//LEX")}
        if "that" in lex_vals:
            flags.add("that")
        if {"whether", "if", "who", "whom", "whose", "what", "which", "where", "when", "why", "how"} & lex_vals:
            flags.add("q")
        if "to" in lex_vals:
            flags.add("to")

        # 2) DESCRIPTION fallback (only if SYNTAX didn't give us anything)
        if not flags:
            desc = frame_el.find(".//DESCRIPTION")
            primary = (desc.attrib.get("primary", "") if desc is not None else "")
            P = primary.upper().replace("_", "-")

            # WH + TO-INF (both cues)
            if any(tag in P for tag in ("WH-TO-INF", "WHAT-TO-INF", "HOW-TO-INF")):
                flags.update({"q", "to"})
            # plain TO-INF
            elif "TO-INF" in P:
                flags.add("to")
            # finite WH
            elif any(tag in P for tag in ("WH-S", "WHAT-S", "HOW-S")):
                flags.add("q")
            # plain S (treat as 'that')
            elif P == "S":
                flags.add("that")
            # else: leave flags empty

        return "+".join(sorted(flags)) or "-"

    @classmethod
    def _pcomp_preps_from_frame_xml(cls, frame_el: ET.Element) -> List[str]:
        """Best-effort: detect PREP heads that take a gerund/clausal complement.

        VN isn't fully consistent across releases: the complement after PREP may
        appear as VBG, or as a wrapper (VP/S/COMP/SBAR/NP) with an 'ing'/'V-ing'/gerund hint.
        VN may also insert trivial interveners (e.g., ADV/ADVP) between PREP and its complement.

        This implementation is:
          - namespace-safe + case-safe for tag comparisons
          - skips ADV/ADVP between PREP and complement
          - optionally recurses into common wrapper nodes (low-risk recall boost)
        """
        syntax = frame_el.find(".//SYNTAX")
        if syntax is None:
            return []

        out: Set[str] = set()

        def _TAG(node: ET.Element) -> str:
            t = (node.tag or "")
            if "}" in t:  # namespace-safe
                t = t.split("}", 1)[-1]
            return t.upper()

        def _has_ing_hint(node: ET.Element) -> bool:
            v = (node.get("value") or "").strip().lower()
            if v and ("v-ing" in v or "gerund" in v or "ing" in v):
                return True
            for sub in node.iter():
                if _TAG(sub) == "VBG":
                    return True
                sv = (sub.get("value") or "").strip().lower()
                if sv and ("v-ing" in sv or "gerund" in sv or "ing" in sv):
                    return True
            return False

        SKIP_BETWEEN_PREP_AND_COMP = {"ADV", "ADVP"}
        # Only recurse into nodes where a linear child-sequence is plausible/useful.
        WRAPPER_TAGS = {"SYNTAX", "PP", "VP", "V", "S", "NP", "COMP", "SBAR"}

        def _scan_seq(children: List[ET.Element]) -> None:
            for i, el in enumerate(children):
                if _TAG(el) == "PREP":
                    raw = (el.get("value") or "").strip().lower()
                    if raw and raw != "none":
                        # VN sometimes inserts ADV/ADVP-like nodes between PREP and its complement.
                        j = i + 1
                        while j < len(children) and _TAG(children[j]) in SKIP_BETWEEN_PREP_AND_COMP:
                            j += 1

                        if j < len(children):
                            nxt = children[j]
                            nxt_tag = _TAG(nxt)

                            is_gerund = False
                            if nxt_tag == "VBG":
                                is_gerund = True
                            elif nxt_tag in {"VP", "V", "S", "NP", "COMP", "SBAR"} and _has_ing_hint(nxt):
                                is_gerund = True

                            if is_gerund:
                                for p in re.split(r"[\/\s\|]+", raw):
                                    p = p.strip()
                                    if p and p != "none":
                                        out.add(p)

                # Recursively scan inside wrapper nodes (safe recall boost; out is a set anyway).
                t = _TAG(el)
                if t in WRAPPER_TAGS:
                    kids = list(el)
                    if kids:
                        _scan_seq(kids)

        _scan_seq(list(syntax))
        return sorted(out)

    @staticmethod
    def _flags_key(flags: FrozenSet[str]) -> str:
        """Return a deterministic compact key for a set of flags.

        We store flags in the VN index as a '|' separated string so callers can
        round-trip via: set(key.split('|')) if key else set().
        """
        if not flags:
            return "-"
        return "|".join(sorted(flags))

    def _frame_signature(self, frame_el: ET.Element) -> Tuple[str, FrozenSet[str], str]:
        """Return (key, prep_set, flags_key) for one VN <FRAME>.

        key in {intrans, trans, ditrans} is based on core object NP count, but CP frames
        are treated as intransitive (v8 behavior) since VN often encodes CP complements
        without an NP object.
        """
        comp_kind = self._comp_kind_from_frame(frame_el)

        obj_nps = self._obj_np_count(frame_el)
        if obj_nps >= 2:
            key = "ditrans"
        elif obj_nps == 1:
            key = "trans"
        else:
            key = "intrans"
        if comp_kind in {"CP_THAT", "CP_WH", "CP_TO", "CP_FOR_TO", "CP_ING"}:
            key = "intrans"

        preps = frozenset(self._preps_from_frame_xml(frame_el))

        base_flags = (self._flags_from_frame_xml(frame_el) or "-").strip()
        flags_set: Set[str] = set(f for f in base_flags.split("+") if f and f != "-")
        for p in self._pcomp_preps_from_frame_xml(frame_el):
            flags_set.add(f"pcomp:{p}")

        fk = self._flags_key(frozenset(flags_set))
        return key, preps, fk

    def _build_index(self) -> None:
        for cid in self._vn.classids():
            root_el = self._as_element(self._vn.vnclass(cid))
            members = self._members_from_xml(root_el)
            if not members:
                continue
            for frame in root_el.findall(".//FRAMES//FRAME"):
                key, prep_set, flags = self._frame_signature(frame)
                comp_kind = self._comp_kind_from_frame(frame)
                bucket = self._index.setdefault((key, comp_kind, prep_set, flags), set())
                for lem in members:
                    if lem:
                        min_objs = 2 if key == "ditrans" else (1 if key == "trans" else 0)
                        if min_objs >= 1 and _wordnet_helper.min_object_count(lem.lower()) < min_objs:
                            continue  # skip intransitives in trans/ditrans frames
                        bucket.add(lem.lower())

    def filter_lemmas(self, keep_fn) -> None:
        """In-place filter of the VerbNet lemma index.

        This is primarily used by the augmenter when `cfg.prefilter_pools_by_allowed_vocab`
        is enabled: we keep only lemmas present in `allowed_lemmas` (derived from
        allowed_vocab surfaces), which reduces rejection-sampling failures.

        `keep_fn` should accept a lowercase lemma and return bool.
        """
        if keep_fn is None:
            return
        try:
            new_index: Dict[Tuple[str, str, FrozenSet[str], str], Set[str]] = {}
            for k, lemmas in self._index.items():
                kept = {l for l in lemmas if l and keep_fn(l)}
                if kept:
                    new_index[k] = kept
            self._index = new_index
            try:
                self.explain_lemma.cache_clear()  # type: ignore[attr-defined]
            except Exception:
                pass
        except Exception:
            # Be conservative: never let a filter failure break initialization.
            return

    # --- complement-kind extraction -----------------------------------
    @staticmethod
    def _first_postverb_complement(frame_el: ET.Element) -> Optional[str]:
        """
        Return the tag name (uppercased) of the first post-VERB complement-like child.
        Handles common VN encodings:
          - direct children NP / S / PP
          - PREP + NP (no <PP> wrapper)  --> treat as PP
          - wrappers like <COMP> / <SBAR> containing S / NP / PREP
        """
        syn = frame_el.find(".//SYNTAX")
        if syn is None:
            return None

        def _contains(node: ET.Element, tag: str) -> bool:
            return node.find(f".//{tag}") is not None

        seen_verb = False
        pending_prep = False

        for el in list(syn):
            tag = (el.tag or "").upper()
            if tag == "VERB":
                seen_verb = True
                pending_prep = False
                continue
            if not seen_verb:
                continue

            # Plain PP node
            if tag == "PP":
                return "PP"

            # VN often encodes PP as PREP followed by NP (no <PP> wrapper)
            if tag == "PREP":
                # We can already classify the first complement as PP
                return "PP"

            if tag == "NP":
                # NP immediately after PREP is PP-internal; otherwise it's a true NP complement
                return "PP" if pending_prep else "NP"

            if tag == "S":
                return "S"

            # Wrappers: COMP/SBAR with an S/NP/PP inside
            if tag in {"COMP", "SBAR"}:
                if _contains(el, "S"):
                    return "S"
                if _contains(el, "PP") or _contains(el, "PREP"):
                    return "PP"
                if _contains(el, "NP"):
                    return "NP"

        return None

    @classmethod
    def _comp_kind_from_frame(cls, frame_el: ET.Element) -> str:
        """
        Map a VN frame to a coarse complement kind:
          NP_OBJ | CP_THAT | CP_WH | CP_TO | CP_FOR_TO | CP_ING | DOUBLE_OBJ | NONE

        Strategy:
          1) If we truly have two core NP objects in SYNTAX -> DOUBLE_OBJ.
          2) Prefer <DESCRIPTION @primary/@secondary> cues (VN/NLTK store CP info there).
          3) Fall back to SYNTAX first-complement heuristics.
        """
        # 1) two core object NPs -> DOUBLE_OBJ
        if cls._obj_np_count(frame_el) >= 2:
            return "DOUBLE_OBJ"

        # 2) DESCRIPTION(primary/secondary) -- this is where VN encodes CP info
        desc = frame_el.find(".//DESCRIPTION")
        primary = (desc.attrib.get("primary", "") if desc is not None else "")
        secondary = (desc.attrib.get("secondary", "") if desc is not None else "")

        prim = primary.upper().replace("_", "-").strip()
        sec = secondary.upper().replace("_", "-").strip()

        # CP families from primary/secondary
        if "FOR-TO-INF" in prim or ("FOR-PP" in sec and "TO-INF" in prim):
            return "CP_FOR_TO"
        if "TO-INF" in prim or "INFINITIVAL" in prim:
            # e.g., "Infinitival Copular Clause" in some VN releases
            return "CP_TO"
        if any(tag in prim for tag in ("WH-S", "WHAT-S", "HOW-S", "WH-TO-INF", "WHAT-TO-INF")):
            return "CP_WH"
        if prim == "S" or prim.endswith("-S"):
            return "CP_THAT"

        # 3) Fall back to SYNTAX
        first = cls._first_postverb_complement(frame_el) or ""
        if first == "S":
            # If SYNTAX shows S, treat as finite CP_THAT by default
            return "CP_THAT"
        if first == "NP":
            return "NP_OBJ"
        if first == "PP":
            # We don't drive PP complements via comp_kind
            return "NONE"

        return "NONE"

    # ---------------- Query ----------------
    def verbs_for_shape(
        self,
        key: str,
        comp_kind: str,
        need_preps: Optional[Iterable[str]] = None,
        flags: Optional[Iterable[str]] = None,
        allow_preps_relax: bool = True,
    ) -> Set[str]:
        """Return lemma set for (key, comp_kind) given required preps + flags.

        Broadening / fallback behavior:
          * Pass 1: exact preps, and full flags subset match.
          * Pass 2: if need_preps given, allow extra preps (if allow_preps_relax), and relax non-critical flags,
                   but NEVER drop *protected* flags:
                      - 'q' (interrogative complement licensing)
                      - 'pcomp:<prep>' (prep + V-ing complement licensing)
          * Pass 3: keep the same preps constraint as pass 1, but relax non-protected flags (still enforcing protected).

        This prevents CP complement broadening from admitting verbs that do not license interrogatives,
        and prevents mixing prep+NP vs prep+V-ing PP frames.
        """
        need_preps = frozenset((p.lower() for p in need_preps) if need_preps else [])
        need_flags = frozenset(flags or [])

        protected_flags = frozenset(f for f in need_flags if (f == "q" or f.startswith("pcomp:")))
        protected = set(protected_flags)

        # pass 1: exact preps and full flags subset match
        cand: Set[str] = set()
        for (k, ck, prep_set, fk), lemmas in self._index.items():
            if k != key or ck != comp_kind:
                continue
            if prep_set != need_preps:
                continue
            frame_flags = set(fk.split("|")) if fk else set()
            if need_flags and not set(need_flags).issubset(frame_flags):
                continue
            cand.update(lemmas)
        if cand:
            return cand

        # pass 2: allow extra preps (if configured), relax non-protected flags, but keep protected flags
        if need_preps:
            cand = set()
            for (k, ck, prep_set, fk), lemmas in self._index.items():
                if k != key or ck != comp_kind:
                    continue
                if not need_preps.issubset(prep_set):
                    continue
                if not allow_preps_relax and (prep_set - need_preps):
                    continue
                frame_flags = set(fk.split("|")) if fk else set()
                if protected and not protected.issubset(frame_flags):
                    continue
                cand.update(lemmas)
            if cand:
                return cand

        # pass 3: relax non-protected flags, keep the same preps constraint
        cand = set()
        for (k, ck, prep_set, fk), lemmas in self._index.items():
            if k != key or ck != comp_kind:
                continue
            if need_preps:
                if prep_set != need_preps:
                    continue
            else:
                if prep_set:
                    continue
            frame_flags = set(fk.split("|")) if fk else set()
            if protected and not protected.issubset(frame_flags):
                continue
            cand.update(lemmas)
        if cand:
            return cand

        return set()
