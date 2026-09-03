"""
Licensor pattern matcher for NPI (Negative Polarity Item) detection.

Compiles licensor patterns from JSONL and provides span/coverage queries.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Set, Tuple

from spacy.tokens import Doc
from spacy.matcher import Matcher, DependencyMatcher


class LicensorMatcher:
    """
    Compiles licensor patterns from JSONL lines with fields:
      id, matcher ("token"|"dep"), strength, pattern, requires_npi_in_clause (bool).
    Provides spans(doc, conservative_guard) and covers(doc, i, conservative_guard).
    """
    def __init__(self, nlp, patterns_path: Optional[str], npi_set: Optional[Set[str]] = None):
        self.nlp = nlp
        self.npi = set(npi_set or [])
        self.token = None
        self.dep = None
        self.meta: Dict[str, Dict[str, object]] = {}
        if patterns_path:
            self._load(patterns_path)

    def _load(self, path: str) -> None:
        self.token = Matcher(self.nlp.vocab)
        self.dep = DependencyMatcher(self.nlp.vocab)

        def _norm_attrs(d: dict) -> dict:
            """Normalize RIGHT_ATTRS for DependencyMatcher nodes."""
            attrs = dict(d) if isinstance(d, dict) else {}
            # DependencyMatcher does NOT support OP on nodes -> remove
            attrs.pop("OP", None)
            # Normalize MORPH: {"HAS": [...]} -> {"IS_SUPERSET": [...]}
            if "MORPH" in attrs and isinstance(attrs["MORPH"], dict):
                morph = dict(attrs["MORPH"])
                if "HAS" in morph:
                    morph = {"IS_SUPERSET": list(morph["HAS"])}
                attrs["MORPH"] = morph
            return attrs

        def _expand_optional_edges(edges):
            """Expand edges with child OP='?' into 2^k variants (with/without).
            We only support '?' here (not '*' or '+')."""
            variants = [[]]
            for edge in edges:
                # edge = [left_id, dep_label, child_spec]
                if (
                        isinstance(edge, (list, tuple))
                        and len(edge) == 3
                        and isinstance(edge[2], dict)
                        and edge[2].get("OP") == "?"
                ):
                    # with-edge variant
                    child_with = dict(edge[2])
                    child_with.pop("OP", None)
                    new_with = [edge[0], edge[1], child_with]
                    variants = [v + [new_with] for v in variants] + [v[:] for v in variants]  # include + exclude
                else:
                    variants = [v + [edge] for v in variants]
            return variants

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                pid = obj.get("id", "LICENSOR")
                matcher = obj.get("matcher", "token")
                strength = obj.get("strength", "strong")
                req_npi = bool(obj.get("requires_npi_in_clause", False))

                # store meta so we can honor strength / guards later
                self.meta[pid] = {
                    "strength": strength,
                    "requires_npi_in_clause": req_npi,
                }

                if matcher == "token":
                    pat = obj.get("pattern", [])
                    if pat:
                        self.token.add(pid, [pat])
                    continue

                if matcher != "dep":
                    continue

                pat = obj.get("pattern", {})
                anchor = pat.get("anchor", {})
                edges = pat.get("edges", [])

                # Build pattern variants (expand optional edges marked with OP='?')
                edge_variants = _expand_optional_edges(edges)
                patterns = []

                for ev in edge_variants:
                    # Start nodes with an anchor so LEFT_ID="anchor" is always valid
                    nodes = [{"RIGHT_ID": "anchor", "RIGHT_ATTRS": _norm_attrs(anchor)}]
                    declared = {"anchor"}
                    used_right_ids = {"anchor"}

                    for i, edge in enumerate(ev, start=1):
                        try:
                            left_id, dep_label, child = edge
                        except Exception:
                            continue

                        attrs = _norm_attrs(child)
                        # Encourage intended relation via child DEP label
                        if isinstance(dep_label, str) and "DEP" not in attrs:
                            attrs["DEP"] = dep_label

                        # Determine RIGHT_ID: explicit child RIGHT_ID > dep label > n{i}
                        rid = child.get("RIGHT_ID") if isinstance(child, dict) else None
                        if not rid:
                            rid = dep_label if isinstance(dep_label, str) else f"n{i}"
                        if rid in used_right_ids:
                            rid = f"{rid}_{i}"
                        # If LEFT_ID hasn't been declared (e.g., we dropped an optional parent), skip this edge
                        if left_id not in declared:
                            continue

                        nodes.append({
                            "LEFT_ID": left_id,
                            "REL_OP": ">",
                            "RIGHT_ID": rid,
                            "RIGHT_ATTRS": attrs,
                        })
                        declared.add(rid)
                        used_right_ids.add(rid)

                    # Only keep non-trivial patterns (anchor-only is allowed but not useful)
                    if len(nodes) >= 1:
                        patterns.append(nodes)

                if not patterns:
                    continue

                try:
                    self.dep.add(pid, patterns)
                except Exception as e:
                    # Helpful debug to pinpoint remaining issues
                    print("[LicensorMatcher][dep.add] failed for pattern id:", pid)
                    for pat_nodes in patterns:
                        for nd in pat_nodes:
                            print("   ", nd)
                        print("   --")
                    raise

    def _sent_has_npi(self, doc: Doc, i: int) -> bool:
        sent = doc[i].sent
        for tok in sent:
            t = tok.text.lower(); l = tok.lemma_.lower()
            if t in self.npi or l in self.npi:
                return True
        return False

    def spans(self, doc: Doc, conservative_guard: bool = False) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        if self.token:
            for mid, s, e in self.token(doc):
                name = self.nlp.vocab.strings[mid]
                meta = self.meta.get(name, {})
                strength = meta.get("strength", "strong")
                if strength == "conservative" and not conservative_guard:
                    continue
                if meta.get("requires_npi_in_clause", False) and not self._sent_has_npi(doc, s):
                    continue
                spans.append((s, e))
        if self.dep:
            for mid, ids in self.dep(doc):
                name = self.nlp.vocab.strings[mid]
                meta = self.meta.get(name, {})
                strength = meta.get("strength", "strong")
                if strength == "conservative" and not conservative_guard:
                    continue
                anchor_i = ids[0] if ids else 0
                if meta.get("requires_npi_in_clause", False) and not self._sent_has_npi(doc, anchor_i):
                    continue
                spans.append((min(ids), max(ids) + 1))
        if not spans:
            return []
        spans.sort()
        merged: List[Tuple[int, int]] = []
        cs, ce = spans[0]
        for s, e in spans[1:]:
            if s <= ce:
                ce = max(ce, e)
            else:
                merged.append((cs, ce)); cs, ce = s, e
        merged.append((cs, ce))
        return merged

    def cover_strengths(self, doc: Doc, conservative_guard: bool = False) -> Dict[int, str]:
        """
        Return mapping token index -> strongest licensor strength covering it.

        Strength order: strong > guarded > conservative > weak.

        Notes:
          * 'conservative' matches are ignored unless conservative_guard=True.
          * Patterns with requires_npi_in_clause are ignored unless an NPI occurs
            in the sentence containing the anchor/token.
        """
        rank = {"weak": 0, "conservative": 1, "guarded": 2, "strong": 3}
        covered: Dict[int, str] = {}

        def _upgrade(i: int, strength: str) -> None:
            s = strength or "strong"
            cur = covered.get(i)
            if cur is None or rank.get(s, 3) > rank.get(cur, 3):
                covered[i] = s

        if self.token:
            for mid, s, e in self.token(doc):
                name = self.nlp.vocab.strings[mid]
                meta = self.meta.get(name, {})
                strength = meta.get("strength", "strong")
                if strength == "conservative" and not conservative_guard:
                    continue
                if meta.get("requires_npi_in_clause", False) and not self._sent_has_npi(doc, s):
                    continue
                for i in range(s, e):
                    _upgrade(i, strength)

        if self.dep:
            for mid, ids in self.dep(doc):
                name = self.nlp.vocab.strings[mid]
                meta = self.meta.get(name, {})
                strength = meta.get("strength", "strong")
                if strength == "conservative" and not conservative_guard:
                    continue
                anchor_i = ids[0] if ids else 0
                if meta.get("requires_npi_in_clause", False) and not self._sent_has_npi(doc, anchor_i):
                    continue
                if not ids:
                    continue
                s, e = min(ids), max(ids) + 1
                for i in range(s, e):
                    _upgrade(i, strength)

        return covered
