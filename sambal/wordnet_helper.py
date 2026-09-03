"""
WordNet helper for verb frame analysis.

Provides methods for:
- Transitivity analysis (min object count, can take object)
- Complement type profiles (that-CP, Q-CP, to-infinitive)
- Reflexive/animate direct object detection
- Bare intransitive frame detection
- PP head extraction from frames
- Quantity noun extraction
"""
from __future__ import annotations

import re
import sys
from functools import lru_cache
from typing import Dict, Set, Tuple


class WordNetHelper:
    """
    Consolidated WordNet functionality for verb frame analysis.
    """

    # Regex patterns for frame matching
    _V = r"(?:----s|[A-Za-z_]+)"  # verb slot: either '----s' or lemma text

    _RE_DITRANS = None  # Initialized in __init__
    _RE_TRANS = None
    _RE_DO_SOMEBODY = None
    _RE_DO_ONESELF = None
    _RE_BARE_INTR = None
    _PREP_WORDS = "to|for|with|at|on|in|into|onto|about|over|under|from|by|against|between|through|of|off|out|up|down|after|before"

    _Q_WORDS = {"whether", "if", "who", "whom", "whose", "what", "which", "where", "when", "why", "how"}

    def __init__(self):
        self._wn = None
        self._wn_ok = False
        self._wn_frames = None

        # Import WordNet and force the lazy corpus load: nltk's
        # LazyCorpusLoader imports fine with no corpus data and raises only on
        # first query, which the per-method fallbacks below would silently
        # absorb as permissive answers. The probe makes missing data surface
        # here, so is_available is honest.
        try:
            from nltk.corpus import wordnet as wn
            wn.synsets("dog")
            self._wn = wn
            self._wn_frames = getattr(wn, "_frame_strings", None) or getattr(wn, "_frames", None)
            self._wn_ok = True
        except Exception as exc:
            print(
                f"WARNING: NLTK WordNet is unavailable ({type(exc).__name__}); "
                "WordNet verb-frame gates would be silently permissive. "
                "Install the corpus data with: "
                "python -c \"import nltk; nltk.download('wordnet')\"",
                file=sys.stderr,
            )

        # Initialize regex patterns
        V = self._V
        self._RE_DITRANS = [
            re.compile(rf"Somebody\s+{V}\s+somebody\s+something\b", re.I),
            re.compile(rf"Somebody\s+{V}\s+something\s+to\s+somebody\b", re.I),
            re.compile(rf"Somebody\s+{V}\s+something\s+for\s+somebody\b", re.I),
            re.compile(rf"Somebody\s+{V}\s+something\s+from\s+somebody\b", re.I),
        ]
        self._RE_TRANS = re.compile(
            rf"Somebody\s+{V}\s+something\b"
            rf"(?!\s+(?:in|on|at|with|to|from|for|of|by)\b)",
            re.I,
        )
        self._RE_DO_SOMEBODY = re.compile(rf"Somebody\s+{V}\s+somebody\b", re.I)
        self._RE_DO_ONESELF = re.compile(rf"Somebody\s+{V}\s+oneself\b", re.I)
        self._RE_BARE_INTR = re.compile(
            rf"^(?:somebody|something|it)\s+{V}\b(?!\s+(?:something|somebody|{self._PREP_WORDS})\b)",
            re.I,
        )

        # Caches
        self._cp_cache: Dict[str, Tuple[bool, bool, bool]] = {}
        self._prep_cache: Dict[str, Set[str]] = {}
        # Per-lemma result caches for the three previously-uncached pure
        # methods. Profiling attributed ~12.8s/118.2s on 25
        # long documents to `min_object_count` alone (141k calls, no cache).
        self._min_obj_cache: Dict[str, int] = {}
        self._refl_animate_cache: Dict[str, bool] = {}
        self._bare_intrans_cache: Dict[str, bool] = {}

    @property
    def is_available(self) -> bool:
        """Return True if WordNet is available."""
        return self._wn_ok

    def _iter_frame_strings_for_lemma(self, lemma_obj):
        """Yield human-readable frame strings for a Lemma, with robust fallbacks."""
        # Preferred API (newer NLTK)
        if hasattr(lemma_obj, "frame_strings"):
            for fs in lemma_obj.frame_strings():
                yield fs
            return
        # Fallback: map frame_ids() using an internal WordNet table if present
        if hasattr(lemma_obj, "frame_ids") and self._wn_frames:
            try:
                for fid in lemma_obj.frame_ids():
                    if 0 <= fid < len(self._wn_frames) and self._wn_frames[fid]:
                        yield self._wn_frames[fid]
            except Exception:
                pass

    def min_object_count(self, lemma: str) -> int:
        """
        Max # of direct objects suggested by WordNet frames for lemma:
          -1 = unknown (no frame strings available)
           0 = seen frames but no object evidence
           1 = transitive
           2 = ditransitive
        """
        if not self._wn_ok or not lemma:
            return -1

        key = lemma.lower().replace(" ", "_")
        cached = self._min_obj_cache.get(key)
        if cached is not None:
            return cached

        best = 0
        saw_any = False

        try:
            for syn in self._wn.synsets(key, pos="v"):
                for lem in syn.lemmas():
                    if lem.name().lower() != key:
                        continue
                    for fs in self._iter_frame_strings_for_lemma(lem):
                        saw_any = True
                        s = str(fs).lower()
                        if any(p.search(s) for p in self._RE_DITRANS):
                            best = max(best, 2)
                        elif self._RE_TRANS.search(s):
                            best = max(best, 1)
        except Exception:
            self._min_obj_cache[key] = -1
            return -1

        result = best if saw_any else -1
        self._min_obj_cache[key] = result
        return result

    def can_take_object(self, lemma: str) -> bool:
        """
        True iff WN suggests >=1 direct-object slot.
        Unknown (-1) is treated as True so we don't drop valid verbs due to API gaps.
        """
        n = self.min_object_count(lemma)
        return (n == -1) or (n >= 1)

    def cp_profile(self, lemma: str) -> Tuple[bool, bool, bool]:
        """
        Return (allows_that_cp, allows_q_cp, allows_to_inf) by scanning WordNet frame strings.
        If WN has no frames for lemma, default to permissive (True, True, True).
        """
        if not self._wn_ok or not lemma:
            return (True, True, True)

        key = lemma.lower().replace(" ", "_")
        if key in self._cp_cache:
            return self._cp_cache[key]

        saw_any = False
        has_that = False
        has_q = False
        has_to = False

        try:
            for syn in self._wn.synsets(key, pos="v"):
                for lem in syn.lemmas():
                    if lem.name().lower() != key:
                        continue
                    for fs in self._iter_frame_strings_for_lemma(lem):
                        s = str(fs).lower()
                        if not s:
                            continue
                        saw_any = True
                        if " that " in f" {s} ":
                            has_that = True
                        if any(f" {w} " in f" {s} " for w in self._Q_WORDS):
                            has_q = True
                        if " infinitive" in s:
                            has_to = True
        except Exception:
            self._cp_cache[key] = (True, True, True)
            return self._cp_cache[key]

        self._cp_cache[key] = (True, True, True) if not saw_any else (has_that, has_q, has_to)
        return self._cp_cache[key]

    def allows_that_cp(self, lemma: str) -> bool:
        return self.cp_profile(lemma)[0]

    def allows_q_cp(self, lemma: str) -> bool:
        return self.cp_profile(lemma)[1]

    def allows_to_inf(self, lemma: str) -> bool:
        return self.cp_profile(lemma)[2]

    def has_reflexive_or_animate_do(self, lemma: str) -> bool:
        """
        True iff WordNet lists at least one frame for `lemma` with an animate
        or reflexive direct object ("Somebody ... somebody" or "Somebody ... oneself").
        If WordNet has no frames for the lemma, return True (don't over-block on gaps).
        """
        if not self._wn_ok or not lemma:
            return True

        key = lemma.lower().replace(" ", "_")
        cached = self._refl_animate_cache.get(key)
        if cached is not None:
            return cached

        saw_any = False

        try:
            for syn in self._wn.synsets(key, pos="v"):
                for lem in syn.lemmas():
                    if lem.name().lower() != key:
                        continue
                    for fs in self._iter_frame_strings_for_lemma(lem):
                        s = str(fs).strip()
                        if not s:
                            continue
                        saw_any = True
                        if self._RE_DO_SOMEBODY.search(s) or self._RE_DO_ONESELF.search(s):
                            self._refl_animate_cache[key] = True
                            return True
        except Exception:
            self._refl_animate_cache[key] = True
            return True

        result = not saw_any  # permissive if WN has no frames at all
        self._refl_animate_cache[key] = result
        return result

    def supports_np_object(self, lemma: str) -> bool:
        """Alias for can_take_object (strict direct-object gate)."""
        return self.can_take_object(lemma)

    def allows_bare_intrans(self, lemma: str) -> bool:
        """
        True iff WordNet suggests at least one *bare* intransitive frame for lemma,
        e.g., 'Somebody ----s' / 'Something ----s' / 'It ----s' with no NP or PP continuation.
        Unknown returns True (don't over-block).
        """
        if not self._wn_ok or not lemma:
            return True

        key = lemma.lower().replace(" ", "_")
        cached = self._bare_intrans_cache.get(key)
        if cached is not None:
            return cached

        saw_any = False

        try:
            for syn in self._wn.synsets(key, pos="v"):
                for lem in syn.lemmas():
                    if lem.name().lower() != key:
                        continue
                    for fs in self._iter_frame_strings_for_lemma(lem):
                        s = str(fs).strip().lower()
                        if not s:
                            continue
                        saw_any = True
                        if self._RE_BARE_INTR.search(s):
                            self._bare_intrans_cache[key] = True
                            return True
        except Exception:
            self._bare_intrans_cache[key] = True
            return True

        result = not saw_any  # if WN has no frames, don't block
        self._bare_intrans_cache[key] = result
        return result

    def prep_after_verb_set(self, lemma: str) -> Set[str]:
        """
        Return the set of PP heads (e.g., 'to', 'for', 'with', ...) that occur
        immediately after the verb in WordNet valence frames, e.g.:
            'Somebody ----s to somebody'  -> {'to'}
            'Somebody ----s about something' -> {'about'}
        """
        if not self._wn_ok or not lemma:
            return set()

        if lemma in self._prep_cache:
            return self._prep_cache[lemma]

        preps: Set[str] = set()
        PREPS = (
            "to for with at on in into onto about over under from by against between "
            "through of off out up down after before since without within beyond like as than"
        ).split()

        pat = re.compile(
            r"----s(?:\s+\w+)*\s+(" + "|".join(PREPS) + r")\s+(?:somebody|something)\b",
            flags=re.IGNORECASE,
        )

        try:
            for ss in self._wn.synsets(lemma, pos=self._wn.VERB):
                for s in ss.lemmas():
                    for fs in ss.frame_strings():
                        m = pat.search(fs)
                        if m:
                            preps.add(m.group(1).lower())
        except Exception:
            pass

        self._prep_cache[lemma] = preps
        return preps

    def supports_preps_after_verb(self, lemma: str, preps: Set[str]) -> bool:
        """True iff WordNet attests the verb with each prep immediately after the verb."""
        wn_preps = self.prep_after_verb_set(lemma)
        return bool(preps) and preps.issubset(wn_preps)

    def get_quantity_lemmas(self) -> Set[str]:
        """
        Return set of quantity-related noun lemmas from WordNet.
        Used to freeze binominal/degree quantity MWEs like 'a lot of NP'.
        """
        if not self._wn_ok:
            return set()

        WN_QUANT_LEX = {"noun.quantity", "noun.group", "noun.measure"}
        q: Set[str] = set()

        try:
            for syn in self._wn.all_synsets(pos="n"):
                if syn.lexname() in WN_QUANT_LEX:
                    for lem_ in syn.lemmas():
                        q.add(lem_.name().replace("_", " ").lower())
        except Exception:
            pass

        return q


# Global WordNet helper instance
_wordnet_helper = WordNetHelper()

# Module-level flag for WordNet availability (for backward compatibility)
_WN_OK = _wordnet_helper.is_available
