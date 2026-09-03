#!/usr/bin/env python3
"""ERG+ACE based to-infinitive lexicon generator.

Produces the two JSONL lexicons the engine's to-infinitive handler reads,
`toinf_adjs.jsonl` and `toinf_verbs.jsonl`, one entry per lemma:

  {
    "lemma": "likely",
    "pos": "ADJ" | "VERB",
    "tags": ["toinf:raising", ...],
    "primary_tag": "toinf:raising",
    "licenses": ["np", "there", "it"],
    "frames": [{"tag": ..., "licenses": [...]}, ...],
    "erg_types": ["aj_vp_i-ssr_le", ...]
  }

One lexicon, runtime filtering: `tags` say which class the lemma belongs to
(tough / raising / control / ECM), `licenses` which surface frames the ERG
accepts for it; the augmenter filters on both at swap time.

How lemmas are classified
-------------------------
Each candidate lemma is parsed by ACE in a set of probe sentences and its ERG
lexical types are read off the derivations, with predicate matching so that
compositional analyses (un- + bound) and ERG over-generations are rejected.
On top of the parser's verdict the script applies curated tables, all defined
near the top of this file: exclusion lists, a there-license blocklist, an
allowlist of true object-control verbs (other ocontrol verbs are additionally
tagged `toinf:purpose` for conservative swap pooling), behavioral ECM
overrides for perception verbs, direct passive probes ("Kim was Xed to leave",
participles via lemminflect) that yield `toinf:raising_passive` + `subjpass`,
force-included lemmas, manual backfills for lemmas the probes cannot reach,
and a few per-lemma patches. The committed lexicons are the output of exactly
these tables; edit the tables, not the files.

Dependencies
------------
  pip install pydelphin lemminflect     (both in the project's pyproject)
  the ACE binary and an ERG grammar image (*.dat) -- see
  sambal/resources/generation/README.md for the versions the committed
  files were built with.

Usage
-----
  # smoke test of the toolchain
  python sambal/resources/generation/gen_toinf_lexicons.py --erg <ERG .dat> --smoke

  # both lexicons, written over the committed files
  python sambal/resources/generation/gen_toinf_lexicons.py --erg <ERG .dat> \
    --adj_in sambal/resources/wordnet_adjectives.jsonl --verb_in verbnet

  # verbs only, reusing an existing adjective lexicon for the shadow logic
  python sambal/resources/generation/gen_toinf_lexicons.py --erg <ERG .dat> \
    --adj_lexicon sambal/resources/toinf_adjs.jsonl --verb_in verbnet
"""


from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Set, DefaultDict

from collections import defaultdict


try:
    from delphin import ace, derivation
except Exception as e:  # pragma: no cover
    ace = None
    derivation = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None

# For proper past participle forms in passive probes
try:
    from lemminflect import getInflection as _getInflection
    def get_past_participle(verb: str) -> str:
        """Get the past participle (VBN) form of a verb using lemminflect."""
        forms = _getInflection(verb, 'VBN')
        if forms:
            return forms[0]
        # Fallback: simple -ed suffix
        if verb.endswith('e'):
            return verb + 'd'
        return verb + 'ed'
except ImportError:
    def get_past_participle(verb: str) -> str:
        """Fallback: simple -ed suffix (inaccurate for irregular verbs)."""
        if verb.endswith('e'):
            return verb + 'd'
        return verb + 'ed'


# ----------------------------
# Predicate/lemma matching
# ----------------------------
#
# Why this exists:
# ERG often analyzes orthographic prefixation (re-, un-, mis-, etc.) compositionally.
# That can yield analyses where the surface form is e.g. "repress" but the lexical
# predicate is _press_v_1 with an added _re-_a_again relation. For our swap lexicons,
# those are dangerous: they allow swaps that humans judge ungrammatical for the intended
# lemma, even though ERG finds a reading.
#
# We therefore require (by default) that the *lexical predicate-like id* for the token
# matches the lemma (with a small amount of orthographic normalization).

_LEMMA_PRED_RE_CACHE: dict[tuple[str, str], re.Pattern] = {}


def _lemma_variants(lemma: str) -> list[str]:
    lemma = (lemma or "").lower()
    vars = {lemma}
    # Very small orthographic variant set: British -> American.
    if lemma.endswith("our"):
        vars.add(lemma[:-3] + "or")
    if lemma.endswith("ise"):
        vars.add(lemma[:-3] + "ize")
    if lemma.endswith("yse"):
        vars.add(lemma[:-3] + "yze")
    return sorted(vars)


def lemma_pred_re(lemma: str, pos: str) -> re.Pattern:
    # Match predicate-ish ids like _feel_v_1 or _easy_a_1.
    # Some derivations omit the leading underscore, so allow it.
    pos = (pos or "").lower()
    key = (lemma.lower(), pos)
    pat = _LEMMA_PRED_RE_CACHE.get(key)
    if pat is None:
        alts = _lemma_variants(lemma)
        # ^_?(alt1|alt2)_(v|a)(_|\d)
        alt_pat = "|".join(re.escape(a) for a in alts if a)
        pat = re.compile(rf"^_?(?:{alt_pat})_{re.escape(pos)}(?:_|\d)", re.IGNORECASE)
        _LEMMA_PRED_RE_CACHE[key] = pat
    return pat


def pred_like_matches_lemma(pred_like: Optional[str], lemma: str, pos: str) -> bool:
    if not pred_like or not lemma:
        return False
    return bool(lemma_pred_re(lemma, pos).match(str(pred_like)))


def _looks_prefixed(lemma: str) -> bool:
    """Heuristic: does this lemma look like it has a productive prefix?
    
    Used to reject compositional analyses where ERG analyzes e.g. "unbound" 
    as "un-" + "bound" and incorrectly inherits bound's raising properties.
    """
    prefixes = ('un', 'over', 'under', 'out', 'mis', 're', 'dis', 'pre', 'non')
    l = lemma.lower()
    for p in prefixes:
        if l.startswith(p) and len(l) > len(p) + 2:
            remainder = l[len(p):]
            if remainder in _KNOWN_ADJ_BASES:
                return True
    return False




# ----------------------------
# Swap-safety heuristics / filters
# ----------------------------

# Some high-frequency "adjectives" are mostly particles/idioms in to-infinitival
# contexts (e.g., "out to get you"). Excluding them here keeps the swap lexicon
# cleaner.
ADJ_EXCLUDE_LEMMAS: Set[str] = {
    "about",
    "out",
    "left",
    # These tend to be high-frequency / compositional polarity variants that aren't
    # helpful augmentation targets and can create odd outputs.
    "overcritical",
    "uncritical",
    "unwell",
    "overcurious",
    # ERG compositionally analyzes un-/over- but these don't actually take to-inf
    "unbound",      # "unbound" means "not restricted", NOT "un-" + "bound to"
    "undue",        # "undue" means "excessive", no to-inf reading
    "overanxious",  # marginal to-inf usage
    "overcareful",  # doesn't naturally take to-inf
    "overjoyed",    # "overjoyed to hear" is resultative, not to-inf control
    "overproud",    # odd in to-inf contexts
}

# Adjectives where ERG grants there-licensing but native speakers reject it
ADJ_THERE_LICENSE_BLOCKLIST: Set[str] = {
    "unsure",       # *"There is unsure to be a problem"
}

# Known adjective bases that commonly get prefixed - used for compositional detection
_KNOWN_ADJ_BASES: Set[str] = {
    "bound", "due", "sure", "able", "happy", "willing", "likely",
    "certain", "anxious", "eager", "careful", "critical", "curious",
    "well", "joyed", "proud",
}


VERB_EXCLUDE_LEMMAS: Set[str] = {
    # Avoid high-risk sense-bleed / idiomatic constructions at lemma level.
    # - "use": can form the "used to" habitual construction when inflected,
    #   which is not a true to-inf control frame.
    # - "resent": rare/odd to-inf senses can contaminate a lemma-level lexicon.
    "use",
    "resent",
    # Exclude extremely common light verbs that explode paraphrase space and can
    # create a lot of unwanted idiomatic/multiword behavior.
    "take",
    "bring",
    # "serve" in "serve to illustrate" is a purpose/result idiom, not true control
    "serve",
    # "compose" in "compose oneself to speak" is reflexive-only and marginal
    "compose",
}

# Verbs where ERG grants there-licensing but native speakers reject it
VERB_THERE_LICENSE_BLOCKLIST: Set[str] = {
    "proceed",      # *"There proceeded to be a problem"
}

# Verbs ERG classifies as ocontrol (oeq) but are semantically ECM
# These accept expletive/weather embedded subjects: "I sensed it to rain"
ECM_BEHAVIORAL_OVERRIDE: Set[str] = {
    "sense",
    "discern",
    "discover",
    "confirm",
    "reckon",
    # Note: NOT "compose" - "compose oneself to speak" is genuinely control-like
}

# New (option B): a passive/participial raising tag for predicates that appear as
# "be VBN to ..." (e.g., "was supposed to", "was bound to"). This is kept
# separate from plain toinf:raising so the augmenter can later choose a
# passive-compatible pool (and avoid ungrammatical outputs like "was seemed to").
TOINF_PASSIVE_RAISING_TAG = "toinf:raising_passive"
# Backwards compatibility: older revisions used this constant name.
TOINF_TAG_RAISING_PASSIVE = TOINF_PASSIVE_RAISING_TAG

# License label used for passive-subject "be VBN to ..." frames.
PASSIVE_SUBJ_LICENSE = "subjpass"

# Plain raising tag constant used in a few helper paths / smoke checks.
TOINF_RAISING_TAG = "toinf:raising"

# Tag priority used when selecting a single primary_tag.
TAG_PRIORITY: List[str] = [
    TOINF_PASSIVE_RAISING_TAG,
    "toinf:raising",
    "toinf:control",
    "toinf:ecm",
    "toinf:ocontrol",
    "toinf:scontrol",
    "toinf:purpose",
    "toinf:tough",
]

def backfill_adj_from_lexicon(_oracle: Any, lemma: str) -> Optional[Dict[str, Any]]:
    """Helper backfill for adjectives.

    The main generation path already uses MANUAL_BACKFILL_ADJS as a last resort.
    The smoke test calls this helper to mirror that behavior.
    """
    bf = MANUAL_BACKFILL_ADJS.get(lemma)
    return dict(bf) if bf else None


def backfill_verb_from_lexicon(_oracle: Any, lemma: str) -> Optional[Dict[str, Any]]:
    """Helper backfill for verbs (mirrors MANUAL_BACKFILL_VERBS)."""
    bf = MANUAL_BACKFILL_VERBS.get(lemma)
    return dict(bf) if bf else None

def pick_primary_tag(tags: Sequence[str]) -> Optional[str]:
    """Select a stable primary tag from a (possibly multi-tag) entry."""
    tset = set(tags)
    for t in TAG_PRIORITY:
        if t in tset:
            return t
    return sorted(tset)[0] if tset else None

# Some high-impact lemmas are worth always attempting even if absent from the input list.
FORCE_INCLUDE_ADJS: Set[str] = {"fun"}
FORCE_INCLUDE_VERBS: Set[str] = {"seem", "decide", "remind", "suspect"}

# Manual backfills only used if classify_* fails (ERG sometimes analyzes "fun" as N).
MANUAL_BACKFILL_ADJS: Dict[str, Dict[str, object]] = {
    "fun": {
        "lemma": "fun",
        "pos": "ADJ",
        "probes": {"manual": True},
        "erg_types": [],
        "tags": ["toinf:tough"],
        "primary_tag": "toinf:tough",
        "licenses": ["np", "it", "gap_pp"],
    }
}

MANUAL_BACKFILL_VERBS: Dict[str, Dict[str, object]] = {
    "seem": {
        "lemma": "seem",
        "pos": "VERB",
        "probes": {"manual": True},
        "erg_types": [],
        "tags": ["toinf:raising"],
        "primary_tag": "toinf:raising",
        "licenses": ["subj", "there"],
    },
    "decide": {
        "lemma": "decide",
        "pos": "VERB",
        "probes": {"manual": True},
        "erg_types": [],
        "tags": ["toinf:scontrol"],
        "primary_tag": "toinf:scontrol",
        "licenses": ["subj"],
    },
    "remind": {
        "lemma": "remind",
        "pos": "VERB",
        "probes": {"manual": True},
        "erg_types": [],
        "tags": ["toinf:ocontrol"],
        "primary_tag": "toinf:ocontrol",
        "licenses": ["obj"],
    },
    "suspect": {
        "lemma": "suspect",
        "pos": "VERB",
        "probes": {"manual": True},
        "erg_types": [],
        "tags": ["toinf:ecm", "toinf:raising_passive"],
        "primary_tag": "toinf:ecm",
        "licenses": ["obj", "ecm_it", "ecm_there", "subj", "subjpass", "there"],
    },
    # ECM verbs with raising_passive - these work in "was Xed to VP" constructions
    "believe": {
        "lemma": "believe",
        "pos": "VERB",
        "probes": {"manual": True},
        "erg_types": [],
        "tags": ["toinf:ecm", "toinf:raising_passive"],
        "primary_tag": "toinf:ecm",
        "licenses": ["obj", "ecm_it", "ecm_there", "subj", "subjpass", "there"],
    },
    "expect": {
        "lemma": "expect",
        "pos": "VERB",
        "probes": {"manual": True},
        "erg_types": [],
        "tags": ["toinf:ecm", "toinf:raising_passive"],
        "primary_tag": "toinf:ecm",
        "licenses": ["obj", "ecm_it", "ecm_there", "subj", "subjpass", "there"],
    },
    "consider": {
        "lemma": "consider",
        "pos": "VERB",
        "probes": {"manual": True},
        "erg_types": [],
        "tags": ["toinf:ecm", "toinf:raising_passive"],
        "primary_tag": "toinf:ecm",
        "licenses": ["obj", "ecm_it", "ecm_there", "subj", "subjpass", "there"],
    },
    "report": {
        "lemma": "report",
        "pos": "VERB",
        "probes": {"manual": True},
        "erg_types": [],
        "tags": ["toinf:ecm", "toinf:raising_passive"],
        "primary_tag": "toinf:ecm",
        "licenses": ["obj", "ecm_it", "ecm_there", "subj", "subjpass", "there"],
    },
    "allege": {
        "lemma": "allege",
        "pos": "VERB",
        "probes": {"manual": True},
        "erg_types": [],
        "tags": ["toinf:ecm", "toinf:raising_passive"],
        "primary_tag": "toinf:ecm",
        "licenses": ["obj", "ecm_it", "ecm_there", "subj", "subjpass", "there"],
    },
    "know": {
        "lemma": "know",
        "pos": "VERB",
        "probes": {"manual": True},
        "erg_types": [],
        "tags": ["toinf:ecm", "toinf:raising_passive"],
        "primary_tag": "toinf:ecm",
        "licenses": ["obj", "ecm_it", "ecm_there", "subj", "subjpass", "there"],
    },
    "understand": {
        "lemma": "understand",
        "pos": "VERB",
        "probes": {"manual": True},
        "erg_types": [],
        "tags": ["toinf:ecm", "toinf:raising_passive"],
        "primary_tag": "toinf:ecm",
        "licenses": ["obj", "ecm_it", "ecm_there", "subj", "subjpass", "there"],
    },
    "assume": {
        "lemma": "assume",
        "pos": "VERB",
        "probes": {"manual": True},
        "erg_types": [],
        "tags": ["toinf:ecm", "toinf:raising_passive"],
        "primary_tag": "toinf:ecm",
        "licenses": ["obj", "ecm_it", "ecm_there", "subj", "subjpass", "there"],
    },
    "think": {
        "lemma": "think",
        "pos": "VERB",
        "probes": {"manual": True},
        "erg_types": [],
        "tags": ["toinf:ecm", "toinf:raising_passive"],
        "primary_tag": "toinf:ecm",
        "licenses": ["obj", "ecm_it", "ecm_there", "subj", "subjpass", "there"],
    },
}


# Verb ERG types that are strong false-positives for to-infinitival control:
# they overwhelmingly correspond to PP+purpose/result ("... to V") rather than
# the NP-VP control frames we want to swap.
BANNED_VERB_ERG_TYPES: Set[str] = {
    "v_pp-vp_oeq-bse_le",
    "v_pp*-vp_oeq_le",
}
BANNED_VERB_ERG_TYPE_RE = re.compile(r"^v_pp\*?-vp_oeq", re.I)

# Conservative allowlist of "true" object-control (directive/causative) verbs.
# Everything else tagged as ocontrol is additionally marked as purpose/result so
# the augmenter can avoid swapping across that boundary.
OCONTROL_SWAP_ALLOWLIST: Set[str] = {
    # Original set - well-established directive/causative verbs
    "advise",
    "allow",
    "ask",
    "beg",
    "bribe",
    "cajole",
    "coax",
    "coerce",
    "compel",
    "convince",
    "encourage",
    "entice",
    "forbid",
    "force",
    "get",
    "help",
    "implore",
    "induce",
    "instruct",
    "invite",
    "motivate",
    "oblige",
    "order",
    "permit",
    "persuade",
    "press",
    "pressure",
    "prompt",
    "remind",
    "request",
    "require",
    "teach",
    "tell",
    "tempt",
    "urge",
    "warn",
    # Additional directive/causative verbs - object controls infinitive
    # These are NOT ECM even if ERG overgenerates expletive parses
    "caution",      # "caution someone to be careful"
    "challenge",    # "challenge someone to do X"
    "command",      # "command someone to do X"
    "commission",   # "commission someone to do X"
    "dare",         # "dare someone to do X" (also has scontrol)
    "defy",         # "defy someone to do X"
    "direct",       # "direct someone to do X"
    "dispatch",     # "dispatch someone to do X"
    "drive",        # "drive someone to do X" (causative sense)
    "elect",        # "elect someone to do X"
    "empower",      # "empower someone to do X"
    "engage",       # "engage someone to do X"
    "goad",         # "goad someone to do X"
    "harass",       # "harass someone to do X"
    "hire",         # "hire someone to do X"
    "impel",        # "impel someone to do X"
    "incite",       # "incite someone to do X"
    "inform",       # "inform someone to do X"
    "inspire",      # "inspire someone to do X"
    "lead",         # "lead someone to do X" (directive sense)
    "move",         # "move someone to do X" (causative sense)
    "nominate",     # "nominate someone to do X"
    "prod",         # "prod someone to do X"
    "provoke",      # "provoke someone to do X"
    "scare",        # "scare someone to do X"
    "seduce",       # "seduce someone to do X"
    "send",         # "send someone to do X"
    "signal",       # "signal someone to do X"
    "spur",         # "spur someone to do X"
    "summon",       # "summon someone to do X"
    "train",        # "train someone to do X"
}


def _filter_verb_types(types: Sequence[str]) -> List[str]:
    """Drop ERG types that are almost always PP+purpose/result, not control."""
    kept: List[str] = []
    for t in types:
        lt = t.lower()
        if lt in BANNED_VERB_ERG_TYPES or BANNED_VERB_ERG_TYPE_RE.match(lt):
            continue
        kept.append(t)
    return kept


# ----------------------------
# JSONL helpers
# ----------------------------


def _jsonl_iter(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_lemmas_jsonl(path: str, *, pos: Optional[str] = None, limit: Optional[int] = None) -> List[str]:
    """Load lemmas from a JSONL file.

    Accepts keys: lemma, form, orth (first one found wins).
    If pos is given, keep only entries where obj['pos'] matches (case-insensitive).
    """
    out: List[str] = []
    seen = set()
    for obj in _jsonl_iter(path):
        if pos is not None:
            p = obj.get("pos")
            if p is None or str(p).upper() != pos.upper():
                continue
        lemma = obj.get("lemma") or obj.get("form") or obj.get("orth")
        if not lemma:
            continue
        lemma = str(lemma).strip()
        if not lemma:
            continue
        lemma_l = lemma.lower()
        if lemma_l in seen:
            continue
        seen.add(lemma_l)
        out.append(lemma_l)
        if limit is not None and len(out) >= limit:
            break
    return out


# ----------------------------
# Participle -> base lemma guessing
# ----------------------------


# Common irregular past-participle -> base-lemma mappings. We keep this deliberately small
# and focused on forms that frequently participate in "be VBN to" to-inf patterns.
IRREGULAR_PARTICIPLE_TO_BASE: Dict[str, str] = {
    # Core cases surfaced in augmentation logs / typical to-inf predicates
    "bound": "bind",
    "meant": "mean",
    "known": "know",
    "said": "say",
    "thought": "think",
    "made": "make",
    "seen": "see",
    "told": "tell",
    "felt": "feel",
    "heard": "hear",
    "kept": "keep",
    "left": "leave",
    "found": "find",
    "held": "hold",
    "given": "give",
    "taken": "take",
    "written": "write",
    "driven": "drive",
    "shown": "show",
    "put": "put",
}

def guess_base_lemma_from_participle(form: str, known_verbs: Set[str]) -> Optional[str]:
    """Best-effort mapping from a participial adjective/verb surface form to a base verb lemma.

    This is used to emit *shadow* VERB entries for participial ADJ predicates (e.g. "supposed",
    "bound") because spaCy often tags them as VERB/VBN and lemmatizes them to the base verb
    (e.g. supposed->suppose, bound->bind).

    Returns None if we can't confidently map to a known verb lemma.
    """
    f = form.lower()
    if f in IRREGULAR_PARTICIPLE_TO_BASE:
        base = IRREGULAR_PARTICIPLE_TO_BASE[f]
        return base if base in known_verbs else None

    # -ied -> -y
    if f.endswith("ied"):
        cand = f[:-3] + "y"
        if cand in known_verbs:
            return cand

    # Regular -ed or -d endings
    if f.endswith("ed"):
        # 1) strip "ed"
        cand0 = f[:-2]
        if cand0 in known_verbs:
            return cand0

        # 2) strip only "d" (verbs ending in "e": suppose->supposed)
        cand1 = f[:-1]
        if cand1 in known_verbs:
            return cand1

        # 3) add back "e" (excited->excite)
        cand2 = cand0 + "e"
        if cand2 in known_verbs:
            return cand2

    # Some participles end in -en; we only trust explicit irregular map for now.
    return None


def add_shadow_verb_entries_from_participial_adjs(
    adj_entries: Sequence[Dict[str, Any]],
    verb_entries_by_lemma: Dict[str, Dict[str, Any]],
    known_verbs: Set[str],
) -> None:
    """Augment the VERB lexicon with "shadow" entries derived from participial ADJs.

    Motivation: spaCy often POS-tags participial adjectives as VERB/VBN in copular/
    passive-looking "be VBN to ..." strings (e.g., "was supposed to", "is bound to"),
    and lemmatizes them to the base verb (supposed->suppose, bound->bind). Without a
    corresponding VERB entry, the augmenter sees a to-inf spine but cannot find a safe
    replacement pool.

    We only create shadow entries when:
    - the adjective entry looks like a participle (regular -ed or known irregular)
    - it participates in a raising-style to-inf frame (tag includes toinf:raising)
    - we have explicit evidence it tolerates expletives (it/there) in the to-inf frame
      (this keeps the derived passive pool narrow)

    Option B behavior:
    We add a *separate* passive/participial tag (TOINF_PASSIVE_RAISING_TAG) rather than
    merging plain toinf:raising into the base verb. This enables the augmenter to later
    pick a passive-compatible pool for "was VBN to ..." contexts (and avoid swaps like
    "was seemed to ...").
    """

    for adj in adj_entries:
        adj_lemma = str(adj.get("lemma", ""))
        adj_tags = set(adj.get("tags", []))
        if "toinf:raising" not in adj_tags:
            continue

        base = guess_base_lemma_from_participle(adj_lemma, known_verbs)
        if not base:
            continue

        # Respect the generator's hard exclusions.
        if base in VERB_EXCLUDE_LEMMAS:
            continue

        adj_licenses = set(adj.get("licenses", []))

        # Keep this narrow: only derive passive/participial raising if the ADJ evidence
        # already allows an expletive subject in the to-inf frame.
        if not ("there" in adj_licenses or "it" in adj_licenses):
            continue

        shadow_licenses = {"subj", "subjpass"}
        if "there" in adj_licenses:
            shadow_licenses.add("there")

        shadow: Dict[str, Any] = {
            "lemma": base,
            "pos": "VERB",
            "tags": [TOINF_PASSIVE_RAISING_TAG],
            "primary_tag": TOINF_PASSIVE_RAISING_TAG,
            "licenses": sorted(shadow_licenses),
            # Keep a breadcrumb for debugging / later refinements.
            "erg_types": [f"shadow_passive_from_adj::{adj_lemma}"],
        }

        if base in verb_entries_by_lemma:
            verb_entries_by_lemma[base] = merge_lex_entries(verb_entries_by_lemma[base], shadow)
        else:
            verb_entries_by_lemma[base] = shadow


def add_passive_raising_frames_from_ecm_verbs(
    adj_entries: Sequence[Dict[str, Any]],
    verb_entries_by_lemma: Dict[str, Dict[str, Any]],
    known_verbs: Set[str],
) -> None:
    """Option B: add a passive/participial raising tag to select ECM verbs.

    This captures patterns like:
      - Kim was supposed to leave.
      - There was believed to be a mistake.

    In these, spaCy commonly tags the participle as VERB/VBN with an auxpass "be".
    The base lemma (suppose, believe, expect, ...) is typically tagged in the ERG
    as an ECM predicate in active voice, but in passive it behaves *like* a raising
    predicate for augmentation purposes (subj/there licensing, no overt object).

    Safety / narrowness criteria:
      1) The verb already has toinf:ecm.
      2) We have some expletive evidence in the ECM frame (ecm_it and/or ecm_there).
      3) We can connect the lemma to an attested participial adjective form (via
         guess_base_lemma_from_participle over the ADJ lexicon). This is a crude but
         effective proxy that the VBN surface is used in "be VBN to ..." strings.

    The derived frame is recorded via:
      - tag: TOINF_PASSIVE_RAISING_TAG
      - licenses: adds subj, subjpass, and (if supported) there
    """

    # Map base verb lemma -> list of participial adjective lemmas observed.
    base_to_part_adjs: Dict[str, List[Dict[str, Any]]] = {}
    for adj in adj_entries:
        adj_lemma = str(adj.get("lemma", ""))
        base = guess_base_lemma_from_participle(adj_lemma, known_verbs)
        if not base:
            continue
        base_to_part_adjs.setdefault(base, []).append(adj)

    for lemma, v in list(verb_entries_by_lemma.items()):
        tags = set(v.get("tags", []))
        if "toinf:ecm" not in tags:
            continue

        licenses = set(v.get("licenses", []))
        has_expletive_evidence = ("ecm_it" in licenses) or ("ecm_it_rain" in licenses) or ("ecm_there" in licenses)
        if not has_expletive_evidence:
            continue

        # Narrowness: only do this if we can link to at least one participial ADJ
        # surface form for the lemma.
        part_adjs = base_to_part_adjs.get(str(lemma), [])
        if not part_adjs:
            continue

        add_licenses = {"subj", "subjpass"}
        if "ecm_there" in licenses:
            add_licenses.add("there")
        else:
            # Sometimes the ADJ evidence is stronger than the ECM probes.
            if any("there" in set(a.get("licenses", [])) for a in part_adjs):
                add_licenses.add("there")

        derived: Dict[str, Any] = {
            "lemma": lemma,
            "pos": "VERB",
            "tags": [TOINF_PASSIVE_RAISING_TAG],
            "primary_tag": TOINF_PASSIVE_RAISING_TAG,
            "licenses": sorted(add_licenses),
            "erg_types": [
                "shadow_passive_from_ecm::" + "/".join(sorted({str(a.get("lemma", "")) for a in part_adjs}))
            ],
        }

        verb_entries_by_lemma[lemma] = merge_lex_entries(v, derived)

def load_verbnet_lemmas(limit: Optional[int] = None) -> List[str]:
    """Load single-token VerbNet member lemmas via NLTK."""
    import nltk
    from nltk.corpus import verbnet as vn  # type: ignore

    try:
        _ = vn.classids()
    except LookupError:
        nltk.download("verbnet")
        _ = vn.classids()

    lemmas: Set[str] = set()
    for cid in vn.classids():
        try:
            xml = vn.vnclass(cid)
        except Exception:
            continue

        for m in xml.findall(".//MEMBER"):
            name = (m.attrib.get("name") or "").strip().lower()
            if name and " " not in name and "_" not in name:
                lemmas.add(name)
                if limit is not None and len(lemmas) >= limit:
                    return sorted(lemmas)

    return sorted(lemmas)


def write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ----------------------------
# Derivation utilities (duck-typed; works across PyDelphin versions)
# ----------------------------


def _node_daughters(node: Any) -> Optional[Sequence[Any]]:
    return getattr(node, "daughters", None)


def _is_node(obj: Any) -> bool:
    return _node_daughters(obj) is not None


def _is_preterminal(node: Any) -> bool:
    ds = _node_daughters(node)
    if not ds:
        return False
    return all(not _is_node(d) for d in ds)


def iter_preterminals(root: Any) -> Iterable[Any]:
    """Yield preterminal nodes (nodes whose daughters are terminals)."""
    if hasattr(root, "preterminals") and callable(getattr(root, "preterminals")):
        yield from root.preterminals()  # type: ignore[attr-defined]
        return
    stack = [root]
    while stack:
        n = stack.pop()
        if not _is_node(n):
            continue
        if _is_preterminal(n):
            yield n
            continue
        ds = _node_daughters(n) or []
        stack.extend(reversed(ds))


def terminals_of(node: Any) -> List[Any]:
    if hasattr(node, "terminals") and callable(getattr(node, "terminals")):
        return list(node.terminals())  # type: ignore[attr-defined]
    out: List[Any] = []
    stack = [node]
    while stack:
        x = stack.pop()
        if _is_node(x):
            ds = _node_daughters(x) or []
            stack.extend(reversed(ds))
        else:
            out.append(x)
    return out


def terminal_form(t: Any) -> str:
    for attr in ("form", "token", "surface", "orth"):
        if hasattr(t, attr):
            v = getattr(t, attr)
            if v is not None:
                return str(v)
    return str(t)


def preterminal_surface(pre: Any) -> str:
    terms = terminals_of(pre)
    return " ".join(terminal_form(t) for t in terms)


def extract_types_for_token(root: Any, token: str) -> List[Tuple[str, str, str]]:
    """Return [(surface, lex_type, entity)] for preterminals matching token."""
    token_l = token.lower()
    matches: List[Tuple[str, str, str]] = []
    for pre in iter_preterminals(root):
        surf = preterminal_surface(pre)
        if surf.lower() != token_l:
            continue
        lex_type = getattr(pre, "type", "") or ""
        entity = getattr(pre, "entity", "") or ""
        matches.append((surf, str(lex_type), str(entity)))
    return matches


def type_selects_vp(lex_type: str) -> bool:
    """Heuristic: does this lexical type name include a VP complement?"""
    lt = lex_type.lower()
    if "vp_i" in lt or "vpslnp" in lt:
        return True

    # Many ERG verb types encode VP complements as a segment like
    #   v_pp-vp_ssr_le
    #   v_vp_ssr-nimp_le
    #   v_np-vp_sor_le
    # i.e., "vp" followed by "_" or "-", not by a non-word character.
    # The previous \bvp\b check fails in these cases because '_' counts as a word char.
    return re.search(r"(?:^|[-_])vp(?:$|[-_])", lt) is not None


def type_is_adj_like(lex_type: str) -> bool:
    """Conservative filter for ADJ entries: accept only adjective-like lexical types."""
    lt = lex_type.lower()
    return lt.startswith("aj_") or lt.startswith("aj-")


def type_is_verb_like(lex_type: str) -> bool:
    """Conservative filter for VERB entries: accept only verb-like lexical types."""
    lt = lex_type.lower()
    return lt.startswith("v_") or lt.startswith("v-")


# ----------------------------
# Lexicon entry helpers
# ----------------------------


def uniq_sorted(xs: Iterable[str]) -> List[str]:
    """Deduplicate + sort while dropping falsy values."""
    return sorted({x for x in xs if x})


def merge_lex_entries(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Union-merge two lexicon entries with the same lemma.

    Keeps the original schema required by the augmenter (lemma/pos/tags/primary_tag/licenses/erg_types)
    and optionally merges an extended 'frames' field if present.
    """
    if not dst:
        return dict(src)

    # Sanity: keep dst lemma/pos if already set.
    out = dict(dst)
    for k in ("lemma", "pos"):
        if k not in out and k in src:
            out[k] = src[k]

    out["tags"] = uniq_sorted(list(out.get("tags", [])) + list(src.get("tags", [])))
    out["licenses"] = uniq_sorted(list(out.get("licenses", [])) + list(src.get("licenses", [])))
    out["erg_types"] = uniq_sorted(list(out.get("erg_types", [])) + list(src.get("erg_types", [])))

    # Frames are optional and ignored by the current augmenter; keep for debugging/future use.
    if "frames" in out or "frames" in src:
        out["frames"] = list(out.get("frames", [])) + list(src.get("frames", []))

    # Primary tag is only informational for the current augmenter.
    # Keep it stable based on TAG_PRIORITY.
    out["primary_tag"] = pick_primary_tag(out.get("tags", []))

    return out


def infer_adj_frames(tags: Sequence[str], licenses: Sequence[str]) -> List[Dict[str, Any]]:
    """Produce a light-weight per-tag view of licenses.

    This is optional metadata: the augmenter ignores it, but it's useful for debugging
    polysemy/multi-tag entries.
    """
    tagset = set(tags)
    licset = set(licenses)
    frames: List[Dict[str, Any]] = []
    if "toinf:raising" in tagset:
        frames.append({"tag": "toinf:raising", "licenses": sorted(licset & {"np", "it", "there"})})
    if "toinf:tough" in tagset:
        frames.append({"tag": "toinf:tough", "licenses": sorted(licset & {"np", "it", "gap_pp"})})
    if "toinf:control" in tagset:
        frames.append({"tag": "toinf:control", "licenses": sorted(licset & {"np"})})
    return frames


def infer_verb_frames(tags: Sequence[str], licenses: Sequence[str]) -> List[Dict[str, Any]]:
    licset = set(licenses or [])
    frames: List[Dict[str, Any]] = []
    if TOINF_PASSIVE_RAISING_TAG in tags:
        frames.append({"tag": TOINF_PASSIVE_RAISING_TAG, "licenses": sorted(licset & {"subj", "subjpass", "there"})})
    if TOINF_RAISING_TAG in tags:
        frames.append({"tag": TOINF_RAISING_TAG, "licenses": sorted(licset & {"subj", "there"})})
    if "toinf:ecm" in tags:
        frames.append({"tag": "toinf:ecm", "licenses": sorted(licset & {"ecm_it", "ecm_it_rain", "ecm_there", "obj"})})
    if "toinf:ocontrol" in tags:
        frames.append({"tag": "toinf:ocontrol", "licenses": sorted(licset & {"obj"})})
    if "toinf:scontrol" in tags:
        frames.append({"tag": "toinf:scontrol", "licenses": sorted(licset & {"subj"})})
    if "toinf:purpose" in tags:
        # Most of these are derived from broad "ocontrol" types; keep the old behavior
        # (prefer subj when available), but fall back to obj so we don't emit empty licenses.
        purp_lic = (licset & {"subj"}) or (licset & {"obj"})
        frames.append({"tag": "toinf:purpose", "licenses": sorted(purp_lic)})
    return frames




# ----------------------------
# ACE wrapper
# ----------------------------


@dataclass
class AceOracle:
    erg_dat: str
    ace_bin: str = "ace"
    # '-1' = first reading only; '--udx' gives derivations.
    cmdargs: List[str] = field(default_factory=lambda: ["-1", "--udx"])
    _parser: Any = field(init=False, repr=False)
    _n_total: int = 0
    _n_parsed: int = 0
    _t_total: float = 0.0

    def __post_init__(self) -> None:
        if ace is None:
            raise RuntimeError(
                "PyDelphin (delphin) is not installed. Install with: pip install pydelphin"
            ) from _IMPORT_ERROR
        self._parser = ace.ACEParser(self.erg_dat, executable=self.ace_bin, cmdargs=self.cmdargs)

    def parse_first_derivation(self, sent: str) -> Optional[Any]:
        """Parse a sentence and return the first derivation root (or None)."""
        t0 = time.time()
        self._n_total += 1

        resp: Optional[Any] = None
        try:
            r0 = self._parser.interact(sent)
        except Exception:
            self._t_total += (time.time() - t0)
            return None

        # PyDelphin's ACEParser.interact() has returned both:
        #   * a single Response/dict-like object, OR
        #   * an iterator yielding Response/dict-like objects (sometimes interleaved with strings)
        if hasattr(r0, "get"):
            resp = r0
        else:
            try:
                for r in r0:
                    if hasattr(r, "get"):
                        resp = r
                        break
            except TypeError:
                # Not iterable and not dict-like
                resp = None

        self._t_total += (time.time() - t0)
        if not resp:
            return None

        results = resp.get("results") or []
        if not results:
            return None

        dstr = results[0].get("derivation")
        if not dstr or derivation is None:
            return None

        try:
            root = derivation.from_string(dstr)
        except Exception:
            return None

        self._n_parsed += 1
        return root

    def stats(self) -> str:
        if self._n_total <= 0:
            return "NOTE: parsed 0 / 0"
        avg_ms = (self._t_total / max(1, self._n_total)) * 1000.0
        return f"NOTE: parsed {self._n_parsed} / {self._n_total} sentences, avg {avg_ms:.0f}ms"


# ----------------------------
# Probes + classification
# ----------------------------


ADJ_PROBES: List[Tuple[str, str]] = [
    # Tough-movement / gap signature (preposition stranding)
    ("gap_pp", "Kim is {X} to talk to."),
    # Canonical NP-subject infinitival complement
    ("np", "Kim is {X} to leave."),
    # Expletive 'there' (raising diagnostic)
    ("there", "There is {X} to be a problem."),
    # Expletive 'it' (keep the probe name explicit so downstream logic
    # doesn't accidentally mismatch probe keys).
    ("it_leave", "It is {X} to leave."),
    ("it_rain", "It is {X} to rain."),
    ("there_trouble", "There is {X} to be trouble.")
]


VERB_PROBES: List[Tuple[str, str]] = [
    # Keep the lemma uninflected: use a modal.
    ("subj", "Kim may {X} to be ready."),
    ("there", "There may {X} to be a problem."),
    ("scontrol", "Kim may {X} to leave."),
    ("obj", "Kim may {X} Dana to leave."),
    ("purpose", "Kim may {X} Dana to leave."),
    ("ecm", "Kim may {X} Dana to be ready."),
    ("ecm_it", "Kim may {X} it to be ready."),
    # Distinguish *weather* "it to rain" ECM from purely propositional "it to be ...".
    ("ecm_it_rain", "Kim may {X} it to rain."),
    ("ecm_there", "Kim may {X} there to be a problem."),
    # Passive probes for direct raising_passive detection (avoids relying on ADJ shadow logic)
    ("passive_subj", "Kim was {X}ed to leave."),
    ("passive_there", "There was {X}ed to be a problem."),
]


def _mn_re(mn: str) -> re.Pattern:
    # Match mnemonics in ERG lexical type names regardless of whether they
    # are delimited by '-' or '_' (both occur in practice).
    # Example: v_pp-vp_ssr_le  → ssr
    return re.compile(r"(?:^|[-_])" + re.escape(mn) + r"(?:$|[-_])", re.I)


def _any_has_mn(types: Sequence[str], mn: str) -> bool:
    rx = _mn_re(mn)
    return any(bool(rx.search(t)) for t in types)


def _is_tough_adj_gap_pp_type(t: str) -> bool:
    """Return True if a type string looks like an ERG tough/PP-gap adjective type.

    We use this to keep the 'gap_pp' license narrow: the 'gap_pp' probe sentence can
    sometimes parse with a raising/control analysis (e.g., *likely*, *certain*), but
    we only want to treat it as evidence for PP-gap/tough movement when ERG picks an
    `aj_pp-*` type.
    """
    s = (t or "").lower()
    return s.startswith("aj_pp") or "aj_pp-vp" in s


def _tags_from_types_adj(types: Sequence[str]) -> List[str]:
    tags: List[str] = []
    # Prefer "hard" evidence from the ERG mnemonic when present.
    # NOTE: PR(TY) adjectives (e.g., "It was nice to meet you") behave like
    # tough/extraposition rather than subject-control, so treat them as tough.
    if _any_has_mn(types, "tgh") or _any_has_mn(types, "wrth") or _any_has_mn(types, "prty"):
        tags.append("toinf:tough")
    if _any_has_mn(types, "ssr"):
        tags.append("toinf:raising")
    # Subject-control (equi) patterns are marked with SEQ.
    if _any_has_mn(types, "seq"):
        tags.append("toinf:control")
    return sorted(set(tags))


def _tags_from_types_verb(types: Sequence[str]) -> List[str]:
    tags: List[str] = []
    if _any_has_mn(types, "ssr"):
        tags.append("toinf:raising")
    if _any_has_mn(types, "sor"):
        tags.append("toinf:ecm")
    if _any_has_mn(types, "oeq"):
        tags.append("toinf:ocontrol")
    if _any_has_mn(types, "seq") or _any_has_mn(types, "aeq"):
        tags.append("toinf:scontrol")
    return sorted(set(tags))


def _primary_tag_from(tags: Sequence[str], *, prefer: Sequence[str]) -> Optional[str]:
    s = set(tags)
    for p in prefer:
        if p in s:
            return p
    return tags[0] if tags else None


def classify_adj(oracle: AceOracle, lemma: str) -> Dict[str, Any]:
    evidence: Dict[str, Any] = {"lemma": lemma, "probes": {}, "erg_types": []}
    types_seen: List[str] = []
    probe_vp_types: Dict[str, List[str]] = {}
    
    # Check if lemma looks like it might be compositionally analyzed (un-/over- prefix)
    is_potentially_prefixed = _looks_prefixed(lemma)

    for probe_name, tmpl in ADJ_PROBES:
        # Some participial adjectives (esp. -ing) can get analyzed as progressive verbs under "be",
        # which can cause false negatives for the license probes (e.g., "troubling").
        # To reduce this, also try a degree-modified variant for probes that use "be".
        sent_variants = [tmpl.format(X=lemma)]
        if lemma.lower().endswith("ing") and probe_name in ("it_leave", "it_rain", "np", "gap_pp"):
            sent_variants.append(tmpl.format(X=f"very {lemma}"))

        parsed_any = False
        matches_best = []
        all_probe_types: List[str] = []
        all_probe_types_pred_ok: List[str] = []
        variant_rows = []
        for sent in sent_variants:
            root = oracle.parse_first_derivation(sent)
            if root is None:
                variant_rows.append({"sent": sent, "parsed": False, "matches": [], "vp_sel_types": []})
                continue
            parsed_any = True
            matches = extract_types_for_token(root, lemma)
            if not matches_best:
                matches_best = matches
            probe_types_all = [m[1] for m in matches if m[1]]
            probe_types = [t for t in probe_types_all if type_selects_vp(t) and type_is_adj_like(t)]
            
            # NEW: Also filter by predicate matching (like we do for verbs)
            # This rejects "unbound" when ERG analyzes it as un- + bound
            probe_types_pred_ok = [
                m[1] for m in matches 
                if m[1] and type_selects_vp(m[1]) and type_is_adj_like(m[1])
                and pred_like_matches_lemma(m[2], lemma, pos="a")
            ]
            
            variant_rows.append({
                "sent": sent, 
                "parsed": True, 
                "matches": matches, 
                "vp_sel_types": probe_types,
                "vp_sel_types_pred_ok": probe_types_pred_ok,
            })
            all_probe_types.extend(probe_types)
            all_probe_types_pred_ok.extend(probe_types_pred_ok)

        # Decision: which types to trust?
        # If predicate matches, prefer those. If prefixed and no pred match, reject.
        if all_probe_types_pred_ok:
            final_probe_types = all_probe_types_pred_ok
        elif is_potentially_prefixed:
            # Looks prefixed but predicate didn't match - this is likely a compositional
            # analysis (e.g., "unbound" = un- + bound). Don't trust it.
            final_probe_types = []
        else:
            # Not prefixed, predicate mismatch might be ERG quirk - use with caution
            final_probe_types = all_probe_types

        # de-dup while preserving order
        vp_sel_types = []
        for t in final_probe_types:
            if t not in vp_sel_types:
                vp_sel_types.append(t)

        evidence["probes"][probe_name] = {
            "sent": sent_variants[0],
            "parsed": parsed_any,
            "matches": matches_best,
            "vp_sel_types": vp_sel_types,
            "variants": variant_rows,
            "pred_matched": bool(all_probe_types_pred_ok),
        }
        probe_vp_types[probe_name] = vp_sel_types
        for t in vp_sel_types:
            if t not in types_seen:
                types_seen.append(t)
# 1) Primary tag from mnemonic-bearing type names
    tags = _tags_from_types_adj(types_seen)

    # Evidence-driven tag augmentation:
    # - a successful gap_pp probe is strong evidence for a tough construction
    # - a successful there / there_trouble probe is strong evidence for raising
    # - it_rain is a better expletive-it diagnostic than plain "it"
    #
    # We do this even when mnemonics are visible, to allow multi-tag entries
    # (e.g., adjectives that participate in both control and raising patterns).
    tagset = set(tags)

    # Note: the gap_pp probe sentence can sometimes parse with a raising/control analysis.
    # Only treat gap_pp as evidence for tough/PP-gap when ERG picks an aj_pp-* type.
    gap_pp_types = probe_vp_types.get("gap_pp") or []
    has_gap_pp = any(_is_tough_adj_gap_pp_type(t) for t in gap_pp_types)

    if has_gap_pp:
        tagset.add("toinf:tough")
    if probe_vp_types.get("there") or probe_vp_types.get("there_trouble"):
        tagset.add("toinf:raising")
    if probe_vp_types.get("it_rain") and not has_gap_pp:
        tagset.add("toinf:raising")
    # As a last resort, "it" without "np" is often raising; keep it conservative.
    if (probe_vp_types.get("it_leave") or probe_vp_types.get("it_rain")) and (not probe_vp_types.get("np")) and (not has_gap_pp):
        tagset.add("toinf:raising")
    tags = sorted(tagset)

    # PR(TY)+SEQ collision: "prty" can co-occur with "seq" on control adjectives.
    # Be conservative in two senses:
    #  - keep recall: if we saw seq, keep the control tag;
    #  - keep precision: only keep the prty→tough tag when the gap_pp probe succeeded.
    # When both prty and seq are present, make control primary.
    has_prty = _any_has_mn(types_seen, "prty")
    has_seq = _any_has_mn(types_seen, "seq")
    gap_ok = has_gap_pp
    if has_prty and has_seq:
        if (not gap_ok) and (not _any_has_mn(types_seen, "tgh")):
            tags = [t for t in tags if t != "toinf:tough"]
        primary_prefer = ("toinf:control", "toinf:raising", "toinf:tough")
    else:
        primary_prefer = ("toinf:raising", "toinf:control", "toinf:tough")

    primary_tag = _primary_tag_from(tags, prefer=primary_prefer)

    # 2) Conservative fallback (only when mnemonics weren't visible):
    #    - gap_pp → tough
    #    - there  → raising
    #    - np     → control
    #    - it-only → raising (safe under Option A because licenses will gate swapping)
    if primary_tag is None:
        if has_gap_pp:
            tags = ["toinf:tough"]
            primary_tag = "toinf:tough"
        elif probe_vp_types.get("there"):
            tags = ["toinf:raising"]
            primary_tag = "toinf:raising"
        elif (probe_vp_types.get("it_leave") or probe_vp_types.get("it_rain")) and not probe_vp_types.get("np"):
            tags = ["toinf:raising"]
            primary_tag = "toinf:raising"
        elif probe_vp_types.get("np"):
            tags = ["toinf:control"]
            primary_tag = "toinf:control"

    # 3) Licenses (used by the augmenter for runtime filtering).
    #    Be conservative: only grant "there" to raising items and only
    #    grant "gap_pp" to tough items.
    # Recompute gap_pp truth with the tough/PP-gap constraint (aj_pp-* types only).
    gap_pp_types = probe_vp_types.get("gap_pp") or []
    has_gap_pp = any(_is_tough_adj_gap_pp_type(t) for t in gap_pp_types)
    has_np = bool(probe_vp_types.get("np") or has_gap_pp)
    has_it = bool(probe_vp_types.get("it_leave") or probe_vp_types.get("it_rain"))
    has_there = bool(probe_vp_types.get("there") or probe_vp_types.get("there_trouble"))
    
    # NEW: Validate there-licensing - reject known false positives
    there_types = (probe_vp_types.get("there") or []) + (probe_vp_types.get("there_trouble") or [])
    if has_there and lemma in ADJ_THERE_LICENSE_BLOCKLIST:
        has_there = False
    # Also require raising mnemonic (ssr) to be more confident
    if has_there and not _any_has_mn(there_types, "ssr"):
        # ERG parsed it but no raising mnemonic - be conservative
        # Keep has_there only if we also saw it in types_seen (overall)
        if not _any_has_mn(types_seen, "ssr"):
            has_there = False

    licenses: List[str] = []
    tagset = set(tags)
    if has_np:
        licenses.append("np")
    if has_it and ("toinf:raising" in tagset or "toinf:tough" in tagset):
        licenses.append("it")
    if has_there and "toinf:raising" in tagset:
        licenses.append("there")
    if has_gap_pp and "toinf:tough" in tagset:
        licenses.append("gap_pp")

    evidence["erg_types"] = types_seen
    evidence["tags"] = sorted(set(tags))
    evidence["primary_tag"] = primary_tag
    evidence["licenses"] = sorted(set(licenses))
    return evidence


def classify_verb(oracle: AceOracle, lemma: str) -> Dict[str, Any]:
    evidence: Dict[str, Any] = {"lemma": lemma, "probes": {}, "erg_types": []}
    types_seen: List[str] = []
    probe_vp_types: Dict[str, List[str]] = {}
    
    # Get proper past participle for passive probes
    past_participle = get_past_participle(lemma)

    for probe_name, tmpl in VERB_PROBES:
        # Handle passive probes specially - use past participle
        if probe_name.startswith("passive_"):
            # Replace {X}ed with the proper past participle
            sent = tmpl.replace("{X}ed", past_participle)
            token_to_match = past_participle
        else:
            sent = tmpl.format(X=lemma)
            token_to_match = lemma
            
        root = oracle.parse_first_derivation(sent)
        if root is None:
            evidence["probes"][probe_name] = {"sent": sent, "parsed": False}
            probe_vp_types[probe_name] = []
            continue

        matches = extract_types_for_token(root, token_to_match)

        # Only trust analyses where the *lexical predicate* corresponds to this lemma.
        # This prevents productive prefix/orthography analyses from sneaking in:
        #   re+press -> press, out+bid -> bid, co+exist -> exist, etc.
        # For passive probes, we still match against the base lemma's predicate.
        probe_types_all = [m[1] for m in matches if m[1]]
        probe_types_pred_ok = [m[1] for m in matches if m[1] and pred_like_matches_lemma(m[2], lemma, pos="v")]

        # Filter out adjunct-purpose analyses etc: require the verb's own lexical type
        # to select a VP complement.
        probe_types_any = [t for t in probe_types_all if type_selects_vp(t) and type_is_verb_like(t)]
        probe_types = [t for t in probe_types_pred_ok if type_selects_vp(t) and type_is_verb_like(t)]

        evidence["probes"][probe_name] = {
            "sent": sent,
            "parsed": True,
            "matches": matches,
            "vp_sel_types": probe_types,
            # for debugging predicate-mismatch cases
            "vp_sel_types_any": probe_types_any,
        }
        probe_vp_types[probe_name] = probe_types

        for t in probe_types:
            if t not in types_seen:
                types_seen.append(t)

    # Drop verb types that are strong PP+purpose/result false positives.
    types_seen = _filter_verb_types(types_seen)

    # Tags come only from visible ERG mnemonics (no probe-based guessing for ECM),
    # because false positives here are very costly for swapping.
    tags = _tags_from_types_verb(types_seen)

    # Evidence-driven tag augmentation: expletive-*there* is strong evidence for raising,
    # but only if ERG analyzes the probe with a raising-type mnemonics (ssr/srs/rse).
    there_types = probe_vp_types.get("there") or []
    has_raising_there = bool(there_types) and (
        _any_has_mn(there_types, "ssr") or _any_has_mn(there_types, "srs") or _any_has_mn(there_types, "rse")
    )
    
    # NEW: Validate there-licensing with blocklist
    if has_raising_there and lemma in VERB_THERE_LICENSE_BLOCKLIST:
        has_raising_there = False
    
    if has_raising_there:
        tags = list(set(tags) | {TOINF_RAISING_TAG})

    # NEW: Behavioral ECM detection
    # If verb is tagged as ocontrol but is in ECM_BEHAVIORAL_OVERRIDE, upgrade to ECM.
    # These are perception/cognition verbs that ERG misclassifies as ocontrol.
    # NOTE: We don't rely on probe success (ecm_it_rain/ecm_there) because ERG
    # overgenerates expletive parses for many verbs that aren't truly ECM.
    if "toinf:ocontrol" in tags and "toinf:ecm" not in tags:
        if lemma in ECM_BEHAVIORAL_OVERRIDE:
            # Upgrade to ECM
            tags = [t for t in tags if t not in ("toinf:ocontrol", "toinf:purpose")]
            tags.append("toinf:ecm")

    # Split broad "ocontrol" into (a) true directive/causative object-control vs
    # (b) PP/purpose-result ("... to V") cases that should not be swapped together.
    if "toinf:ocontrol" in tags and lemma not in OCONTROL_SWAP_ALLOWLIST and "toinf:ecm" not in tags:
        tags = list(set(tags) | {"toinf:purpose"})

    # NEW: Direct passive raising detection from passive probes
    # This avoids relying on ADJ shadow logic
    passive_subj_ok = bool(probe_vp_types.get("passive_subj"))
    passive_there_ok = bool(probe_vp_types.get("passive_there"))
    if "toinf:ecm" in tags and (passive_subj_ok or passive_there_ok):
        if TOINF_PASSIVE_RAISING_TAG not in tags:
            tags.append(TOINF_PASSIVE_RAISING_TAG)

    primary_tag = _primary_tag_from(
        tags,
        prefer=("toinf:raising", "toinf:ecm", "toinf:purpose", "toinf:ocontrol", "toinf:scontrol"),
    )

    licenses: List[str] = []
    if probe_vp_types.get("subj"):
        licenses.append("subj")
    if probe_vp_types.get("obj") and _any_has_mn(probe_vp_types["obj"], "np-vp"):
        licenses.append("obj")

    # Only grant "there" if the expletive-there probe supports a raising-type analysis
    # AND it's not in the blocklist
    if has_raising_there and TOINF_RAISING_TAG in tags:
        licenses.append("there")

    # Only grant ECM licenses if we have SOR evidence.
    if probe_vp_types.get("ecm_it") and (
        "toinf:ecm" in tags or _any_has_mn(probe_vp_types["ecm_it"], "sor")
    ):
        licenses.append("ecm_it")
    if probe_vp_types.get("ecm_it_rain") and (
        "toinf:ecm" in tags or _any_has_mn(probe_vp_types["ecm_it_rain"], "sor")
    ):
        licenses.append("ecm_it_rain")
    if probe_vp_types.get("ecm_there") and (
        "toinf:ecm" in tags or _any_has_mn(probe_vp_types["ecm_there"], "sor")
    ):
        licenses.append("ecm_there")
    
    # Issue 4 fix: ECM verbs should have obj license if they pass ECM probes
    # (they take an NP that is the subject of the embedded clause)
    if "toinf:ecm" in tags and "obj" not in licenses:
        if probe_vp_types.get("ecm_it") or probe_vp_types.get("ecm_there") or probe_vp_types.get("ecm_it_rain"):
            licenses.append("obj")
    
    # Issue 3 fix: scontrol verbs shouldn't have obj license unless they also have ocontrol
    # "promise" is scontrol with an object, but "promise NP to VP" shouldn't swap with
    # true object-control verbs like "persuade NP to VP" (different theta-role for NP)
    if "toinf:scontrol" in tags and "toinf:ocontrol" not in tags and "toinf:ecm" not in tags:
        if "obj" in licenses:
            licenses.remove("obj")
    
    # NEW: Add passive licenses if we detected passive raising
    if TOINF_PASSIVE_RAISING_TAG in tags:
        if "subj" not in licenses:
            licenses.append("subj")
        licenses.append("subjpass")
        if passive_there_ok and "there" not in licenses:
            licenses.append("there")

    evidence["erg_types"] = types_seen
    evidence["tags"] = sorted(set(tags))
    evidence["primary_tag"] = primary_tag
    evidence["licenses"] = sorted(set(licenses))
    return evidence

# ----------------------------
# Smoke test
# ----------------------------


def _print_smoke(oracle: AceOracle) -> None:
    """Lightweight regression test for the generator.

    Important: this smoke test is meant to reflect the *full* generator behavior:
    - manual excludes are applied
    - manual patches (certain/supposed) are applied
    - passive-raising shadow verb logic is applied
    """

    # Include a few excluded lemmas intentionally, so we can verify the exclude lists
    # are respected by the generator pipeline.
    adjs = [
        # canonical tough
        "easy", "tough", "hard", "impossible",
        # canonical raising
        "likely", "bound", "sure", "unlikely",
        # control-ish / problematic in augmentation
        "eager", "unable", "ready",
        # key problem cases
        "certain", "supposed",
        # negatives
        "purple", "wooden",
        # excluded (should not appear in output)
        "overcritical", "uncritical", "unwell", "overcurious",
    ]

    verbs = [
        # raising
        "seem", "appear", "tend", "happen",
        # control
        "try", "plan", "hope", "promise",
        "persuade", "tell", "force",
        # ecm
        "expect", "allow", "believe", "consider", "allege", "intend",
        # key passive-raising bases (must be present to receive shadow/passive frames)
        "suppose", "bind",
        # excluded (should not appear)
        "take", "bring", "use", "resent",
        # negatives
        "banquet", "sleep", "arrive", "put",
    ]

    print("[SMOKE] excludes")
    ex_adj = [a for a in adjs if a in ADJ_EXCLUDE_LEMMAS]
    ex_verb = [v for v in verbs if v in VERB_EXCLUDE_LEMMAS]
    print(f"  excluded_adjs={sorted(ex_adj)}")
    print(f"  excluded_verbs={sorted(ex_verb)}")

    # ---------- build adjective entries (with excludes + backfill) ----------
    out_adjs: List[Dict[str, Any]] = []
    for a in adjs:
        if a in ADJ_EXCLUDE_LEMMAS:
            continue

        ev = classify_adj(oracle, a)
        if not ev.get("tags") and not ev.get("licenses"):
            bf = backfill_adj_from_lexicon(oracle, a)
            if bf:
                ev = bf

        ent: Dict[str, Any] = {
            "lemma": a,
            "pos": "ADJ",
            "tags": ev.get("tags", []),
            "primary_tag": ev.get("primary_tag"),
            "licenses": ev.get("licenses", []),
            "erg_types": ev.get("erg_types", []),
            # keep probes for smoke readability
            "probes": ev.get("probes", {}),
        }
        out_adjs.append(ent)

    # Note: we intentionally do *not* call merge_lex_entries() here, because it drops the
    # per-probe debugging info used by this smoke test.

    # Manual patch for the known problematic adjs (mirrors main()).
    for ent in out_adjs:
        if ent["lemma"] in {"certain", "supposed"}:
            tags = set(ent.get("tags", []))
            lics = set(ent.get("licenses", []))
            tags.add(TOINF_RAISING_TAG)
            lics |= {"it", "there"}
            ent["tags"] = sorted(tags)
            ent["licenses"] = sorted(lics)
            ent["primary_tag"] = pick_primary_tag(ent["tags"])


    # Normalize primary_tag with the same priority function used by the generator output.
    for ent in out_adjs:
        ent["primary_tag"] = pick_primary_tag(ent.get("tags", []))

    # ---------- build verb entries (with excludes + backfill) ----------
    verb_entries_by_lemma: Dict[str, Dict[str, Any]] = {}
    for v in verbs:
        if v in VERB_EXCLUDE_LEMMAS:
            continue

        ev = classify_verb(oracle, v)
        if not ev.get("tags") and not ev.get("licenses"):
            bf = backfill_verb_from_lexicon(oracle, v)
            if bf:
                ev = bf

        verb_entries_by_lemma[v] = {
            "lemma": v,
            "pos": "VERB",
            "tags": ev.get("tags", []),
            "primary_tag": ev.get("primary_tag"),
            "licenses": ev.get("licenses", []),
            "erg_types": ev.get("erg_types", []),
            "probes": ev.get("probes", {}),
        }

    # Passive raising shadow logic (mirrors main()).
    known_verbs = set(verb_entries_by_lemma.keys())
    add_shadow_verb_entries_from_participial_adjs(out_adjs, verb_entries_by_lemma, known_verbs)
    add_passive_raising_frames_from_ecm_verbs(out_adjs, verb_entries_by_lemma, known_verbs)

    # Attach frames (like main()).
    for ent in verb_entries_by_lemma.values():
        ent["frames"] = infer_verb_frames(ent.get("tags", []), ent.get("licenses", []))

    # Note: we intentionally do *not* call merge_lex_entries() here, because it drops the
    # per-probe debugging info used by this smoke test.
    for ent in verb_entries_by_lemma.values():
        ent["primary_tag"] = pick_primary_tag(ent.get("tags", []))
    out_verbs_by_lemma = verb_entries_by_lemma

    # ---------- pretty-print ----------
    def _fmt_probe_status(is_licensed: bool, parsed: bool) -> str:
        if is_licensed and parsed:
            return "LIC✓"
        if is_licensed and not parsed:
            return "LIC✗"
        if (not is_licensed) and parsed:
            return "PARSED"
        return ""

    print("[SMOKE] adjectives (final)")
    for ent in out_adjs:
        lemma = ent["lemma"]
        tags = ent.get("tags", [])
        lic = ent.get("licenses", [])
        print(f"  {lemma:12s} tag={ent.get('primary_tag')} tags={tags} lic={lic} types={ent.get('erg_types', [])[:3]}")
        probes = ent.get("probes", {})

        # Show probe parses, but annotate whether the corresponding license is present.
        # IMPORTANT: For to-inf diagnostics, a probe only counts as evidence if it both
        # parses and yields at least one VP-selecting lexical type (vp_sel_types).
        for pname in ("gap_pp", "np", "there", "there_trouble", "it_leave", "it_rain"):
            info = probes.get(pname, {})
            if not info or not info.get("parsed"):
                continue
            mts = info.get("matches") or []
            if not mts:
                continue

            vp_types = info.get("vp_sel_types") or []
            probe_has_vp = bool(vp_types)

            # Map probe name -> license key (approx)
            lic_key = None
            if pname == "gap_pp":
                lic_key = "gap_pp"
            elif pname == "np":
                lic_key = "np"
            elif pname in {"there", "there_trouble"}:
                lic_key = "there"
            elif pname in {"it_leave", "it_rain"}:
                lic_key = "it"

            lic_set = set(lic)

            # Prefer displaying a VP-selecting match when available.
            chosen = None
            if vp_types:
                for surf, ltype, entid in mts:
                    if ltype in vp_types:
                        chosen = (surf, ltype, entid)
                        break
            if chosen is None:
                chosen = mts[0]

            is_licensed = (lic_key in lic_set) and probe_has_vp if lic_key else False
            surf, ltype, entid = chosen
            status = _fmt_probe_status(is_licensed=is_licensed, parsed=True)
            print(f"    {pname:12s} {status:6s} :: {info['sent']} :: type={ltype} entity={entid}")

        # Highlight if a license exists but we never saw a parsed probe for it.
        for needed in ("gap_pp", "np", "it", "there"):
            if needed not in set(lic):
                continue
            # any probe corresponding to license is VP-selecting?
            if needed == "it":
                ok = bool((probes.get("it_leave", {}).get("vp_sel_types")) or (probes.get("it_rain", {}).get("vp_sel_types")))
            elif needed == "there":
                ok = bool((probes.get("there", {}).get("vp_sel_types")) or (probes.get("there_trouble", {}).get("vp_sel_types")))
            else:
                ok = bool(probes.get(needed, {}).get("vp_sel_types"))
            if not ok:
                print(f"    WARNING: license={needed} present but no corresponding probe parsed (may be manual override).")

    print("[SMOKE] verbs (final)")
    for lemma in sorted(out_verbs_by_lemma.keys()):
        ent = out_verbs_by_lemma[lemma]
        tags = ent.get("tags", [])
        lic = ent.get("licenses", [])
        frames = ent.get("frames", [])
        print(f"  {lemma:10s} tag={ent.get('primary_tag')} tags={tags} lic={lic} frames={frames} types={ent.get('erg_types', [])[:3]}")
        probes = ent.get("probes", {})
        for pname in ("there", "ecm_it", "ecm_there", "obj", "subj"):
            info = probes.get(pname, {})
            if not info or not info.get("parsed"):
                continue
            vp_types = info.get("vp_sel_types") or []
            if not vp_types:
                continue
            is_licensed = pname in set(lic)
            status = _fmt_probe_status(is_licensed=is_licensed, parsed=True)
            print(f"    {pname:9s} {status:6s} :: {info['sent']} :: vp_sel_types={vp_types[:2]}")

    # ---------- targeted checks ----------
    def _check(desc: str, cond: bool) -> None:
        print(f"[SMOKE][CHECK] {desc}: {'PASS' if cond else 'FAIL'}")

    # likely/bound should be raising (not tough) and should not be granted gap_pp just because the gap_pp probe parses.
    if "likely" in {e['lemma'] for e in out_adjs}:
        e = next(e for e in out_adjs if e["lemma"] == "likely")
        _check("likely primary_tag is raising", e.get("primary_tag") == TOINF_RAISING_TAG)
        _check("likely has there license", "there" in set(e.get("licenses", [])))
        _check("likely does NOT have gap_pp license", "gap_pp" not in set(e.get("licenses", [])))

    if "bound" in {e['lemma'] for e in out_adjs}:
        e = next(e for e in out_adjs if e["lemma"] == "bound")
        _check("bound primary_tag is raising", e.get("primary_tag") == TOINF_RAISING_TAG)
        _check("bound has there license", "there" in set(e.get("licenses", [])))
        _check("bound does NOT have gap_pp license", "gap_pp" not in set(e.get("licenses", [])))

    # certain/supposed should be patched to raising + (it/there)
    for a in ("certain", "supposed"):
        if a in {e['lemma'] for e in out_adjs}:
            e = next(e for e in out_adjs if e["lemma"] == a)
            _check(f"{a} has raising tag", TOINF_RAISING_TAG in set(e.get("tags", [])))
            _check(f"{a} has it license", "it" in set(e.get("licenses", [])))
            _check(f"{a} has there license", "there" in set(e.get("licenses", [])))

    # Suppose/bind should pick up passive-raising behavior via shadows/ECM logic.
    for v in ("suppose", "bind"):
        if v in out_verbs_by_lemma:
            e = out_verbs_by_lemma[v]
            _check(f"{v} has raising_passive tag", TOINF_PASSIVE_RAISING_TAG in set(e.get("tags", [])))
            _check(f"{v} has subjpass license", "subjpass" in set(e.get("licenses", [])))

    # Excluded items should not appear in final outputs.
    _check("take excluded from verb output", "take" not in out_verbs_by_lemma)
    _check("bring excluded from verb output", "bring" not in out_verbs_by_lemma)
    _check("use excluded from verb output", "use" not in out_verbs_by_lemma)
    _check("resent excluded from verb output", "resent" not in out_verbs_by_lemma)
    for a in ("overcritical", "uncritical", "unwell", "overcurious"):
        _check(f"{a} excluded from adj output", a not in {e['lemma'] for e in out_adjs})

    print(oracle.stats())
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--erg", required=True, help="Path to ERG .dat grammar image")
    ap.add_argument("--ace", default="ace", help="Path to ACE binary (default: ace)")

    ap.add_argument("--adj_in", help="Input adjectives JSONL (lemma list to process)")
    ap.add_argument("--adj_lexicon", help="Pre-generated adjective lexicon JSONL (skip adj processing, use for verb shadow logic)")
    ap.add_argument("--verb_in", help="Input verbs JSONL, or the literal string \"verbnet\"")
    ap.add_argument("--adj_out", default="sambal/resources/toinf_adjs.jsonl", help="Output adjectives JSONL")
    ap.add_argument("--verb_out", default="sambal/resources/toinf_verbs.jsonl", help="Output verbs JSONL")
    ap.add_argument("--out-adj-missed", required=False, help="write missed adjective lemmas (one per line)")
    ap.add_argument("--out-verb-missed", required=False, help="write missed verb lemmas (one per line)")

    ap.add_argument("--limit", type=int, default=None, help="Process at most N lemmas from each input")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for shuffling (0 = no shuffle)")
    ap.add_argument("--smoke", action="store_true", help="Run a small built-in smoke test and exit")

    args = ap.parse_args(argv)

    oracle = AceOracle(args.erg, ace_bin=args.ace)

    if args.smoke:
        _print_smoke(oracle)
        return 0

    # Verbs-only mode: use pre-generated adj lexicon, skip adj processing
    verbs_only = bool(args.adj_lexicon)
    
    if verbs_only:
        if not args.verb_in:
            ap.error("--verb_in is required when using --adj_lexicon")
        # Load pre-generated adj lexicon directly
        out_adjs: List[Dict[str, Any]] = []
        with open(args.adj_lexicon, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out_adjs.append(json.loads(line))
        print(f"[VERBS-ONLY] Loaded {len(out_adjs)} adjective entries from {args.adj_lexicon}")
        adjs = []  # Empty - won't process
    else:
        if not args.adj_in or not args.verb_in:
            ap.error("--adj_in and --verb_in are required unless --smoke or --adj_lexicon is used")
        adjs = load_lemmas_jsonl(args.adj_in, pos="ADJ", limit=args.limit)
    # verbs: JSONL file OR the literal string "verbnet" (load via NLTK)
    if args.verb_in.strip().lower() == "verbnet":
        verbs = load_verbnet_lemmas(limit=args.limit)
    else:
        verbs = load_lemmas_jsonl(args.verb_in, pos="VERB", limit=args.limit)

    # (IV) Force-include: ensure high-impact lemmas are attempted even if absent from inputs.
    # Assumes FORCE_INCLUDE_ADJS / FORCE_INCLUDE_VERBS are defined above.
    if not verbs_only:
        for a in FORCE_INCLUDE_ADJS:
            if a not in adjs:
                adjs.append(a)
    for v in FORCE_INCLUDE_VERBS:
        if v not in verbs:
            verbs.append(v)

    # Used for participle→base mapping when creating passive/participial raising entries.
    # (Keep this as the *full* candidate verb list, not just the verbs that survive filtering.)
    known_verbs = set(verbs)

    if args.seed:
        rng = random.Random(args.seed)
        if not verbs_only:
            rng.shuffle(adjs)
        rng.shuffle(verbs)

    # Build ADJ entries into a lemma->entry map to avoid duplicate JSONL entries and
    # to support multi-tag / multi-evidence adjectives.
    adj_entries_by_lemma: Dict[str, Dict[str, Any]] = {}
    missed_adjs: List[str] = []
    missed_verbs: List[str] = []
    kept_a = 0
    
    if verbs_only:
        # Skip adj processing - out_adjs already loaded from --adj_lexicon
        kept_a = len(out_adjs)
    else:
        for i, a in enumerate(adjs):
            if i % 500 == 0:
                print(f"[PROGRESS] ADJ processed {i}/{len(adjs)} (kept={kept_a})")
            if a in ADJ_EXCLUDE_LEMMAS:
                missed_adjs.append(a)
                continue

            ev = classify_adj(oracle, a)
            # (IV) Backfill on classify failure / empty evidence.
            if (not ev) or (not ev.get("tags")) or (not ev.get("licenses")):
                ev = MANUAL_BACKFILL_ADJS.get(a) or ev

            # Keep iff it lands in any toinf class (tag) and has at least one VP-selecting license.
            if (not ev) or (not ev.get("tags")) or (not ev.get("licenses")):
                missed_adjs.append(a)
                continue

            kept_a += 1
            new_ent = {
                "lemma": a,
                "pos": "ADJ",
                "tags": ev["tags"],
                "primary_tag": ev["primary_tag"],
                "licenses": ev["licenses"],
                "frames": infer_adj_frames(ev["tags"], ev["licenses"]),
                "erg_types": ev.get("erg_types", []),
            }

            if a in adj_entries_by_lemma:
                adj_entries_by_lemma[a] = merge_lex_entries(adj_entries_by_lemma[a], new_ent)
            else:
                adj_entries_by_lemma[a] = new_ent

        # Light manual fixes for a handful of high-impact adjectival predicates where ERG
        # evidence can skew toward the "control" reading (and spaCy often uses them in
        # raising/expletive contexts).
        ADJ_MANUAL_PATCHES = {
            # "There/It was certain to ..." is often treated as raising in augmentation.
            "certain": {"add_tags": ["toinf:raising"], "add_licenses": ["it", "there"]},
            # "be supposed to" behaves like raising/expletive-OK in practice.
            "supposed": {"add_tags": ["toinf:raising"], "add_licenses": ["it", "there"]},
            # "unsure" is NOT raising - *"There is unsure to be a problem"
            # ERG over-generates here; fix to control only
            "unsure": {
                "set_tags": ["toinf:control"],
                "set_licenses": ["np"],
            },
        }
        for lemma, patch in ADJ_MANUAL_PATCHES.items():
            if lemma not in adj_entries_by_lemma:
                continue
            ent = adj_entries_by_lemma[lemma]
            
            # Handle set_* (replaces entirely)
            if "set_tags" in patch:
                ent["tags"] = sorted(set(patch["set_tags"]))
            else:
                # Handle add/remove
                tags = set(ent.get("tags", []))
                tags |= set(patch.get("add_tags", []))
                tags -= set(patch.get("remove_tags", []))
                ent["tags"] = sorted(tags)
            
            if "set_licenses" in patch:
                ent["licenses"] = sorted(set(patch["set_licenses"]))
            else:
                lics = set(ent.get("licenses", []))
                lics |= set(patch.get("add_licenses", []))
                lics -= set(patch.get("remove_licenses", []))
                ent["licenses"] = sorted(lics)
            
            ent["primary_tag"] = pick_primary_tag(ent.get("tags", []))
            ent["frames"] = infer_adj_frames(ent.get("tags", []), ent.get("licenses", []))

        out_adjs = sorted(adj_entries_by_lemma.values(), key=lambda d: d["lemma"])

    # Build VERB entries into a lemma->entry map so we can merge multiple "frames" for
    # polysemous lemmas (e.g., ECM + passive/participial raising).
    verb_entries_by_lemma: Dict[str, Dict[str, Any]] = {}
    kept_v = 0
    for i, v in enumerate(verbs):
        if i % 500 == 0:
            print(f"[PROGRESS] VERB processed {i}/{len(verbs)} (kept={kept_v})")
        if v in VERB_EXCLUDE_LEMMAS:
            missed_verbs.append(v)
            continue

        ev = classify_verb(oracle, v)
        # (IV) Backfill on classify failure / empty evidence.
        if (not ev) or (not ev.get("tags")) or (not ev.get("licenses")):
            ev = MANUAL_BACKFILL_VERBS.get(v) or ev

        if (not ev) or (not ev.get("tags")) or (not ev.get("licenses")):
            missed_verbs.append(v)
            continue

        kept_v += 1
        entry = {
            "lemma": v,
            "pos": "VERB",
            "tags": ev["tags"],
            "primary_tag": ev["primary_tag"],
            "licenses": ev["licenses"],
            "erg_types": ev.get("erg_types", []),
        }
        if v in verb_entries_by_lemma:
            verb_entries_by_lemma[v] = merge_lex_entries(verb_entries_by_lemma[v], entry)
        else:
            verb_entries_by_lemma[v] = entry

    # Option B (passive raising): add a passive-only raising tag for VBN predicates in
    # "be VBN to ..." contexts. This is derived from:
    #   1) participial ADJ entries that behave like raising predicates (e.g., bound), and
    #   2) ECM verbs with strong expletive evidence (e.g., suppose/believe/expect).
    add_shadow_verb_entries_from_participial_adjs(out_adjs, verb_entries_by_lemma, known_verbs)
    add_passive_raising_frames_from_ecm_verbs(out_adjs, verb_entries_by_lemma, known_verbs)

    # Finalize per-tag frame views for debug (augmenter ignores this field).
    for _ent in verb_entries_by_lemma.values():
        _ent["frames"] = infer_verb_frames(_ent.get("tags", []), _ent.get("licenses", []))

    out_verbs: List[Dict[str, Any]] = [
        verb_entries_by_lemma[k] for k in sorted(verb_entries_by_lemma.keys())
    ]

    if not verbs_only:
        write_jsonl(args.adj_out, out_adjs)
    write_jsonl(args.verb_out, out_verbs)

    if args.out_adj_missed and not verbs_only:
        with open(args.out_adj_missed, "w", encoding="utf-8") as f:
            for a in missed_adjs:
                f.write(a + "\n")
    if args.out_verb_missed:
        with open(args.out_verb_missed, "w", encoding="utf-8") as f:
            for v in missed_verbs:
                f.write(v + "\n")

    if verbs_only:
        print(f"[VERBS-ONLY] Using {kept_a} adjectives from {args.adj_lexicon}")
        print(f"[DONE] verbs kept {len(out_verbs)}/{len(verbs)} -> {args.verb_out}")
    else:
        print(f"[DONE] adjectives kept {kept_a}/{len(adjs)} -> {args.adj_out}")
        print(f"[DONE] verbs kept {len(out_verbs)}/{len(verbs)} -> {args.verb_out}")
    print(oracle.stats())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
