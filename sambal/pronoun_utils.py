"""
Pronoun utilities for sambal.

Contains pronoun pools, gender mappings, and classification functions
used for pronoun replacement and gender-aware processing.
"""
from __future__ import annotations

from typing import Optional

from spacy.tokens import Token


# ---- Pronoun Pools by Case/Function ----

PRON_POOLS = {
    "SUBJ": ["i", "you", "he", "she", "it", "we", "they"],
    "OBJ": ["me", "you", "him", "her", "it", "us", "them"],
    "POSSDET": ["my", "your", "his", "her", "its", "our", "their"],
    "POSSPRON": ["mine", "yours", "his", "hers", "ours", "theirs"],
    "REFL": ["myself", "yourself", "himself", "herself", "itself", "ourselves", "yourselves", "themselves"],
}

# Keep expletives/wh/dummy fixed
PRON_LOCKED = {"there", "it", "what", "which", "who", "whom", "whose", "whoever", "whomever", "whatever", "whichever"}


# ----------------------- Gender-aware helpers -----------------------
# We use a simple tri-valued gender label:
#   "m" = masculine, "f" = feminine, "n" = gender-neutral/unspecified.
# This matches `ecmonsen/gendered_words` (m/f/n/o); we ignore "o".

GENDER_TAGS = {"m", "f", "n"}

PRON_GENDER = {
    # masculine
    "he": "m", "him": "m", "his": "m", "himself": "m",
    # feminine
    "she": "f", "her": "f", "hers": "f", "herself": "f",
    # neutral / unspecified (including 1st/2nd person)
    "they": "n", "them": "n", "their": "n", "theirs": "n", "themselves": "n",
    "i": "n", "me": "n", "my": "n", "mine": "n", "myself": "n",
    "you": "n", "your": "n", "yours": "n", "yourself": "n", "yourselves": "n",
    "we": "n", "us": "n", "our": "n", "ours": "n", "ourselves": "n",
    "it": "n", "its": "n", "itself": "n",
}


def pronoun_gender(form: str) -> Optional[str]:
    """Get the gender tag for a pronoun form.

    Returns:
        "m" for masculine, "f" for feminine, "n" for neutral, or None if not a pronoun.
    """
    if not form:
        return None
    return PRON_GENDER.get(form.lower())


def pron_class(tok: Token) -> Optional[str]:
    """Classify a pronoun token by its grammatical function.

    Returns one of: "SUBJ", "OBJ", "POSSDET", "POSSPRON", "REFL", or None.
    """
    if tok.lemma_.lower() in PRON_LOCKED:
        return None
    if tok.morph.get("Reflex") == ["Yes"]:
        return "REFL"
    case = tok.morph.get("Case")
    poss = tok.morph.get("Poss")
    if poss == ["Yes"]:
        # possessive pronoun vs determiner via POS tag
        return "POSSPRON" if tok.tag_ in {"PRP$"} and tok.dep_ in {"attr", "nsubj", "dobj", "obj"} else "POSSDET"
    if case == ["Acc"] or tok.dep_ in {"dobj", "obj", "iobj"}:
        return "OBJ"
    if tok.dep_ in {"nsubj", "nsubjpass", "csubj"} or case == ["Nom"]:
        return "SUBJ"
    if tok.dep_ in {"dobj", "obj", "iobj", "pobj"}:
        return "OBJ"
    return None


