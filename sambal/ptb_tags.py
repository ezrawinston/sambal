"""
PTB (Penn Treebank) tag inference functions.

These functions infer appropriate PTB tags for tokens based on morphology
and syntactic context. Extracted from Augmenter to reduce class complexity.
"""
from __future__ import annotations
from typing import Optional, Literal
from spacy.tokens import Token


# =============================================================================
# Tag Constants
# =============================================================================

VERB_TAGS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
NOUN_TAGS = {"NN", "NNS", "NNP", "NNPS"}
ADJ_TAGS = {"JJ", "JJR", "JJS"}
ADV_TAGS = {"RB", "RBR", "RBS"}


# =============================================================================
# Degree-Based Tag Inference (Parameterized for ADJ/ADV)
# =============================================================================

def _ptb_tag_by_degree(
    tok: Token,
    existing_tags: set,
    base_tag: str,
    comparative_tag: str,
    superlative_tag: str
) -> str:
    """
    Infer PTB tag based on morphological Degree feature.

    This is the parameterized version that works for both ADJ and ADV,
    eliminating the code duplication between _ptb_tag_for_adj and _ptb_tag_for_adv.

    Args:
        tok: The token to analyze
        existing_tags: Valid existing tags to preserve (e.g., {"JJ", "JJR", "JJS"})
        base_tag: Tag for positive degree (e.g., "JJ" or "RB")
        comparative_tag: Tag for comparative degree (e.g., "JJR" or "RBR")
        superlative_tag: Tag for superlative degree (e.g., "JJS" or "RBS")

    Returns:
        The appropriate PTB tag
    """
    if tok.tag_ in existing_tags:
        return tok.tag_

    deg = tok.morph.get("Degree")
    if deg == ["Cmp"]:
        return comparative_tag
    if deg == ["Sup"]:
        return superlative_tag
    return base_tag


# =============================================================================
# POS-Specific Tag Functions
# =============================================================================

def ptb_tag_for_adj(tok: Token) -> str:
    """
    Infer PTB adjective tag (JJ/JJR/JJS).

    Preserves existing valid tags, otherwise infers from morphological Degree.
    """
    return _ptb_tag_by_degree(
        tok,
        existing_tags=ADJ_TAGS,
        base_tag="JJ",
        comparative_tag="JJR",
        superlative_tag="JJS"
    )


def ptb_tag_for_adv(tok: Token) -> str:
    """
    Infer PTB adverb tag (RB/RBR/RBS).

    Preserves existing valid tags, otherwise infers from morphological Degree.
    """
    return _ptb_tag_by_degree(
        tok,
        existing_tags=ADV_TAGS,
        base_tag="RB",
        comparative_tag="RBR",
        superlative_tag="RBS"
    )


def ptb_tag_for_verb(tok: Token) -> str:
    """
    Infer PTB verb tag (VB/VBD/VBG/VBN/VBP/VBZ).

    Uses local AUX/COP + complement cues for structure-driven decisions,
    falling back to morphological features.
    """
    if tok.tag_ in VERB_TAGS:
        return tok.tag_

    morph = tok.morph
    vform = set(morph.get("VerbForm"))
    tense = set(morph.get("Tense"))
    aspect = set(morph.get("Aspect"))
    txt = tok.text.lower()

    # Children by dep
    auxes = [ch for ch in tok.children if ch.dep_ in {"aux", "auxpass"}]
    cops = [ch for ch in tok.children if ch.dep_ == "cop"]

    has_modal = any(a.tag_ == "MD" for a in auxes)
    has_do = any(a.lemma_.lower() == "do" for a in auxes)
    has_have = any(a.lemma_.lower() == "have" for a in auxes)
    has_be_aux = any(a.lemma_.lower() == "be" for a in auxes)
    has_pass = any(a.dep_ == "auxpass" for a in auxes)

    has_be_cop = any(c.lemma_.lower() == "be" for c in cops)

    # Object/complements and RC are strong signals of verbhood
    comp_deps = {"obj", "dobj", "ccomp", "xcomp", "obl", "pobj"}
    has_obj_or_comp = any(ch.dep_ in comp_deps for ch in tok.children)
    in_relcl = ("relcl" in tok.dep_.lower())

    # Treat 'be' as progressive even if mislabeled as 'cop' ONLY when there is evidence
    # that this is a verbal predicate (object/complement or it's a relcl) AND it looks participial
    looks_participial = txt.endswith("ing") or ("Part" in vform) or ("Prog" in aspect)
    be_progressive = has_be_aux or (has_be_cop and (has_obj_or_comp or in_relcl) and looks_participial)

    # ---- Strong, structure-driven decisions ----
    if has_modal or has_do:
        return "VB"  # could/do/does/did + VB
    if has_pass or (has_have and not be_progressive):
        return "VBN"  # passive, or perfect (no progressive)
    if be_progressive:
        return "VBG"  # be + -ing (true progressive)

    # ---- Morphological fallbacks ----
    if "Part" in vform:
        return "VBG" if ("Prog" in aspect or txt.endswith("ing")) else "VBN"
    if "Past" in tense:
        return "VBD"
    if "Inf" in vform:
        return "VB"

    # ---- Default finite present ----
    is_3sg = (tok.morph.get("Number") == ["Sing"] and tok.morph.get("Person") == ["3"])
    return "VBZ" if is_3sg else "VBP"


def ptb_tag_for_noun(tok: Token, guess_number_fn=None) -> str:
    """
    Infer PTB noun tag (NN/NNS).

    Preserves existing valid tags. For unknown tags, uses the provided
    number-guessing function or defaults to NN.

    Args:
        tok: The token to analyze
        guess_number_fn: Optional function(tok) -> "Sing"|"Plur" for number guessing

    Returns:
        NN or NNS tag
    """
    if tok.tag_ in {"NN", "NNS", "NNP", "NNPS"}:
        return tok.tag_

    if guess_number_fn is not None:
        return "NNS" if guess_number_fn(tok) == "Plur" else "NN"

    return "NN"


# =============================================================================
# Number Forcing by Context
# =============================================================================

# Determiners/quantifiers that force singular
_SINGULAR_FORCERS = {"a", "an", "each", "every", "another", "either", "neither", "this", "that"}

# Determiners that force plural (by form)
_PLURAL_DET_FORMS = {"these", "those"}

# Adjectival modifiers that force plural
_PLURAL_AMOD_LEMMAS = {"many", "several", "few", "numerous", "multiple", "various"}


def force_noun_number_by_context(tok: Token, tag: Optional[str]) -> Optional[str]:
    """
    Force noun number (NN/NNS) based on local determiners/quantifiers.

    This overrides the PTB tag when there's unambiguous local evidence
    from determiners (a/an/these/those), quantifiers (many/several),
    or numerals (one vs other numbers).

    Args:
        tok: The noun token
        tag: The current PTB tag (NN or NNS)

    Returns:
        The (possibly overridden) tag
    """
    if tag not in {"NN", "NNS"}:
        return tag  # only common nouns

    dets = [c for c in tok.children if c.dep_ == "det"]
    det_lemmas = {d.lemma_.lower() for d in dets}
    det_forms = {d.text.lower() for d in dets}
    det_nums = set().union(*(set(d.morph.get("Number")) for d in dets))

    amod_lemmas = {c.lemma_.lower() for c in tok.children if c.dep_ == "amod"}
    nummods = [c for c in tok.children if c.dep_ == "nummod"]

    force_pl = (
        bool(_PLURAL_DET_FORMS & det_forms) or
        bool(_PLURAL_DET_FORMS & det_lemmas) or
        ("Plur" in det_nums) or
        bool(_PLURAL_AMOD_LEMMAS & amod_lemmas)
    )

    force_sg = bool(
        _SINGULAR_FORCERS & (det_forms | det_lemmas)
    ) or ("Sing" in det_nums)

    if nummods:
        is_one = any(n.lemma_.lower() == "one" or (n.like_num and n.text.strip() == "1") for n in nummods)
        if is_one:
            force_sg = True
        else:
            force_pl = True

    if force_pl and not force_sg:
        return "NNS"
    if force_sg and not force_pl:
        return "NN"
    return tag
