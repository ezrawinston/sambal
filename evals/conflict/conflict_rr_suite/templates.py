"""Template builders for conflict + control minimal pairs.

All *conflict* pairs are swap-only:
  good = grammatical but implausible (role reversed)
  bad  = ungrammatical but plausible

Additionally, each item includes a *control* pair to ensure the grammatical phenomenon
is easy in a non-adversarial setting (plausible semantics):
  control_good = grammatical & plausible
  control_bad  = the same ungrammatical sentence as conflict_bad

This is critical for your use-case: you observed that some phenomena were too hard
even without the lure. Controls let you filter those out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .lexicons import Frame


@dataclass(frozen=True)
class ItemSentences:
    family: str
    sentence_good: str
    sentence_bad: str
    control_good: str
    control_bad: str
    meta: Dict


def _cap(s: str) -> str:
    # capitalize first character
    if not s:
        return s
    return s[0].upper() + s[1:]


def sva_role_reversal(frame: Frame, A_pl: str, B_sg: str) -> ItemSentences:
    """Family: subject–verb agreement, induced by role reversal.

    conflict_good (grammatical, implausible):  The B_sg V_sg3 the A_pl.
    conflict_bad  (ungrammatical, plausible):  The A_pl V_sg3 the B_sg.

    control_good  (grammatical, plausible):    The A_pl V_pl  the B_sg.
    control_bad   = conflict_bad
    """
    V3 = frame.verbs.sg3
    Vp = frame.verbs.pl

    good = _cap(f"the {B_sg} {V3} the {A_pl}.")
    bad = _cap(f"the {A_pl} {V3} the {B_sg}.")
    ctrl_good = _cap(f"the {A_pl} {Vp} the {B_sg}.")
    ctrl_bad = bad

    meta = {
        "frame": frame.name,
        "A_pl": A_pl,
        "B_sg": B_sg,
        "verb_pl": Vp,
        "verb_sg3": V3,
    }
    return ItemSentences(
        family="sva_role_reversal",
        sentence_good=good,
        sentence_bad=bad,
        control_good=ctrl_good,
        control_bad=ctrl_bad,
        meta=meta,
    )


def detnum_role_reversal(frame: Frame, A_pl: str, B_sg: str, det_sg: str = "a", det_flex: str = "some") -> ItemSentences:
    """Family: determiner–noun number agreement, induced by role reversal.

    We use an *asymmetric determiner pair*:
      det_sg  (singular-only): 'a' or 'each'
      det_flex (sing/pl OK):   'some'

    conflict_good (grammatical, implausible):  A/Each B_sg V_past some A_pl.
    conflict_bad  (ungrammatical, plausible):  A/Each A_pl V_past some B_sg.
      (only one agreement violation: det_sg + plural)

    control_good  (grammatical, plausible):    Some A_pl V_past some B_sg.
    control_bad   = conflict_bad
    """
    Vpast = frame.verbs.past

    # Use 'an' if det_sg=='a' and B_sg begins with vowel sound (simple heuristic)
    det_sg_form = det_sg
    if det_sg == "a" and B_sg[:1] in "aeiou":
        det_sg_form = "an"

    good = _cap(f"{det_sg_form} {B_sg} {Vpast} {det_flex} {A_pl}.")
    bad = _cap(f"{det_sg_form} {A_pl} {Vpast} {det_flex} {B_sg}.")
    ctrl_good = _cap(f"{det_flex} {A_pl} {Vpast} {det_flex} {B_sg}.")
    ctrl_bad = bad

    meta = {
        "frame": frame.name,
        "A_pl": A_pl,
        "B_sg": B_sg,
        "verb_past": Vpast,
        "det_sg": det_sg_form,
        "det_flex": det_flex,
    }
    return ItemSentences(
        family="detnum_role_reversal",
        sentence_good=good,
        sentence_bad=bad,
        control_good=ctrl_good,
        control_bad=ctrl_bad,
        meta=meta,
    )


def passive_role_reversal(frame: Frame, A_pl: str, B_sg: str) -> ItemSentences:
    """Family: passive voice + auxiliary agreement, induced by role reversal.

    This is a very *easy* grammatical phenomenon (auxiliary agreement) that both
    of your models handle well (BLiMP passive suites are high), while still
    allowing a strong, human-evident semantic lure via role reversal.

    We keep the structure extremely local:
      "The X were V_pp by the Y."

    conflict_good (grammatical, implausible):  The A_pl were V_pp by the B_sg.
      - grammatical because A_pl is plural (matches 'were')
      - implausible because A_pl becomes the *patient* and B_sg becomes the *agent*

    conflict_bad  (ungrammatical, plausible):  The B_sg were V_pp by the A_pl.
      - ungrammatical because B_sg is singular but uses 'were'
      - plausible because B_sg is the *patient* and A_pl is the *agent*

    control_good  (grammatical, plausible):    The B_sg was V_pp by the A_pl.
    control_bad   = conflict_bad
    """
    Vpp = frame.verbs.pp

    good = _cap(f"the {A_pl} were {Vpp} by the {B_sg}.")
    bad = _cap(f"the {B_sg} were {Vpp} by the {A_pl}.")
    ctrl_good = _cap(f"the {B_sg} was {Vpp} by the {A_pl}.")
    ctrl_bad = bad

    meta = {
        "frame": frame.name,
        "A_pl": A_pl,
        "B_sg": B_sg,
        "verb_pp": Vpp,
        "aux_conflict": "were",
        "aux_control": "was",
    }

    return ItemSentences(
        family="passive_role_reversal",
        sentence_good=good,
        sentence_bad=bad,
        control_good=ctrl_good,
        control_bad=ctrl_bad,
        meta=meta,
    )
