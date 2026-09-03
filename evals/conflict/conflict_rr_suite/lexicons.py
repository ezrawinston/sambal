"""Curated micro-frames for *human-obvious* semantic plausibility.

We avoid PMI-driven noun selection because:
  - small corpora (10M tokens) produce sparse/noisy PMI
  - heuristic 'noun candidates' can be POS-noisy (e.g., 'other', 'same', 'use')
  - plausibility needs to be *obvious to humans* to support the ICML 2026 paper's argument

Instead, we define small semantic frames where the sentence
  'The SUBJECT VERB the OBJECT'
is strongly plausible for many SUBJECT/OBJECT fillers, while the role-reversed
version is strongly implausible.

The baseline LM is still used as the final plausibility oracle; these frames just
ensure the lure is interpretable.

All words are lower-case and intended to be single-token-ish in typical BPEs, but
the suite does not *require* single-token words unless you enable that option.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Sequence

ObjKind = Literal["inanimate", "animate"]


@dataclass(frozen=True)
class VerbForms:
    pl: str     # plural/base present: "eat"
    sg3: str    # 3sg present: "eats"
    past: str   # past: "ate" / "chased"
    pp: str     # past participle: "eaten" / "chased" (for passive family)


@dataclass(frozen=True)
class Frame:
    name: str
    verbs: VerbForms
    subj_pl: Sequence[str]   # plausible plural subjects
    obj_sg: Sequence[str]    # plausible singular objects
    obj_kind: ObjKind


FRAMES: List[Frame] = [
    Frame(
        name="eat_food",
        verbs=VerbForms(pl="eat", sg3="eats", past="ate", pp="eaten"),
        subj_pl=["children","kids","boys","girls","people","dogs","cats","wolves","lions","students"],
        obj_sg=["cake","pizza","sandwich","apple","meat","fish","bread","cookie","carrot"],
        obj_kind="inanimate",
    ),
    Frame(
        name="read_text",
        verbs=VerbForms(pl="read", sg3="reads", past="read", pp="read"),
        subj_pl=["students","kids","children","teachers","readers","parents"],
        obj_sg=["book","story","newspaper","letter","email","magazine"],
        obj_kind="inanimate",
    ),
    Frame(
        name="drive_vehicle",
        verbs=VerbForms(pl="drive", sg3="drives", past="drove", pp="driven"),
        subj_pl=["drivers","tourists","workers","people","students"],
        obj_sg=["car","truck","bus","taxi","van"],
        obj_kind="inanimate",
    ),
    Frame(
        name="fix_device",
        verbs=VerbForms(pl="fix", sg3="fixes", past="fixed", pp="fixed"),
        subj_pl=["mechanics","workers","engineers","technicians","students"],
        obj_sg=["car","engine","computer","phone","machine"],
        obj_kind="inanimate",
    ),
    Frame(
        name="paint_surface",
        verbs=VerbForms(pl="paint", sg3="paints", past="painted", pp="painted"),
        subj_pl=["artists","painters","kids","students","workers"],
        obj_sg=["wall","house","picture","fence","door"],
        obj_kind="inanimate",
    ),
    Frame(
        name="kick_ball",
        verbs=VerbForms(pl="kick", sg3="kicks", past="kicked", pp="kicked"),
        subj_pl=["players","kids","boys","girls","soldiers"],
        obj_sg=["ball","door","rock"],
        obj_kind="inanimate",
    ),
    Frame(
        name="carry_object",
        verbs=VerbForms(pl="carry", sg3="carries", past="carried", pp="carried"),
        subj_pl=["workers","students","people","porters","children"],
        obj_sg=["box","bag","book","chair","table"],
        obj_kind="inanimate",
    ),
    # Optional: more human-object frames (less "inanimate" but still clear)
    Frame(
        name="chase_prey",
        verbs=VerbForms(pl="chase", sg3="chases", past="chased", pp="chased"),
        subj_pl=["dogs","cats","wolves","lions","hunters","kids","police"],
        obj_sg=["cat","mouse","rabbit","thief","deer"],
        obj_kind="animate",
    ),
]


# Determiner pairs for the determiner-number family:
#   - det_sg: singular-only (hard grammatical constraint)
#   - det_flex: OK with singular/plural (keeps only one mismatch under swap)
DET_SG_OPTIONS = ["a", "each"]
DET_FLEX = "some"
