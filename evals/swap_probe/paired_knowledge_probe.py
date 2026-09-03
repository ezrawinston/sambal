#!/usr/bin/env python3
"""paired_knowledge_probe.py

Frequency-controlled "no world knowledge" probe, extended with a grammar swap control.

What this script measures
-------------------------
For each 2x2 item (x1->y1 and x2->y2), we build:

(1) Knowledge swap (relational / semantic):
      GOOD:    template(x1,y1,x2,y2)
      SWAPPED: template(x1,y2,x2,y1)

    Margin: ΔK = logP(GOOD) - logP(SWAPPED)

    Both candidates use the same words, paired differently, which largely cancels
    unigram-frequency confounds.

(2) Grammar swap (structural / syntactic), for items where grammar_swap=True:
      GRAMMAR_GOOD: template(x1,y1,x2,y2)
      GRAMMAR_BAD:  template(y1,x1,y2,x2)   # swap x<->y within each clause

    Margin: ΔG = logP(GRAMMAR_GOOD) - logP(GRAMMAR_BAD)

Grammar modes (--grammar_mode)
------------------------------
  auto   (default): if --counterbalance_order then 'sem2' else 'primary'
  primary: compare only (x1,y1,x2,y2) vs (y1,x1,y2,x2)
  sem2   : average across the two semantics-good clause orders:
           (x1,y1,x2,y2) and (x2,y2,x1,y1), each vs its grammar-bad.
  all4   : average across all four grammatical arrangements:
           (x1,y1,x2,y2), (x1,y2,x2,y1), (x2,y2,x1,y1), (x2,y1,x1,y2),
           each vs its grammar-bad created by swapping x<->y within each clause.
  nonsense_gram: average of 2 nonsense-but-grammatical arrangements:
           (x1,y2,x2,y1) and (x2,y1,x1,y2)
           vs average of 4 ungrammatical arrangements:
           (y1,x1,y2,x1), (y1,x2,y2,x2), (y2,x1,y1,x1), (y2,x2,y1,x2)
  none   : skip grammar evaluation entirely.

Scoring
-------
A sentence's score is the sum of the backend's per-token log-probabilities over the
scored positions, so the scoring rule is the backend's.

By default we score only "critical" tokens (ΔK: y1,y2; ΔG: x1,x2,y1,y2).
Use --score_all_tokens to score the entire sentence.

Outputs
-------
- Prints a table with ΔK and (if applicable) ΔG for sambal and normal models.
- Writes optional JSON/CSV.
- Optional matplotlib plots for both ΔK and ΔG distributions (+ scatter ΔK vs ΔG).
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
from collections import Counter
from matplotlib import pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from evals.backends import LanguageModel  # noqa: E402
from evals.common import add_backend_arguments, backend_from_args  # noqa: E402


plt.rcParams.update({
    # "text.usetex": True,            # Use LaTeX to render text
    "font.family": "serif",         # Use serif fonts (like Computer Modern)
    # "font.serif": ["Computer Modern Roman"],
    "font.size": 7,                # Match your document's font size (usually 10pt or 11pt)
    "axes.labelsize": 7,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
        # DATA LINES & MARKERS
    "lines.linewidth": 1.0,  # Thickness of the plot lines
    "lines.markersize": 4.0,  # Size of the markers (dots, squares, etc.)

    # THE BOX AROUND THE CHART (Spines)
    "axes.linewidth": 1,  # Thickness of the 4 edges (spines)

    # TICK MARKS
    "xtick.major.width": 1,  # Thickness of X-axis major ticks
    "ytick.major.width": 1,  # Thickness of Y-axis major ticks
    "xtick.minor.width": 0.8,  # Thickness of X-axis minor ticks (if used)
    "ytick.minor.width": 0.8,  # Thickness of Y-axis minor ticks (if used)
    "figure.figsize": (3.5, 2.5)    # Default size (width, height) in inches
})
# -----------------------------
# Data: counterbalanced 2x2 items
# -----------------------------


@dataclass(frozen=True)
class AssocItem:
    knowledge_type: str
    template: str
    x1: str
    y1: str
    x2: str
    y2: str
    note: str = ""
    grammar_swap: bool = False

    def fmt(self, x1: str, y1: str, x2: str, y2: str) -> str:
        return self.template.format(x1=x1, y1=y1, x2=x2, y2=y2)

    # --- Knowledge (semantic) swap ---
    def good(self) -> str:
        return self.fmt(self.x1, self.y1, self.x2, self.y2)

    def swapped(self) -> str:
        # Swap y1 and y2 (the association targets)
        return self.fmt(self.x1, self.y2, self.x2, self.y1)

    # --- Clause-order counterbalance for knowledge ---
    def good_order_swapped(self) -> str:
        # Reverse clause order: (x2,y2) then (x1,y1)
        return self.fmt(self.x2, self.y2, self.x1, self.y1)

    def swapped_order_swapped(self) -> str:
        # Reverse clause order, then swap y within that order: template(x2,y1,x1,y2)
        return self.fmt(self.x2, self.y1, self.x1, self.y2)


@dataclass(frozen=True)
class SinglePairItem:
    """Single-pair item with just one x-y association to swap."""
    knowledge_type: str
    template: str
    x: str
    y: str
    note: str = ""

    def fmt(self, x: str, y: str) -> str:
        return self.template.format(x=x, y=y)

    def orig(self) -> str:
        """Original sentence with correct x-y pairing."""
        return self.fmt(self.x, self.y)

    def swapped(self) -> str:
        """Swapped sentence with x and y reversed."""
        return self.fmt(self.y, self.x)


def make_items() -> List[AssocItem]:
    """Starter set.

    Grammar swap is only enabled for the 10 VerbObj items (no additional VerbObj examples
    beyond those 10).
    """
    items: List[AssocItem] = []

    # # Clothing / body part
    # wear_t = "You wear a {x1} on your {y1} and you wear a {x2} on your {y2}"
    # items += [
    #     AssocItem("Clothing", wear_t, "hat", "head", "shoe", "foot"),
    #     AssocItem("Clothing", wear_t, "glove", "hand", "ring", "finger"),
    #     AssocItem("Clothing", wear_t, "watch", "wrist", "belt", "waist"),
    #     AssocItem("Clothing", wear_t, "necklace", "neck", "socks", "feet"),
    # ]
    #
    # # Tool affordances (selectional preference)
    # tool_t = "To {x1} you use the {y1} and to {x2} you use the {y2}"
    # items += [
    #     AssocItem("Tools", tool_t, "cut paper", "scissors", "hammer nails", "hammer"),
    #     AssocItem("Tools", tool_t, "write", "pen", "paint", "brush"),
    #     AssocItem("Tools", tool_t, "sweep", "broom", "dig", "shovel"),
    #     AssocItem("Tools", tool_t, "unlock a door", "key", "drink soup", "spoon"),
    # ]
    #
    # # Storage / location
    # store_t = "You store {x1} in the {y1} and you store {x2} in the {y2}"
    # items += [
    #     AssocItem("Storage", store_t, "milk", "fridge", "clothes", "closet"),
    #     AssocItem("Storage", store_t, "books", "shelf", "trash", "bin"),
    #     AssocItem("Storage", store_t, "cars", "garage", "money", "bank"),
    # ]
    #
    # # Material properties
    # prop_t = "The material {x1} is {y1} and the material {x2} is {y2}"
    # items += [
    #     AssocItem("MaterialProps", prop_t, "glass", "brittle", "rubber", "elastic"),
    #     AssocItem("MaterialProps", prop_t, "iron", "magnetic", "paper", "flammable"),
    #     AssocItem("MaterialProps", prop_t, "stone", "hard", "cotton", "soft"),
    # ]
    #
    # # Typical colors / temperature
    # col_t = "Because {x1} is {y1} and {x2} is {y2}"
    # items += [
    #     AssocItem("Colors", col_t, "snow", "white", "grass", "green"),
    #     AssocItem("Temp", col_t, "fire", "hot", "ice", "cold"),
    # ]

    # --- Grammar-swap minimal sentences (10 items) ---
    g_t = "The boy {x1} the {y1} and {x2} the {y2}."
    items += [
        AssocItem("1", g_t, "sang", "song", "ate", "bread", grammar_swap=True),
        AssocItem("2", g_t, "drank", "coffee", "wore", "hat", grammar_swap=True),
        AssocItem("3", g_t, "flew", "plane", "played", "game", grammar_swap=True),
        AssocItem("4", g_t, "played", "piano", "read", "books", grammar_swap=True),
        AssocItem("5", g_t, "wrote", "letter", "ate", "bagel", grammar_swap=True),
        AssocItem("6", g_t, "ran", "race", "opened", "door", grammar_swap=True),
        AssocItem("7", g_t, "kicked", "ball", "rode", "bike", grammar_swap=True),
        AssocItem("8", g_t, "climbs", "mountain", "breaks", "glass", grammar_swap=True),
        AssocItem("9", g_t, "sails", "boat", "catches", "mouse", grammar_swap=True),
        AssocItem("10", g_t, "wears", "shoes", "cleans", "room", grammar_swap=True),
        AssocItem("111", g_t, "drives", "car", "plays", "music", grammar_swap=True),
        AssocItem("112", g_t, "takes", "bus", "sees", "movie", grammar_swap=True),
        AssocItem("113", g_t, "changes", "shirt", "tells", "joke", grammar_swap=True),
        AssocItem("114", g_t, "opens", "window", "drinks", "coffee", grammar_swap=True),
        AssocItem("115", g_t, "drinks", "water", "opens", "bag", grammar_swap=True),
        AssocItem("116", g_t, "writes", "report", "asks", "question", grammar_swap=True),
        AssocItem("117", g_t, "starts", "engine", "says", "word", grammar_swap=True),
        AssocItem("118", g_t, "breaks", "window", "drinks", "milk", grammar_swap=True),
        AssocItem("119", g_t, "pays", "bill", "turns", "key", grammar_swap=True),
        AssocItem("121", g_t, "makes", "dinner", "tells", "story", grammar_swap=True),
        AssocItem("122", g_t, "drinks", "tea", "writes", "note", grammar_swap=True),
        AssocItem("123", g_t, "bakes", "cake", "takes", "plane", grammar_swap=True),
        AssocItem("125", g_t, "opens", "door", "stops", "car", grammar_swap=True),
        AssocItem("126", g_t, "breaks", "window", "gives", "answer", grammar_swap=True),
        AssocItem("127", g_t, "builds", "house", "says", "word", grammar_swap=True),
        AssocItem("128", g_t, "pays", "bill", "sees", "movie", grammar_swap=True),
        AssocItem("130", g_t, "drives", "bus", "asks", "question", grammar_swap=True),
        AssocItem("131", g_t, "beats", "game", "drinks", "water", grammar_swap=True),
        AssocItem("132", g_t, "runs", "race", "turns", "key", grammar_swap=True),
        AssocItem("133", g_t, "opens", "box", "takes", "bath", grammar_swap=True),
        AssocItem("134", g_t, "drinks", "milk", "wins", "game", grammar_swap=True),
        AssocItem("135", g_t, "writes", "letter", "makes", "bed", grammar_swap=True),
        AssocItem("136", g_t, "throws", "ball", "stops", "car", grammar_swap=True),
        AssocItem("137", g_t, "breaks", "glass", "puts", "money", grammar_swap=True),
        AssocItem("138", g_t, "pays", "bill", "gives", "answer", grammar_swap=True),
        AssocItem("139", g_t, "drives", "bus", "changes", "shirt", grammar_swap=True),
        AssocItem("140", g_t, "sees", "movie", "tells", "story", grammar_swap=True),
        AssocItem("142", g_t, "turns", "key", "asks", "question", grammar_swap=True),
        AssocItem("143", g_t, "starts", "day", "opens", "bottle", grammar_swap=True),
        AssocItem("144", g_t, "opens", "bottle", "takes", "flight", grammar_swap=True),
        AssocItem("145", g_t, "runs", "race", "writes", "name", grammar_swap=True),
        AssocItem("146", g_t, "drinks", "water", "plays", "music", grammar_swap=True),
        AssocItem("147", g_t, "breaks", "window", "makes", "dinner", grammar_swap=True),
        AssocItem("148", g_t, "cut", "cake", "said", "word", grammar_swap=True),
        AssocItem("149", g_t, "gives", "answer", "stops", "car", grammar_swap=True),
        AssocItem("152", g_t, "wore", "coat", "bought", "food", grammar_swap=True),
        AssocItem("153", g_t, "read", "paper", "met", "teacher", grammar_swap=True),
        AssocItem("154", g_t, "took", "train", "sent", "message", grammar_swap=True),
        AssocItem("155", g_t, "ate", "meat", "kept", "secret", grammar_swap=True),
        AssocItem("156", g_t, "sold", "book", "cut", "fruit", grammar_swap=True),
        AssocItem("120", g_t, "told", "story", "ran", "race", grammar_swap=True),
        AssocItem("157", g_t, "ate", "apple", "met", "friend", grammar_swap=True),
        AssocItem("158", g_t, "wrote", "note", "made", "coffee", grammar_swap=True),
        AssocItem("159", g_t, "opened", "door", "ate", "bread", grammar_swap=True),
        AssocItem("160", g_t, "locked", "door", "bought", "ticket", grammar_swap=True),
        AssocItem("162", g_t, "read", "news", "carried", "bag", grammar_swap=True),
        AssocItem("163", g_t, "took", "flight", "left", "home", grammar_swap=True),
        AssocItem("164", g_t, "opened", "bottle", "sent", "message", grammar_swap=True),
        AssocItem("165", g_t, "wrote", "report", "broke", "glass", grammar_swap=True),
        AssocItem("166", g_t, "built", "house", "ate", "cake", grammar_swap=True),
        AssocItem("167", g_t, "kept", "secret", "fixed", "door", grammar_swap=True),
        AssocItem("169", g_t, "ate", "fish", "held", "hand", grammar_swap=True),
        AssocItem("170", g_t, "closed", "box", "broke", "glass", grammar_swap=True),
        AssocItem("171", g_t, "wrote", "name", "shut", "window", grammar_swap=True),
        AssocItem("172", g_t, "sent", "letter", "wore", "shoes", grammar_swap=True),
        AssocItem("173", g_t, "read", "paper", "carried", "bag", grammar_swap=True),
        AssocItem("174", g_t, "turned", "key", "met", "friend", grammar_swap=True),
        AssocItem("175", g_t, "said", "word", "started", "car", grammar_swap=True),
        AssocItem("176", g_t, "wore", "pants", "kicked", "ball", grammar_swap=True),
        AssocItem("177", g_t, "paid", "bill", "told", "story", grammar_swap=True),
        AssocItem("178", g_t, "changed", "shirt", "drove", "bus", grammar_swap=True),
        AssocItem("179", g_t, "ate", "rice", "spoke", "truth", grammar_swap=True),
        AssocItem("180", g_t, "dug", "hole", "climbed", "ladder", grammar_swap=True),
        AssocItem("181", g_t, "gave", "answer", "built", "house", grammar_swap=True),
        AssocItem("182", g_t, "opened", "bottle", "ate", "orange", grammar_swap=True),
        AssocItem("183", g_t, "sold", "book", "kept", "secret", grammar_swap=True),
        AssocItem("184", g_t, "bought", "ticket", "ate", "bread", grammar_swap=True),
        AssocItem("185", g_t, "shut", "door", "met", "teacher", grammar_swap=True),
        AssocItem("186", g_t, "closed", "window", "paid", "bill", grammar_swap=True),
        AssocItem("187", g_t, "took", "train", "wrote", "report", grammar_swap=True),
        AssocItem("188", g_t, "read", "news", "broke", "window", grammar_swap=True),
        AssocItem("189", g_t, "made", "bed", "ate", "rice", grammar_swap=True),
        AssocItem("190", g_t, "closed", "lid", "did", "work", grammar_swap=True),
        AssocItem("191", g_t, "sent", "message", "built", "house", grammar_swap=True),
        AssocItem("192", g_t, "sold", "book", "broke", "glass", grammar_swap=True),
        AssocItem("193", g_t, "climbed", "stairs", "lost", "money", grammar_swap=True),
        AssocItem("194", g_t, "wore", "hat", "stopped", "car", grammar_swap=True),
        AssocItem("195", g_t, "ate", "egg", "found", "key", grammar_swap=True),
        AssocItem("196", g_t, "changed", "shirt", "watched", "movie", grammar_swap=True),
        AssocItem("198", g_t, "drove", "bus", "played", "game", grammar_swap=True),
        AssocItem("199", g_t, "ran", "race", "fixed", "door", grammar_swap=True),
        AssocItem("201", g_t, "took", "flight", "said", "word", grammar_swap=True),
        AssocItem("202", g_t, "wrote", "note", "locked", "door", grammar_swap=True),
        AssocItem("203", g_t, "read", "paper", "asked", "question", grammar_swap=True),
        AssocItem("204", g_t, "broke", "glass", "left", "home", grammar_swap=True),
        AssocItem("205", g_t, "made", "dinner", "kept", "secret", grammar_swap=True),
        AssocItem("206", g_t, "bought", "ticket", "washed", "dishes", grammar_swap=True),
        AssocItem("207", g_t, "shut", "window", "met", "friend", grammar_swap=True),
        AssocItem("208", g_t, "carried", "box", "cleaned", "kitchen", grammar_swap=True),
        AssocItem("209", g_t, "opened", "bag", "built", "house", grammar_swap=True),
        #
        # AssocItem("11", g_t, "brushed", "teeth", "locked", "cabinet", grammar_swap=True),
        # AssocItem("12", g_t, "peeled", "banana", "folded", "napkin", grammar_swap=True),
        # AssocItem("13", g_t, "inflated", "mattress", "baked", "muffin", grammar_swap=True),
        # AssocItem("14", g_t, "shaved", "mustache", "painted", "ceiling", grammar_swap=True),
        # AssocItem("15", g_t, "charged", "battery", "mopped", "hallway", grammar_swap=True),
        # AssocItem("16", g_t, "mowed", "lawn", "dusted", "shelf", grammar_swap=True),
        # AssocItem("17", g_t, "took", "medicine", "wrote", "poem", grammar_swap=True),
        # AssocItem("18", g_t, "drove", "tractor", "hung", "curtain", grammar_swap=True),
        # AssocItem("19", g_t, "fixed", "toaster", "lit", "candle", grammar_swap=True),
        # AssocItem("20", g_t, "tuned", "piano", "grilled", "steak", grammar_swap=True),
        # AssocItem("21", g_t, "sealed", "envelope", "solved", "problem", grammar_swap=True),
        # AssocItem("22", g_t, "poured", "coffee", "answered", "doorbell", grammar_swap=True),
        # AssocItem("23", g_t, "chopped", "onion", "mailed", "postcard", grammar_swap=True),
        # AssocItem("24", g_t, "fried", "bacon", "typed", "homework", grammar_swap=True),
        # AssocItem("25", g_t, "filled", "bucket", "swept", "stairs", grammar_swap=True),
        # AssocItem("26", g_t, "pushed", "wagon", "fixed", "engine", grammar_swap=True),
        # AssocItem("27", g_t, "planted", "tomato", "ironed", "shirt", grammar_swap=True),
        # AssocItem("28", g_t, "folded", "laundry", "broke", "vase", grammar_swap=True),
        # AssocItem("29", g_t, "studied", "history", "ate", "cookie", grammar_swap=True),
        # AssocItem("30", g_t, "painted", "portrait", "counted", "money", grammar_swap=True),
        # AssocItem("31", g_t, "cleaned", "kitchen", "wrote", "song", grammar_swap=True),
        # AssocItem("32", g_t, "opened", "refrigerator", "cooked", "dinner", grammar_swap=True),
        # AssocItem("33", g_t, "zipped", "jacket", "peeled", "orange", grammar_swap=True),
        # AssocItem("34", g_t, "combed", "hair", "sharpened", "scissors", grammar_swap=True),
        # AssocItem("35", g_t, "fed", "kitten", "canceled", "meeting", grammar_swap=True),
        # AssocItem("36", g_t, "delivered", "pizza", "solved", "mystery", grammar_swap=True),
        # AssocItem("37", g_t, "wrote", "diary", "replaced", "bulb", grammar_swap=True),
        # AssocItem("38", g_t, "drove", "car", "decorated", "cupcake", grammar_swap=True),
        # AssocItem("39", g_t, "caught", "cold", "grabbed", "umbrella", grammar_swap=True),
        # AssocItem("40", g_t, "packed", "luggage", "roasted", "turkey", grammar_swap=True),
        # AssocItem("41", g_t, "rinsed", "lettuce", "boiled", "pasta", grammar_swap=True),
        # AssocItem("42", g_t, "sliced", "cucumber", "stirred", "soup", grammar_swap=True),
        # AssocItem("43", g_t, "mixed", "dough", "charged", "battery", grammar_swap=True),
        # AssocItem("44", g_t, "wiped", "window", "paid", "fee", grammar_swap=True),
        # AssocItem("45", g_t, "washed", "hair", "built", "shelf", grammar_swap=True),
        # AssocItem("46", g_t, "restarted", "computer", "peeled", "carrot", grammar_swap=True),
        # AssocItem("47", g_t, "closed", "zipper", "poured", "tea", grammar_swap=True),
        # AssocItem("48", g_t, "hung", "poster", "toasted", "muffin", grammar_swap=True),
        # AssocItem("49", g_t, "tied", "ribbon", "ate", "cereal", grammar_swap=True),
        # AssocItem("50", g_t, "unpacked", "luggage", "shaved", "mustache", grammar_swap=True),
        # AssocItem("51", g_t, "turned", "knob", "sliced", "tomato", grammar_swap=True),
        # AssocItem("52", g_t, "fed", "puppy", "painted", "ceiling", grammar_swap=True),
        # AssocItem("53", g_t, "closed", "drawer", "fried", "bacon", grammar_swap=True),
        # AssocItem("54", g_t, "drew", "cartoon", "broke", "camera", grammar_swap=True),
        # AssocItem("55", g_t, "planted", "cabbage", "fixed", "printer", grammar_swap=True),
        # AssocItem("56", g_t, "measured", "temperature", "decorated", "cookie", grammar_swap=True),
        # AssocItem("57", g_t, "washed", "car", "wrote", "speech", grammar_swap=True),
        # AssocItem("58", g_t, "carried", "luggage", "trimmed", "hair", grammar_swap=True),
        # AssocItem("59", g_t, "scrubbed", "toilet", "ate", "muffin", grammar_swap=True),
        # AssocItem("60", g_t, "hugged", "cousin", "boiled", "pasta", grammar_swap=True),
        # AssocItem("61", g_t, "checked", "wallet", "paid", "fee", grammar_swap=True),
        # AssocItem("62", g_t, "watched", "movie", "chopped", "onion", grammar_swap=True),
        # AssocItem("63", g_t, "changed", "battery", "ate", "cookie", grammar_swap=True),
        # AssocItem("64", g_t, "played", "guitar", "solved", "problem", grammar_swap=True),
        # AssocItem("65", g_t, "made", "breakfast", "locked", "cabinet", grammar_swap=True),
        # AssocItem("66", g_t, "finished", "homework", "washed", "window", grammar_swap=True),
        # AssocItem("67", g_t, "fed", "sheep", "painted", "portrait", grammar_swap=True),
        # AssocItem("68", g_t, "rented", "apartment", "borrowed", "umbrella", grammar_swap=True),
        # AssocItem("69", g_t, "found", "wallet", "mowed", "lawn", grammar_swap=True),
        # AssocItem("70", g_t, "wore", "pajamas", "fixed", "printer", grammar_swap=True),
        # AssocItem("71", g_t, "swept", "garage", "played", "chess", grammar_swap=True),
        # AssocItem("72", g_t, "kept", "secret", "sliced", "cucumber", grammar_swap=True),
        # AssocItem("73", g_t, "paid", "debt", "dusted", "shelf", grammar_swap=True),
        # AssocItem("74", g_t, "watched", "news", "chopped", "garlic", grammar_swap=True),
        # AssocItem("75", g_t, "zipped", "suitcase", "cooked", "dinner", grammar_swap=True),
        # AssocItem("76", g_t, "borrowed", "money", "mopped", "hallway", grammar_swap=True),
        # AssocItem("77", g_t, "returned", "umbrella", "filled", "bucket", grammar_swap=True),
        # AssocItem("78", g_t, "knitted", "sweater", "locked", "cabinet", grammar_swap=True),
        # AssocItem("79", g_t, "vacuumed", "sofa", "sliced", "tomato", grammar_swap=True),
        # AssocItem("80", g_t, "opened", "notebook", "roasted", "turkey", grammar_swap=True),
        # AssocItem("81", g_t, "scrubbed", "bathroom", "counted", "money", grammar_swap=True),
        # AssocItem("82", g_t, "locked", "drawer", "served", "breakfast", grammar_swap=True),
        # AssocItem("83", g_t, "watered", "lawn", "fixed", "computer", grammar_swap=True),
        # AssocItem("84", g_t, "reset", "computer", "grilled", "steak", grammar_swap=True),
        # AssocItem("85", g_t, "wiped", "oven", "watched", "movie", grammar_swap=True),
        # AssocItem("86", g_t, "carved", "pumpkin", "typed", "homework", grammar_swap=True),
        # AssocItem("87", g_t, "peeled", "potato", "wrote", "poem", grammar_swap=True),
        # AssocItem("88", g_t, "drank", "tea", "painted", "ceiling", grammar_swap=True),
        # AssocItem("89", g_t, "polished", "helmet", "boiled", "pasta", grammar_swap=True),
        # AssocItem("90", g_t, "trimmed", "eyebrow", "packed", "suitcase", grammar_swap=True),
        # AssocItem("91", g_t, "filed", "complaint", "fried", "bacon", grammar_swap=True),
        # AssocItem("92", g_t, "bought", "medicine", "trimmed", "eyebrow", grammar_swap=True),
        # AssocItem("93", g_t, "checked", "passport", "boiled", "pasta", grammar_swap=True),
        # AssocItem("94", g_t, "saved", "money", "shaved", "mustache", grammar_swap=True),
        # AssocItem("95", g_t, "tasted", "soup", "checked", "engine", grammar_swap=True),
        # AssocItem("96", g_t, "wiped", "refrigerator", "played", "chess", grammar_swap=True),
        # AssocItem("97", g_t, "picked", "apple", "solved", "mystery", grammar_swap=True),
        # AssocItem("98", g_t, "washed", "grape", "kept", "secret", grammar_swap=True),
        # AssocItem("99", g_t, "broke", "window", "roasted", "turkey", grammar_swap=True),
        # AssocItem("100", g_t, "practiced", "piano", "started", "engine", grammar_swap=True),
        # AssocItem("101", g_t, "enjoyed", "music", "mopped", "bathroom", grammar_swap=True),
        # AssocItem("102", g_t, "learned", "math", "grilled", "steak", grammar_swap=True),
        # AssocItem("103", g_t, "watched", "sunrise", "folded", "laundry", grammar_swap=True),
        # AssocItem("104", g_t, "sent", "postcard", "sharpened", "scissors", grammar_swap=True),
        # AssocItem("105", g_t, "accepted", "apology", "peeled", "orange", grammar_swap=True),
        # AssocItem("106", g_t, "ignored", "advice", "fixed", "toaster", grammar_swap=True),
        # AssocItem("107", g_t, "lowered", "curtain", "made", "breakfast", grammar_swap=True),
        # AssocItem("108", g_t, "followed", "recipe", "locked", "drawer", grammar_swap=True),
        # AssocItem("109", g_t, "filled", "bathtub", "watched", "movie", grammar_swap=True),
        # AssocItem("110", g_t, "visited", "zoo", "mowed", "lawn", grammar_swap=True),

    ]
    c = Counter()
    for i in items:
        c.update([(i.x1, i.y1), (i.x2, i.y2)])
    print(c)
    return items


def make_single_pair_items() -> List[SinglePairItem]:
    """Single-pair items: sentences with one x-y association to test.

    Each item compares:
      ORIG:    template(x, y)
      SWAPPED: template(y, x)

    Margin: ΔS = logP(ORIG) - logP(SWAPPED)
    """
    items: List[SinglePairItem] = []

    # Idiomatic/temporal expressions
    # items += [
    #     SinglePairItem("Idiom", "Once upon a {x}, there was a boy who loved reading {y}.", "time", "books"),
    #     SinglePairItem("Idiom", "In the blink of an {x}, she learned to play the {y}.", "eye", "piano"),
    #     SinglePairItem("Idiom", "At the crack of {x}, the rooster heard the {y}.", "dawn", "alarm"),
    # ]
    #
    # # Typical associations
    # items += [
    #     SinglePairItem("Association", "The {x} flew through the {y}.", "bird", "sky"),
    #     SinglePairItem("Association", "The {x} swam in the {y}.", "fish", "ocean"),
    #     SinglePairItem("Association", "The {x} drove on the {y}.", "car", "road"),
    #     SinglePairItem("Association", "The {x} grew in the {y}.", "flower", "garden"),
    #     SinglePairItem("Association", "The {x} sailed on the {y}.", "ship", "sea"),
    # ]
    #
    # # Action-instrument pairings
    # items += [
    #     SinglePairItem("Instrument", "She used the {x} to write the {y}.", "pen", "letter"),
    #     SinglePairItem("Instrument", "He used the {x} to cut the {y}.", "knife", "bread"),
    #     SinglePairItem("Instrument", "They used the {x} to paint the {y}.", "brush", "wall"),
    #     SinglePairItem("Instrument", "She used the {x} to unlock the {y}.", "key", "door"),
    # ]
    #
    # Location/storage
    # items += [
    #     SinglePairItem("Location", "Once upon a {x}, there was a boy who loved {y}.", "times", "book"),
    #     SinglePairItem("Location", "Every {x}, the employee drives to {y}.", "days", "work"),
    #     SinglePairItem("Location", "They hold an {x} every 24 {y}.", "elections", "hour"),
    #     # SinglePairItem("Location", "In the past few {x}, the total increased by {y}.", "year", "millions"),
    #     # SinglePairItem("Location", "A {x} in the hand is worth two in the {y}.", "birds", "bush"),
    #     # SinglePairItem("Location", "A {x} is worth a thousand {y}.", "pictures", "word"),
    #     # SinglePairItem("Location", "The {x} was placed in the {y}.", "money", "wallet"),
    # ]

    return items


# -----------------------------
# Args
# -----------------------------


def parse_args():
    p = argparse.ArgumentParser()
    add_backend_arguments(p, prefix="sambal-", role="the ablated model")
    add_backend_arguments(p, prefix="normal-", role="the comparison model", with_device=False)

    p.add_argument(
        "--score_all_tokens",
        action="store_true",
        help="Score all tokens in the sentence (slower). Default scores only critical swapped spans.",
    )

    p.add_argument(
        "--counterbalance_order",
        action="store_true",
        help="For ΔK: average margins across both clause orders (reduces position confounds).",
    )

    p.add_argument(
        "--grammar_mode",
        type=str,
        default="auto",
        choices=["auto", "none", "primary", "sem2", "all4", "nonsense_gram"],
        help=(
            "How to compute ΔG for grammar_swap items. "
            "auto: if --counterbalance_order then sem2 else primary. "
            "primary: (x1,y1,x2,y2) vs (y1,x1,y2,x2). "
            "sem2: average across (x1,y1,x2,y2) and (x2,y2,x1,y1), each vs its grammar-bad. "
            "all4: average across all 4 grammatical arrangements (both semantic mappings × both clause orders), each vs its grammar-bad. "
            "nonsense_gram: average of 2 nonsense-grammatical (x1,y2,x2,y1) and (x2,y1,x1,y2) vs average of 4 ungrammatical (y,x,y,x) patterns. "
            "none: skip ΔG entirely."
        ),
    )

    p.add_argument("--max_items", type=int, default=None, help="Debug: limit number of items.")

    p.add_argument("--out_csv", type=str, default=None)
    p.add_argument("--out_json", type=str, default=None)

    p.add_argument(
        "--plot_base",
        type=str,
        default=None,
        help="Base filename (prefix) for plots. Produces *_knowledge_*.png and *_grammar_*.png (+ *_scatter.png).",
    )
    p.add_argument("--plot_bins", type=int, default=30)
    p.add_argument("--plot_dpi", type=int, default=200)
    return p.parse_args()


# -----------------------------
# Token span helpers
# -----------------------------

def _find_subseq(haystack: Sequence[int], needle: Sequence[int]) -> List[int]:
    """Return all start indices where needle occurs in haystack."""
    if len(needle) == 0 or len(needle) > len(haystack):
        return []
    out: List[int] = []
    for i in range(len(haystack) - len(needle) + 1):
        if list(haystack[i : i + len(needle)]) == list(needle):
            out.append(i)
    return out


def find_word_token_positions(backend: LanguageModel, sent_ids: List[int], word: str) -> List[int]:
    """Return token indices in sent_ids that correspond to `word` (all tokens in its span).

    Tries surface forms to handle tokenizers that encode leading spaces or punctuation.
    """
    variants = [
        word,
        " " + word,
        word + ".",
        " " + word + ".",
        word + ",",
        " " + word + ",",
        word + ";",
        " " + word + ";",
        word + ":",
        " " + word + ":",
    ]
    for v in variants:
        needle = backend.encode(v)
        starts = _find_subseq(sent_ids, needle)
        if starts:
            s = starts[0]
            return list(range(s, s + len(needle)))
    return []


def critical_positions(backend: LanguageModel, sentence: str, critical_words: Sequence[str]) -> Optional[List[int]]:
    """Return token indices in `sentence` for all `critical_words`.

    If no words are found (tokenization mismatch), returns None.
    """
    ids = backend.encode(sentence)
    pos: List[int] = []
    for w in critical_words:
        pos += find_word_token_positions(backend, ids, w)
    pos = sorted(set(pos))
    return pos if pos else None


# -----------------------------
# Scoring
# -----------------------------

def score_sentence(
    sentence: str,
    backend: LanguageModel,
    positions: Optional[List[int]] = None,
) -> float:
    """Sum of the backend's token log-probabilities over `positions` (all tokens when None).

    Returns:
      Sum log-prob (higher = more preferred).
    """
    ids = backend.encode(sentence)
    if len(ids) == 0:
        return float("-inf")

    if positions is None:
        return backend.token_logprobs([ids], [1.0])[0][0].sum().item()
    logps = backend.token_logprobs([ids], [1.0], positions=[positions])[0][0]
    return logps[positions].sum().item()


def score_pair(
    backend: LanguageModel,
    good_sent: str,
    bad_sent: str,
    score_all_tokens: bool,
    critical_words: Sequence[str],
) -> Tuple[float, float, float, Optional[List[int]], Optional[List[int]]]:
    """Score GOOD vs BAD and return (s_good, s_bad, margin, pos_good, pos_bad)."""
    if score_all_tokens:
        pos_good = None
        pos_bad = None
    else:
        pos_good = critical_positions(backend, good_sent, critical_words)
        pos_bad = critical_positions(backend, bad_sent, critical_words)
        if pos_good is None or pos_bad is None:
            # token-span matching failed; fall back to full-sentence scoring
            pos_good = None
            pos_bad = None

    s_good = score_sentence(good_sent, backend, positions=pos_good)
    s_bad = score_sentence(bad_sent, backend, positions=pos_bad)
    return s_good, s_bad, (s_good - s_bad), pos_good, pos_bad


# -----------------------------
# Evaluation
# -----------------------------

def _effective_grammar_mode(grammar_mode: str, counterbalance_order: bool) -> str:
    if grammar_mode == "auto":
        return "sem2" if counterbalance_order else "primary"
    return grammar_mode


def _grammar_arrangements(item: AssocItem, mode: str) -> List[Tuple[str, str, str, str]]:
    """Return a list of (xa, ya, xb, yb) arrangements that are *grammatical* under the template."""
    if mode == "primary":
        return [(item.x1, item.y1, item.x2, item.y2)]
    if mode == "sem2":
        return [
            (item.x1, item.y1, item.x2, item.y2),
            (item.x2, item.y2, item.x1, item.y1),
        ]
    if mode == "all4":
        return [
            (item.x1, item.y1, item.x2, item.y2),
            (item.x1, item.y2, item.x2, item.y1),
            (item.x2, item.y2, item.x1, item.y1),
            (item.x2, item.y1, item.x1, item.y2),
        ]
    raise ValueError(f"Unknown grammar mode: {mode}")


def evaluate_single_pair(
    items: List[SinglePairItem],
    backend: LanguageModel,
    score_all_tokens: bool,
) -> List[Dict]:
    """Evaluate single-pair items for one model.

    Returns one row per item with:
      - single_pair_margin: ΔS = logP(ORIG) - logP(SWAPPED)
    """
    rows: List[Dict] = []

    for item in items:
        orig_sent = item.orig()
        swap_sent = item.swapped()

        # Critical words are both x and y
        crit = [item.x, item.y]

        s_orig, s_swap, margin, pos_orig, pos_swap = score_pair(
            backend, orig_sent, swap_sent, score_all_tokens, crit
        )

        rows.append({
            "knowledge_type": item.knowledge_type,
            "note": item.note,
            "orig": orig_sent,
            "swapped": swap_sent,
            "single_pair_score_orig": s_orig,
            "single_pair_score_swapped": s_swap,
            "single_pair_margin": float(margin),
            "single_pair_prefers_orig": bool(margin > 0),
            "single_pair_scored_positions_orig": pos_orig,
            "single_pair_scored_positions_swapped": pos_swap,
            "x": item.x,
            "y": item.y,
        })

    return rows


def evaluate(
    items: List[AssocItem],
    backend: LanguageModel,
    score_all_tokens: bool,
    counterbalance_order: bool,
    grammar_mode: str,
) -> List[Dict]:
    """Evaluate all items for one model.

    Returns one row per item with:
      - knowledge: ΔK scores/margins
      - grammar : ΔG scores/margins (if grammar_swap=True and grammar_mode != 'none')

    Grammar ΔG is computed according to --grammar_mode (primary/sem2/all4).
    """
    rows: List[Dict] = []

    g_mode_eff = _effective_grammar_mode(grammar_mode, counterbalance_order)

    for item in items:
        # --- Knowledge swap ---
        good1 = item.good()
        badK1 = item.swapped()
        critK = [item.y1, item.y2]  # knowledge swap touches only y's

        k_good1, k_bad1, k_m1, pos_k_good1, pos_k_bad1 = score_pair(
            backend, good1, badK1, score_all_tokens, critK
        )

        k_good2 = k_bad2 = k_m2 = None
        pos_k_good2 = pos_k_bad2 = None
        if counterbalance_order:
            good2 = item.good_order_swapped()
            badK2 = item.swapped_order_swapped()
            k_good2, k_bad2, k_m2, pos_k_good2, pos_k_bad2 = score_pair(
                backend, good2, badK2, score_all_tokens, critK
            )
            k_margin = 0.5 * (k_m1 + k_m2)
        else:
            k_margin = k_m1

        # --- Grammar swap ---
        g_margin: Optional[float] = None
        g_cases: Optional[List[Dict]] = None

        # Convenience per-case fields (useful in analysis)
        g_m_primary = g_m_swapped = g_m_cb = g_m_swapped_cb = None

        if item.grammar_swap and g_mode_eff != "none":
            critG = [item.x1, item.y1, item.x2, item.y2]

            # Special handling for nonsense_gram mode
            if g_mode_eff == "nonsense_gram":
                # Two nonsense but grammatical: (x1,y2,x2,y1) and (x2,y1,x1,y2)
                gram_arrangements = [
                    (item.x1, item.y2, item.x2, item.y1),
                    (item.x2, item.y1, item.x1, item.y2),
                ]

                # Four ungrammatical: (y1,x1,y2,x1), (y1,x2,y2,x2), (y2,x1,y1,x1), (y2,x2,y1,x2)
                ungram_arrangements = [
                    (item.y1, item.x1, item.y2, item.x1),
                    (item.y1, item.x2, item.y2, item.x2),
                    (item.y2, item.x1, item.y1, item.x1),
                    (item.y2, item.x2, item.y1, item.x2),
                ]

                g_cases = []
                gram_scores: List[float] = []
                ungram_scores: List[float] = []

                # Check if we should force full-sentence scoring
                if score_all_tokens:
                    force_all = True
                else:
                    force_all = False
                    for (xa, ya, xb, yb) in gram_arrangements + ungram_arrangements:
                        sent = item.fmt(xa, ya, xb, yb)
                        if critical_positions(backend, sent, critG) is None:
                            force_all = True
                            break

                # Score grammatical nonsense sentences
                for idx, (xa, ya, xb, yb) in enumerate(gram_arrangements):
                    sent = item.fmt(xa, ya, xb, yb)
                    if force_all or score_all_tokens:
                        pos = None
                    else:
                        pos = critical_positions(backend, sent, critG)
                    score = score_sentence(sent, backend, positions=pos)
                    gram_scores.append(score)
                    g_cases.append({
                        "case_index": idx,
                        "type": "grammatical_nonsense",
                        "sentence": sent,
                        "score": score,
                        "scored_positions": pos,
                    })

                # Score ungrammatical sentences
                for idx, (xa, ya, xb, yb) in enumerate(ungram_arrangements):
                    sent = item.fmt(xa, ya, xb, yb)
                    if force_all or score_all_tokens:
                        pos = None
                    else:
                        pos = critical_positions(backend, sent, critG)
                    score = score_sentence(sent, backend, positions=pos)
                    ungram_scores.append(score)
                    g_cases.append({
                        "case_index": len(gram_arrangements) + idx,
                        "type": "ungrammatical",
                        "sentence": sent,
                        "score": score,
                        "scored_positions": pos,
                    })

                # Compute margin: avg(grammatical) - avg(ungrammatical)
                avg_gram = sum(gram_scores) / len(gram_scores)
                avg_ungram = sum(ungram_scores) / len(ungram_scores)
                g_margin = avg_gram - avg_ungram

                # Store named fields for analysis
                g_m_swapped = gram_scores[0] if len(gram_scores) > 0 else None  # (x1,y2,x2,y1)
                g_m_swapped_cb = gram_scores[1] if len(gram_scores) > 1 else None  # (x2,y1,x1,y2)

            else:
                # Original logic for other modes
                arrangements = _grammar_arrangements(item, g_mode_eff)

                g_cases = []
                margins: List[float] = []

                # To avoid mixing "critical spans" vs "full sentence" across cases,
                # we do a single pass of position-finding. If any sentence fails, we score
                # *all* grammar cases with full-sentence PLL.
                if score_all_tokens:
                    force_all = True
                else:
                    force_all = False
                    for (xa, ya, xb, yb) in arrangements:
                        gs = item.fmt(xa, ya, xb, yb)
                        bs = item.fmt(ya, xa, yb, xb)
                        if critical_positions(backend, gs, critG) is None or critical_positions(backend, bs, critG) is None:
                            force_all = True
                            break

                for idx, (xa, ya, xb, yb) in enumerate(arrangements):
                    gs = item.fmt(xa, ya, xb, yb)
                    bs = item.fmt(ya, xa, yb, xb)
                    s_g, s_b, m, pos_g, pos_b = score_pair(
                        backend,
                        gs,
                        bs,
                        score_all_tokens=(score_all_tokens or force_all),
                        critical_words=critG,
                    )
                    g_cases.append({
                        "case_index": idx,
                        "good": gs,
                        "bad": bs,
                        "score_good": s_g,
                        "score_bad": s_b,
                        "margin": m,
                        "scored_positions_good": pos_g,
                        "scored_positions_bad": pos_b,
                    })
                    margins.append(float(m))

                    # Fill named fields for common cases (helps when comparing modes)
                    if (xa, ya, xb, yb) == (item.x1, item.y1, item.x2, item.y2):
                        g_m_primary = float(m)
                    elif (xa, ya, xb, yb) == (item.x1, item.y2, item.x2, item.y1):
                        g_m_swapped = float(m)
                    elif (xa, ya, xb, yb) == (item.x2, item.y2, item.x1, item.y1):
                        g_m_cb = float(m)
                    elif (xa, ya, xb, yb) == (item.x2, item.y1, item.x1, item.y2):
                        g_m_swapped_cb = float(m)

                g_margin = sum(margins) / len(margins) if margins else None

        rows.append({
            "knowledge_type": item.knowledge_type,
            "note": item.note,

            # Knowledge
            "good": good1,
            "swapped": badK1,
            "knowledge_score_good": k_good1,
            "knowledge_score_swapped": k_bad1,
            "knowledge_margin_primary": float(k_m1),
            "knowledge_margin_cb": (None if k_m2 is None else float(k_m2)),
            "knowledge_margin": float(k_margin),
            "knowledge_prefers_good": bool(k_margin > 0),
            "knowledge_scored_positions_good": pos_k_good1,
            "knowledge_scored_positions_swapped": pos_k_bad1,
            "knowledge_scored_positions_good_cb": pos_k_good2,
            "knowledge_scored_positions_swapped_cb": pos_k_bad2,

            # Grammar
            "has_grammar": bool(item.grammar_swap),
            "grammar_mode": (None if (not item.grammar_swap or g_mode_eff == "none") else g_mode_eff),
            "grammar_margin": g_margin,
            "grammar_prefers_good": (None if g_margin is None else bool(g_margin > 0)),
            "grammar_cases": g_cases,
            "grammar_margin_primary": g_m_primary,
            "grammar_margin_swapped": g_m_swapped,
            "grammar_margin_cb": g_m_cb,
            "grammar_margin_swapped_cb": g_m_swapped_cb,

            # Copy raw fields (helpful for debugging / filtering)
            "x1": item.x1,
            "y1": item.y1,
            "x2": item.x2,
            "y2": item.y2,
        })

    return rows


def summarize(rows: List[Dict], margin_key: str) -> Dict:
    """Summarize rows using the specified margin_key.

    Skips rows where margin is None.
    """
    filt = [r for r in rows if r.get(margin_key) is not None]
    if not filt:
        return {"overall": {"n": 0, "accuracy": float("nan"), "mean_margin": float("nan")}, "by_type": {}}

    def _acc(rs: List[Dict]) -> float:
        good = sum(1 for r in rs if float(r[margin_key]) > 0)
        return good / max(1, len(rs))

    def _mean(rs: List[Dict]) -> float:
        return sum(float(r[margin_key]) for r in rs) / max(1, len(rs))

    by_type: Dict[str, List[Dict]] = {}
    for r in filt:
        by_type.setdefault(r["knowledge_type"], []).append(r)

    return {
        "overall": {"n": len(filt), "accuracy": _acc(filt), "mean_margin": _mean(filt)},
        "by_type": {
            k: {"n": len(v), "accuracy": _acc(v), "mean_margin": _mean(v)}
            for k, v in sorted(by_type.items(), key=lambda kv: kv[0])
        },
    }


DUAL_KIND = "dual_pair"
SINGLE_KIND = "single_pair"


def evaluate_swap_probe(
    backend: LanguageModel,
    items: Optional[List[AssocItem]] = None,
    single_items: Optional[List[SinglePairItem]] = None,
    score_all_tokens: bool = False,
    counterbalance_order: bool = False,
    grammar_mode: str = "auto",
) -> Dict:
    """Run both probes on one model and return summary / rows / protocol."""
    items = make_items() if items is None else list(items)
    single_items = make_single_pair_items() if single_items is None else list(single_items)

    dual_rows = evaluate(items, backend, score_all_tokens, counterbalance_order, grammar_mode)
    single_rows = evaluate_single_pair(single_items, backend, score_all_tokens)

    knowledge = summarize(dual_rows, "knowledge_margin")
    grammar = summarize(dual_rows, "grammar_margin")
    single = summarize(single_rows, "single_pair_margin")

    summary: Dict[str, float] = {
        "swap_probe/accuracy_grammar": grammar["overall"]["accuracy"],
        "swap_probe/accuracy_semantic": knowledge["overall"]["accuracy"],
        "swap_probe/mean_delta_g": grammar["overall"]["mean_margin"],
        "swap_probe/mean_delta_k": knowledge["overall"]["mean_margin"],
        "swap_probe/n_grammar": grammar["overall"]["n"],
        "swap_probe/n_semantic": knowledge["overall"]["n"],
        "swap_probe/single_pair/accuracy": single["overall"]["accuracy"],
        "swap_probe/single_pair/mean_margin": single["overall"]["mean_margin"],
    }
    for knowledge_type, stats in knowledge["by_type"].items():
        summary[f"swap_probe/by_type/{knowledge_type}/accuracy_semantic"] = stats["accuracy"]
        summary[f"swap_probe/by_type/{knowledge_type}/mean_delta_k"] = stats["mean_margin"]

    rows = [dict(r, kind=DUAL_KIND) for r in dual_rows] + [dict(r, kind=SINGLE_KIND) for r in single_rows]
    return {
        "summary": summary,
        "rows": rows,
        "protocol": {
            "score_all_tokens": bool(score_all_tokens),
            "counterbalance_order": bool(counterbalance_order),
            "grammar_mode": str(grammar_mode),
            "effective_grammar_mode": _effective_grammar_mode(grammar_mode, counterbalance_order),
        },
    }


def rows_of_kind(rows: List[Dict], kind: str) -> List[Dict]:
    """The rows of one probe, without the ``kind`` tag."""
    return [{k: v for k, v in r.items() if k != "kind"} for r in rows if r["kind"] == kind]


def print_table(sambal_rows: List[Dict], norm_rows: List[Dict], max_width: int = 88):
    print("\n" + "=" * 160)
    print("PAIRED KNOWLEDGE SWAP (ΔK) + GRAMMAR SWAP (ΔG) PROBE")
    print("ΔK = logP(good) - logP(swapped-y);  ΔG = logP(grammar-good) - logP(grammar-bad)")
    print("=" * 160)

    header = (
        f"{'#':<3} {'Type':<12} {'Prompt (truncated)':<{max_width}} | "
        f"{'SAMBAL ΔK':>8} {'Norm ΔK':>8} | {'SAMBAL ΔG':>8} {'Norm ΔG':>8}"
    )
    print(header)
    print("-" * len(header))

    for i, (fr, nr) in enumerate(zip(sambal_rows, norm_rows), start=1):
        prompt = fr["good"]
        if len(prompt) > max_width:
            prompt = prompt[: max_width - 3] + "..."

        fk = float(fr["knowledge_margin"])
        nk = float(nr["knowledge_margin"])

        fg = fr.get("grammar_margin")
        ng = nr.get("grammar_margin")

        fg_str = f"{fg:8.2f}" if fg is not None else f"{'n/a':>8}"
        ng_str = f"{ng:8.2f}" if ng is not None else f"{'n/a':>8}"

        print(
            f"{i:<3} {fr['knowledge_type']:<12} {prompt:<{max_width}} | {fk:8.2f} {nk:8.2f} | {fg_str} {ng_str}"
        )

    print("-" * len(header))


def print_single_pair_table(sambal_rows: List[Dict], norm_rows: List[Dict], max_width: int = 88):
    print("\n" + "=" * 120)
    print("SINGLE-PAIR SWAP (ΔS) PROBE")
    print("ΔS = logP(orig) - logP(swapped)")
    print("=" * 120)

    header = (
        f"{'#':<3} {'Type':<12} {'Orig (truncated)':<{max_width}} | "
        f"{'SAMBAL ΔS':>8} {'Norm ΔS':>8}"
    )
    print(header)
    print("-" * len(header))

    for i, (fr, nr) in enumerate(zip(sambal_rows, norm_rows), start=1):
        prompt = fr["orig"]
        if len(prompt) > max_width:
            prompt = prompt[: max_width - 3] + "..."

        fs = float(fr["single_pair_margin"])
        ns = float(nr["single_pair_margin"])

        print(
            f"{i:<3} {fr['knowledge_type']:<12} {prompt:<{max_width}} | {fs:8.2f} {ns:8.2f}"
        )

    print("-" * len(header))



# -----------------------------
# Plotting
# -----------------------------

def plot_margin_distributions(
    sambal_rows: List[Dict],
    norm_rows: List[Dict],
    out_base: str,
    margin_key: str,
    suffix: str,
    xlabel: str,
    bins: int = 30,
    dpi: int = 200,
):
    """Histogram + violin + mean-by-type bars for a given margin key."""
    if plt is None:
        return

    base = out_base[:-4] if out_base.lower().endswith(".png") else out_base
    base = f"{base}_{suffix}"

    sambal_m = [float(r[margin_key]) for r in sambal_rows if r.get(margin_key) is not None]
    norm_m = [float(r[margin_key]) for r in norm_rows if r.get(margin_key) is not None]

    if not sambal_m or not norm_m:
        print(f"WARNING: no rows with margin_key='{margin_key}' for plots '{suffix}'.")
        return

    # 1) Histogram overlay
    plt.figure()
    plt.hist(norm_m, bins=bins, alpha=0.5, label="Normal LM")
    plt.hist(sambal_m, bins=bins, alpha=0.5, label="SAMBAL")
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    hist_path = base + "_hist.png"
    plt.savefig(hist_path, dpi=dpi)
    plt.close()

    # 2) Violin plot
    plt.figure()
    plt.violinplot([sambal_m, norm_m], showmeans=True, showextrema=True)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xticks([1, 2], ["SAMBAL", "Normal LM"])
    plt.ylabel(xlabel)
    plt.tight_layout()
    viol_path = base + "_violin.png"
    plt.savefig(viol_path, dpi=dpi)
    plt.close()

    # 3) Mean margin by type (grouped bars)
    types = sorted(set(r["knowledge_type"] for r in sambal_rows if r.get(margin_key) is not None))
    by_type_path = None
    if types:
        f_means = []
        n_means = []
        for t in types:
            fm = [float(r[margin_key]) for r in sambal_rows if r["knowledge_type"] == t and r.get(margin_key) is not None]
            nm = [float(r[margin_key]) for r in norm_rows if r["knowledge_type"] == t and r.get(margin_key) is not None]
            f_means.append(sum(fm) / len(fm) if fm else float("nan"))
            n_means.append(sum(nm) / len(nm) if nm else float("nan"))

        xs = list(range(len(types)))
        width = 0.42
        plt.figure(figsize=(max(6.0, 0.55 * len(types)), 4.0))
        plt.bar([x - width / 2 for x in xs], f_means, width=width, label="SAMBAL")
        plt.bar([x + width / 2 for x in xs], n_means, width=width, label="Normal LM")
        plt.axhline(0.0, linestyle="--", linewidth=1)
        plt.xticks(xs, types, rotation=45, ha="right")
        plt.ylabel(f"Mean {xlabel}")
        plt.legend()
        plt.tight_layout()
        by_type_path = base + "_by_type.png"
        plt.savefig(by_type_path, dpi=dpi)
        plt.close()

    paths = [hist_path, viol_path] + ([by_type_path] if by_type_path else [])
    print("Wrote plots:", ", ".join(p for p in paths if p))


def plot_scatter_knowledge_vs_grammar(
    sambal_rows: List[Dict],
    norm_rows: List[Dict],
    out_base: str,
    dpi: int = 200,
):
    """Scatter for items with ΔG: x=ΔK, y=ΔG."""


    base = out_base[:-4] if out_base.lower().endswith(".png") else out_base

    f_pts = [(float(r["knowledge_margin"]), float(r["grammar_margin"])) for r in sambal_rows if r.get("grammar_margin") is not None]
    n_pts = [(float(r["knowledge_margin"]), float(r["grammar_margin"])) for r in norm_rows if r.get("grammar_margin") is not None]

    if not f_pts or not n_pts:
        print("WARNING: no grammar rows for scatter plot; skipping.")
        return

    fx, fy = zip(*f_pts)
    nx, ny = zip(*n_pts)

    plt.figure()
    plt.scatter(nx, ny, alpha=0.7, label="Normal LM")
    plt.scatter(fx, fy, alpha=0.7, label="SAMBAL")
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Semantics margin ΔK ")
    plt.ylabel("Syntax margin ΔG")
    plt.legend()
    plt.tight_layout()
    path = base + "_scatter.pdf"
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Wrote plot: {path}")


# -----------------------------
# Main
# -----------------------------

def main():
    args = parse_args()

    # Load dual-pair items
    items = make_items()
    if args.max_items is not None:
        items = items[: args.max_items]

    # Load single-pair items
    single_items = make_single_pair_items()
    if args.max_items is not None:
        single_items = single_items[: args.max_items]

    eff_g = _effective_grammar_mode(args.grammar_mode, args.counterbalance_order)

    print("Loading SAMBAL model ...")
    sambal_backend = backend_from_args(args, prefix="sambal-")

    print("Loading Normal model ...")
    normal_backend = backend_from_args(args, prefix="normal-")

    print(
        f"Loaded {len(items)} dual-pair items and {len(single_items)} single-pair items. "
        f"Device={sambal_backend.device}. "
        f"score_all_tokens={args.score_all_tokens}. counterbalance_order={args.counterbalance_order}. "
        f"grammar_mode={args.grammar_mode} (effective={eff_g})."
    )

    print("Scoring items (SAMBAL) ...")
    sambal = evaluate_swap_probe(
        sambal_backend,
        items=items,
        single_items=single_items,
        score_all_tokens=args.score_all_tokens,
        counterbalance_order=args.counterbalance_order,
        grammar_mode=args.grammar_mode,
    )

    print("Scoring items (Normal LM) ...")
    normal = evaluate_swap_probe(
        normal_backend,
        items=items,
        single_items=single_items,
        score_all_tokens=args.score_all_tokens,
        counterbalance_order=args.counterbalance_order,
        grammar_mode=args.grammar_mode,
    )

    sambal_rows = rows_of_kind(sambal["rows"], DUAL_KIND)
    norm_rows = rows_of_kind(normal["rows"], DUAL_KIND)
    sambal_single_rows = rows_of_kind(sambal["rows"], SINGLE_KIND)
    norm_single_rows = rows_of_kind(normal["rows"], SINGLE_KIND)

    assert len(sambal_rows) == len(norm_rows)

    print_table(sambal_rows, norm_rows)

    # Summaries
    sambal_K = summarize(sambal_rows, "knowledge_margin")
    norm_K = summarize(norm_rows, "knowledge_margin")
    sambal_G = summarize(sambal_rows, "grammar_margin")
    norm_G = summarize(norm_rows, "grammar_margin")

    print("\nSummary (Knowledge swaps):")
    print(
        f"  SAMBAL:  acc={sambal_K['overall']['accuracy']*100:.1f}%  "
        f"meanΔK={sambal_K['overall']['mean_margin']:.3f}  n={sambal_K['overall']['n']}"
    )
    print(
        f"  Normal LM:  acc={norm_K['overall']['accuracy']*100:.1f}%  "
        f"meanΔK={norm_K['overall']['mean_margin']:.3f}  n={norm_K['overall']['n']}"
    )

    if sambal_G["overall"]["n"] > 0:
        print(f"\nSummary (Grammar swaps; mode={args.grammar_mode}, effective={eff_g}):")
        print(
            f"  SAMBAL:  acc={sambal_G['overall']['accuracy']*100:.1f}%  "
            f"meanΔG={sambal_G['overall']['mean_margin']:.3f}  n={sambal_G['overall']['n']}"
        )
        print(
            f"  Normal LM:  acc={norm_G['overall']['accuracy']*100:.1f}%  "
            f"meanΔG={norm_G['overall']['mean_margin']:.3f}  n={norm_G['overall']['n']}"
        )
    else:
        print("\n(No grammar_swap=True items found, or grammar_mode=none; grammar summary skipped.)")

    # ===== Single-pair results =====
    assert len(sambal_single_rows) == len(norm_single_rows)

    print_single_pair_table(sambal_single_rows, norm_single_rows)

    # Single-pair summaries
    sambal_S = summarize(sambal_single_rows, "single_pair_margin")
    norm_S = summarize(norm_single_rows, "single_pair_margin")

    print("\nSummary (Single-pair swaps):")
    print(
        f"  SAMBAL:  acc={sambal_S['overall']['accuracy']*100:.1f}%  "
        f"meanΔS={sambal_S['overall']['mean_margin']:.3f}  n={sambal_S['overall']['n']}"
    )
    print(
        f"  Normal LM:  acc={norm_S['overall']['accuracy']*100:.1f}%  "
        f"meanΔS={norm_S['overall']['mean_margin']:.3f}  n={norm_S['overall']['n']}"
    )

    # Output packing
    out_pack = {
        "dual_pair": {
            "items": [item.__dict__ for item in items],
            "sambal": {"rows": sambal_rows, "knowledge_summary": sambal_K, "grammar_summary": sambal_G},
            "normal": {"rows": norm_rows, "knowledge_summary": norm_K, "grammar_summary": norm_G},
        },
        "single_pair": {
            "items": [item.__dict__ for item in single_items],
            "sambal": {"rows": sambal_single_rows, "summary": sambal_S},
            "normal": {"rows": norm_single_rows, "summary": norm_S},
        },
        "score_all_tokens": sambal["protocol"]["score_all_tokens"],
        "counterbalance_order": sambal["protocol"]["counterbalance_order"],
        "grammar_mode": sambal["protocol"]["grammar_mode"],
    }

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(out_pack, f, indent=2)
        print(f"\nWrote {args.out_json}")

    if args.out_csv:
        # Dual-pair CSV
        dual_csv = args.out_csv.replace(".csv", "_dual_pair.csv")
        with open(dual_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "knowledge_type",
                    "good",
                    "swapped_y",
                    "sambal_ΔK",
                    "normal_ΔK",
                    "grammar_mode",
                    "sambal_ΔG",
                    "normal_ΔG",
                ]
            )
            for fr, nr in zip(sambal_rows, norm_rows):
                w.writerow(
                    [
                        fr["knowledge_type"],
                        fr["good"],
                        fr["swapped"],
                        fr["knowledge_margin"],
                        nr["knowledge_margin"],
                        fr.get("grammar_mode"),
                        fr.get("grammar_margin"),
                        nr.get("grammar_margin"),
                    ]
                )
        print(f"Wrote {dual_csv}")

        # Single-pair CSV
        single_csv = args.out_csv.replace(".csv", "_single_pair.csv")
        with open(single_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "knowledge_type",
                    "orig",
                    "swapped",
                    "sambal_ΔS",
                    "normal_ΔS",
                ]
            )
            for fr, nr in zip(sambal_single_rows, norm_single_rows):
                w.writerow(
                    [
                        fr["knowledge_type"],
                        fr["orig"],
                        fr["swapped"],
                        fr["single_pair_margin"],
                        nr["single_pair_margin"],
                    ]
                )
        print(f"Wrote {single_csv}")

    if args.plot_base:
        # Dual-pair plots
        plot_margin_distributions(
            sambal_rows,
            norm_rows,
            out_base=args.plot_base,
            margin_key="knowledge_margin",
            suffix="knowledge",
            xlabel="ΔK = logP(good) - logP(swapped-y)",
            bins=args.plot_bins,
            dpi=args.plot_dpi,
        )

        plot_margin_distributions(
            sambal_rows,
            norm_rows,
            out_base=args.plot_base,
            margin_key="grammar_margin",
            suffix="grammar",
            xlabel="ΔG = logP(grammar-good) - logP(grammar-bad)",
            bins=args.plot_bins,
            dpi=args.plot_dpi,
        )

        plot_scatter_knowledge_vs_grammar(
            sambal_rows,
            norm_rows,
            out_base=args.plot_base,
            dpi=args.plot_dpi,
        )

        # Single-pair plots
        plot_margin_distributions(
            sambal_single_rows,
            norm_single_rows,
            out_base=args.plot_base,
            margin_key="single_pair_margin",
            suffix="single_pair",
            xlabel="ΔS = logP(orig) - logP(swapped)",
            bins=args.plot_bins,
            dpi=args.plot_dpi,
        )


if __name__ == "__main__":
    main()
