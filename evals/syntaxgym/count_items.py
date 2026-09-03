#!/usr/bin/env python3
import json
from pathlib import Path

SUITE_ORDER = [
    "number_orc", "number_prep", "number_src",
    "reflexive_orc_fem", "reflexive_orc_masc",
    "reflexive_prep_fem", "reflexive_prep_masc",
    "reflexive_src_fem", "reflexive_src_masc",
    "npi_orc_any", "npi_orc_ever", "npi_src_any", "npi_src_ever",
    "fgd_hierarchy", "fgd_object", "fgd_pp", "fgd_subject",
    "center_embed", "center_embed_mod",
    "cleft", "cleft_modifier",
    "subordination", "subordination_orc-orc", "subordination_pp-pp", "subordination_src-src",
]

data_dir = Path(__file__).parent / "data/syntaxgym"

for suite in SUITE_ORDER:
    path = data_dir / f"{suite}.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        count = len(data.get("items", []))
        print(f"{suite}: {count}")
    else:
        print(f"{suite}: FILE NOT FOUND")
