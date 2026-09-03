import json, random
from pathlib import Path

random.seed(0)

NEUTRAL_HEADS = [
    "manager","teacher","doctor","author","engineer","student","artist","journalist","lawyer","scientist",
    "nurse","pilot","chef","musician","athlete","driver","researcher","designer","editor","director",
    "writer","photographer","producer","analyst","consultant","developer","architect","librarian","farmer","mechanic"
]

# Verbs that plausibly take reflexive objects without adding extra nouns.
VERBS = [
    "introduced", "injured", "criticized", "praised", "blamed", "defended",
    "helped", "trusted", "embarrassed", "congratulated", "reminded", "forgave"
]

# Keep sentences short and noun-free besides the subject.
ADVERBS = ["", "", "", "quietly", "openly", "privately", "carefully"]  # biased toward empty

def make_sentence(head, verb, adv, refl):
    if adv:
        return f"The {head} {adv} {verb} {refl}."
    else:
        return f"The {head} {verb} {refl}."

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def build_pairs(uid, good_refl, bad_refl="themselves", n_items=200, lexically_identical=False):
    rows = []
    pair_id = 0
    # sample with replacement to get enough items
    for _ in range(n_items):
        head = random.choice(NEUTRAL_HEADS)
        verb = random.choice(VERBS)
        adv  = random.choice(ADVERBS)

        good = make_sentence(head, verb, adv, good_refl)
        bad  = make_sentence(head, verb, adv, bad_refl)

        pair_id += 1
        rows.append({
            "sentence_good": good,
            "sentence_bad": bad,
            "field": "syntax",
            "linguistics_term": "anaphora_reflexive_gender_underspec_no_attractor",
            "UID": uid,
            "simple_LM_method": True,
            "one_prefix_method": False,
            "two_prefix_method": False,
            "lexically_identical": lexically_identical,  # these differ in the reflexive word
            "pair_id": pair_id,
        })
    return rows

out_dir = Path(__file__).resolve().parent / "reflexive_probes"
files = [
    ("reflexive_no_attractor_neutralNP_himself_vs_themselves.jsonl",
     "reflexive_no_attractor_neutralNP_himself_vs_themselves",
     "himself"),
    ("reflexive_no_attractor_neutralNP_herself_vs_themselves.jsonl",
     "reflexive_no_attractor_neutralNP_herself_vs_themselves",
     "herself"),
    # Optional:
    ("reflexive_no_attractor_neutralNP_themself_vs_themselves.jsonl",
     "reflexive_no_attractor_neutralNP_themself_vs_themselves",
     "themself"),
]

for fname, uid, good_refl in files:
    rows = build_pairs(uid=uid, good_refl=good_refl, n_items=200)
    write_jsonl(out_dir / fname, rows)
    print("wrote", fname, "with", len(rows), "pairs")
