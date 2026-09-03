#!/usr/bin/env python
"""Build the inputs the evaluator goldens run on.

Everything here is deterministic, so re-running rewrites identical files:

* ``fixtures/tiny_model.pt`` -- a GPT-BERT state dict for
  ``lm/gpt-bert/configs/tiny.json``. Its matrices are drawn from a fixed
  numpy ``RandomState`` stream (stable across numpy versions) and rounded to
  float16, so the loaded weights are exactly the same on every platform; the
  file is not committed -- ``ensure_fixtures()`` rebuilds it on demand and
  checks it against the committed digest in ``fixtures/tiny_model.sha256``;
* one small input set per evaluator format, written from the sentence
  material below (no benchmark rows are copied), except the conflict pool and
  controls, which are a slice of this repository's own committed suite.

Usage: ``python tests/evals/make_fixtures.py``
"""

import hashlib
import json
import pathlib
import random
import sys
from types import SimpleNamespace

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIX = HERE / "fixtures"
TINY_CONFIG = REPO / "lm/gpt-bert/configs/tiny.json"
TINY_MODEL = FIX / "tiny_model.pt"
TINY_DIGEST = FIX / "tiny_model.sha256"
PRETRAINING = REPO / "lm/gpt-bert/pretraining"
SEED = 0


def state_digest(state):
    """sha256 over every tensor's name, dtype, shape and raw bytes (sorted by name)."""
    h = hashlib.sha256()
    for key in sorted(state):
        t = state[key].detach().cpu().contiguous()
        h.update(f"{key}|{t.dtype}|{tuple(t.shape)}|".encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def tiny_state():
    """The tiny model's state dict, generated without touching torch's RNG."""
    import numpy as np
    import torch

    sys.path.insert(0, str(PRETRAINING))
    from model_extra import Bert  # noqa: E402  (the training-side class; only its shapes are used)

    cfg = SimpleNamespace(**json.loads(TINY_CONFIG.read_text()))
    model = Bert(cfg)
    rng = np.random.RandomState(SEED)
    std = (2.0 / (5.0 * cfg.hidden_size)) ** 0.5  # the model's own init scale
    state, by_storage = {}, {}
    for key, value in model.state_dict(keep_vars=True).items():
        value = value.detach()
        if not value.is_floating_point():
            # position-index buffers: deterministic, small ints -> int16
            assert int(value.max()) < 32768 and int(value.min()) >= 0
            state[key] = value.to(torch.int16).clone()
            continue
        if value.ndim < 2:
            # biases, layer-norm gains, depth-mixing weights: the model's deterministic init
            state[key] = value.to(torch.float16).clone()
            continue
        ptr = value.data_ptr()
        if ptr in by_storage:  # tied weights (the output projection reuses the word embedding)
            state[key] = state[by_storage[ptr]]
            continue
        arr = rng.standard_normal(tuple(value.shape)).clip(-2.0, 2.0) * std
        state[key] = torch.from_numpy(arr.astype(np.float16))
        by_storage[ptr] = key
    return state


def build_tiny_model():
    import torch

    state = tiny_state()
    FIX.mkdir(parents=True, exist_ok=True)
    torch.save(state, TINY_MODEL)
    digest = state_digest(state)
    TINY_DIGEST.write_text(digest + "\n")
    n_params = sum(v.numel() for v in state.values() if v.is_floating_point())
    return TINY_MODEL, n_params, digest


def ensure_fixtures():
    """Rebuild the tiny model when missing and verify it against the committed digest."""
    import torch

    if not TINY_MODEL.exists():
        build_tiny_model()
    if TINY_DIGEST.exists():
        got = state_digest(torch.load(TINY_MODEL, map_location="cpu"))
        want = TINY_DIGEST.read_text().strip()
        if got != want:
            raise RuntimeError(
                f"{TINY_MODEL} does not match {TINY_DIGEST} ({got} vs {want}); "
                "delete the .pt to rebuild it, or rerun make_fixtures.py if the generator changed")
    return TINY_MODEL


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# BLiMP-format minimal pairs (three paradigms + one supplemental file)
# ---------------------------------------------------------------------------
BLIMP = {
    "agreement_subject_verb": ("morphology", "subject_verb_agreement", [
        ("The dogs run across the yard.", "The dogs runs across the yard."),
        ("My sister bakes bread on Sundays.", "My sister bake bread on Sundays."),
        ("Those children weren't listening.", "Those children wasn't listening."),
        ("The teacher who met the students smiles.", "The teacher who met the students smile."),
        ("Sam's cousins have arrived.", "Sam's cousins has arrived."),
    ]),
    "anaphor_agreement": ("syntax", "anaphor_agreement", [
        ("The girl saw herself in the mirror.", "The girl saw himself in the mirror."),
        ("The boys can't help themselves.", "The boys can't help himself."),
        ("Maria bought herself a coat.", "Maria bought themselves a coat."),
        ("The men praised themselves loudly.", "The men praised herself loudly."),
        ("Our neighbor hurt himself yesterday.", "Our neighbor hurt themselves yesterday."),
    ]),
    "determiner_noun": ("morphology", "determiner_noun_agreement", [
        ("These books are new.", "This books are new."),
        ("That apple looks ripe.", "Those apple looks ripe."),
        ("I read this letter twice.", "I read these letter twice."),
        ("Kim lost those keys again.", "Kim lost that keys again."),
        ("Every guest brought a gift.", "Every guests brought a gift."),
    ]),
}

# Supplemental file: no ``field`` key, like the reflexive probe files.
SUPPLEMENTAL = [
    ("The manager who visited the students congratulated himself.",
     "The manager who visited the students congratulated themselves."),
    ("The nurses that helped the doctor trusted themselves.",
     "The nurses that helped the doctor trusted herself."),
    ("The pilot near the passengers blamed himself.",
     "The pilot near the passengers blamed themselves."),
    ("The authors who thanked the editor described themselves.",
     "The authors who thanked the editor described himself."),
    ("The child beside the parents amused herself.",
     "The child beside the parents amused themselves."),
]


def write_blimp():
    out = FIX / "blimp"
    for uid, (field, term, pairs) in BLIMP.items():
        rows = []
        for i, (good, bad) in enumerate(pairs):
            rows.append({
                "sentence_good": good,
                "sentence_bad": bad,
                "field": field,
                "linguistics_term": term,
                "UID": uid,
                "simple_LM_method": True,
                "one_prefix_method": False,
                "two_prefix_method": False,
                "lexically_identical": False,
                "pair_id": i,
            })
        write_jsonl(out / f"{uid}.jsonl", rows)
    write_jsonl(out / "reflexive_probe_tiny.jsonl",
                [{"sentence_good": g, "sentence_bad": b} for g, b in SUPPLEMENTAL])


# ---------------------------------------------------------------------------
# SyntaxGym-format suites: one file per suite name the trainers select, two
# items each, a formula family per name prefix so every operator the formula
# parser supports (+, -, <, >, =, &, |, region ``*``) is exercised.
# ---------------------------------------------------------------------------
SUITE_NAMES = [
    "number_orc", "number_prep", "number_src",
    "reflexive_orc_fem", "reflexive_orc_masc",
    "reflexive_prep_fem", "reflexive_prep_masc",
    "reflexive_src_fem", "reflexive_src_masc",
    "npi_orc_any", "npi_orc_ever", "npi_src_any", "npi_src_ever",
    "fgd_hierarchy", "fgd_object", "fgd_pp", "fgd_subject",
    "center_embed", "center_embed_mod",
    "cleft", "cleft_modifier",
    "subordination", "subordination_orc-orc", "subordination_pp-pp", "subordination_src-src",
    "mvrr", "mvrr_mod",
    "npz_ambig", "npz_ambig_mod", "npz_obj", "npz_obj_mod",
]

NOUNS = ["author", "farmer", "pilot", "nurse", "senator", "teacher", "actor", "dancer"]
PL = {"author": "authors", "farmer": "farmers", "pilot": "pilots", "nurse": "nurses",
      "senator": "senators", "teacher": "teachers", "actor": "actors", "dancer": "dancers"}
VERBS_SG = ["laughs", "waits", "smiles", "leaves"]
VERBS_PL = ["laugh", "wait", "smile", "leave"]


def regions(*contents):
    return [{"region_number": i + 1, "content": c} for i, c in enumerate(contents)]


def item(number, conds):
    return {"item_number": number,
            "conditions": [{"condition_name": n, "regions": regions(*r)} for n, r in conds]}


def make_suite(name, rng):
    n1, n2, n3 = rng.sample(NOUNS, 3)
    vi = rng.randrange(len(VERBS_SG))
    vs, vp = VERBS_SG[vi], VERBS_PL[vi]
    items, preds = [], []
    if name.startswith("number"):
        for k in range(2):
            a, b = (n1, n2) if k == 0 else (n2, n3)
            items.append(item(k + 1, [
                ("match", ("The", a, "that the", PL[b], "admired", vs, "quietly", ".")),
                ("mismatch", ("The", a, "that the", PL[b], "admired", vp, "quietly", ".")),
            ]))
        preds = ["(6;%match%) < (6;%mismatch%)"]
    elif name.startswith("reflexive"):
        for k in range(2):
            a, b = (n1, n2) if k == 0 else (n3, n1)
            items.append(item(k + 1, [
                ("match_sg", ("The", a, "near the", PL[b], "praised", "herself", ".")),
                ("mismatch_sg", ("The", a, "near the", PL[b], "praised", "themselves", ".")),
                ("match_pl", ("The", PL[a], "near the", b, "praised", "themselves", ".")),
                ("mismatch_pl", ("The", PL[a], "near the", b, "praised", "herself", ".")),
            ]))
        preds = ["((6;%match_sg%) < (6;%mismatch_sg%)) & ((6;%match_pl%) < (6;%mismatch_pl%))"]
    elif name.startswith("npi"):
        for k in range(2):
            a, b = (n2, n1) if k == 0 else (n3, n2)
            items.append(item(k + 1, [
                ("good", ("No", a, "that the", PL[b], "liked has", "ever", "left", ".")),
                ("bad", ("The", a, "that the", PL[b], "liked has", "ever", "left", ".")),
            ]))
        preds = ["(*;%good%) < (*;%bad%)"]
    elif name.startswith("fgd"):
        for k in range(2):
            a, b = (n1, n3) if k == 0 else (n2, n1)
            items.append(item(k + 1, [
                ("what_gap", ("I know", "what", "the", a, "sent", "", "to the", b, "yesterday", ".")),
                ("that_gap", ("I know", "that", "the", a, "sent", "", "to the", b, "yesterday", ".")),
                ("what_nogap", ("I know", "what", "the", a, "sent", "the letter", "to the", b, "yesterday", ".")),
                ("that_nogap", ("I know", "that", "the", a, "sent", "the letter", "to the", b, "yesterday", ".")),
            ]))
        preds = ["((7;%what_gap%) - (7;%that_gap%)) < ((7;%what_nogap%) - (7;%that_nogap%))"]
    elif name.startswith("center_embed"):
        for k in range(2):
            a, b = (n3, n2) if k == 0 else (n1, n3)
            items.append(item(k + 1, [
                ("plaus", ("The", "letter", "that the", a, "wrote", "arrived", ".")),
                ("implaus", ("The", "letter", "that the", a, "arrived", "wrote", ".")),
            ]))
        preds = ["( (5;%plaus%) + (6;%plaus%) ) < ( (5;%implaus%) + (6;%implaus%) )"]
    elif name.startswith("cleft"):
        for k in range(2):
            a = n1 if k == 0 else n2
            items.append(item(k + 1, [
                ("np_match", ("What the", a, "saw", "was", "a bird", ".")),
                ("np_mismatch", ("What the", a, "did", "was", "a bird", ".")),
                ("vp_match", ("What the", a, "did", "was", "see a bird", ".")),
                ("vp_mismatch", ("What the", a, "saw", "was", "see a bird", ".")),
            ]))
        preds = ["((5;%np_match%) < (5;%np_mismatch%)) & ((5;%vp_match%) < (5;%vp_mismatch%))"]
    elif name.startswith("subordination"):
        for k in range(2):
            a, b = (n2, n3) if k == 0 else (n3, n1)
            items.append(item(k + 1, [
                ("sub_no", ("", "The", a, "saw the", b, ",", "the play began", ".")),
                ("sub_yes", ("After", "the", a, "saw the", b, ",", "the play began", ".")),
                ("no_sub_no", ("", "The", a, "saw the", b, ".", "", "")),
                ("no_sub_yes", ("After", "the", a, "saw the", b, ".", "", "")),
            ]))
        preds = ["((7;%sub_yes%) - (7;%sub_no%)) < ((7;%no_sub_yes%) - (7;%no_sub_no%))"]
    else:  # mvrr / npz: an "or" of two region comparisons
        for k in range(2):
            a, b = (n1, n2) if k == 0 else (n3, n2)
            items.append(item(k + 1, [
                ("ambig", ("While the", a, "hunted", "the", b, "ran", "away", ".")),
                ("unambig", ("While the", a, "hunted", ",", "the", b, "ran", "away", ".")),
            ]))
        preds = ["((6;%ambig%) > (6;%unambig%)) | ((7;%ambig%) > (7;%unambig%))"]
    return {
        "meta": {"name": name, "metric": "sum", "comment": "tiny fixture", "reference": "", "author": ""},
        "region_meta": {},
        "predictions": [{"type": "formula", "formula": f} for f in preds],
        "items": items,
    }


def write_syntaxgym():
    out = FIX / "syntaxgym"
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    for name in SUITE_NAMES:
        (out / f"{name}.json").write_text(json.dumps(make_suite(name, rng), indent=1) + "\n")


# ---------------------------------------------------------------------------
# EWoK-format items (the fields the scorer reads plus the concept columns)
# ---------------------------------------------------------------------------
EWOK = [
    ("spatial-relations", "above", "below", "direct", "antonym",
     "The lamp is above the desk.", "The lamp is below the desk.",
     "The lamp hangs over the desk.", "The lamp sits under the desk."),
    ("spatial-relations", "inside", "outside", "direct", "antonym",
     "The cat is inside the box.", "The cat is outside the box.",
     "The box holds the cat.", "The box is empty."),
    ("social-relations", "friend", "enemy", "indirect", "variable swap",
     "Mia helps Noah every day.", "Noah helps Mia every day.",
     "Mia is kind to Noah.", "Noah is kind to Mia."),
    ("social-relations", "teacher", "student", "direct", "antonym",
     "Ana teaches Ben to read.", "Ben teaches Ana to read.",
     "Ana is the teacher.", "Ben is the teacher."),
    ("physical-relations", "heavy", "light", "direct", "antonym",
     "The stone is heavy.", "The stone is light.",
     "The stone sinks in the pond.", "The stone floats on the pond."),
    ("physical-relations", "hot", "cold", "indirect", "antonym",
     "The soup was just cooked.", "The soup was left in the snow.",
     "The soup is hot.", "The soup is cold."),
    ("material-properties", "glass", "wood", "direct", "concept swap",
     "The bowl is made of glass.", "The bowl is made of wood.",
     "The bowl can shatter.", "The bowl can burn."),
    ("material-properties", "wet", "dry", "direct", "antonym",
     "The towel is wet.", "The towel is dry.",
     "The towel drips on the floor.", "The towel stays folded."),
    ("agent-properties", "believe", "doubt", "direct", "concept swap",
     "Lee sees the bird on the roof.", "Lee sees the bird fly away.",
     "Lee believes the bird is on the roof.", "Lee doubts the bird is on the roof."),
    ("agent-properties", "want", "avoid", "indirect", "antonym",
     "Sam is very hungry.", "Sam is very full.",
     "Sam wants the sandwich.", "Sam avoids the sandwich."),
]


def write_ewok():
    rows = []
    for i, (dom, ca, cb, ctype, cdiff, c1, c2, t1, t2) in enumerate(EWOK):
        for flip in (False, True):
            rows.append({
                "Domain": dom, "ConceptA": ca, "ConceptB": cb,
                "ContextType": ctype, "ContextDiff": cdiff, "TargetDiff": "concept swap",
                "Context1": c2 if flip else c1, "Context2": c1 if flip else c2,
                "Target1": t2 if flip else t1, "Target2": t1 if flip else t2,
            })
    write_jsonl(FIX / "ewok" / "tiny_domains.jsonl", rows)


# ---------------------------------------------------------------------------
# SNLI-format pairs: premise x {entailment, neutral, contradiction}
# ---------------------------------------------------------------------------
SNLI = [
    ("A man is playing a guitar on the street.",
     "A man is making music outdoors.", "The man is a professional musician.", "A man is sleeping in a bed."),
    ("Two dogs are running through a field.",
     "Some animals are outside.", "The dogs are chasing a ball.", "The dogs are asleep indoors."),
    ("A woman in a red coat is crossing the road.",
     "A person is walking across the street.", "The woman is late for work.", "The woman is swimming in a lake."),
    ("Children are building a sandcastle at the beach.",
     "Kids are playing in the sand.", "The children are siblings.", "The children are doing homework in class."),
    ("An old man reads a newspaper on a bench.",
     "A man is reading.", "The man is reading about sports.", "The man is running a marathon."),
    ("A chef is chopping vegetables in a kitchen.",
     "Someone is preparing food.", "The chef is making soup.", "The kitchen is empty."),
    ("A girl is riding a bicycle down a hill.",
     "A child is on a bike.", "The girl is racing her brother.", "The girl is sitting on a couch."),
    ("Three people are waiting at a bus stop.",
     "People are standing outside.", "The bus is running late.", "Nobody is at the bus stop."),
    ("A cat sleeps on a warm windowsill.",
     "A cat is resting.", "The cat is dreaming of mice.", "The cat is chasing a bird outside."),
    ("A boy throws a ball to his dog.",
     "A child plays with a pet.", "The dog is a golden retriever.", "The boy is reading quietly."),
    ("Workers are repairing the roof of a house.",
     "People are working on a building.", "The roof was damaged by a storm.", "The house has no roof."),
    ("A couple dances at a wedding.",
     "Two people are dancing.", "The couple just got married.", "Everyone at the wedding is seated."),
    ("A student writes notes in a library.",
     "Someone is writing.", "The student is studying for an exam.", "The library is closed."),
    ("A farmer feeds the chickens at dawn.",
     "Animals are being fed.", "The farmer owns many chickens.", "The chickens are being sold."),
    ("Snow covers the quiet mountain village.",
     "The village is snowy.", "The villagers are skiing.", "The village is hot and sunny."),
    ("A pilot inspects the plane before takeoff.",
     "A person checks an aircraft.", "The flight is going overseas.", "The plane has already landed."),
    ("A baker sells bread at the market.",
     "Bread is for sale.", "The bread is freshly baked.", "The market is empty of food."),
    ("Two friends share an umbrella in the rain.",
     "It is raining.", "The friends are going to a movie.", "The sun is shining brightly."),
]


def write_snli():
    out = FIX / "snli"
    labels = ["entailment", "neutral", "contradiction"]
    splits = {"train": SNLI[:10], "dev": SNLI[10:14], "test": SNLI[14:18]}
    for split, rows_in in splits.items():
        rows = []
        for i, (premise, ent, neu, con) in enumerate(rows_in):
            for label, hyp in zip(labels, (ent, neu, con)):
                pid = f"{split}{i:03d}{label[0]}"
                rows.append({"annotator_labels": [label], "captionID": f"cap{i:03d}",
                             "gold_label": label, "pairID": pid,
                             "sentence1": premise, "sentence2": hyp})
        if split == "train":  # one unlabeled row, skipped by the loader
            rows.append({"annotator_labels": ["-"], "captionID": "capx", "gold_label": "-",
                         "pairID": "trainx", "sentence1": rows_in[0][0], "sentence2": "Nothing is happening."})
        write_jsonl(out / f"snli_1.0_{split}.jsonl", rows)


# ---------------------------------------------------------------------------
# A small CoNLL-U file (words, UPOS, heads, deprels; one multiword token line)
# ---------------------------------------------------------------------------
CONLLU = """# sent_id = tiny-1
# text = The cat sleeps on the mat.
1\tThe\tthe\tDET\tDT\t_\t2\tdet\t_\t_
2\tcat\tcat\tNOUN\tNN\t_\t3\tnsubj\t_\t_
3\tsleeps\tsleep\tVERB\tVBZ\t_\t0\troot\t_\t_
4\ton\ton\tADP\tIN\t_\t6\tcase\t_\t_
5\tthe\tthe\tDET\tDT\t_\t6\tdet\t_\t_
6\tmat\tmat\tNOUN\tNN\t_\t3\tobl\t_\t_
7\t.\t.\tPUNCT\t.\t_\t3\tpunct\t_\t_

# sent_id = tiny-2
# text = The dog chased the cat quickly.
1\tThe\tthe\tDET\tDT\t_\t2\tdet\t_\t_
2\tdog\tdog\tNOUN\tNN\t_\t3\tnsubj\t_\t_
3\tchased\tchase\tVERB\tVBD\t_\t0\troot\t_\t_
4\tthe\tthe\tDET\tDT\t_\t5\tdet\t_\t_
5\tcat\tcat\tNOUN\tNN\t_\t3\tobj\t_\t_
6\tquickly\tquickly\tADV\tRB\t_\t3\tadvmod\t_\t_
7\t.\t.\tPUNCT\t.\t_\t3\tpunct\t_\t_

# sent_id = tiny-3
# text = The dogs don't like the rain.
1\tThe\tthe\tDET\tDT\t_\t2\tdet\t_\t_
2\tdogs\tdog\tNOUN\tNNS\t_\t5\tnsubj\t_\t_
3-4\tdon't\t_\t_\t_\t_\t_\t_\t_\t_
3\tdo\tdo\tAUX\tVBP\t_\t5\taux\t_\t_
4\tn't\tnot\tPART\tRB\t_\t5\tadvmod\t_\t_
5\tlike\tlike\tVERB\tVB\t_\t0\troot\t_\t_
6\tthe\tthe\tDET\tDT\t_\t7\tdet\t_\t_
7\train\train\tNOUN\tNN\t_\t5\tobj\t_\t_
8\t.\t.\tPUNCT\t.\t_\t5\tpunct\t_\t_

# sent_id = tiny-4
# text = A cat and a dog sat on the mat.
1\tA\ta\tDET\tDT\t_\t2\tdet\t_\t_
2\tcat\tcat\tNOUN\tNN\t_\t6\tnsubj\t_\t_
3\tand\tand\tCCONJ\tCC\t_\t5\tcc\t_\t_
4\ta\ta\tDET\tDT\t_\t5\tdet\t_\t_
5\tdog\tdog\tNOUN\tNN\t_\t2\tconj\t_\t_
6\tsat\tsit\tVERB\tVBD\t_\t0\troot\t_\t_
7\ton\ton\tADP\tIN\t_\t9\tcase\t_\t_
8\tthe\tthe\tDET\tDT\t_\t9\tdet\t_\t_
9\tmat\tmat\tNOUN\tNN\t_\t6\tobl\t_\t_
10\t.\t.\tPUNCT\t.\t_\t6\tpunct\t_\t_

# sent_id = tiny-5
# text = The rain stopped and the dog ran.
1\tThe\tthe\tDET\tDT\t_\t2\tdet\t_\t_
2\train\train\tNOUN\tNN\t_\t3\tnsubj\t_\t_
3\tstopped\tstop\tVERB\tVBD\t_\t0\troot\t_\t_
4\tand\tand\tCCONJ\tCC\t_\t7\tcc\t_\t_
5\tthe\tthe\tDET\tDT\t_\t6\tdet\t_\t_
6\tdog\tdog\tNOUN\tNN\t_\t7\tnsubj\t_\t_
7\tran\trun\tVERB\tVBD\t_\t3\tconj\t_\t_
8\t.\t.\tPUNCT\t.\t_\t3\tpunct\t_\t_

# sent_id = tiny-6
# text = Cats like the mat more than dogs do.
1\tCats\tcat\tNOUN\tNNS\t_\t2\tnsubj\t_\t_
2\tlike\tlike\tVERB\tVBP\t_\t0\troot\t_\t_
3\tthe\tthe\tDET\tDT\t_\t4\tdet\t_\t_
4\tmat\tmat\tNOUN\tNN\t_\t2\tobj\t_\t_
5\tmore\tmore\tADV\tRBR\t_\t2\tadvmod\t_\t_
6\tthan\tthan\tSCONJ\tIN\t_\t8\tmark\t_\t_
7\tdogs\tdog\tNOUN\tNNS\t_\t8\tnsubj\t_\t_
8\tdo\tdo\tAUX\tVBP\t_\t5\tadvcl\t_\t_
9\t.\t.\tPUNCT\t.\t_\t2\tpunct\t_\t_
"""


def write_conllu():
    out = FIX / "ud" / "tiny.conllu"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(CONLLU)


# ---------------------------------------------------------------------------
# Conflict pool / controls: a family-balanced slice of the committed suite
# ---------------------------------------------------------------------------
def write_conflict_slice():
    per_family = {"sva_role_reversal": 4, "detnum_role_reversal": 3, "passive_role_reversal": 3}
    for src, dst, suffix in (("evals/conflict/pool/conflict_pool.jsonl", "pool.jsonl", ""),
                             ("evals/conflict/controls.jsonl", "controls.jsonl", "_control")):
        taken = {k: 0 for k in per_family}
        kept = []
        with open(REPO / src, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                fam = json.loads(line)["linguistics_term"].replace(suffix, "")
                if fam in taken and taken[fam] < per_family[fam]:
                    taken[fam] += 1
                    kept.append(line)
        out = FIX / "conflict" / dst
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            fh.writelines(kept)


# ---------------------------------------------------------------------------
# Sentences for the tensor-level goldens (varied length and punctuation)
# ---------------------------------------------------------------------------
SENTENCES = [
    "The cat sat on the mat.",
    "Katherine can't help herself.",
    "Go home.",
    "Where did the students who arrived late sit?",
    "The painting that the artist painted deteriorated, sadly.",
    "Sam's friends don't know it's raining; they left anyway.",
]


def main():
    FIX.mkdir(parents=True, exist_ok=True)
    write_blimp()
    write_syntaxgym()
    write_ewok()
    write_snli()
    write_conllu()
    write_conflict_slice()
    (FIX / "sentences.json").write_text(json.dumps(SENTENCES, indent=1) + "\n")
    out, n_params, digest = build_tiny_model()
    print(f"wrote {out} ({n_params} float parameters, float16; sha256 {digest[:12]}...)")
    print(f"fixtures under {FIX}")


if __name__ == "__main__":
    main()
