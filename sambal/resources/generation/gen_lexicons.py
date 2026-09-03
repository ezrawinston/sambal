#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lexicon resource generator
==========================

Regenerates the bundled sambal lexicon resources from their upstream sources.
Every stage fetches its own inputs (URLs recorded in SOURCES.txt); nothing is
hand-typed. See generation/README.md for the stage -> resource -> source and
license map. Countability lists, to-infinitive lexicons, gendered-word,
proper-noun and vocabulary gates have their own generators in this directory.

Outputs (default: sambal/resources -- the committed files, overwritten in place):
  - wordnet_nouns.jsonl / wordnet_adjectives.jsonl / wordnet_adverbs.jsonl
  - human_nouns.jsonl              (WordNet noun.person inventory)
  - given_names_ssa.txt
  - function_words_ud_ewt.txt
  - fixed_mwes_ud_ewt.txt          (UD `fixed` MWEs; function-word complexes)
  - phrasal_verbs_ud_ewt.txt       (UD `compound:prt` verb-particle MWEs)
  - streusle_vmwe_patterns.jsonl / streusle_vmwes.txt
  - english_npis_wiktionary.tsv    (Wiktionary NPI category via API)
  - fixed_mwes_all.txt             (union of the three MWE lists)
  - SOURCES.txt                    (provenance log)
"""
import argparse
import collections
import io
import os
import re
import sys
import json
import zipfile
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Set, Optional
from collections import defaultdict
import requests
from conllu import parse_incr
from tqdm import tqdm

# downloaded inputs (including the per-year SSA files) live with the other
# fetched inputs, not with the generated resources; the SSA zip itself is
# committed in-repo because its endpoint rejects scripted requests
DOWNLOADS_DIR = Path("sambal/resources/generation/downloads")
SSA_ZIP = Path(__file__).resolve().parent / "ssa_names.zip"


# >>> HUMAN-NOUNS: imports
import re
try:
    import nltk
    from nltk.corpus import wordnet as wn
except Exception as _e:
    wn = None  # we'll check at runtime

try:
    # optional: only used to filter to lemmas whose head can realize NN & NNS
    from lemminflect import getInflection as _getInflection
    _HAVE_LEMMINFLECT = True
except Exception:
    _HAVE_LEMMINFLECT = False

# -------------------- utilities & provenance --------------------

def _cap_class(s: str) -> str:
    if s.islower(): return "lower"
    if s.istitle(): return "title"
    if s.isupper(): return "upper"
    return "mixed"

def write_jsonl_lexicon(out_path: str, entries: list[dict]) -> None:
    """
    entries: list of dicts with at least:
      - form: str     (EXACT token as you want it consumed later, case preserved)
      - pos:  "NOUN"|"ADJ"|"ADV"
      - tags: list[str] (optional; [] if you have none)
    Writes one JSON object per line. Keeps duplicates if 'form' differs by case.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for e in entries:
            if "tags" not in e:  # keep any tags you already computed
                e["tags"] = []
            # Store case info as a tag, so the augmenter can reason without recomputing
            if not any(t.startswith("CASE=") for t in e["tags"]):
                e["tags"].append(f"CASE={_cap_class(e['form'])}")
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def write_sources(outdir: Path, lines: List[str]):
    (outdir / "SOURCES.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

def write_list(path: Path, items: Iterable[str]):
    uniq = sorted({x.strip() for x in items if x and x.strip()}, key=str.lower)
    path.write_text("\n".join(uniq) + ("\n" if uniq else ""), encoding="utf-8")

# The UD extractor's token regex needs a word character first, so the clitic
# forms the engine needs cannot come out of it; appended after the sorted core.
FUNCTION_WORDS_HAND_ADDED = ("'s", "'t", "’s", "’t")

def write_function_words(path: Path, items: Iterable[str]):
    write_list(path, items)
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(FUNCTION_WORDS_HAND_ADDED) + "\n")

def export_wordnet_lemmas_jsonl(outdir: Path, underscore: str = "-") -> Tuple[Path, Path, Path]:
    """
    Write WordNet lemmas to JSONL with case-preserving 'form', 'pos' ∈ {NOUN, ADJ, ADV},
    and tags including wn:lexname + wn:offset. Underscores are replaced by `underscore`
    (use '_' to keep underscores; default '-' keeps outputs single-token).
    """
    import nltk
    from nltk.corpus import wordnet as wn
    try:
        wn.ensure_loaded()
    except LookupError:
        nltk.download("wordnet")
        wn.ensure_loaded()

    def _entries_for_pos(wn_pos_list: List[str], upos: str) -> List[dict]:
        seen = set()
        out: List[dict] = []
        # Accept letters, hyphen, apostrophe, and spaces (if underscore=" ")
        pat = re.compile(r"^[A-Za-z](?:[A-Za-z' -]*[A-Za-z])?$")
        for wn_pos in wn_pos_list:
            for syn in wn.all_synsets(pos=wn_pos):
                lex_tag = f"wn:{syn.lexname()}"
                off_tag = f"wn:offset:{syn.offset()}"
                for lem in syn.lemmas():
                    form = lem.name().replace("_", underscore)
                    if not pat.fullmatch(form):
                        continue
                    key = (upos, form)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append({"form": form, "pos": upos, "tags": [lex_tag, off_tag]})
        return out

    noun_entries = _entries_for_pos([wn.NOUN], "NOUN")
    # Include satellite adjectives ('s') in addition to 'a'
    adj_entries  = _entries_for_pos([wn.ADJ, wn.ADJ_SAT], "ADJ")
    adv_entries  = _entries_for_pos([wn.ADV], "ADV")

    nouns_jsonl = outdir / "wordnet_nouns.jsonl"
    adjs_jsonl  = outdir / "wordnet_adjectives.jsonl"
    advs_jsonl  = outdir / "wordnet_adverbs.jsonl"

    write_jsonl_lexicon(str(nouns_jsonl), noun_entries)
    write_jsonl_lexicon(str(adjs_jsonl),  adj_entries)
    write_jsonl_lexicon(str(advs_jsonl),  adv_entries)

    return nouns_jsonl, adjs_jsonl, advs_jsonl

def write_human_nouns_jsonl(out_path: str, require_both_numbers: bool = False) -> Path:
    """
    Emit human nouns to JSONL using your WordNet noun.person harvest.
    'form' is lowercased (WordNet is lowercase); tags include 'human' and source.
    """
    # WordNet files these two under noun.Tops / noun.group rather than
    # noun.person, so the harvest cannot reach the most generic human nouns.
    hand_added = {"person", "people"}
    entries = []
    for w in sorted(set(collect_human_nouns(require_both_numbers=require_both_numbers)) | hand_added):
        entries.append({"form": w, "pos": "NOUN", "tags": ["human", "source=wordnet.person"]})
    write_jsonl_lexicon(out_path, entries)
    return Path(out_path)


# -------------------- WordNet lemmas ----------------------------

def build_ssa_given_names(outdir: Path, sources: List[str]):
    zip_url = "https://www.ssa.gov/oact/babynames/names.zip"
    dest_zip = SSA_ZIP if SSA_ZIP.exists() else DOWNLOADS_DIR / "ssa_names.zip"
    dest_txt = outdir / "given_names_ssa.txt"

    if not dest_zip.exists():
        raise SystemExit(
            f"{SSA_ZIP} is missing (it ships with the repository). Restore it, "
            f"or download {zip_url} by hand to {DOWNLOADS_DIR / 'ssa_names.zip'} "
            "— the endpoint answers scripted requests with HTTP 403.")

    yob_dir = DOWNLOADS_DIR / "ssa_names"
    ensure_dir(yob_dir)
    names = set()
    with zipfile.ZipFile(dest_zip) as zf:
        for nm in zf.namelist():
            if nm.startswith("yob") and nm.endswith(".txt"):
                # keep the per-year files on disk for gen_gender_names.py
                zf.extract(nm, yob_dir)
                with zf.open(nm) as fh:
                    for line in io.TextIOWrapper(fh, encoding="utf-8"):
                        name = line.split(",")[0].strip()
                        if re.fullmatch(r"[A-Za-z][A-Za-z'-]*", name):
                            names.add(name)
    write_list(dest_txt, names)
    sources.append(f"SSA names (national)\t{zip_url}")
    return dest_txt


# -------------------- UD English-EWT ----------------------------

UD_EWT_BASE = "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/{ref}"
UD_FILES = ["en_ewt-ud-train.conllu", "en_ewt-ud-dev.conllu", "en_ewt-ud-test.conllu"]
FUNCTION_UPOS = {"ADP", "DET", "AUX", "PRON", "PART", "SCONJ", "CCONJ"}

def fetch_ud_ewt(outdir: Path, ref: str, sources: List[str]) -> List[Path]:
    local = []
    for fname in UD_FILES:
        url = f"{UD_EWT_BASE.format(ref=ref)}/{fname}"
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        p = outdir / fname
        p.write_bytes(r.content)
        local.append(p)
        sources.append(f"UD-EWT {fname}\t{url}")
    return local

def extract_ud_function_words(conllu_paths: List[Path]) -> Set[str]:
    forms = set()
    for p in conllu_paths:
        with p.open("r", encoding="utf-8") as fh:
            for sent in parse_incr(fh):
                for tok in sent:
                    if isinstance(tok["id"], int) and tok["upostag"] in FUNCTION_UPOS:
                        form = tok["form"]
                        if re.fullmatch(r"[^\W_]+(?:[-’'][^\W_]+)*", form):
                            forms.add(form.lower())
    return forms

def extract_ud_fixed_mwes(conllu_paths: List[Path]) -> Set[str]:
    mwes = set()
    for p in conllu_paths:
        with p.open("r", encoding="utf-8") as fh:
            for sent in parse_incr(fh):
                by_head = {}
                for tok in sent:
                    if isinstance(tok["id"], int) and tok.get("deprel") == "fixed":
                        head = tok["head"]
                        by_head.setdefault(head, []).append(tok)
                idx2tok = {tok["id"]: tok for tok in sent if isinstance(tok["id"], int)}
                for head_id, kids in by_head.items():
                    parts = [idx2tok.get(head_id)] + kids
                    parts = [t for t in parts if t is not None]
                    parts.sort(key=lambda t: t["id"])
                    phrase = " ".join(t["form"] for t in parts)
                    if len(phrase.split()) >= 2:
                        mwes.add(phrase.lower())
    return mwes

def extract_ud_phrasal_verbs(conllu_paths: List[Path]) -> Set[str]:
    vp = set()
    for p in conllu_paths:
        with p.open("r", encoding="utf-8") as fh:
            for sent in parse_incr(fh):
                id2tok = {tok["id"]: tok for tok in sent if isinstance(tok["id"], int)}
                for tok in sent:
                    if not isinstance(tok["id"], int):
                        continue
                    if tok.get("deprel") == "compound:prt":
                        head_id = tok.get("head")
                        if isinstance(head_id, int) and head_id in id2tok:
                            head = id2tok[head_id]
                            if head.get("upostag") in {"VERB", "AUX"}:
                                v_lemma = (head.get("lemma") or head.get("form") or "").lower()
                                prt = (tok.get("form") or "").lower()
                                if v_lemma and prt:
                                    vp.add(f"{v_lemma} {prt}")
    return vp


# -------------------- STREUSLE v4.7.1 (CONLLULEX → spaCy Matcher patterns) --------------------
# Source: https://github.com/nert-nlp/streusle (tag v4.7.1 ships CoNLL-U-Lex)
# Columns authority: CONLLULEX (SMWE, LEXCAT, LEXLEMMA) + base CoNLL-U columns (UPOS/LEMMA/FORM)

import io, json, re
from typing import List, Tuple, Dict, Set, Optional
import requests
from pathlib import Path

STREUSLE_CONLLULEX_URL = "https://raw.githubusercontent.com/nert-nlp/streusle/{ref}/streusle.conllulex"

# verbal MWE classes (PARSEME-style) retained
_V_LEXCAT_RE = re.compile(r"^V\.(?:VID|VPC\.full|VPC\.semi|LVC\.full|LVC\.cause|IAV)$", re.I)

# placeholders in the phrase string & generalized matchers
_PLACEHOLDER_FOR_UPOS = {"PRON": "PRON"}  # e.g., "work for PRON"

# POS allowed in the optional wildcard gap (to enable patterns like "trust * gut")
_WILDCARD_FILLER_POS = {"DET", "ADJ", "ADV", "NUM", "PRON", "PART"}

def _read_conllulex_sentences(blob: bytes) -> List[List[str]]:
    sents, cur = [], []
    for raw in io.StringIO(blob.decode("utf-8", errors="replace")):
        line = raw.rstrip("\n")
        if not line or line.startswith("#"):
            if cur: sents.append(cur); cur = []
            continue
        cur.append(line)
    if cur: sents.append(cur)
    return sents

def _conllulex_column_indices(header_cols: Optional[List[str]]) -> Tuple[int,int,int,int,int,int,int]:
    if header_cols:
        cols = [c.upper() for c in header_cols]
        try:
            i_ID      = cols.index("ID")
            i_FORM    = cols.index("FORM")
            i_LEMMA   = cols.index("LEMMA")
            i_UPOS    = cols.index("UPOS")
            i_SMWE    = cols.index("SMWE")
            i_LEXCAT  = cols.index("LEXCAT")
            i_LEXLEMMA= cols.index("LEXLEMMA")
            return i_ID, i_FORM, i_LEMMA, i_UPOS, i_SMWE, i_LEXCAT, i_LEXLEMMA
        except ValueError as e:
            raise RuntimeError("STREUSLE .conllulex header missing required columns") from e
    # Documented fallback order (0-based indices):
    # ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC  SMWE LEXCAT LEXLEMMA SS SS2 WMWE WCAT WLEMMA LEXTAG
    return 0, 1, 2, 3, 10, 11, 12

def _parse_conllulex(blob: bytes) -> List[Dict]:
    header_cols: Optional[List[str]] = None
    for raw in io.StringIO(blob.decode("utf-8", errors="replace")):
        if raw.startswith("# global.columns"):
            m = re.search(r"=\s*(.+)$", raw)
            if m: header_cols = m.group(1).strip().split()
            break
    i_ID, i_FORM, i_LEMMA, i_UPOS, i_SMWE, i_LEXCAT, i_LEXLEMMA = _conllulex_column_indices(header_cols)

    groups: List[Dict] = []
    for lines in _read_conllulex_sentences(blob):
        toks = []
        for line in lines:
            cols = line.split("\t")
            if len(cols) <= i_LEXLEMMA: raise RuntimeError("STREUSLE .conllulex row too few columns")
            if "-" in cols[i_ID] or "." in cols[i_ID]: continue
            try: tid = int(cols[i_ID])
            except ValueError: continue
            smwe = cols[i_SMWE]
            if not smwe or smwe == "_" or ":" not in smwe: continue
            gid, gpos = smwe.split(":", 1)
            try: gpos_i = int(gpos)
            except ValueError: gpos_i = 10**6
            toks.append({
                "gid": gid, "gpos": gpos_i, "id": tid,
                "form": cols[i_FORM],
                "lemma": cols[i_LEMMA].lower() if cols[i_LEMMA] != "_" else "",
                "upos": cols[i_UPOS],
                "lexcat": cols[i_LEXCAT],
                "lexlemma": cols[i_LEXLEMMA].lower() if cols[i_LEXLEMMA] != "_" else "",
            })
        by_gid: Dict[str, List[Dict]] = {}
        for t in toks: by_gid.setdefault(t["gid"], []).append(t)
        for gid, members in by_gid.items():
            members.sort(key=lambda x: x["gpos"])
            lexcat0 = members[0]["lexcat"] if members else ""
            if not _V_LEXCAT_RE.match(lexcat0 or ""): continue
            groups.append({
                "lexcat": lexcat0,
                "tokens": [{
                    "id": m["id"], "upos": m["upos"], "lemma": m["lemma"], "form": m["form"],
                    "smwe_pos": m["gpos"],
                } for m in members]
            })
    return groups

def _phrase_from_group(g: Dict) -> str:
    parts = []
    for t in g["tokens"]:
        upos = t["upos"]
        lemma = t["lemma"] or t["form"]
        parts.append(_PLACEHOLDER_FOR_UPOS.get(upos, lemma))
    return " ".join(w.strip() for w in parts if w.strip()).lower()

def _patterns_spacy_matcher_from_group(g: Dict) -> Dict[str, List[List[Dict]]]:
    """
    spaCy Matcher patterns (sequence). Emit:
      - 'tight': exact token sequence (LEMMA; pronouns generalized by POS)
      - 'wildcard': if group begins with VERB/AUX and ends with NOUN/PROPN,
                    allow a gap of filler POS between them (DET/ADJ/ADV/NUM/PRON/PART).
    """
    tight: List[Dict] = []
    for t in g["tokens"]:
        upos = t["upos"]; lemma = (t["lemma"] or t["form"])
        if upos == "PRON":
            tight.append({"POS": "PRON"})
        elif upos in {"VERB", "AUX"}:
            tight.append({"LEMMA": lemma, "POS": {"IN": ["VERB", "AUX"]}})
        else:
            tight.append({"LEMMA": lemma})
    out = {"tight": [tight]}

    toks = g["tokens"]
    if len(toks) >= 2 and toks[0]["upos"] in {"VERB","AUX"} and toks[-1]["upos"] in {"NOUN","PROPN"}:
        head = {"LEMMA": toks[0]["lemma"] or toks[0]["form"], "POS": {"IN": ["VERB","AUX"]}}
        tail = {"LEMMA": toks[-1]["lemma"] or toks[-1]["form"]}
        wildcard = {"POS": {"IN": sorted(_WILDCARD_FILLER_POS)}, "OP": "*"}
        out["wildcard"] = [[head, wildcard, tail]]
    return out

def build_streusle_vmwes(outdir: Path, ref: str, sources: List[str]) -> Path:
    """
    Fetch STREUSLE .conllulex for the given tag (e.g., v4.7.1), extract verbal MWEs, and write:
      • streusle_vmwes.txt               (space-separated, pronouns→PRON)
      • streusle_vmwe_patterns.jsonl     (spaCy Matcher patterns; 'tight' + 'wildcard')
    """
    url = STREUSLE_CONLLULEX_URL.format(ref=ref)
    r = requests.get(url, timeout=180); r.raise_for_status()
    groups = _parse_conllulex(r.content)

    phrases: Set[str] = set()
    jsonl_path = outdir / "streusle_vmwe_patterns.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as jf:
        for i, g in enumerate(groups):
            phrase = _phrase_from_group(g)
            if phrase: phrases.add(phrase)
            pats = _patterns_spacy_matcher_from_group(g)
            jf.write(json.dumps({
                "id": f"streusle:{i}",
                "lexcat": g["lexcat"],
                "phrase": phrase,
                "dialect": "spacy-matcher",
                "patterns": pats
            }, ensure_ascii=False) + "\n")

    if not phrases:
        raise RuntimeError("STREUSLE extraction produced zero verbal MWEs. Check the tag or parser.")
    txt_path = outdir / "streusle_vmwes.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        for p in sorted(phrases):
            f.write(p + "\n")

    sources.append(f"STREUSLE vMWEs\t{url}")
    sources.append(f"STREUSLE patterns\t{jsonl_path.name} (spacy-matcher)")
    return txt_path

# >>> HUMAN-NOUNS: helpers
_TOKEN_RE = re.compile(r"^[a-z]+(?: [a-z]+)*$")  # letters + single spaces

def _ensure_wordnet_loaded():
    if wn is None:
        raise RuntimeError(
            "WordNet not available. Install NLTK and data: `pip install nltk` then in Python: "
            ">>> import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
        )
    try:
        wn.ensure_loaded()
    except LookupError:
        import nltk as _nltk
        _nltk.download("wordnet")
        _nltk.download("omw-1.4")
        wn.ensure_loaded()

def _normalize_lemma(s: str) -> str:
    # WordNet uses underscores for multiwords
    s = s.replace("_", " ").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _ok_form(s: str) -> bool:
    # allow single/multi tokens; ban punctuation/digits
    return bool(_TOKEN_RE.fullmatch(s))

def _head_can_inflect_both(entry: str) -> bool:
    """Return True iff the head token (last word) can realize both NN and NNS (uses lemminflect)."""
    if not _HAVE_LEMMINFLECT:
        return True
    head = entry.split()[-1]
    return bool(_getInflection(head, tag="NN")) and bool(_getInflection(head, tag="NNS"))

def collect_human_nouns(require_both_numbers: bool = False) -> list[str]:
    """
    Authoritative source: Princeton WordNet (via NLTK), lexname == 'noun.person'.
    Returns lemmas with WordNet casing; normalizes only by replacing '_' with a single space
    and collapsing internal whitespace. Multiword lemmas are kept as space-separated.
    """
    _ensure_wordnet_loaded()
    out: set[str] = set()
    for syn in wn.all_synsets(pos="n"):
        if syn.lexname() != "noun.person":
            continue
        for lem in syn.lemma_names():
            # normalize like other nouns: underscores -> space, squeeze spaces; keep case
            entry = lem.replace("_", " ").strip()
            entry = re.sub(r"\s+", " ", entry)
            # validate using lowercase copy, but keep original case in output
            if not _ok_form(entry.lower()):
                continue
            if require_both_numbers and not _head_can_inflect_both(entry.lower()):
                continue
            out.add(entry)
    return sorted(out)



# --- add near your UD helpers in gen_lexicons_4.py ---
# -------------------- Wiktionary NPIs ---------------------------

WIKTIONARY_API = "https://en.wiktionary.org/w/api.php"

def fetch_wiktionary_npis_via_api(outdir: Path, sources: List[str]) -> Path:
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": "Category:English_negative_polarity_items",
        "cmlimit": "max",
        "format": "json",
    }
    sess = requests.Session()
    sess.headers.update({"User-Agent": "npigen/1.0 (categorymembers API)"})
    rows = []
    cont = None
    while True:
        q = dict(params)
        if cont:
            q.update(cont)
        r = sess.get(WIKTIONARY_API, params=q, timeout=60)
        r.raise_for_status()
        data = r.json()
        for item in data.get("query", {}).get("categorymembers", []):
            title = item.get("title", "")
            if ":" in title:
                continue
            url = f"https://en.wiktionary.org/wiki/{title.replace(' ', '_')}"
            rows.append((title, url))
        cont = data.get("continue")
        if not cont:
            break
    if not rows:
        raise RuntimeError("Wiktionary API returned zero NPIs")
    out = outdir / "english_npis_wiktionary.tsv"
    with out.open("w", encoding="utf-8") as f:
        f.write("expression\tsource_url\n")
        for t,u in sorted(rows, key=lambda x: x[0].lower()):
            f.write(f"{t}\t{u}\n")
    sources.append("Wiktionary NPIs via MediaWiki API\thttps://en.wiktionary.org/w/api.php")
    return out

def merge_fixed_mwes(outdir: Path, sources: List[str]) -> Path:
    a = outdir / "fixed_mwes_ud_ewt.txt"
    b = outdir / "phrasal_verbs_ud_ewt.txt"
    c = outdir / "streusle_vmwes.txt"
    if not (a.exists() and b.exists() and c.exists()):
        raise RuntimeError("Missing one of fixed_mwes_ud_ewt.txt / phrasal_verbs_ud_ewt.txt / streusle_vmwes.txt")
    merged = set()
    for p in (a,b,c):
        for ln in p.read_text(encoding="utf-8").splitlines():
            w = ln.strip()
            if w:
                merged.add(w)
    out = outdir / "fixed_mwes_all.txt"
    write_list(out, merged)
    sources.append("Merged MWEs\tfixed_mwes_all.txt = UD fixed ∪ UD VPC ∪ STREUSLE vMWEs")
    return out


# -------------------- CLI main ---------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Regenerate the sambal bundled lexicon resources from their "
                    "upstream sources. Writes the committed files in place -- git shows "
                    "what changed. See generation/README.md for the "
                    "stage -> resource -> source/license map.")
    ap.add_argument("--out", "--outdir", dest="outdir", type=str,
                    default="sambal/resources")
    ap.add_argument("--ud-ewt-ref", type=str,
                    default="4c89b5833a70aa5ed3a00bad2f23f57992cc7df8",
                    help="UD_English-EWT commit, tag or branch. Defaults to the "
                         "UD 2.16 commit (tag r2.16) that the committed resources "
                         "were built from; pass 'master' for current upstream UD.")
    ap.add_argument("--streusle-ref", type=str, default="v4.7.1",
                    help="STREUSLE tag/branch (e.g., master).")
    ap.add_argument("--skip-wiktionary-npi", action="store_true",
                    help="Skip the Wiktionary NPI category fetch.")
    ap.add_argument("--human-require-both-numbers", action="store_true",
                    help="Keep only human-noun entries whose head inflects to "
                         "both NN and NNS (requires lemminflect).")
    ap.add_argument("--wn-underscore", type=str, default="-",
                    help="Replacement for '_' in WordNet lemmas for JSONL outputs. "
                         "Use '_' to keep underscores; default '-' keeps outputs single-token.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    sources = []

    print("Exporting WordNet lemmas (JSONL)...")
    wn_nouns_jsonl, wn_adjs_jsonl, wn_advs_jsonl = export_wordnet_lemmas_jsonl(outdir, underscore=args.wn_underscore)
    print(f"Wrote JSONL: {wn_nouns_jsonl}, {wn_adjs_jsonl}, {wn_advs_jsonl}")

    print("Collecting human nouns (WordNet noun.person, JSONL)...")
    outp = write_human_nouns_jsonl(str(outdir / "human_nouns.jsonl"),
                                   require_both_numbers=args.human_require_both_numbers)
    print(f"[human_nouns.jsonl] wrote {outp}")

    print("Building SSA given names...")
    build_ssa_given_names(outdir, sources)

    print(f"Fetching UD-EWT ({args.ud_ewt_ref}) and extracting function words & MWEs...")
    conllu_paths = fetch_ud_ewt(outdir, args.ud_ewt_ref, sources)
    fwords = extract_ud_function_words(conllu_paths)
    fixed = extract_ud_fixed_mwes(conllu_paths)
    vpcs = extract_ud_phrasal_verbs(conllu_paths)
    write_function_words(outdir / "function_words_ud_ewt.txt", fwords)
    write_list(outdir / "fixed_mwes_ud_ewt.txt", fixed)
    write_list(outdir / "phrasal_verbs_ud_ewt.txt", vpcs)

    print(f"Fetching STREUSLE ({args.streusle_ref}) and extracting vMWEs...")
    build_streusle_vmwes(outdir, args.streusle_ref, sources)

    if not args.skip_wiktionary_npi:
        print("Fetching lexical NPIs via MediaWiki API...")
        fetch_wiktionary_npis_via_api(outdir, sources)

    print("Merging UD fixed + UD VPC + STREUSLE vMWEs...")
    merge_fixed_mwes(outdir, sources)

    write_sources(outdir, sources)
    print(f"Done. Review outputs in {outdir.resolve()}, then copy the resource "
          f"files into sambal/resources/.")


if __name__ == "__main__":
    main()
