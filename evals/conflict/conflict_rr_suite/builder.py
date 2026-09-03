"""Candidate generation + model-in-the-loop selection.

The selection logic is designed to address two failure modes you observed:

1) "Phenomenon too hard" — even without the adversarial lure, models fail.
   -> We filter with a control minimal pair that isolates the grammatical phenomenon
      but keeps semantics plausible.

2) "Lure not obvious" — PMI-driven selection yields pairs where neither variant is
   clearly more plausible.
   -> We use curated micro-frames (Frame) so plausibility is interpretable, then
      select items where the *baseline model* prefers the ungrammatical-but-plausible
      sentence (conflict), despite passing the control.

Sentences are scored through ``LanguageModel.sequence_logprobs``: the sum of the
token log-probabilities of a sentence at one temperature.

Output is minimal-pair JSONL.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from evals.backends import LanguageModel

from .lexicons import DET_FLEX, DET_SG_OPTIONS, FRAMES
from .templates import ItemSentences, detnum_role_reversal, sva_role_reversal, passive_role_reversal
from .text_utils import is_swap_only


@dataclass
class ScoredItem:
    item: ItemSentences
    # baseline scores
    base_good: float
    base_bad: float
    base_ctrl_good: float
    # derived
    base_delta_conf: float   # good - bad (negative => prefers bad)
    base_delta_ctrl: float   # ctrl_good - bad (positive => prefers ctrl_good)

    # optional secondary model (e.g., SAMBAL)
    other_good: Optional[float] = None
    other_bad: Optional[float] = None
    other_ctrl_good: Optional[float] = None
    other_delta_conf: Optional[float] = None
    other_delta_ctrl: Optional[float] = None


def score_sentences(model: LanguageModel, sentences: Sequence[str], temperature: float) -> List[float]:
    """Sentence log-probability sums for ``sentences`` at one temperature."""
    if len(sentences) == 0:
        return []
    seqs = [model.encode(s) for s in sentences]
    return [row[0] for row in model.sequence_logprobs(seqs, [temperature]).tolist()]


def _score_triplets(
    *,
    model: LanguageModel,
    triplets: Sequence[Tuple[str,str,str]],
    temperature: float,
) -> List[Tuple[float,float,float]]:
    """Score (good, bad, ctrl_good) for a batch of items."""
    flat: List[str] = []
    for g,b,cg in triplets:
        flat.extend([g,b,cg])
    scores = score_sentences(model, flat, temperature)
    return [(scores[i], scores[i + 1], scores[i + 2]) for i in range(0, len(scores), 3)]


def sample_items(
    *,
    rng: random.Random,
    n: int,
    family: str,
    only_inanimate_objects: bool = True,
) -> List[ItemSentences]:
    """Sample raw (unscored) items for a given family."""
    items: List[ItemSentences] = []
    attempts = 0
    while len(items) < n and attempts < n * 50:
        attempts += 1
        frame = rng.choice(FRAMES)
        if only_inanimate_objects and frame.obj_kind != "inanimate":
            continue
        A_pl = rng.choice(list(frame.subj_pl))
        B_sg = rng.choice(list(frame.obj_sg))
        if A_pl == B_sg:
            continue

        if family == "sva":
            it = sva_role_reversal(frame, A_pl=A_pl, B_sg=B_sg)
        elif family == "det":
            det_sg = rng.choice(DET_SG_OPTIONS)
            it = detnum_role_reversal(frame, A_pl=A_pl, B_sg=B_sg, det_sg=det_sg, det_flex=DET_FLEX)
        elif family == "passive":
            it = passive_role_reversal(frame, A_pl=A_pl, B_sg=B_sg)
        else:
            raise ValueError("family must be 'sva', 'det', or 'passive'")

        ok, _, _ = is_swap_only(it.sentence_good, it.sentence_bad)
        if not ok:
            # should not happen with these templates; keep as safety
            continue
        items.append(it)
    return items


def build_suite(
    *,
    # baseline model
    base_model: LanguageModel,
    # optional second model
    other_model: Optional[LanguageModel] = None,
    # generation
    n_sva: int,
    n_det: int,
    n_passive: int,
    seed: int,
    pool_multiplier: int = 200,   # how many raw candidates to sample per desired item
    only_inanimate_objects: bool = True,
    # scoring
    temperature: float = 1.0,
    batch_items: int = 64,
    # selection thresholds
    ctrl_margin: float = 1.0,
    conf_margin: float = 0.5,
    # require secondary model to pass control?
    require_other_ctrl: bool = False,
    other_ctrl_margin: float = 0.5,
    # require secondary model NOT to be lured?
    require_other_prefers_good: bool = False,
    # diversity
    max_per_frame: int = 30,
    max_per_noun_pair: int = 3,
) -> Tuple[List[ScoredItem], List[ScoredItem]]:
    """Return (conflict_items, control_items) as scored lists.

    conflict_items are BLiMP-style pairs:
        sentence_good (grammatical but implausible)
        sentence_bad  (ungrammatical but plausible)

    control_items are BLiMP-style pairs:
        control_good (grammatical & plausible)
        control_bad  (same as sentence_bad)
    """
    rng = random.Random(seed)

    # Create candidate pools
    sva_pool = sample_items(rng=rng, n=n_sva * pool_multiplier, family="sva", only_inanimate_objects=only_inanimate_objects)
    det_pool = sample_items(rng=rng, n=n_det * pool_multiplier, family="det", only_inanimate_objects=only_inanimate_objects)
    passive_pool = sample_items(rng=rng, n=n_passive * pool_multiplier, family="passive", only_inanimate_objects=only_inanimate_objects)

    # Score + select with diversity constraints
    def select_from_pool(pool: List[ItemSentences], target: int) -> Tuple[List[ScoredItem], List[ScoredItem]]:
        conflicts: List[ScoredItem] = []
        controls: List[ScoredItem] = []
        frame_counts: Dict[str,int] = {}
        pair_counts: Dict[Tuple[str,str],int] = {}
        seen: set = set()

        # We process pool in batches and keep high-confidence items
        idx = 0
        while idx < len(pool) and len(conflicts) < target:
            print(f"selected {len(conflicts)}/{target}", flush=True)
            batch = pool[idx: idx + batch_items]
            idx += batch_items
            triplets = [(it.sentence_good, it.sentence_bad, it.control_good) for it in batch]
            base_scores = _score_triplets(
                model=base_model,
                triplets=triplets,
                temperature=temperature,
            )

            other_scores = None
            if other_model is not None:
                other_scores = _score_triplets(
                    model=other_model,
                    triplets=triplets,
                    temperature=temperature,
                )

            for j,it in enumerate(batch):
                bg, bb, bcg = base_scores[j]
                delta_conf = bg - bb
                delta_ctrl = bcg - bb

                # Hard requirements: easy control + conflict lure
                if delta_ctrl < ctrl_margin:
                    continue
                if delta_conf > -conf_margin:
                    continue

                # Diversity constraints
                fr = it.meta.get("frame","?")
                frame_counts.setdefault(fr, 0)
                if frame_counts[fr] >= max_per_frame:
                    continue

                pair_key = (it.meta.get("A_pl","?"), it.meta.get("B_sg","?"))
                pair_counts.setdefault(pair_key, 0)
                if pair_counts[pair_key] >= max_per_noun_pair:
                    continue

                # De-dup exact (family, frame, A, B, verb)
                verb_key = it.meta.get("verb_sg3") or it.meta.get("verb_past") or it.meta.get("verb_pp")
                dedup_key = (it.family, fr, it.meta.get("A_pl"), it.meta.get("B_sg"), verb_key)
                if dedup_key in seen:
                    continue

                sc = ScoredItem(
                    item=it,
                    base_good=bg,
                    base_bad=bb,
                    base_ctrl_good=bcg,
                    base_delta_conf=delta_conf,
                    base_delta_ctrl=delta_ctrl,
                )

                if other_scores is not None:
                    og, ob, ocg = other_scores[j]
                    sc.other_good = og
                    sc.other_bad = ob
                    sc.other_ctrl_good = ocg
                    sc.other_delta_conf = og - ob
                    sc.other_delta_ctrl = ocg - ob

                    if require_other_ctrl and sc.other_delta_ctrl < other_ctrl_margin:
                        continue
                    if require_other_prefers_good and sc.other_delta_conf is not None and sc.other_delta_conf < 0:
                        continue

                # Keep it
                seen.add(dedup_key)
                frame_counts[fr] += 1
                pair_counts[pair_key] += 1
                conflicts.append(sc)
                controls.append(sc)

                if len(conflicts) >= target:
                    break

        # Sort conflicts by strength of lure (baseline prefers bad strongly)
        conflicts.sort(key=lambda x: x.base_delta_conf)  # most negative first
        controls.sort(key=lambda x: x.base_delta_ctrl, reverse=True)
        return conflicts[:target], controls[:target]

    sva_conf, sva_ctrl = select_from_pool(sva_pool, n_sva)
    det_conf, det_ctrl = select_from_pool(det_pool, n_det)
    pass_conf, pass_ctrl = select_from_pool(passive_pool, n_passive)

    # Combine
    conflicts = sva_conf + det_conf + pass_conf
    controls = sva_ctrl + det_ctrl + pass_ctrl
    return conflicts, controls


def write_jsonl_conflicts(path: Path, items: Sequence[ScoredItem], prefix: str = "rr") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i,sc in enumerate(items):
            it = sc.item
            ex = {
                "sentence_good": it.sentence_good,
                "sentence_bad": it.sentence_bad,
                "field": "syntax_semantics",
                "linguistics_term": it.family,
                "UID": f"{prefix}_{it.family}_{i:06d}",
                "metadata": {
                    **it.meta,
                    "family": it.family,
                    "control_good": it.control_good,
                    "baseline": {
                        "score_good": sc.base_good,
                        "score_bad": sc.base_bad,
                        "delta_conf": sc.base_delta_conf,
                        "score_control_good": sc.base_ctrl_good,
                        "delta_ctrl": sc.base_delta_ctrl,
                    },
                },
            }
            if sc.other_good is not None:
                ex["metadata"]["other_model"] = {
                    "score_good": sc.other_good,
                    "score_bad": sc.other_bad,
                    "delta_conf": sc.other_delta_conf,
                    "score_control_good": sc.other_ctrl_good,
                    "delta_ctrl": sc.other_delta_ctrl,
                }

            # also store swap info for auditing
            ok, idxs, toks = is_swap_only(it.sentence_good, it.sentence_bad)
            ex["metadata"]["swap_only"] = ok
            ex["metadata"]["swap_indices"] = idxs
            ex["metadata"]["swap_tokens"] = toks

            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def write_jsonl_controls(path: Path, items: Sequence[ScoredItem], prefix: str = "rr_ctrl") -> None:
    """Write control pairs (plausible semantics, isolates grammar)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i,sc in enumerate(items):
            it = sc.item
            ex = {
                "sentence_good": it.control_good,
                "sentence_bad": it.control_bad,
                "field": "control",
                "linguistics_term": f"{it.family}_control",
                "UID": f"{prefix}_{it.family}_{i:06d}",
                "metadata": {
                    **it.meta,
                    "family": it.family,
                    "baseline": {
                        "score_good": sc.base_ctrl_good,
                        "score_bad": sc.base_bad,  # ctrl_bad == bad
                        "delta": sc.base_delta_ctrl,
                    },
                },
            }
            if sc.other_good is not None:
                ex["metadata"]["other_model"] = {
                    "score_good": sc.other_ctrl_good,
                    "score_bad": sc.other_bad,
                    "delta": sc.other_delta_ctrl,
                }
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
