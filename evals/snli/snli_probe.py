"""SNLI probe: train a multinomial logistic-regression probe on frozen
sentence representations and evaluate 3-way NLI accuracy.

For each sentence, the representation is pooled from the final-layer hidden
states of the frozen model (``mean`` over the sentence's tokens, the ICML 2026 paper's
protocol, or the state at its last token); a premise/hypothesis pair is
featurized as [u_premise, u_hypothesis, |u_premise - u_hypothesis|,
u_premise * u_hypothesis, cos] with L2-normalized u. The probe's C is
selected on the dev split from {0.1, 1.0, 10.0} by (accuracy, macro-F1).

Writes, with --output_dir, per model metrics.json, per-split predictions_*.tsv
and the labels-only predictions_test_labels.tsv under <output_dir>/<model>/snli/;
with --record_prefix P, the committed record's two files P_metrics.json and
P_predictions_test_labels.tsv. SNLI-hard metrics are computed from the test
predictions restricted to the official hard-subset pair ids, either here
(``--hard_pair_ids``) or afterwards by the table generator under reproduce/.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from evals.backends import LanguageModel  # noqa: E402
from evals.common import add_backend_arguments, backend_from_args  # noqa: E402

SNLI_LABEL_TO_ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
SNLI_ID_TO_LABEL = {v: k for k, v in SNLI_LABEL_TO_ID.items()}
LABELS = [0, 1, 2]
POOLINGS = ("mean", "final_token")
CANDIDATE_C = [0.1, 1.0, 10.0]
FEATURE_SPEC = ("[u_premise, u_hypothesis, abs(u_premise-u_hypothesis), u_premise*u_hypothesis, cos] "
                "where u is {pooling}-pooled sentence representation")


# ----------------------------
# CLI / config
# ----------------------------

def parse_arguments():
    parser = argparse.ArgumentParser()
    add_backend_arguments(parser)
    parser.add_argument("--output_dir", type=pathlib.Path,
                        help="run directory: metrics and every prediction file under <output_dir>/<model>/snli/")
    parser.add_argument("--record_prefix", type=pathlib.Path,
                        help="write <record_prefix>_metrics.json and <record_prefix>_predictions_test_labels.tsv")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Sentences per representation batch.")
    parser.add_argument(
        "--snli_dir",
        default=pathlib.Path("downloads/snli_1.0"),
        type=pathlib.Path,
        help="Directory containing snli_1.0_train/dev/test.jsonl files.",
    )
    parser.add_argument(
        "--snli_train_limit",
        default=100000,
        type=int,
        help="Maximum number of SNLI train examples to use. Set <=0 to use the full split.",
    )
    parser.add_argument(
        "--snli_dev_limit",
        default=0,
        type=int,
        help="Maximum number of SNLI dev examples to use. Set <=0 to use the full split.",
    )
    parser.add_argument(
        "--snli_test_limit",
        default=0,
        type=int,
        help="Maximum number of SNLI test examples to use. Set <=0 to use the full split.",
    )
    parser.add_argument(
        "--pooling",
        default="mean",
        choices=list(POOLINGS),
        help="How a sentence representation is pooled from its token states.",
    )
    parser.add_argument(
        "--hard_pair_ids",
        default=None,
        type=pathlib.Path,
        help="Optional file of SNLI-hard pair ids, one per line; adds the hard-subset metrics.",
    )
    args = parser.parse_args()
    if args.output_dir is None and args.record_prefix is None:
        parser.error("give --output_dir and/or --record_prefix")
    return args


# ----------------------------
# Metrics
# ----------------------------

def accuracy(pred, gold) -> float:
    pred = np.asarray(pred)
    gold = np.asarray(gold)
    return float(np.mean(pred == gold))


def f1_macro(pred, gold, labels=None) -> Tuple[float, Dict[int, float]]:
    pred = np.asarray(pred).astype(np.int64)
    gold = np.asarray(gold).astype(np.int64)
    if labels is None:
        labels = sorted(set(pred.tolist()) | set(gold.tolist()))
    per_label = {}
    f1s = []
    for label in labels:
        tp = int(np.sum((pred == label) & (gold == label)))
        fp = int(np.sum((pred == label) & (gold != label)))
        fn = int(np.sum((pred != label) & (gold == label)))
        if tp == 0:
            f1 = 0.0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)
        per_label[int(label)] = float(f1)
        f1s.append(f1)
    return float(np.mean(f1s)), per_label


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    denom = float(np.linalg.norm(u) * np.linalg.norm(v))
    if denom == 0.0:
        return 0.0
    return float(np.dot(u, v) / denom)


# ----------------------------
# Representations
# ----------------------------

def pool_states(states: torch.Tensor, pooling: str) -> np.ndarray:
    """One sentence's ``[len, width]`` final-layer states -> its representation."""
    if states.size(0) <= 0:
        raise ValueError("cannot pool an empty sentence")
    vector = states.mean(dim=0) if pooling == "mean" else states[-1]
    return vector.numpy().astype(np.float32)


def sentence_representations(backend: LanguageModel, texts: Sequence[str], pooling: str = "mean",
                             batch_size: int = 32) -> List[np.ndarray]:
    """Pooled final-layer representation of each text, in order."""
    if pooling not in POOLINGS:
        raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}")
    reps: List[np.ndarray] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Sentence reps", leave=False):
        seqs = [backend.encode(text) for text in texts[start:start + batch_size]]
        if any(len(seq) == 0 for seq in seqs):
            raise ValueError("a sentence encoded to no tokens")
        for states in backend.hidden_states(seqs, "final", add_special_tokens=True):
            reps.append(pool_states(states[0], pooling))
    return reps


# ----------------------------
# Data
# ----------------------------

def load_snli_split(path: pathlib.Path, limit: int = 0) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            gold = row.get("gold_label", "")
            if gold not in SNLI_LABEL_TO_ID:
                continue
            items.append(
                {
                    "pair_id": row.get("pairID", ""),
                    "premise": row["sentence1"],
                    "hypothesis": row["sentence2"],
                    "label": SNLI_LABEL_TO_ID[gold],
                    "label_name": gold,
                }
            )
            if limit and limit > 0 and len(items) >= limit:
                break
    return items


def load_pair_ids(path: pathlib.Path) -> List[str]:
    """The pair ids in a one-id-per-line file, in file order."""
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ----------------------------
# Features
# ----------------------------

def l2_normalize(rep: np.ndarray) -> np.ndarray:
    denom = float(np.linalg.norm(rep))
    if denom == 0.0:
        return rep.astype(np.float32)
    return (rep / denom).astype(np.float32)


def pair_features_for_snli(rep1: np.ndarray, rep2: np.ndarray) -> np.ndarray:
    rep1 = l2_normalize(rep1)
    rep2 = l2_normalize(rep2)
    abs_diff = np.abs(rep1 - rep2)
    prod = rep1 * rep2
    feats = [
        rep1,
        rep2,
        abs_diff,
        prod,
        np.array([cosine(rep1, rep2)], dtype=np.float32),
    ]
    return np.concatenate(feats, axis=0).astype(np.float32)


# ----------------------------
# Evaluation
# ----------------------------

def evaluate_snli(backend: LanguageModel, snli_dir: pathlib.Path, train_limit: int = 100000,
                  dev_limit: int = 0, test_limit: int = 0, pooling: str = "mean", batch_size: int = 32,
                  hard_pair_ids: Optional[Iterable[str]] = None) -> dict:
    """Fit the probe on train, pick C on dev, score every split.

    Returns ``{"summary", "rows", "protocol"}``; with ``hard_pair_ids`` the
    summary also carries the test metrics restricted to those pairs.
    """
    if pooling not in POOLINGS:
        raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}")
    snli_dir = pathlib.Path(snli_dir)
    limits = {"train": int(train_limit), "dev": int(dev_limit), "test": int(test_limit)}
    splits = {name: load_snli_split(snli_dir / f"snli_1.0_{name}.jsonl", limit)
              for name, limit in limits.items()}

    def attach_features(items: List[dict]):
        unique_texts: List[str] = []
        seen = set()
        for item in items:
            for text in (item["premise"], item["hypothesis"]):
                if text not in seen:
                    seen.add(text)
                    unique_texts.append(text)
        reps = sentence_representations(backend, unique_texts, pooling, batch_size)
        rep_map = dict(zip(unique_texts, reps))

        X = []
        y = []
        for item in items:
            prem_rep = rep_map[item["premise"]]
            hyp_rep = rep_map[item["hypothesis"]]
            item["pair_cosine"] = cosine(prem_rep, hyp_rep)
            X.append(pair_features_for_snli(prem_rep, hyp_rep))
            y.append(item["label"])
        return np.stack(X, axis=0), np.asarray(y, dtype=np.int64)

    features = {name: attach_features(items) for name, items in splits.items()}

    best = None
    for C in CANDIDATE_C:
        clf = LogisticRegression(C=C, max_iter=3000, solver="lbfgs", random_state=0)
        clf.fit(*features["train"])
        dev_pred = clf.predict(features["dev"][0])
        dev_acc = accuracy(dev_pred, features["dev"][1])
        dev_macro_f1, _ = f1_macro(dev_pred, features["dev"][1], labels=LABELS)
        score_key = (dev_acc, dev_macro_f1)
        if best is None or score_key > best["score_key"]:
            best = {"score_key": score_key, "C": C, "clf": clf}

    clf = best["clf"]

    summary: Dict[str, float] = {}
    for name, items in splits.items():
        X, y = features[name]
        pred = clf.predict(X)
        prob = clf.predict_proba(X)
        macro_f1, _ = f1_macro(pred, y, labels=LABELS)
        for item, p, pr in zip(items, pred.tolist(), prob.tolist()):
            item["pred_label"] = int(p)
            item["pred_label_name"] = SNLI_ID_TO_LABEL[int(p)]
            item["prob_entailment"] = float(pr[SNLI_LABEL_TO_ID["entailment"]])
            item["prob_neutral"] = float(pr[SNLI_LABEL_TO_ID["neutral"]])
            item["prob_contradiction"] = float(pr[SNLI_LABEL_TO_ID["contradiction"]])
        summary[f"snli/{name}/accuracy"] = accuracy(pred, y)
        summary[f"snli/{name}/macro_f1"] = macro_f1
        summary[f"snli/{name}/mean_pair_cosine"] = float(np.mean([item["pair_cosine"] for item in items]))
        summary[f"snli/{name}/n_items"] = len(items)
    summary["snli/probe/selected_C"] = float(best["C"])

    if hard_pair_ids is not None:
        wanted = set(hard_pair_ids)
        hard = [item for item in splits["test"] if item["pair_id"] in wanted]
        hard_pred = [item["pred_label"] for item in hard]
        hard_gold = [item["label"] for item in hard]
        summary["snli/hard/accuracy"] = accuracy(hard_pred, hard_gold)
        summary["snli/hard/macro_f1"] = f1_macro(hard_pred, hard_gold, labels=LABELS)[0]
        summary["snli/hard/n_items"] = len(hard)

    protocol = {
        "pooling": pooling,
        "train_limit": limits["train"],
        "dev_limit": limits["dev"],
        "test_limit": limits["test"],
        "candidate_C": list(CANDIDATE_C),
        "feature_spec": FEATURE_SPEC.format(pooling=pooling),
    }
    return {"summary": summary, "rows": splits, "protocol": protocol}


def metrics_document(result: dict) -> dict:
    """The metrics.json record: the probe's settings and one block per split."""
    summary, rows, protocol = result["summary"], result["rows"], result["protocol"]

    def block(name: str) -> dict:
        items = rows[name]
        _, per_label_f1 = f1_macro([item["pred_label"] for item in items],
                                   [item["label"] for item in items], labels=LABELS)
        return {
            "n_items": summary[f"snli/{name}/n_items"],
            "accuracy": summary[f"snli/{name}/accuracy"],
            "macro_f1": summary[f"snli/{name}/macro_f1"],
            "per_label_f1": {SNLI_ID_TO_LABEL[k]: v for k, v in per_label_f1.items()},
            "mean_pair_cosine": summary[f"snli/{name}/mean_pair_cosine"],
        }

    document = {
        "task": "snli",
        "probe": {
            "family": "multinomial_logistic_regression",
            "selected_C": summary["snli/probe/selected_C"],
            "feature_spec": protocol["feature_spec"],
            "train_limit": protocol["train_limit"],
            "dev_limit": protocol["dev_limit"],
            "test_limit": protocol["test_limit"],
        },
        "train": block("train"),
        "dev": block("dev"),
        "test": block("test"),
    }
    if "snli/hard/accuracy" in summary:
        document["hard"] = {
            "n_items": summary["snli/hard/n_items"],
            "accuracy": summary["snli/hard/accuracy"],
            "macro_f1": summary["snli/hard/macro_f1"],
        }
    return document


# ----------------------------
# I/O helpers
# ----------------------------

def ensure_dir(path: pathlib.Path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: pathlib.Path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# The labels-only projection of the test predictions: pair id, gold and
# predicted labels, without the sentence text and probabilities.
LABEL_COLUMNS = ["pair_id", "label", "label_name", "pred_label", "pred_label_name"]


def save_rows_tsv(path: pathlib.Path, rows):
    if isinstance(rows, dict):
        for split, split_rows in rows.items():
            split_path = path.parent / f"{path.stem}_{split}{path.suffix}"
            save_rows_tsv(split_path, split_rows)
        return

    if not rows:
        return

    # Use the union of keys across all rows so DictWriter never fails on a
    # row that carries a column absent from the first row.
    keys = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            safe_row = {}
            for k, v in row.items():
                if isinstance(v, np.ndarray):
                    safe_row[k] = json.dumps(v.tolist())
                elif isinstance(v, (list, tuple, dict)):
                    safe_row[k] = json.dumps(v, ensure_ascii=False)
                else:
                    safe_row[k] = v
            writer.writerow(safe_row)


# ----------------------------
# Main
# ----------------------------

def main():
    args = parse_arguments()
    backend = backend_from_args(args)

    hard_pair_ids = load_pair_ids(args.hard_pair_ids) if args.hard_pair_ids else None
    result = evaluate_snli(backend, args.snli_dir, train_limit=args.snli_train_limit,
                           dev_limit=args.snli_dev_limit, test_limit=args.snli_test_limit,
                           pooling=args.pooling, batch_size=args.batch_size,
                           hard_pair_ids=hard_pair_ids)
    results = metrics_document(result)

    label_rows = [{k: item.get(k, "") for k in LABEL_COLUMNS} for item in result["rows"].get("test", [])]
    if args.output_dir is not None:
        task_out = args.output_dir / backend.name / "snli"
        ensure_dir(task_out)
        save_json(task_out / "metrics.json", results)
        save_rows_tsv(task_out / "predictions.tsv", result["rows"])
        save_rows_tsv(task_out / "predictions_test_labels.tsv", label_rows)
    if args.record_prefix is not None:
        prefix = args.record_prefix
        ensure_dir(prefix.parent)
        save_json(prefix.parent / f"{prefix.name}_metrics.json", results)
        save_rows_tsv(prefix.parent / f"{prefix.name}_predictions_test_labels.tsv", label_rows)
    print("\n=== SNLI ===")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
