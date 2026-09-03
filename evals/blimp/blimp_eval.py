#!/usr/bin/env python
"""Minimal-pair evaluation: BLiMP, its subsets, and any directory of files in its format.

Every ``.jsonl`` file at the top level of the input directory is a paradigm;
each line holds ``sentence_good`` / ``sentence_bad`` and, optionally,
``field``, ``linguistics_term`` and ``UID`` (a file without them counts as
"supplemental", named after the file). A pair is correct when the model
assigns the grammatical sentence the higher log-probability. Everything is
scored over the temperature grid at once; the reported temperature is the one
with the highest mean per-paradigm accuracy (first on ties), and T=1.0 is
reported beside it.

Library::

    from evals.blimp.blimp_eval import evaluate_blimp
    result = evaluate_blimp(backend, "evals/blimp/data")   # summary / rows / protocol

Command line (from the repository root)::

    python evals/blimp/blimp_eval.py --input_path evals/blimp/data --output_dir OUT \\
        --backend gptbert --backend-arg checkpoint=lm/gpt-bert/trained_models/<ckpt>.bin [--predict]

writes ``OUT/<backend name>/<input dir name>/{best_temperature_report.txt,temperature_1_report.txt}``
and, with ``--predict``, the chosen sentence per pair at T=1.0 and at the best temperature.
``--record PATH`` (with or instead of ``--output_dir``) writes the metrics the trainers log
(``training_metrics``) as one JSON: the best temperature, the average and per-field accuracies
at the best temperature and at T=1.0, and ``per_uid``, the per-paradigm accuracies at the best
temperature. ``--record_prefix P`` writes the two reports as ``P_best_temperature_report.txt``
and ``P_temperature_1_report.txt``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter, OrderedDict
from typing import Dict, Iterable, List, Optional, Sequence

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from evals.backends import LanguageModel  # noqa: E402
from evals.common import TEMPERATURE_INDEX_1, TEMPERATURE_STEP, add_backend_arguments, backend_from_args, temperature_grid  # noqa: E402

REPORT_SECTIONS = (("field", "FIELD ACCURACY"), ("linguistics_term", "LINGUISTIC TERM ACCURACY"),
                   ("uid", "UID ACCURACY"))


def load_pairs(data_dir) -> List[Dict]:
    """Every pair of every top-level ``.jsonl`` file, in file order (files sorted by name)."""
    data_dir = pathlib.Path(data_dir)
    pairs: List[Dict] = []
    for path in sorted(p for p in data_dir.iterdir() if p.suffix == ".jsonl"):
        with open(path, "r", encoding="utf-8") as fh:
            for index, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                field = row.get("field", "supplemental")
                if field == "syntax_semantics":
                    field = "syntax/semantics"
                pairs.append({
                    "file": path.stem, "index": index,
                    "good": row["sentence_good"], "bad": row["sentence_bad"],
                    "field": field,
                    "linguistics_term": row.get("linguistics_term", "supplemental"),
                    "uid": row.get("UID", path.stem),
                })
    return pairs


@torch.no_grad()
def evaluate_blimp(backend: LanguageModel, data_dir, temperatures: Optional[Sequence[float]] = None,
                   progress: bool = False) -> Dict:
    """Score every pair under ``data_dir``; returns ``summary`` / ``rows`` / ``protocol``."""
    temps = torch.as_tensor(list(temperatures) if temperatures is not None else temperature_grid().tolist(),
                            dtype=torch.float32)
    n_temps = int(temps.numel())
    t1 = _index_of_one(temps)
    pairs = load_pairs(data_dir)
    if not pairs:
        raise ValueError(f"no .jsonl pairs under {data_dir}")

    counts = {key: {"correct": [Counter() for _ in range(n_temps)], "total": [Counter() for _ in range(n_temps)]}
              for key, _ in REPORT_SECTIONS}
    rows: List[Dict] = []
    iterator = pairs
    if progress:
        from tqdm import tqdm
        iterator = tqdm(pairs, desc="pairs")
    for pair in iterator:
        scores = backend.sequence_logprobs([backend.encode(pair["good"]), backend.encode(pair["bad"])], temps.tolist())
        ranking = torch.argsort(scores.t(), dim=1, descending=True)           # [n_temps, 2]
        correct = (ranking[:, 0] == 0)
        for t in range(n_temps):
            for key, _ in REPORT_SECTIONS:
                counts[key]["total"][t][pair[key]] += 1
                if bool(correct[t]):
                    counts[key]["correct"][t][pair[key]] += 1
        rows.append(dict(pair, logprob_good=float(scores[0, t1]), logprob_bad=float(scores[1, t1]),
                         correct_by_temperature=[int(c) for c in correct.tolist()]))

    accuracy = {key: [_accuracies(counts[key]["correct"][t], counts[key]["total"][t]) for t in range(n_temps)]
                for key, _ in REPORT_SECTIONS}
    averages = [sum(a.values()) / len(a) for a in accuracy["uid"]]
    best = int(torch.argmax(torch.tensor(averages)).item())
    for row in rows:
        row["correct_at_temperature_1"] = row["correct_by_temperature"][t1]
        row["correct_at_best_temperature"] = row["correct_by_temperature"][best]

    summary: Dict[str, float] = OrderedDict()
    summary["blimp/best_temperature"] = round(best * TEMPERATURE_STEP, 10) if _is_grid(temps) else float(temps[best])
    summary["blimp/best_temperature_index"] = best
    summary["blimp/best_temp_avg_uid_accuracy"] = averages[best]
    summary["blimp/temp_1_avg_uid_accuracy"] = averages[t1]
    for tag, t in (("best_temp", best), ("temp_1", t1)):
        for key, _ in REPORT_SECTIONS:
            for name, value in accuracy[key][t].items():
                summary[f"blimp/{tag}/{key}/{name}"] = value
    summary["blimp/n_pairs"] = len(pairs)
    return {
        "summary": summary,
        "rows": rows,
        "protocol": {"temperatures": temps.tolist(), "temperature_1_index": t1, "best_temperature_index": best,
                     "data_dir": str(data_dir), "n_pairs": len(pairs), "backend": backend.name},
        "accuracy_by_temperature": accuracy,
        "average_accuracy_by_temperature": averages,
    }


def _accuracies(correct: Counter, total: Counter) -> Dict[str, float]:
    return OrderedDict((name, correct[name] / total[name] * 100.0) for name in total)


def _index_of_one(temps: torch.Tensor) -> int:
    return int((temps - 1.0).abs().argmin().item())


def _is_grid(temps: torch.Tensor) -> bool:
    return temps.numel() == 61 and torch.allclose(temps.clamp(min=TEMPERATURE_STEP / 2),
                                                  torch.arange(61, dtype=torch.float32).clamp(min=0.5) * TEMPERATURE_STEP,
                                                  atol=1e-6)


def training_metrics(result: Dict) -> Dict[str, float]:
    """The flat metric dict the trainers log during training (their existing key names)."""
    best, t1 = result["protocol"]["best_temperature_index"], result["protocol"]["temperature_1_index"]
    accuracy = result["accuracy_by_temperature"]
    out = {
        "blimp/best_temperature": float(torch.tensor(result["protocol"]["temperatures"])[best]),
        "blimp/best_temp_avg_uid_accuracy": result["average_accuracy_by_temperature"][best],
        "blimp/temp_1_avg_uid_accuracy": result["average_accuracy_by_temperature"][t1],
    }
    for field, value in accuracy["field"][best].items():
        out[f"blimp/best_temp_field_{field.replace('/', '_')}"] = value
    for field, value in accuracy["field"][t1].items():
        out[f"blimp/temp_1_field_{field.replace('/', '_')}"] = value
    for uid, value in accuracy["uid"][best].items():
        out[f"blimp/per_uid/{uid}"] = value
    return out


# ---------------------------------------------------------------------------
# report and record writers
# ---------------------------------------------------------------------------
def report_text(result: Dict, temperature_index: int, printed_temperature: float) -> str:
    """The report for one temperature: per-field, per-term and per-paradigm accuracies, then the average."""
    lines = [f"TEMPERATURE: {printed_temperature:.2f}", ""]
    for key, title in REPORT_SECTIONS:
        lines.append(f"### {title}")
        lines.extend(f"{name}: {value:.2f}" for name, value in result["accuracy_by_temperature"][key][temperature_index].items())
        lines.append("")
    lines += ["### AVERAGE ACCURACY", f"{result['average_accuracy_by_temperature'][temperature_index]:.2f}", ""]
    return "\n".join(lines) + "\n"


def parse_report(text: str) -> Dict:
    """Inverse of ``report_text`` (values only, order-insensitive)."""
    out: Dict = {"sections": {}}
    section = None
    for line in text.splitlines():
        if line.startswith("TEMPERATURE: "):
            out["temperature"] = float(line.split(": ", 1)[1])
        elif line.startswith("### "):
            section = line[4:]
            out["sections"][section] = {}
        elif line.strip() and section == "AVERAGE ACCURACY":
            out["average"] = float(line)
        elif line.strip() and section:
            name, _, value = line.rpartition(": ")
            out["sections"][section][name] = float(value)
    return out


def predictions(result: Dict, temperature_index: int) -> Dict:
    """Per paradigm, the sentence chosen for each pair at one temperature."""
    out: Dict[str, Dict] = {}
    for row in result["rows"]:
        chosen = row["good"] if row["correct_by_temperature"][temperature_index] else row["bad"]
        out.setdefault(row["uid"], {"predictions": []})["predictions"].append(
            {"id": f"{row['uid']}_{row['index']}", "pred": " " + chosen})
    return out


def write_outputs(result: Dict, out_dir, predict: bool = False) -> pathlib.Path:
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best, t1 = result["protocol"]["best_temperature_index"], result["protocol"]["temperature_1_index"]
    (out_dir / "best_temperature_report.txt").write_text(report_text(result, best, result["summary"]["blimp/best_temperature"]))
    (out_dir / "temperature_1_report.txt").write_text(report_text(result, t1, 1.0))
    if predict:
        (out_dir / "predictions.json").write_text(json.dumps(predictions(result, t1)))
        (out_dir / "predictions_at_best_temperature.json").write_text(json.dumps(predictions(result, best)))
    return out_dir


def write_reports(result: Dict, prefix) -> list:
    """The two reports as ``<prefix>_best_temperature_report.txt`` and ``<prefix>_temperature_1_report.txt``."""
    prefix = pathlib.Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    best, t1 = result["protocol"]["best_temperature_index"], result["protocol"]["temperature_1_index"]
    paths = [prefix.parent / f"{prefix.name}_best_temperature_report.txt",
             prefix.parent / f"{prefix.name}_temperature_1_report.txt"]
    paths[0].write_text(report_text(result, best, result["summary"]["blimp/best_temperature"]))
    paths[1].write_text(report_text(result, t1, 1.0))
    return paths


def write_record(result: Dict, path) -> pathlib.Path:
    """``training_metrics`` as one JSON: the scalars at the top level, ``per_uid`` nested, keys sorted."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record: Dict = {"per_uid": {}}
    for key, value in training_metrics(result).items():
        key = key[len("blimp/"):]
        if key.startswith("per_uid/"):
            record["per_uid"][key[len("per_uid/"):]] = value
        else:
            record[key] = value
    record["per_uid"] = dict(sorted(record["per_uid"].items()))
    with path.open("w") as fh:
        json.dump(dict(sorted(record.items())), fh, indent=1)
    return path


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_path", required=True, type=pathlib.Path, help="directory of .jsonl minimal-pair files")
    parser.add_argument("--output_dir", type=pathlib.Path,
                        help="run directory: reports under <output_dir>/<backend name>/<input dir name>/")
    parser.add_argument("--record", type=pathlib.Path,
                        help="write the metrics as one JSON: best temperature, average and per-field "
                             "accuracies at the best temperature and at T=1.0, per-paradigm accuracies "
                             "at the best temperature")
    parser.add_argument("--record_prefix", type=pathlib.Path,
                        help="write the reports as <record_prefix>_best_temperature_report.txt and "
                             "<record_prefix>_temperature_1_report.txt")
    parser.add_argument("--predict", action="store_true",
                        help="also write the chosen sentence per pair (needs --output_dir)")
    parser.add_argument("--no_progress", action="store_true")
    add_backend_arguments(parser)
    args = parser.parse_args(argv)
    if args.output_dir is None and args.record is None and args.record_prefix is None:
        parser.error("give --output_dir, --record and/or --record_prefix")
    if args.predict and args.output_dir is None:
        parser.error("--predict needs --output_dir")
    backend = backend_from_args(args)
    result = evaluate_blimp(backend, args.input_path, progress=not args.no_progress)
    written = []
    if args.output_dir is not None:
        written.append(write_outputs(result, args.output_dir / backend.name / args.input_path.stem, predict=args.predict))
    if args.record is not None:
        written.append(write_record(result, args.record))
    if args.record_prefix is not None:
        written.extend(write_reports(result, args.record_prefix))
    print(report_text(result, result["protocol"]["best_temperature_index"], result["summary"]["blimp/best_temperature"]))
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
