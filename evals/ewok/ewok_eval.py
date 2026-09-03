#!/usr/bin/env python
"""EWoK: world-knowledge evaluation over contexts and targets.

Every ``.jsonl`` file at the top level of the input directory holds items with
``Context1`` / ``Context2`` / ``Target1`` / ``Target2`` and four grouping
fields: ``Domain``, ``ContextType``, ``ContextDiff`` and ``TargetDiff``. An
item is scored as the two concatenations ``"Context1 Target1"`` and
``"Context1 Target2"`` (joined by one space, the tokens of ``Context1`` alone
passed as the prefix hint) and is correct when the first ranks higher.

Everything is scored over the temperature grid at once. Accuracies are
aggregated per domain and per contrast field, the reported temperature is the
one with the highest mean per-domain accuracy (first on ties), and T=1.0 is
reported beside it. A group with no correct item at a temperature is absent
from that temperature's table, and therefore from that temperature's mean.

Library::

    from evals.ewok.ewok_eval import evaluate_ewok
    result = evaluate_ewok(backend, "evals/ewok/downloads/ewok_filtered")   # summary / rows / protocol

Command line (from the repository root)::

    python evals/ewok/ewok_eval.py --input_path evals/ewok/downloads/ewok_filtered \\
        --output_dir OUT --backend-arg checkpoint=lm/gpt-bert/trained_models/<ckpt>.bin [--predict]

writes (``--record_prefix P`` writes the best-temperature report as ``P_report.txt``, with or
instead of ``--output_dir``) ``OUT/<backend name>/<input dir name>/{best_temperature_report.txt,
temperature_1_report.txt}`` and, with ``--predict``, the chosen target per item
at T=1.0 and at the best temperature.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from collections import Counter, OrderedDict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from evals.backends import LanguageModel  # noqa: E402
from evals.common import TEMPERATURE_STEP, add_backend_arguments, backend_from_args, temperature_grid  # noqa: E402

REPORT_SECTIONS = (("domain", "DOMAIN ACCURACY"), ("context_type", "CONTEXT TYPE ACCURACY"),
                   ("context_diff", "CONTEXT CONTRAST ACCURACY"), ("target_diff", "TARGET CONTRAST ACCURACY"))


def load_items(data_dir) -> List[Dict]:
    """Every item of every top-level ``.jsonl`` file, in directory order, indexed within its file."""
    data_dir = pathlib.Path(data_dir)
    items: List[Dict] = []
    for name in os.listdir(data_dir):
        if not name.endswith(".jsonl"):
            continue
        index = 0
        with open(data_dir / name, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                items.append({
                    "file": pathlib.Path(name).stem, "index": index,
                    "id": f"{row['Domain']}_{index}",
                    "context1": row["Context1"], "context2": row["Context2"],
                    "target1": row["Target1"], "target2": row["Target2"],
                    "domain": row["Domain"], "context_type": row["ContextType"],
                    "context_diff": row["ContextDiff"], "target_diff": row["TargetDiff"],
                })
                index += 1
    return items


def item_sequences(backend: LanguageModel, item: Dict) -> Tuple[List[str], List[int]]:
    """The two strings ranked for one item, and the prefix hint (the context's token count) for each."""
    texts = [" ".join([item["context1"], target]) for target in (item["target1"], item["target2"])]
    return texts, [len(backend.encode(item["context1"]))] * len(texts)


@torch.no_grad()
def evaluate_ewok(backend: LanguageModel, data_dir, temperatures: Optional[Sequence[float]] = None,
                  progress: bool = False) -> Dict:
    """Score every item under ``data_dir``; returns ``summary`` / ``rows`` / ``protocol``."""
    temps = torch.as_tensor(list(temperatures) if temperatures is not None else temperature_grid().tolist(),
                            dtype=torch.float32)
    n_temps = int(temps.numel())
    t1 = _index_of_one(temps)
    items = load_items(data_dir)
    if not items:
        raise ValueError(f"no .jsonl items under {data_dir}")

    counts = {key: {"correct": [Counter() for _ in range(n_temps)], "total": [Counter() for _ in range(n_temps)]}
              for key, _ in REPORT_SECTIONS}
    rows: List[Dict] = []
    scores_by_item: List[torch.Tensor] = []
    iterator = items
    if progress:
        from tqdm import tqdm
        iterator = tqdm(items, desc="items")
    for item in iterator:
        texts, prefix_lens = item_sequences(backend, item)
        scores = backend.sequence_logprobs([backend.encode(t) for t in texts], temps.tolist(),
                                           prefix_lens=prefix_lens)          # [n_seqs, n_temps]
        ranking = torch.argsort(scores.t(), dim=1, descending=True)          # [n_temps, n_seqs]
        correct = (ranking[:, 0] == 0)
        for t in range(n_temps):
            hit = bool(correct[t])
            for key, _ in REPORT_SECTIONS:
                counts[key]["total"][t][item[key]] += 1
                if hit:
                    counts[key]["correct"][t][item[key]] += 1
        scores_by_item.append(scores)
        rows.append(dict(item, sequences=texts, correct_by_temperature=[int(c) for c in correct.tolist()]))

    accuracy = {key: [_accuracies(counts[key]["correct"][t], counts[key]["total"][t]) for t in range(n_temps)]
                for key, _ in REPORT_SECTIONS}
    averages = torch.tensor([sum(a.values()) / len(a) for a in accuracy["domain"]])
    best = int(torch.argmax(averages).item())
    for row, scores in zip(rows, scores_by_item):
        row["logprobs_at_temperature_1"] = scores[:, t1].tolist()
        row["logprobs_at_best_temperature"] = scores[:, best].tolist()
        row["correct_at_temperature_1"] = row["correct_by_temperature"][t1]
        row["correct_at_best_temperature"] = row["correct_by_temperature"][best]

    summary: Dict[str, float] = OrderedDict()
    summary["ewok/best_temperature"] = _printed_temperature(temps, best)
    summary["ewok/best_temperature_index"] = best
    summary["ewok/best_temp_avg_domain_accuracy"] = float(averages[best])
    summary["ewok/temp_1_avg_domain_accuracy"] = float(averages[t1])
    for key, _ in REPORT_SECTIONS:
        for name, value in accuracy[key][best].items():
            summary[f"ewok/{key}/{name}"] = value
    summary["ewok/n_items"] = len(items)
    return {
        "summary": summary,
        "rows": rows,
        "protocol": {"sequences": "context1+target1 vs context1+target2, context as prefix",
                     "temperatures": temps.tolist(), "temperature_1_index": t1,
                     "best_temperature_index": best, "data_dir": str(data_dir),
                     "files": sorted({row["file"] for row in rows}), "n_items": len(items),
                     "backend": backend.name},
        "accuracy_by_temperature": accuracy,
        "average_accuracy_by_temperature": [float(a) for a in averages],
    }


def _accuracies(correct: Counter, total: Counter) -> Dict[str, float]:
    """Percentage per group, in first-correct order; a group with no correct item is absent."""
    return OrderedDict((name, correct[name] / total[name] * 100.0) for name in correct)


def _index_of_one(temps: torch.Tensor) -> int:
    return int((temps - 1.0).abs().argmin().item())


def _printed_temperature(temps: torch.Tensor, index: int) -> float:
    """The label the reports carry: the grid position on the standard grid, else the value itself."""
    grid = torch.arange(temps.numel(), dtype=torch.float32) * TEMPERATURE_STEP
    if torch.allclose(temps.clamp(min=TEMPERATURE_STEP / 2), grid.clamp(min=TEMPERATURE_STEP / 2), atol=1e-6):
        return round(index * TEMPERATURE_STEP, 10)
    return float(temps[index])


# ---------------------------------------------------------------------------
# report writers (the layout of the committed ``*_report.txt`` records)
# ---------------------------------------------------------------------------
def report_text(result: Dict, temperature_index: int, printed_temperature: float) -> str:
    """The report for one temperature: per-domain and per-contrast accuracies, then the average."""
    lines = [f"TEMPERATURE: {printed_temperature:.2f}", ""]
    for key, title in REPORT_SECTIONS:
        lines.append(f"### {title}")
        lines.extend(f"{name}: {value:.2f}"
                     for name, value in result["accuracy_by_temperature"][key][temperature_index].items())
        lines.append("")
    lines += ["### AVERAGE ACCURACY",
              f"{result['average_accuracy_by_temperature'][temperature_index]:.2f}", ""]
    return "\n".join(lines) + "\n"


def predictions(result: Dict, temperature_index: int) -> Dict:
    """Per domain, the target chosen for each item at one temperature."""
    out: Dict[str, Dict] = {}
    for row in result["rows"]:
        chosen = row["target1"] if row["correct_by_temperature"][temperature_index] else row["target2"]
        out.setdefault(row["domain"], {"predictions": []})["predictions"].append(
            {"id": row["id"], "pred": " " + chosen})
    return out


def write_outputs(result: Dict, out_dir, predict: bool = False) -> pathlib.Path:
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best, t1 = result["protocol"]["best_temperature_index"], result["protocol"]["temperature_1_index"]
    (out_dir / "best_temperature_report.txt").write_text(
        report_text(result, best, result["summary"]["ewok/best_temperature"]))
    (out_dir / "temperature_1_report.txt").write_text(report_text(result, t1, 1.0))
    if predict:
        (out_dir / "predictions.json").write_text(json.dumps(predictions(result, t1)))
        (out_dir / "predictions_at_best_temperature.json").write_text(
            json.dumps(predictions(result, best)))
    return out_dir


def write_record(result: Dict, prefix) -> pathlib.Path:
    """The best-temperature report as ``<prefix>_report.txt``."""
    prefix = pathlib.Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    best = result["protocol"]["best_temperature_index"]
    path = prefix.parent / f"{prefix.name}_report.txt"
    path.write_text(report_text(result, best, result["summary"]["ewok/best_temperature"]))
    return path


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_path", required=True, type=pathlib.Path, help="directory of .jsonl EWoK files")
    parser.add_argument("--output_dir", type=pathlib.Path,
                        help="run directory: reports under <output_dir>/<backend name>/<input dir name>/")
    parser.add_argument("--record_prefix", type=pathlib.Path,
                        help="write the best-temperature report as <record_prefix>_report.txt")
    parser.add_argument("--predict", action="store_true",
                        help="also write the chosen target per item (needs --output_dir)")
    parser.add_argument("--no_progress", action="store_true")
    add_backend_arguments(parser)
    args = parser.parse_args(argv)
    if args.output_dir is None and args.record_prefix is None:
        parser.error("give --output_dir and/or --record_prefix")
    if args.predict and args.output_dir is None:
        parser.error("--predict needs --output_dir")
    backend = backend_from_args(args)
    result = evaluate_ewok(backend, args.input_path, progress=not args.no_progress)
    written = []
    if args.output_dir is not None:
        written.append(write_outputs(result, args.output_dir / backend.name / args.input_path.stem, predict=args.predict))
    if args.record_prefix is not None:
        written.append(write_record(result, args.record_prefix))
    print(report_text(result, result["protocol"]["best_temperature_index"],
                      result["summary"]["ewok/best_temperature"]))
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
