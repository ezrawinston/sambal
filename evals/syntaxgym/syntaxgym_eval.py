#!/usr/bin/env python
"""Targeted syntactic evaluation on SyntaxGym-style test suites.

A suite is a json file of items; each item has several conditions, each a
sequence of regions of text, and the suite's prediction formulas compare
region surprisals across conditions (``(6;%mismatch%) > (6;%match%)``). A
sentence's per-token log-probabilities come from the model backend; a region's
surprisal (bits) is the sum over the tokens whose character span falls in it.
Everything is scored over the temperature grid at once: per-suite accuracies
are reported at T=1.0, the aggregate at the temperature with the highest
item-weighted accuracy (first on ties) and at T=1.0.

Library::

    from evals.syntaxgym.syntaxgym_eval import evaluate_syntaxgym
    result = evaluate_syntaxgym(backend, "evals/syntaxgym/data/syntaxgym", suites="core_only")

Command line (from the repository root), one record per model in the layout of
the committed ``records/syntaxgym/*.json`` files::

    python evals/syntaxgym/syntaxgym_eval.py --input_path evals/syntaxgym/data/syntaxgym \\
        --output_json OUT.json --backend gptbert --backend-arg config=lm/gpt-bert/configs/small.json \\
        --models baseline:checkpoint=lm/gpt-bert/trained_models/gptbert_babycosmofine_long.bin \\
        --models sambal:checkpoint=lm/gpt-bert/trained_models/gptbert_sambal_long_ema.bin

Each ``--models LABEL:key=value[,key=value...]`` names one model by the backend
options that differ from the common ``--backend-arg`` ones; without ``--models``
the common options describe the single model and the record key is its name.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import math
import pathlib
import re
import sys
import urllib.request
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from evals.backends import LanguageModel, load_backend, parse_backend_args  # noqa: E402
from evals.common import add_backend_arguments, temperature_grid  # noqa: E402

# ---------------------------------------------------------------------------
# suites
# ---------------------------------------------------------------------------
CORE_IN_SCOPE_SUITES: List[str] = [
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

BORDERLINE_LEXICOSYNTACTIC: List[str] = [
    "mvrr", "mvrr_mod",
    "npz_ambig", "npz_ambig_mod", "npz_obj", "npz_obj_mod",
]

ALL_HU2020_SYNTAXGYM: List[str] = CORE_IN_SCOPE_SUITES + BORDERLINE_LEXICOSYNTACTIC

SUITE_SETS = {"core_only": CORE_IN_SCOPE_SUITES, "borderline_only": BORDERLINE_LEXICOSYNTACTIC,
              "all": ALL_HU2020_SYNTAXGYM, "in_scope": ALL_HU2020_SYNTAXGYM}

DEFAULT_DOWNLOAD_BASE = ("https://raw.githubusercontent.com/cpllab/syntactic-generalization/"
                         "2f42038477a9f03ab66f308a963ed49864cd8111/test_suites/json/")


def download_suite_json(suite_name: str, cache_dir: pathlib.Path, base_url: str = DEFAULT_DOWNLOAD_BASE) -> pathlib.Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{suite_name}.json"
    if out_path.exists():
        return out_path
    url = f"{base_url}{suite_name}.json"
    print(f"[syntaxgym] downloading {suite_name} from {url}")
    try:
        urllib.request.urlretrieve(url, out_path)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"failed to download suite {suite_name!r} from {url}: {e}")
    return out_path


def select_suites(suites: Union[str, Sequence[str]]) -> List[str]:
    """A suite-set name (core_only / borderline_only / all) or an explicit list of suite names."""
    if isinstance(suites, str):
        if suites not in SUITE_SETS:
            raise ValueError(f"unknown suite set {suites!r}; one of {sorted(SUITE_SETS)}")
        return list(SUITE_SETS[suites])
    return list(suites)


# ---------------------------------------------------------------------------
# prediction formulas
# ---------------------------------------------------------------------------
_REGION_REF_RE = re.compile(r"\((\*|\d+);%([^%]+)%\)")


class FormulaError(RuntimeError):
    pass


def _safe_eval_arith(expr: str) -> float:
    """Evaluate a simple arithmetic expression with +, -, parentheses, and floats."""
    node = ast.parse(expr, mode="eval")

    def _eval(n):
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)):
                return float(n.value)
            raise FormulaError(f"Unsupported constant: {n.value!r}")
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            val = _eval(n.operand)
            return +val if isinstance(n.op, ast.UAdd) else -val
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub)):
            left = _eval(n.left)
            right = _eval(n.right)
            return left + right if isinstance(n.op, ast.Add) else left - right
        raise FormulaError(f"Unsupported expression node: {ast.dump(n)}")

    return float(_eval(node))


def _split_top_level(s: str, sep: str) -> List[str]:
    """Split by a separator only when not inside parentheses."""
    parts: List[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == sep and depth == 0:
            parts.append(s[start:i].strip())
            start = i + 1
    parts.append(s[start:].strip())
    return [p for p in parts if p != ""]


def _find_top_level_comparator(s: str) -> Tuple[int, str]:
    """Return (index, comparator) for the first top-level comparator in s."""
    depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif depth == 0 and ch in (">", "<", "="):
            return i, ch
    raise FormulaError(f"No top-level comparator found in: {s}")


def _strip_outer_parens(s: str) -> str:
    """Strip a single layer of redundant outer parentheses."""
    s = s.strip()
    while s.startswith("(") and s.endswith(")"):
        depth = 0
        fully_wrapped = True
        for i, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    fully_wrapped = False
                    break
            if depth == 0 and i != len(s) - 1:
                fully_wrapped = False
                break
        if fully_wrapped and depth == 0:
            s = s[1:-1].strip()
        else:
            break
    return s


def evaluate_formula(formula: str, region_totals: Dict[Tuple[str, int], object], condition_names: Sequence[str],
                     tol: float = 1e-6, temp_idx: Optional[int] = None) -> bool:
    """Evaluate a SyntaxGym prediction formula against region totals."""

    def _as_float(v: object) -> float:
        if torch.is_tensor(v):
            if temp_idx is None:
                raise FormulaError("temp_idx must be provided when region_totals stores tensors")
            return float(v[int(temp_idx)].item())
        return float(v)

    def region_value(region: str, cond: str) -> float:
        if region == "*":
            vals = [v for (c, _r), v in region_totals.items() if c == cond]
            if not vals:
                return 0.0
            if torch.is_tensor(vals[0]):
                if temp_idx is None:
                    raise FormulaError("temp_idx must be provided when region_totals stores tensors")
                return float(torch.stack([v for v in vals], dim=0).sum(dim=0)[int(temp_idx)].item())
            return float(sum(_as_float(v) for v in vals))
        rnum = int(region)
        key = (cond, rnum)
        if key not in region_totals:
            raise FormulaError(f"Missing region total for {key}")
        return _as_float(region_totals[key])

    def repl(m: re.Match) -> str:
        region, cond = m.group(1), m.group(2)
        if cond not in condition_names:
            raise FormulaError(f"Unknown condition {cond!r} in formula {formula!r}")
        return str(region_value(region, cond))

    expr = _REGION_REF_RE.sub(repl, formula.strip())
    expr = _strip_outer_parens(expr)

    or_parts = _split_top_level(expr, "|")
    if len(or_parts) > 1:
        return any(evaluate_formula(p, region_totals, condition_names, tol=tol) for p in or_parts)

    and_parts = _split_top_level(expr, "&")
    if len(and_parts) > 1:
        return all(evaluate_formula(p, region_totals, condition_names, tol=tol) for p in and_parts)

    atom = _strip_outer_parens(expr.strip())
    idx, comp = _find_top_level_comparator(atom)
    left = _safe_eval_arith(atom[:idx].strip())
    right = _safe_eval_arith(atom[idx + 1:].strip())
    if comp == ">":
        return left > right
    if comp == "<":
        return left < right
    return abs(left - right) <= tol


# ---------------------------------------------------------------------------
# regions
# ---------------------------------------------------------------------------
PUNCT_NO_SPACE = tuple(",.!?:;") + ("'", "'")


def build_condition_text_and_edges(regions: Sequence[Dict]) -> Tuple[str, List[int]]:
    """Build a full sentence string from region contents and return region start edges (utf-8 bytes)."""
    parts: List[str] = []
    edges: List[int] = []
    cursor = 0
    for i, r in enumerate(regions):
        content = (r.get("content") or "").lstrip()
        if i != 0 and content.strip() != "" and not content.startswith(PUNCT_NO_SPACE):
            parts.append(" ")
            cursor += 1
        edges.append(cursor)
        parts.append(content)
        cursor += len(content.encode("utf-8"))
    return "".join(parts), edges


def map_tokens_to_regions(offsets: Sequence[Tuple[int, int]], region_edges: Sequence[int], sent_len: int) -> List[int]:
    """Map each token index to a 1-based region number (0 for empty spans)."""
    ends = list(region_edges[1:]) + [sent_len]
    region_for_token: List[int] = []
    r_cursor = 0
    for (start, end) in offsets:
        if start == end == 0:
            region_for_token.append(0)
            continue
        pos = (end - 1) if end > start else start
        while r_cursor < len(ends) and pos >= ends[r_cursor]:
            r_cursor += 1
        region_for_token.append(r_cursor + 1)
    return region_for_token


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def load_suite(path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@torch.no_grad()
def evaluate_suite(backend: LanguageModel, suite: Dict, temperatures: Sequence[float]) -> Dict:
    """Score one suite: per-prediction and overall accuracy at every temperature."""
    pred_objs = suite.get("predictions", [])
    formulas = [p.get("formula") if isinstance(p, dict) else str(p) for p in pred_objs]
    if not formulas:
        raise RuntimeError("Suite JSON has no predictions.")
    items = suite.get("items", [])
    if not items:
        raise RuntimeError("Suite JSON has no items.")

    temps = [float(t) for t in temperatures]
    n_temps = len(temps)
    per_pred_correct = torch.zeros((n_temps, len(formulas)), dtype=torch.long)
    per_pred_total = torch.zeros((n_temps, len(formulas)), dtype=torch.long)

    for item in items:
        conditions = item.get("conditions", [])
        cond_names = [c["condition_name"] for c in conditions]
        texts, encoded = [], []
        for cond in conditions:
            text, _ = build_condition_text_and_edges(cond.get("regions", []))
            texts.append(text)
            encoded.append(backend.encode_with_offsets(text))
        logps = backend.token_logprobs([ids for ids, _ in encoded], temps)

        region_totals: Dict[Tuple[str, int], torch.Tensor] = {}
        for cond, text, (ids, offsets), lp in zip(conditions, texts, encoded, logps):
            cname = cond["condition_name"]
            regions = cond.get("regions", [])
            _, edges = build_condition_text_and_edges(regions)
            token2reg = map_tokens_to_regions(offsets, edges, sent_len=len(text.encode("utf-8")))
            surprisal = -lp / math.log(2.0)
            for r in regions:
                region_totals[(cname, int(r["region_number"]))] = torch.zeros((n_temps,), dtype=torch.float32)
            for tok_i, rnum in enumerate(token2reg):
                if rnum <= 0:
                    continue
                key = (cname, int(rnum))
                if key in region_totals:
                    region_totals[key] += surprisal[:, tok_i]

        for j, fml in enumerate(formulas):
            for ti in range(n_temps):
                try:
                    ok = evaluate_formula(fml, region_totals, cond_names, temp_idx=ti)
                    per_pred_total[ti, j] += 1
                    per_pred_correct[ti, j] += int(ok)
                except FormulaError:
                    pass

    total_correct = per_pred_correct.sum(dim=1)
    total_preds = per_pred_total.sum(dim=1)
    return {
        "suite_name": suite.get("meta", {}).get("name", ""),
        "temperatures": temps,
        "overall_accuracy_by_temperature": (total_correct.float() / total_preds.clamp(min=1).float()).tolist(),
        "total_correct_by_temperature": total_correct.tolist(),
        "total_predictions": int(total_preds[0].item()) if n_temps > 0 else 0,
        "n_items": len(items),
        "n_predictions_per_item": len(formulas),
        "per_prediction_accuracy_by_temperature": (per_pred_correct.float() / per_pred_total.clamp(min=1).float()).tolist(),
        "formulas": formulas,
    }


def evaluate_syntaxgym(backend: LanguageModel, data_dir, suites: Union[str, Sequence[str]] = "core_only",
                       temperatures: Optional[Sequence[float]] = None, progress: bool = False) -> Dict:
    """Score a set of suites under ``data_dir``; returns ``summary`` / ``rows`` / ``protocol`` / ``suites``."""
    data_dir = pathlib.Path(data_dir)
    names = select_suites(suites)
    missing = [s for s in names if not (data_dir / f"{s}.json").exists()]
    if missing:
        raise RuntimeError(f"SyntaxGym suite files missing under {data_dir}: " + ", ".join(missing)
                           + ". Fetch the pinned test suites once with: python evals/syntaxgym/fetch_data.py")
    temps = torch.as_tensor(list(temperatures) if temperatures is not None else temperature_grid().tolist(),
                            dtype=torch.float32)
    n_temps = int(temps.numel())
    t1 = int((temps - 1.0).abs().argmin().item())

    iterator = names
    if progress:
        from tqdm import tqdm
        iterator = tqdm(names, desc="suites")
    per_suite: Dict[str, Dict] = OrderedDict()
    total_correct = torch.zeros(n_temps, dtype=torch.long)
    total_preds = torch.zeros(n_temps, dtype=torch.long)
    for name in iterator:
        result = evaluate_suite(backend, load_suite(data_dir / f"{name}.json"), temps.tolist())
        per_suite[name] = result
        total_correct += torch.tensor(result["total_correct_by_temperature"], dtype=torch.long)
        total_preds += result["total_predictions"]
    weighted = (total_correct.float() / total_preds.clamp(min=1).float()).tolist()
    best = int(torch.argmax(torch.tensor(weighted)).item())

    summary: Dict[str, float] = OrderedDict()
    summary["syntaxgym/best_temperature"] = float(temps[best])
    summary["syntaxgym/best_temperature_index"] = best
    summary["syntaxgym/best_temp_avg_accuracy"] = weighted[best] * 100.0
    summary["syntaxgym/temp_1_avg_accuracy"] = weighted[t1] * 100.0
    for name, result in per_suite.items():
        summary[f"syntaxgym/per_suite/{name}"] = result["overall_accuracy_by_temperature"][t1] * 100.0
    for name, result in per_suite.items():
        summary[f"syntaxgym/per_suite_at_best/{name}"] = result["overall_accuracy_by_temperature"][best] * 100.0
    summary["syntaxgym/n_suites"] = len(per_suite)
    summary["syntaxgym/n_predictions"] = int(total_preds[0].item()) if n_temps else 0
    rows = [{"suite": name, "n_items": r["n_items"], "n_predictions": r["total_predictions"],
             "accuracy_at_temperature_1": r["overall_accuracy_by_temperature"][t1] * 100.0,
             "accuracy_at_best_temperature": r["overall_accuracy_by_temperature"][best] * 100.0,
             "accuracy_by_temperature": r["overall_accuracy_by_temperature"]} for name, r in per_suite.items()]
    return {
        "summary": summary, "rows": rows,
        "protocol": {"suites": names, "suite_set": suites if isinstance(suites, str) else "explicit",
                     "temperatures": temps.tolist(), "temperature_1_index": t1, "best_temperature_index": best,
                     "data_dir": str(data_dir), "backend": backend.name},
        "suites": per_suite,
        "weighted_accuracy_by_temperature": weighted,
    }


# ---------------------------------------------------------------------------
# the trainers' metric dicts and the batch record
# ---------------------------------------------------------------------------
def training_metrics(result: Dict) -> Dict[str, float]:
    """Per-suite accuracy at T=1.0 plus the item-weighted aggregates (the single-process trainers' keys)."""
    s = result["summary"]
    out = OrderedDict((k, v) for k, v in s.items() if k.startswith("syntaxgym/per_suite/"))
    out["syntaxgym/best_temperature"] = s["syntaxgym/best_temperature"]
    out["syntaxgym/best_temp_avg_accuracy"] = s["syntaxgym/best_temp_avg_accuracy"]
    out["syntaxgym/temp_1_avg_accuracy"] = s["syntaxgym/temp_1_avg_accuracy"]
    return out


def training_metrics_weighted(result: Dict) -> Dict[str, float]:
    """Aggregates plus per-suite accuracy at both temperatures (the multi-process trainers' keys)."""
    s, best, t1 = result["summary"], result["protocol"]["best_temperature_index"], result["protocol"]["temperature_1_index"]
    out = OrderedDict()
    out["syntaxgym/best_temperature"] = s["syntaxgym/best_temperature"]
    out["syntaxgym/best_temp_weighted_accuracy"] = s["syntaxgym/best_temp_avg_accuracy"]
    out["syntaxgym/temp_1_weighted_accuracy"] = s["syntaxgym/temp_1_avg_accuracy"]
    for name, r in result["suites"].items():
        out[f"syntaxgym/best_temp_suite_{name}"] = r["overall_accuracy_by_temperature"][best] * 100.0
        out[f"syntaxgym/temp_1_suite_{name}"] = r["overall_accuracy_by_temperature"][t1] * 100.0
    return out


def record_entry(result: Dict, path: Optional[str] = None) -> Dict:
    """One model's entry in the committed record: aggregates, per-suite T=1.0 values, the trainer metric dict."""
    raw = training_metrics(result)
    entry = {"raw": raw,
             "per_suite": {k[len("syntaxgym/per_suite/"):]: v for k, v in raw.items() if k.startswith("syntaxgym/per_suite/")},
             "best_temperature": raw["syntaxgym/best_temperature"],
             "best_temp_avg_accuracy": raw["syntaxgym/best_temp_avg_accuracy"],
             "temp_1_avg_accuracy": raw["syntaxgym/temp_1_avg_accuracy"]}
    if path is not None:
        entry["path"] = str(path)
    return entry


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------
def parse_models(specs: Sequence[str]) -> List[Tuple[str, Dict[str, str]]]:
    """``LABEL:key=value[,key=value...]`` -> (label, options)."""
    out = []
    for spec in specs:
        label, sep, rest = spec.partition(":")
        if not sep or not label:
            raise ValueError(f"--models expects LABEL:key=value[,key=value...], got {spec!r}")
        out.append((label, parse_backend_args([p for p in rest.split(",") if p])))
    return out


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_path", required=True, type=pathlib.Path, help="directory of suite json files")
    parser.add_argument("--output_json", required=True, type=pathlib.Path)
    parser.add_argument("--suite_set", default="core_only", choices=sorted(SUITE_SETS))
    parser.add_argument("--suites", default=None, help="comma-separated suite names (overrides --suite_set)")
    parser.add_argument("--temperatures", default=None,
                        help="comma-separated temperatures instead of the 0.00..3.00 step 0.05 grid")
    parser.add_argument("--models", action="append", default=[], metavar="LABEL:KEY=VALUE[,KEY=VALUE...]",
                        help="one model per flag, by the backend options that differ from --backend-arg")
    parser.add_argument("--no_progress", action="store_true")
    add_backend_arguments(parser)
    args = parser.parse_args(argv)

    suites = [s.strip() for s in args.suites.split(",") if s.strip()] if args.suites else args.suite_set
    temps = [float(t) for t in args.temperatures.split(",")] if args.temperatures else None
    device = torch.device(args.device) if args.device else None
    common = parse_backend_args(args.backend_arg)
    models = parse_models(args.models) or [(None, {})]

    record: Dict = {"meta": {"timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                             "input_path": str(args.input_path), "suites": suites if isinstance(suites, str) else ",".join(suites),
                             "backend": args.backend, "backend_args": common, "device": str(device or "auto")},
                    "checkpoints": {}}
    for label, options in models:
        backend = load_backend(args.backend, dict(common, **options), device=device)
        print(f"[{record['meta']['timestamp']}] scoring {label or backend.name}")
        result = evaluate_syntaxgym(backend, args.input_path, suites=suites, temperatures=temps,
                                    progress=not args.no_progress)
        entry = record_entry(result, path=options.get("checkpoint"))
        print(f"  best temperature {entry['best_temperature']:.2f} -> {entry['best_temp_avg_accuracy']:.2f}%; "
              f"T=1.0 -> {entry['temp_1_avg_accuracy']:.2f}%")
        record["checkpoints"][label or backend.name] = entry
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(record, indent=2, sort_keys=True))
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
