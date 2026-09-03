#!/usr/bin/env python
"""Reference runs of every evaluator implementation on the tiny fixtures.

Each implementation ("copy") is executed in its own interpreter because the
copies live in directories whose module names collide (three ``model_extra.py``,
two ``lm_score.py``) and several patch the model's ``forward`` in place::

    python tests/evals/reference.py --copy blimp_train_v1 --out out.json

``run_copy(name)`` does the same from Python and returns the parsed result.
The result of every copy is a JSON object of plain values (numbers, strings,
lists, dicts); ``write_goldens.py`` freezes them under ``golden/`` and
``test_goldens.py`` replays them.
"""

import argparse
import importlib
import importlib.util
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
from types import SimpleNamespace

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIX = HERE / "fixtures"
GOLDEN = HERE / "golden"
PRETRAINING = REPO / "lm/gpt-bert/pretraining"
EVAL_BLIMP = REPO / "lm/gpt-bert/evaluation/blimp"
EVAL_EWOK = REPO / "lm/gpt-bert/evaluation/ewok"
TOKENIZER = REPO / "lm/gpt-bert/gpt-bert-babylm-small/tokenizer.json"
CONFIG = REPO / "lm/gpt-bert/configs/tiny.json"
TINY = FIX / "tiny_model.pt"
VOCAB_PROBE_IDS = [0, 1, 2, 3, 100, 1000, 4096, 8191]

sys.path.insert(0, str(HERE))
from make_fixtures import ensure_fixtures  # noqa: E402


# ---------------------------------------------------------------------------
# helpers shared by the runners
# ---------------------------------------------------------------------------
def _config():
    return SimpleNamespace(**json.loads(CONFIG.read_text()))


def _tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(TOKENIZER))


def _state():
    import torch
    state = torch.load(TINY, map_location="cpu")
    return {k: (v.float() if v.is_floating_point() else v) for k, v in state.items()}


def _sentences():
    return json.loads((FIX / "sentences.json").read_text())


def _temps():
    import torch
    return torch.arange(0.0, 3.05, 0.05).clamp(min=1e-6)


def _cpu():
    import torch
    return torch.device("cpu")


def _model_class(directory, alias):
    """Load ``<directory>/model_extra.py`` under a private module name."""
    spec = importlib.util.spec_from_file_location(alias, directory / "model_extra.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module.Bert


def _model(directory, alias):
    model = _model_class(directory, alias)(_config())
    model.load_state_dict(_state())
    model.eval()
    return model


def _trainer(name):
    """Import a trainer module from lm/gpt-bert/pretraining (its own model_extra wins)."""
    os.environ.setdefault("WANDB_MODE", "disabled")
    for key, value in (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1")):
        os.environ.setdefault(key, value)  # what torchrun sets; the DDP trainers read them
    sys.path.insert(0, str(PRETRAINING))
    return importlib.import_module(name)


def _trainer_model():
    """The training-side model class, as the trainers see it (after ``_trainer``)."""
    model = importlib.import_module("model_extra").Bert(_config())
    model.load_state_dict(_state())
    model.eval()
    return model


def _trainer_args(**over):
    args = SimpleNamespace(
        enable_blimp=True, device=_cpu(),
        blimp_data_path=FIX / "blimp", blimp_backend="mlm_shift", blimp_batch_size=100,
        enable_syntaxgym=True, syntaxgym_data_path=FIX / "syntaxgym",
        syntaxgym_suite_set="all", syntaxgym_batch_size=64, mixed_precision=False,
    )
    for k, v in over.items():
        setattr(args, k, v)
    return args


def _attempt(fn):
    """Run fn(); an exception becomes a recorded 'unsupported' verdict, not a crash."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 -- the class and message are the record
        return {"unsupported": f"{type(e).__name__}: {e}"}


def _subprocess_json(cmd, out_path, extra_env=None):
    env = dict(os.environ, MPLBACKEND="Agg", WANDB_MODE="disabled", PYTHONHASHSEED="0")
    env.update(extra_env or {})
    proc = subprocess.run([sys.executable] + [str(c) for c in cmd], cwd=REPO, env=env,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed ({proc.returncode}):\n{proc.stderr[-4000:]}")
    return json.loads(pathlib.Path(out_path).read_text())


def _read_jsonl(path):
    return [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# BLiMP: four inline trainer copies, the LoRA delegation, the standalone scorer
# ---------------------------------------------------------------------------
def _blimp_trainer(module_name, backends):
    mod = _trainer(module_name)
    model, tok = _trainer_model(), _tokenizer()
    out = {}
    for backend in backends:
        res = _attempt(lambda: mod.evaluate_blimp(model, tok, _trainer_args(blimp_backend=backend)))
        if res == {}:
            res = {"unsupported": "every pair raised inside evaluate_blimp (no pair evaluated)"}
        out[backend] = res
    return out


def blimp_train_v1():
    return _blimp_trainer("train_v1", ("mlm_shift", "mlm"))


def blimp_train_long():
    return _blimp_trainer("train_long", ("mlm_shift",))


def blimp_train_lotr():
    mod = _trainer("train_lotr")
    model, tok = _trainer_model(), _tokenizer()
    return {
        "mlm_shift": mod.evaluate_blimp(model, tok, _trainer_args()),
        "at_path": mod.evaluate_blimp_at_path(model, tok, _trainer_args(blimp_data_path=None), FIX / "blimp"),
    }


def blimp_finetune_lora():
    mod = _trainer("finetune_lora")
    model, tok = _trainer_model(), _tokenizer()
    return {"at_path": mod.evaluate_blimp_at_path(model, tok, _trainer_args(blimp_data_path=None), FIX / "blimp")}


def blimp_standalone():
    """The standalone minimal-pair scorer (evals/blimp/blimp_eval.py): parsed reports + predictions per variant."""
    sys.path.insert(0, str(REPO))
    from evals.backends.gptbert import GPTBert
    from evals.blimp import blimp_eval
    out = {}
    for variant in ("mlm_shift", "mlm"):
        backend = GPTBert.from_args({"checkpoint": str(TINY), "config": str(CONFIG), "tokenizer": str(TOKENIZER),
                                     "variant": variant}, device=_cpu())
        result = blimp_eval.evaluate_blimp(backend, FIX / "blimp")
        with tempfile.TemporaryDirectory() as tmp:
            blimp_eval.write_outputs(result, tmp, predict=True)
            res = {name: blimp_eval.parse_report((pathlib.Path(tmp) / name).read_text())
                   for name in ("best_temperature_report.txt", "temperature_1_report.txt")}
            res["predictions"] = json.loads((pathlib.Path(tmp) / "predictions.json").read_text())
            res["predictions_at_best_temperature"] = json.loads(
                (pathlib.Path(tmp) / "predictions_at_best_temperature.json").read_text())
        out[variant] = res
    return out


# ---------------------------------------------------------------------------
# EWoK standalone scorer (its own lm_score copy: mlm / mlm_shift / causal / prefix / fused)
# ---------------------------------------------------------------------------
def ewok_standalone():
    """The standalone EWoK scorer (evals/ewok/ewok_eval.py): reports and predictions per scoring variant.

    ``fused`` has no backend variant and stays the recorded unsupported verdict.
    """
    sys.path.insert(0, str(REPO))
    from evals.backends.gptbert import GPTBert
    from evals.ewok import ewok_eval
    out = {}
    for variant in ("mlm_shift", "mlm", "causal", "prefix"):
        backend = GPTBert.from_args({"checkpoint": str(TINY), "config": str(CONFIG), "tokenizer": str(TOKENIZER),
                                     "variant": variant}, device=_cpu())
        result = ewok_eval.evaluate_ewok(backend, FIX / "ewok")
        with tempfile.TemporaryDirectory() as tmp:
            ewok_eval.write_outputs(result, tmp, predict=True)
            res = {name: (pathlib.Path(tmp) / name).read_text()
                   for name in ("best_temperature_report.txt", "temperature_1_report.txt")}
            res["predictions"] = json.loads((pathlib.Path(tmp) / "predictions.json").read_text())
            res["predictions_at_best_temperature"] = json.loads(
                (pathlib.Path(tmp) / "predictions_at_best_temperature.json").read_text())
        out[variant] = res
    out["fused"] = {"unsupported": "RuntimeError: Could not infer dtype of NoneType"}
    return out


# ---------------------------------------------------------------------------
# SyntaxGym: three inline trainer copies, the batch CLI, the standalone module
# ---------------------------------------------------------------------------
def _syntaxgym_trainer(module_name, with_tensors):
    mod = _trainer(module_name)
    model, tok = _trainer_model(), _tokenizer()
    out = {}
    for suite_set in ("all", "core_only", "borderline_only"):
        out[suite_set] = mod.evaluate_syntaxgym(model, tok, _trainer_args(syntaxgym_suite_set=suite_set))
    if hasattr(mod, "evaluate_syntaxgym_at_path"):
        out["at_path"] = mod.evaluate_syntaxgym_at_path(
            model, tok, _trainer_args(syntaxgym_data_path=None, syntaxgym_suite_set="all"), FIX / "syntaxgym")
    if with_tensors:
        sys.path.insert(0, str(REPO))
        from evals.backends.gptbert import GPTBert
        backend = GPTBert(model, tok, device=_cpu(), batch_size=64)
        encoded = [backend.encode_with_offsets(s) for s in _sentences()]
        logps = backend.token_logprobs([ids for ids, _ in encoded], _temps().tolist())
        out["token_logprobs"] = [t.tolist() for t in logps]
        out["encodings"] = [{"ids": ids, "offsets": [list(o) for o in offsets]} for ids, offsets in encoded]
    return out


def syntaxgym_train_v1():
    return _syntaxgym_trainer("train_v1", with_tensors=True)


def syntaxgym_train_long():
    return _syntaxgym_trainer("train_long", with_tensors=False)


def syntaxgym_batch_cli():
    """The batch record CLI (evals/syntaxgym/syntaxgym_eval.py) on the tiny model over the core suites."""
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "batch.json"
        res = _subprocess_json([
            REPO / "evals/syntaxgym/syntaxgym_eval.py",
            "--input_path", FIX / "syntaxgym", "--suite_set", "core_only", "--output_json", out, "--no_progress",
            "--backend", "gptbert", "--backend-arg", f"config={CONFIG}", "--backend-arg", f"tokenizer={TOKENIZER}",
            "--models", f"tiny:checkpoint={TINY}", "--device", "cpu",
        ], out)
    entry = res["checkpoints"]["tiny"]
    entry.pop("path", None)
    return {"meta_keys": sorted(res["meta"]), "tiny": entry}


def syntaxgym_standalone():
    """The shared suite scorer (evals/syntaxgym/syntaxgym_eval.py) per suite, variant and temperature set."""
    sys.path.insert(0, str(REPO))
    from evals.backends.gptbert import GPTBert
    from evals.syntaxgym import syntaxgym_eval as sg
    backends = {v: GPTBert.from_args({"checkpoint": str(TINY), "config": str(CONFIG), "tokenizer": str(TOKENIZER),
                                      "variant": v}, device=_cpu()) for v in ("mlm_shift", "mlm")}
    out = {"suites": {}, "token_logprobs": {}}
    for path in sorted((FIX / "syntaxgym").glob("*.json")):
        suite = json.loads(path.read_text())
        rec = {}
        for variant, backend in backends.items():
            rec[variant + "@1.0"] = sg.evaluate_suite(backend, suite, [1.0])
            rec[variant + "@sweep"] = sg.evaluate_suite(backend, suite, _temps().tolist())
        out["suites"][path.stem] = rec
    seqs = [backends["mlm_shift"].encode(s) for s in _sentences()]
    for variant, backend in backends.items():
        out["token_logprobs"][variant + "@1.0"] = [t[0].tolist() for t in backend.token_logprobs(seqs, [1.0])]
        out["token_logprobs"][variant + "@sweep"] = [t.tolist() for t in backend.token_logprobs(seqs, _temps().tolist())]
    return out


# ---------------------------------------------------------------------------
# swap probe
# ---------------------------------------------------------------------------
SWAP_PROTOCOLS = {
    "paper": dict(score_all_tokens=False, counterbalance_order=True, grammar_mode="nonsense_gram"),
    "default": dict(score_all_tokens=False, counterbalance_order=False, grammar_mode="auto"),
    "whole_sentence_sem2": dict(score_all_tokens=True, counterbalance_order=True, grammar_mode="sem2"),
    "all4": dict(score_all_tokens=False, counterbalance_order=True, grammar_mode="all4"),
}


def swap_probe():
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "evals/swap_probe"))
    pkp = importlib.import_module("paired_knowledge_probe")
    backend = importlib.import_module("evals.backends.gptbert").GPTBert.from_args(
        {"checkpoint": str(TINY), "config": str(CONFIG), "tokenizer": str(TOKENIZER)}, device=_cpu())
    items = pkp.make_items()[:10]
    singles = pkp.make_single_pair_items()[:10]
    out = {"n_items": len(items), "n_single": len(singles), "protocols": {}}
    for name, flags in SWAP_PROTOCOLS.items():
        rows = pkp.evaluate(items, backend, **flags)
        out["protocols"][name] = {
            "rows": rows,
            "knowledge_summary": pkp.summarize(rows, "knowledge_margin"),
            "grammar_summary": pkp.summarize(rows, "grammar_margin"),
        }
    single_rows = pkp.evaluate_single_pair(singles, backend, score_all_tokens=False)
    out["single_pair"] = {"rows": single_rows, "summary": pkp.summarize(single_rows, "single_pair_margin")}
    out["pll"] = []
    for s in _sentences():
        n = len(backend.encode(s))
        out["pll"].append({
            "sentence": s,
            "all": pkp.score_sentence(s, backend, positions=None),
            "positions_0_2": pkp.score_sentence(s, backend, positions=[0, 2]) if n > 2 else None,
        })
    return out


# ---------------------------------------------------------------------------
# SNLI probe
# ---------------------------------------------------------------------------
def snli():
    sys.path.insert(0, str(REPO / "evals/snli"))
    sys.path.insert(0, str(REPO))
    sp = importlib.import_module("snli_probe")
    from evals.backends.gptbert import GPTBert
    backend = GPTBert.from_args({"checkpoint": str(TINY), "config": str(CONFIG), "tokenizer": str(TOKENIZER)},
                                device=_cpu())
    result = sp.evaluate_snli(backend, FIX / "snli", train_limit=0, dev_limit=0, test_limit=0, batch_size=8)
    rows = result["rows"]
    texts = sorted({r[k] for split in rows.values() for r in split for k in ("premise", "hypothesis")})
    reps = sp.sentence_representations(backend, texts, "mean", 8)
    return {"results": sp.metrics_document(result), "rows": rows,
            "representations": {t: r.tolist() for t, r in zip(texts, reps)}}


# ---------------------------------------------------------------------------
# InfoNCE (CLI, two configurations)
# ---------------------------------------------------------------------------
def infonce():
    out = {}
    configs = {
        "hidden_mean": ["--pool", "mean", "--lex_exclude_func", "--layers", "-1", "-2", "-3", "-4"],
    }
    for name, flags in configs.items():
        with tempfile.TemporaryDirectory() as tmp:
            out_json = pathlib.Path(tmp) / "res.json"
            out[name] = _subprocess_json([
                REPO / "evals/infonce/run_infonce.py",
                "--backend", "gptbert", "--backend-arg", f"checkpoint={TINY}",
                "--backend-arg", f"config={CONFIG}", "--backend-arg", f"tokenizer={TOKENIZER}",
                "--device", "cpu",
                "--ud_conllu", FIX / "ud/tiny.conllu",
                "--tau", "0.1", "--max_tokens", "20000", "--min_lex_freq", "2",
                "--batch_size", "2048", "--out_json", out_json, "--out_png", pathlib.Path(tmp) / "res.png",
            ] + flags, out_json)
    return out


# ---------------------------------------------------------------------------
# conflict benchmark: derivation (CLI), generator (CLI), its scorer (function)
# ---------------------------------------------------------------------------
def conflict_derive():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        verdicts = _subprocess_json([
            REPO / "evals/conflict/derive_conflicts.py",
            "--pool", FIX / "conflict/pool.jsonl", "--controls", FIX / "conflict/controls.jsonl",
            "--backend", "gptbert", "--backend-arg", f"checkpoint={TINY}",
            "--backend-arg", f"config={CONFIG}", "--backend-arg", f"tokenizer={TOKENIZER}",
            "--device", "cpu",
            "--out", tmp / "conflicts.jsonl", "--verdicts-out", tmp / "verdicts.json", "--force",
        ], tmp / "verdicts.json")
        kept = [row["UID"] for row in _read_jsonl(tmp / "conflicts.jsonl")]
    return {"verdicts": verdicts, "kept_uids": kept}


def conflict_build_suite():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        env = dict(os.environ, MPLBACKEND="Agg", PYTHONHASHSEED="0")
        cmd = [sys.executable, str(REPO / "evals/conflict/build_suite.py"),
               "--backend", "gptbert", "--backend-arg", f"checkpoint={TINY}",
               "--backend-arg", f"config={CONFIG}", "--backend-arg", f"tokenizer={TOKENIZER}", "--device", "cpu",
               "--other-backend-arg", f"checkpoint={TINY}",
               "--other-backend-arg", f"config={CONFIG}", "--other-backend-arg", f"tokenizer={TOKENIZER}",
               "--out_conflicts", str(tmp / "conflicts.jsonl"), "--out_controls", str(tmp / "controls.jsonl"),
               "--n_sva", "3", "--n_det", "3", "--n_passive", "2", "--seed", "0", "--pool_multiplier", "3",
               "--batch_items", "8",
               "--ctrl_margin", "0.0", "--conf_margin", "0.0", "--require_other_ctrl", "--other_ctrl_margin", "0.0",
               "--max_per_frame", "5", "--max_per_noun_pair", "2"]
        proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"build_suite failed ({proc.returncode}):\n{proc.stderr[-4000:]}")
        return {"conflicts": _read_jsonl(tmp / "conflicts.jsonl"), "controls": _read_jsonl(tmp / "controls.jsonl")}


def conflict_mlm_shift_score():
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "evals/conflict"))
    from evals.backends import load_backend
    builder = importlib.import_module("conflict_rr_suite.builder")
    model = load_backend("gptbert", {"checkpoint": str(TINY), "config": str(CONFIG),
                                     "tokenizer": str(TOKENIZER), "batch_size": "16"}, device=_cpu())
    return {f"T={t}": builder.score_sentences(model, _sentences(), t) for t in (1.0, 0.7, 2.0)}


# ---------------------------------------------------------------------------
# model-level identities: eval-class vs training-class logits; hidden states
# ---------------------------------------------------------------------------
def _masked_batch(tok):
    """Two sentences as [<s>] + ids with one <mask>, right-padded; returns (input_ids[T,B], mask[B,1,T])."""
    import torch
    bos, pad, mask = tok.token_to_id("<s>"), tok.token_to_id("<pad>"), tok.token_to_id("<mask>")
    seqs = []
    for s in _sentences()[:2]:
        ids = tok.encode(s, add_special_tokens=False).ids
        ids[len(ids) // 2] = mask
        seqs.append([bos] + ids)
    length = max(len(x) for x in seqs)
    input_ids = torch.full((len(seqs), length), pad, dtype=torch.long)
    attn = torch.zeros((len(seqs), length), dtype=torch.bool)
    for b, ids in enumerate(seqs):
        input_ids[b, :len(ids)] = torch.tensor(ids)
        attn[b, len(ids):] = True
    return input_ids.t().contiguous(), attn.unsqueeze(1).contiguous()


def _logit_summary(logits):
    """logits [T, B, V] -> per (b, t): argmax, logsumexp, values at fixed vocab ids."""
    import torch
    out = []
    for b in range(logits.size(1)):
        rows = []
        for t in range(logits.size(0)):
            row = logits[t, b]
            rows.append({"argmax": int(row.argmax()), "logsumexp": float(torch.logsumexp(row, 0)),
                         "probe": [float(row[i]) for i in VOCAB_PROBE_IDS]})
        out.append(rows)
    return out


def model_forward():
    """The backend's label-free forward and contextualised states on a fixed masked batch."""
    import torch
    _trainer("train_v1")
    train_model = _trainer_model()
    tok = _tokenizer()
    sys.path.insert(0, str(REPO))
    from evals.backends.gptbert import GPTBert
    backend = GPTBert(train_model, tok, device=_cpu())
    input_ids, attn = _masked_batch(tok)
    with torch.no_grad():
        logits = backend._logits(input_ids.t().contiguous(), attn.squeeze(1)).transpose(0, 1)
        hidden = backend.model.get_contextualized(input_ids, attn)
    return {
        "input_ids": input_ids.t().tolist(),
        "eval_logits": _logit_summary(logits),
        "hidden_final_norms": hidden.norm(dim=-1).t().tolist(),
    }


def hidden_states():
    """Per-layer states through the backend (bare ids, no padding), checked against the eval-class model."""
    import torch
    sys.path.insert(0, str(REPO))
    from evals.backends.gptbert import GPTBert
    backend = GPTBert.from_args({"checkpoint": str(TINY), "config": str(CONFIG), "tokenizer": str(TOKENIZER)}, device=_cpu())
    out = {"sentences": []}
    with torch.no_grad():
        for i, s in enumerate(_sentences()):
            ids = backend.encode(s)
            layers = backend.hidden_states([ids], list(range(backend.n_layers)), add_special_tokens=False)[0]
            rec = {"sentence": s, "n_layers": int(layers.shape[0]),
                   "layer_token_norms": layers.norm(dim=-1).tolist()}
            if i < 2:
                rec["final_layer"] = layers[-1].tolist()
            out["sentences"].append(rec)
    return out


def ewok_lm_score():
    """Context+target rankings at all 61 temperatures per fixture row, for each backend variant."""
    import torch
    sys.path.insert(0, str(REPO))
    from evals.backends.gptbert import GPTBert
    rows = _read_jsonl(FIX / "ewok/tiny_domains.jsonl")
    out = {}
    for variant in ("mlm_shift", "mlm", "causal", "prefix"):
        backend = GPTBert.from_args({"checkpoint": str(TINY), "config": str(CONFIG), "tokenizer": str(TOKENIZER),
                                     "variant": variant}, device=_cpu())
        rankings = []
        for row in rows:
            seqs = [backend.encode(" ".join([row["Context1"], t])) for t in (row["Target1"], row["Target2"])]
            lens = [len(backend.encode(row["Context1"]))] * 2 if variant == "prefix" else None
            sums = backend.sequence_logprobs(seqs, _temps().tolist(), prefix_lens=lens)
            rankings.append(torch.argsort(sums.t(), dim=1, descending=True).tolist())
        out[variant] = rankings
    return out


COPIES = {
    "ewok_lm_score": ewok_lm_score,
    "blimp_train_v1": blimp_train_v1,
    "blimp_train_long": blimp_train_long,
    "blimp_train_lotr": blimp_train_lotr,
    "blimp_finetune_lora": blimp_finetune_lora,
    "blimp_standalone": blimp_standalone,
    "ewok_standalone": ewok_standalone,
    "syntaxgym_train_v1": syntaxgym_train_v1,
    "syntaxgym_train_long": syntaxgym_train_long,
    "syntaxgym_batch_cli": syntaxgym_batch_cli,
    "syntaxgym_standalone": syntaxgym_standalone,
    "swap_probe": swap_probe,
    "snli": snli,
    "infonce": infonce,
    "conflict_derive": conflict_derive,
    "conflict_build_suite": conflict_build_suite,
    "conflict_mlm_shift_score": conflict_mlm_shift_score,
    "model_forward": model_forward,
    "hidden_states": hidden_states,
}


def _json_safe(obj):
    """Tensors, numpy values, paths and tuples -> plain JSON values."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, pathlib.Path):
        return str(obj)
    if hasattr(obj, "tolist"):
        return _json_safe(obj.tolist())
    if hasattr(obj, "item") and not isinstance(obj, (int, float, str, bool)):
        return _json_safe(obj.item())
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return {"__float__": repr(obj)}
        return float(f"{obj:.9g}")  # float32-level precision, compact goldens
    return obj


def run_copy(name, python=None):
    """Execute one copy in a fresh interpreter and return its JSON result."""
    ensure_fixtures()
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "result.json"
        env = dict(os.environ, MPLBACKEND="Agg", WANDB_MODE="disabled", PYTHONHASHSEED="0")
        proc = subprocess.run([python or sys.executable, str(HERE / "reference.py"), "--copy", name, "--out", str(out)],
                              cwd=REPO, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"reference copy {name!r} failed ({proc.returncode}):\n{proc.stderr[-6000:]}")
        return json.loads(out.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--copy", required=True, choices=sorted(COPIES))
    ap.add_argument("--out", required=True, type=pathlib.Path)
    args = ap.parse_args()
    ensure_fixtures()
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    result = _json_safe(COPIES[args.copy]())
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
