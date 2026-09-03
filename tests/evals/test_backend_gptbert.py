"""The gptbert backend: interface conformance, and identity with the frozen scorer goldens."""

import json
import math
import pathlib
import sys

import pytest
import torch

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

from make_fixtures import ensure_fixtures  # noqa: E402
from reference import CONFIG, FIX, GOLDEN, TINY, TOKENIZER  # noqa: E402

from evals.backends import load_backend  # noqa: E402
from evals.backends.conformance import assert_conformant  # noqa: E402
from evals.backends.gptbert import GPTBert  # noqa: E402
from evals.common import temperature_grid  # noqa: E402

# float32 reproducibility across platforms: the low-temperature rows of the log-prob sweep are
# sums of magnitude ~1e6, where one ulp is 0.25, so the sweep comparisons are relative as well
# as absolute
ATOL = 1e-4
RTOL = 1e-5


def golden(name):
    return json.loads((GOLDEN / f"{name}.json").read_text())["values"]


@pytest.fixture(scope="module")
def sentences():
    return json.loads((FIX / "sentences.json").read_text())


def make_backend(variant="mlm_shift", batch_size=64):
    ensure_fixtures()
    return GPTBert.from_args({"checkpoint": str(TINY), "config": str(CONFIG), "tokenizer": str(TOKENIZER),
                              "variant": variant, "batch_size": str(batch_size)}, device=torch.device("cpu"))


@pytest.fixture(scope="module")
def backend():
    return make_backend()


def test_registry_alias_and_module_spec():
    ensure_fixtures()
    args = {"checkpoint": str(TINY), "config": str(CONFIG), "tokenizer": str(TOKENIZER)}
    a = load_backend("gptbert", args, device=torch.device("cpu"))
    b = load_backend("evals.backends.gptbert:GPTBert", args, device=torch.device("cpu"))
    assert type(a) is type(b) is GPTBert and a.name == "tiny_model" and a.max_seq_len == 512 and a.n_layers == 4
    with pytest.raises(ValueError):
        load_backend("gptbert", dict(args, bogus="1"), device=torch.device("cpu"))


@pytest.mark.parametrize("variant", ["mlm_shift", "mlm", "causal", "prefix"])
def test_conformance(variant, sentences):
    assert_conformant(make_backend(variant, batch_size=5), sentences)


def test_mlm_shift_matches_trainer_token_logprobs(backend, sentences):
    """Per-token log-probs over the 61-temperature grid == train_v1.token_logprobs_syntaxgym."""
    want = golden("syntaxgym_train_v1")["token_logprobs"]
    got = backend.token_logprobs([backend.encode(s) for s in sentences], temperature_grid().tolist())
    for g, w in zip(got, want):
        assert g.shape == (61, len(w[0]))
        assert torch.allclose(g, torch.tensor(w, dtype=g.dtype), rtol=RTOL, atol=ATOL)


def test_mlm_matches_standalone_pll(sentences):
    want = golden("syntaxgym_standalone")["token_logprobs"]
    backend = make_backend("mlm")
    seqs = [backend.encode(s) for s in sentences]
    sweep = backend.token_logprobs(seqs, temperature_grid().tolist())
    for g, w in zip(sweep, want["mlm@sweep"]):
        assert torch.allclose(g, torch.tensor(w, dtype=g.dtype), rtol=RTOL, atol=ATOL)
    single = backend.token_logprobs(seqs, [1.0])
    for g, w in zip(single, want["mlm@1.0"]):
        assert torch.allclose(g[0], torch.tensor(w, dtype=g.dtype), rtol=RTOL, atol=ATOL)


def test_sequence_sums_match_conflict_scorer_and_swap_probe(backend, sentences):
    seqs = [backend.encode(s) for s in sentences]
    sums = backend.sequence_logprobs(seqs, [1.0, 0.7, 2.0])
    want = golden("conflict_mlm_shift_score")
    for j, key in enumerate(("T=1.0", "T=0.7", "T=2.0")):
        assert float((sums[:, j] - torch.tensor(want[key])).abs().max()) <= 1e-4
    pll = golden("swap_probe")["pll"]
    per_token = backend.token_logprobs(seqs, [1.0])
    for t, rec in zip(per_token, pll):
        assert abs(float(t[0].sum()) - rec["all"]) <= 1e-4
        if rec["positions_0_2"] is not None:
            assert abs(float(t[0][[0, 2]].sum()) - rec["positions_0_2"]) <= 1e-4


def _ewok_rankings(backend, rows, prefix):
    """The EWoK copy's protocol: rank context1+target1 vs context1+target2 at every temperature."""
    temps = temperature_grid().tolist()
    out = []
    for row in rows:
        seqs = [backend.encode(" ".join([row["Context1"], t])) for t in (row["Target1"], row["Target2"])]
        lens = [len(backend.encode(row["Context1"]))] * 2 if prefix else None
        sums = backend.sequence_logprobs(seqs, temps, prefix_lens=lens)   # [2, 61]
        out.append(torch.argsort(sums.t(), dim=1, descending=True).tolist())
    return out


@pytest.mark.parametrize("variant", ["mlm_shift", "mlm", "causal", "prefix"])
def test_ewok_rankings_match_ewok_copy(variant):
    """Rankings at all 61 temperatures == the vendored EWoK scorer's rank_* for the same variant."""
    rows = [json.loads(l) for l in (FIX / "ewok/tiny_domains.jsonl").read_text().splitlines() if l.strip()]
    want = golden("ewok_lm_score")[variant]
    got = _ewok_rankings(make_backend(variant), rows, prefix=(variant == "prefix"))
    assert got == want


def test_hidden_states_match_model_copies(backend, sentences):
    """All layers == the frozen per-layer states; <s>-prefixed mean pooling == the SNLI probe's representations."""
    want = golden("hidden_states")["sentences"]
    seqs = [backend.encode(s) for s in sentences]
    all_layers = backend.hidden_states(seqs, list(range(backend.n_layers)), add_special_tokens=False)
    for h, rec in zip(all_layers, want):
        norms = torch.tensor(rec["layer_token_norms"])
        assert h.shape[0] == rec["n_layers"] == norms.shape[0]
        assert float((h.norm(dim=-1) - norms).abs().max()) <= 1e-4
        if "final_layer" in rec:
            assert float((h[-1] - torch.tensor(rec["final_layer"])).abs().max()) <= 1e-4
    reps = golden("snli")["representations"]
    texts = sorted(reps)
    pooled = [h[0].mean(dim=0) for h in backend.hidden_states([backend.encode(t) for t in texts], "final")]
    for t, vec in zip(texts, pooled):
        assert float((vec - torch.tensor(reps[t])).abs().max()) <= 1e-4


def test_offsets_and_incremental_decode_fallback(backend, sentences):
    """The tokenizer's own offsets agree with the base-class fallback on these texts."""
    from evals.backends.base import LanguageModel
    for text in sentences:
        ids, offsets = backend.encode_with_offsets(text)
        ids2, fallback = LanguageModel.encode_with_offsets(backend, text)
        assert ids == ids2
        for (s, e), (fs, fe), tok in zip(offsets, fallback, ids):
            assert text[fs:fe].strip() == text[s:e].strip(), (text, tok, (s, e), (fs, fe))
