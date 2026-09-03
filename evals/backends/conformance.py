"""Conformance checks for a ``LanguageModel`` backend.

    from evals.backends.conformance import assert_conformant
    assert_conformant(backend, ["The cat sat on the mat.", ...])

The checks are about the interface contract, not about model quality: text
round-trips, offsets cover the text, log-probabilities have the promised
shape, sign and temperature behaviour, and a sequence scores the same alone
as inside a batch (the padding check). A new backend's tests should run them
on a small instance of its model.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch

from .base import LanguageModel


def run_conformance(backend: LanguageModel, texts: Sequence[str], atol: float = 1e-4) -> Dict[str, Tuple[bool, str]]:
    """Every check name -> (passed, detail)."""
    checks: Dict[str, Tuple[bool, str]] = {}
    texts = list(texts)
    if len(texts) < 2:
        raise ValueError("give at least two texts of different lengths")

    def record(name: str, ok: bool, detail: str = "") -> None:
        checks[name] = (bool(ok), detail)

    # --- text <-> ids -----------------------------------------------------------
    seqs: List[List[int]] = []
    ok_round, ok_offsets, detail = True, True, ""
    for text in texts:
        ids = backend.encode(text)
        seqs.append(ids)
        if " ".join(backend.decode(ids).split()) != " ".join(text.split()):
            ok_round, detail = False, f"decode(encode(t)) != t for {text!r}"
        ids2, offsets = backend.encode_with_offsets(text)
        if ids2 != ids or len(offsets) != len(ids):
            ok_offsets, detail = False, f"offsets misaligned for {text!r}"
            continue
        covered = set()
        last_start = -1
        for start, end in offsets:
            if not (0 <= start <= end <= len(text)) or start < last_start:
                ok_offsets, detail = False, f"bad span {(start, end)} in {text!r}"
            last_start = max(last_start, start)
            covered.update(range(start, end))
        uncovered = [i for i, ch in enumerate(text) if not ch.isspace() and i not in covered]
        if uncovered:
            ok_offsets, detail = False, f"characters {uncovered[:5]} of {text!r} not covered by any token"
    record("encode_decode_round_trip", ok_round, detail if not ok_round else "")
    record("offsets_cover_text", ok_offsets, detail if not ok_offsets else "")

    # --- token log-probabilities -------------------------------------------------
    temps = [1.0, 0.5, 2.0, 1e6]
    scored = backend.token_logprobs(seqs, temps)
    ok_shape = all(t.shape == (len(temps), len(s)) and t.dtype == torch.float32 and t.device.type == "cpu"
                   for t, s in zip(scored, seqs))
    record("token_logprobs_shape", ok_shape, "" if ok_shape else str([tuple(t.shape) for t in scored]))
    finite = all(torch.isfinite(t).all() for t in scored)
    nonpos = all((t <= 1e-6).all() for t in scored)
    record("token_logprobs_finite_nonpositive", finite and nonpos, "" if finite and nonpos else "inf/nan or > 0 found")
    spreads = [float(t[-1].max() - t[-1].min()) for t in scored]
    record("high_temperature_flattens", all(sp < 1e-2 for sp in spreads),
           f"spread of token log-probs at T=1e6: {max(spreads):.3g}")
    single = backend.token_logprobs([seqs[0]], [1.0])[0][0]
    record("temperature_1_matches_default", torch.allclose(single, backend.token_logprobs([seqs[0]])[0][0], atol=atol),
           "token_logprobs with the default temperatures differs from T=1.0")

    # --- batch invariance (padding) ---------------------------------------------
    longest = max(seqs, key=len)
    shortest = min(seqs, key=len)
    alone = backend.token_logprobs([shortest], [1.0])[0]
    batched = backend.token_logprobs([longest, shortest, longest], [1.0])[1]
    diff = float((alone - batched).abs().max())
    record("batch_invariance", diff <= atol, f"max |alone - batched| = {diff:.3g}")

    # --- determinism -------------------------------------------------------------
    again = backend.token_logprobs([shortest], [1.0])[0]
    record("deterministic", torch.allclose(alone, again, atol=1e-6), "two identical calls differ")

    # --- sequence sums -----------------------------------------------------------
    sums = backend.sequence_logprobs(seqs, [1.0, 2.0])
    expect = torch.stack([t[[0, 2]].sum(-1) for t in scored])
    record("sequence_logprobs_are_sums", sums.shape == (len(seqs), 2) and torch.allclose(sums, expect, atol=atol),
           "sequence_logprobs != sum of token_logprobs")

    # --- hints ---------------------------------------------------------------------
    try:
        backend.token_logprobs(seqs[:2], [1.0], prefix_lens=[1, min(2, len(seqs[1]))])
        record("prefix_lens_accepted", True)
    except Exception as e:  # noqa: BLE001
        record("prefix_lens_accepted", False, f"{type(e).__name__}: {e}")
    try:
        keep = [0, len(longest) - 1]
        partial = backend.token_logprobs([longest], [1.0], positions=[keep])[0][0]
        full = backend.token_logprobs([longest], [1.0])[0][0]
        rest = [i for i in range(len(longest)) if i not in keep]
        ok = (partial.shape == full.shape and float((partial[keep] - full[keep]).abs().max()) <= atol
              and (not rest or float(partial[rest].abs().max()) == 0.0))
        record("positions_hint", ok, "scored positions must match the full call; unscored ones must be 0")
    except Exception as e:  # noqa: BLE001
        record("positions_hint", False, f"{type(e).__name__}: {e}")

    # --- hidden states ------------------------------------------------------------
    try:
        final = backend.hidden_states(seqs, "final")
        ok = all(h.ndim == 3 and h.shape[0] == 1 and h.shape[1] == len(s) for h, s in zip(final, seqs))
        record("hidden_states_final_shape", ok, "" if ok else str([tuple(h.shape) for h in final]))
        last = backend.hidden_states([seqs[0]], [-1])[0]
        record("hidden_states_layer_minus_one_is_final", torch.allclose(last, final[0], atol=atol),
               "layers=[-1] differs from 'final'")
        if backend.n_layers > 1:
            two = backend.hidden_states([seqs[0]], [0, -1])[0]
            record("hidden_states_layer_selection", two.shape[0] == 2 and torch.allclose(two[1], final[0], atol=atol),
                   "layers=[0, -1] wrong shape or last layer differs")
        h_alone = backend.hidden_states([shortest], "final")[0]
        h_batched = backend.hidden_states([longest, shortest], "final")[1]
        d = float((h_alone - h_batched).abs().max())
        record("hidden_states_batch_invariance", d <= atol, f"max |alone - batched| = {d:.3g}")
        bare = backend.hidden_states([seqs[0]], "final", add_special_tokens=False)[0]
        record("hidden_states_without_special_tokens", bare.shape == final[0].shape,
               f"{tuple(bare.shape)} vs {tuple(final[0].shape)}")
    except NotImplementedError:
        record("hidden_states_final_shape", False, "hidden_states not implemented")

    # --- properties ---------------------------------------------------------------
    record("properties", isinstance(backend.name, str) and isinstance(backend.device, torch.device)
           and (backend.max_seq_len is None or backend.max_seq_len > 0) and backend.n_layers >= 1,
           f"name={backend.name!r} device={backend.device} max_seq_len={backend.max_seq_len} n_layers={backend.n_layers}")
    return checks


def assert_conformant(backend: LanguageModel, texts: Sequence[str], atol: float = 1e-4) -> None:
    """Raise AssertionError listing every failed check."""
    failed = [f"{name}: {detail}" for name, (ok, detail) in run_conformance(backend, texts, atol).items() if not ok]
    if failed:
        raise AssertionError(f"{type(backend).__name__} fails {len(failed)} conformance check(s):\n  " + "\n  ".join(failed))
