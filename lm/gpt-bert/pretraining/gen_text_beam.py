#!/usr/bin/env python
import argparse, json, math, os, random
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from model_extra import Bert

# Optional: LoRA
try:
    from lora import add_lora_gptbert, mark_only_lora_trainable
    HAVE_LORA = True
except Exception:
    HAVE_LORA = False


# ----------------- utils -----------------

def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_config(path: str) -> dict:
    with open(path, "r") as f: return json.load(f)

def build_causal_mask(T: int, device: torch.device) -> torch.BoolTensor:
    m = torch.ones(T, T, dtype=torch.bool, device=device)
    m = torch.tril(m)
    m = ~m  # True = masked
    return m.unsqueeze(0)  # [1,T,T]

def logits_last_pos(model: Bert, ids_1d: torch.LongTensor, attn_mask_btt: torch.BoolTensor) -> torch.Tensor:
    """
    Return logits for next token at last position for a single sequence.
    """
    T = ids_1d.size(0)
    inp = ids_1d.view(T, 1)
    labels = torch.full((T, 1), -100, dtype=torch.long, device=ids_1d.device)
    labels[-1, 0] = 0  # request logits at last position
    with torch.no_grad():
        contextual = model.get_contextualized(inp, attn_mask_btt)
        logits_sel = model.classifier(contextual, labels)  # [1,V]
    return logits_sel[0]  # [V]

def apply_repetition_penalty_(logits: torch.Tensor, history_ids: list[int], penalty: float):
    if penalty <= 1.0 or not history_ids: return
    for tid in set(history_ids):
        v = logits[tid].item()
        logits[tid] = (v / penalty) if v > 0 else (v * penalty)

def block_repeated_ngrams_(logits: torch.Tensor, generated: list[int], n: int):
    if n <= 0 or len(generated) < n - 1: return
    prefix = tuple(generated[-(n - 1):]) if n > 1 else tuple()
    bans = set()
    for i in range(len(generated) - n + 1):
        if n == 1 or tuple(generated[i:i + n - 1]) == prefix:
            bans.add(generated[i + n - 1])
    for tid in bans:
        logits[tid] = float("-inf")


# ----------------- decoding: sampling -----------------

def sample_decode(model, tokenizer, args, device):
    pad_id  = tokenizer.token_to_id("<pad>")
    mask_id = tokenizer.token_to_id("<mask>")
    bos_id  = tokenizer.token_to_id("<s>")
    eos_id  = tokenizer.token_to_id("</s>")

    if args.prompt:
        prompt_ids = tokenizer.encode(args.prompt).ids
    else:
        prompt_ids = []
    if not args.no_bos:
        if not prompt_ids or prompt_ids[0] != bos_id:
            prompt_ids = [bos_id] + prompt_ids

    generated = list(prompt_ids)
    for _ in range(args.max_new_tokens):
        ids = torch.tensor(generated, dtype=torch.long, device=device)
        attn = build_causal_mask(ids.size(0), device)
        logits = logits_last_pos(model, ids, attn)

        # block specials
        if pad_id is not None:  logits[pad_id] = float("-inf")
        if mask_id is not None: logits[mask_id] = float("-inf")
        if bos_id is not None and len(generated) > 0: logits[bos_id] = float("-inf")
        # block EOS if min_len not reached
        if eos_id is not None and len(generated) < args.min_len:
            logits[eos_id] = float("-inf")

        # repetition
        hist = generated[-args.rep_window:] if args.rep_window > 0 else generated
        apply_repetition_penalty_(logits, hist, args.repetition_penalty)

        # n-gram block
        block_repeated_ngrams_(logits, generated, args.no_repeat_ngram_size)

        # temperature
        if args.temperature != 1.0:
            logits = logits / max(1e-8, args.temperature)

        # top-k / top-p
        if args.top_k and args.top_k > 0:
            v, _ = torch.topk(logits, k=min(args.top_k, logits.numel()))
            logits[logits < v[-1]] = float("-inf")
        if args.top_p and args.top_p < 1.0:
            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum > args.top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0]  = False
            logits[sorted_idx[mask]] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()

        generated.append(next_id)
        if eos_id is not None and next_id == eos_id:
            if len(generated) >= args.min_len: break
            # otherwise continue sampling (already blocked above)

    to_decode = generated[1:] if (not args.no_bos and generated and generated[0] == bos_id) else generated
    return tokenizer.decode(to_decode)


# ----------------- decoding: beam search -----------------

def beam_search_decode(model, tokenizer, args, device):
    """
    Deterministic beam search with length penalty, repetition penalty, and no-repeat-ngram constraint.
    """
    pad_id  = tokenizer.token_to_id("<pad>")
    mask_id = tokenizer.token_to_id("<mask>")
    bos_id  = tokenizer.token_to_id("<s>")
    eos_id  = tokenizer.token_to_id("</s>")

    if args.prompt:
        prompt_ids = tokenizer.encode(args.prompt).ids
    else:
        prompt_ids = []
    if not args.no_bos:
        if not prompt_ids or prompt_ids[0] != bos_id:
            prompt_ids = [bos_id] + prompt_ids

    # Each beam: (tokens:list[int], raw_logprob:float, norm_score:float, ended:bool)
    def length_norm(L: int, alpha: float) -> float:
        # HF-style length penalty
        return ((5 + L) / 6) ** alpha if alpha != 0.0 else 1.0

    beams = [{
        "tokens": list(prompt_ids),
        "raw": 0.0,
        "score": 0.0,
        "ended": False
    }]
    completed = []

    for step in range(args.max_new_tokens):
        candidates = []
        for b in beams:
            if b["ended"]:
                candidates.append((b["score"], b["raw"], b["tokens"], True))
                continue

            ids = torch.tensor(b["tokens"], dtype=torch.long, device=device)
            attn = build_causal_mask(ids.size(0), device)
            logits = logits_last_pos(model, ids, attn)

            # block specials / EOS (min_len)
            if pad_id is not None:  logits[pad_id] = float("-inf")
            if mask_id is not None: logits[mask_id] = float("-inf")
            if bos_id is not None and len(b["tokens"]) > 0: logits[bos_id] = float("-inf")
            if eos_id is not None and len(b["tokens"]) < args.min_len:
                logits[eos_id] = float("-inf")

            # repetition (local window)
            hist = b["tokens"][-args.rep_window:] if args.rep_window > 0 else b["tokens"]
            apply_repetition_penalty_(logits, hist, args.repetition_penalty)

            # no-repeat n-gram
            block_repeated_ngrams_(logits, b["tokens"], args.no_repeat_ngram_size)

            # Optional per-beam top-k to prune expansions
            beam_k = args.beam_top_k if args.beam_top_k and args.beam_top_k > 0 else args.num_beams
            logprobs = torch.log_softmax(logits, dim=-1)
            topv, topi = torch.topk(logprobs, k=min(beam_k, logprobs.numel()))

            for v, idx in zip(topv.tolist(), topi.tolist()):
                new_tokens = b["tokens"] + [idx]
                new_raw = b["raw"] + v
                ended = (eos_id is not None and idx == eos_id and len(new_tokens) >= args.min_len)
                norm = new_raw / length_norm(len(new_tokens), args.length_penalty)
                candidates.append((norm, new_raw, new_tokens, ended))

        # select next beams
        candidates.sort(key=lambda x: x[0], reverse=True)
        new_beams = []
        for norm, raw, toks, ended in candidates:
            if ended:
                completed.append((norm, raw, toks))
            else:
                new_beams.append({"tokens": toks, "raw": raw, "score": norm, "ended": False})
            if len(new_beams) >= args.num_beams:
                break

        beams = new_beams

        # stopping
        if args.early_stopping and (len(completed) >= args.num_beams or len(beams) == 0):
            break

    # choose best output
    if completed:
        completed.sort(key=lambda x: x[0], reverse=True)
        best = completed[0][2]
    else:
        # fall back to best ongoing beam
        beams.sort(key=lambda x: x["score"], reverse=True)
        best = beams[0]["tokens"]

    # drop BOS for printing
    bos_id = tokenizer.token_to_id("<s>")
    to_decode = best[1:] if (not args.no_bos and best and best[0] == bos_id) else best
    return tokenizer.decode(to_decode)


# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser("Text generation (sampling or beam) for GPT-BERT")
    # Model/tokenizer
    ap.add_argument("--config_file", required=True)
    ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--checkpoint", required=True)

    # LoRA adapter (optional)
    ap.add_argument("--lora_adapter", default=None)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_scope", type=str, default="attn,mlp")
    ap.add_argument("--train_embeddings", action="store_true",
                    help="If set, load trained embeddings from LoRA adapter (use when fine-tuning trained embeddings).")
    ap.add_argument("--embeddings_path", default=None,
                    help="Optional separate embeddings .pt file (for --train_only_embeddings mode).")

    # Mode
    ap.add_argument("--strategy", choices=["sample", "beam"], default="sample")

    # Common decoding args
    ap.add_argument("--prompt", type=str, default="")
    ap.add_argument("--max_new_tokens", type=int, default=100)
    ap.add_argument("--no_bos", action="store_true")
    ap.add_argument("--min_len", type=int, default=0)
    ap.add_argument("--repetition_penalty", type=float, default=1.2)
    ap.add_argument("--rep_window", type=int, default=128)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=0)

    # Sampling-only
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--top_p", type=float, default=1.0)

    # Beam-only
    ap.add_argument("--num_beams", type=int, default=4)
    ap.add_argument("--length_penalty", type=float, default=1.0)
    ap.add_argument("--early_stopping", action="store_true")
    ap.add_argument("--beam_top_k", type=int, default=0, help="Per-beam top-k pruning (0=disable)")

    # System
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # tokenizer + config
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    cfg = load_config(args.config_file)
    cfg["vocab_size"] = tokenizer.get_vocab_size()
    config = SimpleNamespace(**cfg)

    # special IDs
    pad_id  = tokenizer.token_to_id("<pad>")
    mask_id = tokenizer.token_to_id("<mask>")
    bos_id  = tokenizer.token_to_id("<s>")
    eos_id  = tokenizer.token_to_id("</s>")

    # model
    model = Bert(config).to(device)
    state = torch.load(args.checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("[warn] load_state_dict: missing:", missing, " unexpected:", unexpected, flush=True)

    # optional LoRA
    if args.lora_adapter is not None:
        if not HAVE_LORA:
            raise RuntimeError("lora.py not found; cannot load --lora_adapter")
        add_lora_gptbert(model, r=args.lora_r, alpha=args.lora_alpha, scope=args.lora_scope)
        # If embeddings were trained during fine-tuning, pass train_embeddings=True
        mark_only_lora_trainable(model, train_layernorm=False, train_embeddings=args.train_embeddings)
        model.to(device)
        lora_sd = torch.load(args.lora_adapter, map_location="cpu")
        missing, unexpected = model.load_state_dict(lora_sd, strict=False)
        if unexpected:
            print("[warn] unexpected LoRA keys:", unexpected)
        if missing:
            # only warn if an expected lora key is missing
            miss_lora = [k for k in missing if "lora_" in k]
            if miss_lora:
                print("[warn] missing LoRA keys:", miss_lora)

        # Show what was actually loaded from the adapter
        loaded_keys = [k for k in lora_sd.keys()]
        lora_keys = [k for k in loaded_keys if "lora_" in k]
        emb_keys = [k for k in loaded_keys if "embedding" in k.lower()]
        print(f"[info] Loaded {len(lora_keys)} LoRA keys, {len(emb_keys)} embedding keys from adapter")
        if emb_keys:
            print(f"[info] Embedding keys loaded: {emb_keys}")

    # Optionally load trained embeddings from a separate file
    if args.embeddings_path is not None:
        print(f"Loading trained embeddings from {args.embeddings_path}...")
        emb_sd = torch.load(args.embeddings_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(emb_sd, strict=False)
        if unexpected:
            print("[warn] unexpected embedding keys:", unexpected)
        emb_loaded = [k for k in emb_sd.keys() if "embedding" in k.lower()]
        print(f"Loaded embedding weights: {emb_loaded}")

    model.eval()

    # decode
    if args.strategy == "beam":
        text = beam_search_decode(model, tokenizer, args, device)
    else:
        text = sample_decode(model, tokenizer, args, device)

    print(text)


if __name__ == "__main__":
    main()
