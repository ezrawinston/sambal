#!/usr/bin/env python
import argparse
import json
import math
import os
import random
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from tokenizers import Tokenizer
from model_extra import Bert

# Optional: LoRA adapter support (if you fine-tuned with LoRA)
try:
    from lora import add_lora_gptbert, mark_only_lora_trainable
    HAVE_LORA = True
except Exception:
    HAVE_LORA = False


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r") as f:
        return json.load(f)


def build_causal_mask(T: int, device: torch.device) -> torch.BoolTensor:
    """
    Your Attention expects a boolean mask where True = masked (disallowed).
    In training, you did: mask = ones -> tril -> ~. We replicate that.
    Shape: [1, T, T] for a single example.
    """
    m = torch.ones(T, T, dtype=torch.bool, device=device)
    m = torch.tril(m)
    m = ~m  # True means 'mask'
    return m.unsqueeze(0)


def logits_last_pos(model: Bert,
                    input_ids_1d: torch.LongTensor,
                    attn_mask_btt: torch.BoolTensor) -> torch.Tensor:
    """
    Compute logits for the next token (one step) by selecting only the last position.
    - input_ids_1d: [T]
    - attn_mask_btt: [1, T, T] boolean (True = masked)
    Returns: logits [vocab_size]
    """
    T = input_ids_1d.size(0)
    # Model wants shapes: input_ids [T, B], attention_mask [B, T, T]
    inp = input_ids_1d.view(T, 1)

    # Labels are only used as a *selector* inside the classifier; values don't matter.
    labels = torch.full((T, 1), fill_value=-100, dtype=torch.long, device=input_ids_1d.device)
    labels[-1, 0] = 0  # request logits at the last position

    # get contextual states
    with torch.no_grad():
        contextual = model.get_contextualized(inp, attn_mask_btt)
        # classifier returns logits only for positions where labels != -100
        logits_selected = model.classifier(contextual, labels)  # shape [1, vocab_size]
    return logits_selected[0]


def apply_repetition_penalty_(logits: torch.Tensor,
                              generated_ids: list[int],
                              penalty: float) -> None:
    """
    HuggingFace-style repetition penalty:
      if logit > 0: logit /= penalty
      else:         logit *= penalty
    We apply it once per unique seen token id.
    """
    if penalty <= 1.0 or not generated_ids:
        return
    uniq = set(generated_ids)
    # Vectorized scatter is overkill; simple loop is fine for a small vocab.
    for tid in uniq:
        val = logits[tid].item()
        if val > 0:
            logits[tid] = logits[tid] / penalty
        else:
            logits[tid] = logits[tid] * penalty


def top_k_filter_(logits: torch.Tensor, k: int) -> None:
    if k is None or k <= 0 or k >= logits.numel():
        return
    v, _ = torch.topk(logits, k)
    cutoff = v[-1]
    logits[logits < cutoff] = float("-inf")


def top_p_filter_(logits: torch.Tensor, p: float) -> None:
    if p is None or p >= 1.0:
        return
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    # keep smallest set with cumulative >= p
    mask = cumsum > p
    # shift mask right to always keep at least 1 token
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    # map back to original indices
    logits[sorted_idx[mask]] = float("-inf")

def block_repeated_ngrams_(logits, generated, n):
    if n <= 0 or len(generated) < n-1:
        return
    # collect next-token bans from history
    bans = set()
    for i in range(len(generated) - n + 1):
        prefix = tuple(generated[i:i+n-1])
        nxt    = generated[i+n-1]
        if tuple(generated[-(n-1):]) == prefix:
            bans.add(nxt)
    for tid in bans:
        logits[tid] = float("-inf")

def main():
    ap = argparse.ArgumentParser(
        "GPT-BERT text generation (causal) with repetition penalty",
        description="""
        Generation modes:
        1. Base model: --checkpoint <base.bin>
        2. Base + LoRA: --checkpoint <base.bin> --lora_adapter <adapter.pt> --lora_r R --lora_alpha A --lora_scope attn,mlp
        3. Base + LoRA + trained embeddings: Add --train_embeddings (embeddings in adapter)
        4. Base + separate embeddings: --checkpoint <base.bin> --embeddings_path <embeddings.pt>
        """
    )
    # Model + tokenizer
    ap.add_argument("--config_file",     required=True)
    ap.add_argument("--tokenizer_path",  required=True)
    ap.add_argument("--checkpoint",      required=True,
                    help="Model state_dict (e.g., pretraining checkpoint or merged LoRA).")
    # Optional LoRA adapter (if you saved LoRA-only weights)
    ap.add_argument("--lora_adapter",    default=None,
                    help="Path to LoRA adapter .pt (optional). Requires lora.py and matching r/alpha.")
    ap.add_argument("--lora_r",          type=int, default=8)
    ap.add_argument("--lora_alpha",      type=int, default=16)
    ap.add_argument("--lora_scope",      type=str, default="attn,mlp",
                    help="Comma-list among {attn,mlp}.")
    ap.add_argument("--train_embeddings", action="store_true",
                    help="If set, load trained embeddings from LoRA adapter (use when fine-tuning trained embeddings).")
    ap.add_argument("--embeddings_path", default=None,
                    help="Optional separate embeddings .pt file (for --train_only_embeddings mode).")
    # Decoding
    ap.add_argument("--prompt",          type=str, default="",
                    help="Prompt text. If empty, generation starts from <s> only.")
    ap.add_argument("--max_new_tokens",  type=int, default=100)
    ap.add_argument("--temperature",     type=float, default=1.0)
    ap.add_argument("--top_k",           type=int,   default=0,
                    help="0 disables top-k.")
    ap.add_argument("--top_p",           type=float, default=1.0,
                    help="1.0 disables nucleus.")
    ap.add_argument("--rep_window", type=int, default=64)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=0)
    ap.add_argument("--no_bos",          action="store_true",
                    help="Do not prepend <s> to the prompt.")
    ap.add_argument("--seed",            type=int, default=42)
    ap.add_argument("--device",          type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # Load tokenizer + config
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    cfg = load_config(args.config_file)
    cfg["vocab_size"] = tokenizer.get_vocab_size()
    config = SimpleNamespace(**cfg)

    # Special token ids
    pad_id  = tokenizer.token_to_id("<pad>")
    mask_id = tokenizer.token_to_id("<mask>")
    bos_id  = tokenizer.token_to_id("<s>")
    eos_id  = tokenizer.token_to_id("</s>")

    # Build model
    model = Bert(config).to(device)
    state = torch.load(args.checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("[warn] load_state_dict: missing:", missing, " unexpected:", unexpected, flush=True)

    # Optionally inject LoRA and load adapter weights
    if args.lora_adapter is not None:
        if not HAVE_LORA:
            raise RuntimeError("lora.py not found or import failed, but --lora_adapter was provided.")
        add_lora_gptbert(model, r=args.lora_r, alpha=args.lora_alpha, scope=args.lora_scope)
        # keep base frozen; LoRA params are eval anyway
        # If embeddings were trained during fine-tuning, pass train_embeddings=True
        mark_only_lora_trainable(model, train_layernorm=False, train_embeddings=args.train_embeddings)
        model.to(device)
        lora_sd = torch.load(args.lora_adapter, map_location="cpu")
        # The adapter state_dict contains keys like "...lora_A.weight"/"...lora_B.weight"
        # and may also contain embedding weights if --train_embeddings was used during fine-tuning
        missing, unexpected = model.load_state_dict(lora_sd, strict=False)

        if unexpected:
            print("[warn] unexpected adapter keys:", unexpected)
        missing_lora = [k for k in missing if "lora_" in k]
        if missing_lora:
            print("[warn] missing expected LoRA keys:", missing_lora)

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

    # Encode prompt
    if args.prompt:
        enc = tokenizer.encode(args.prompt)
        prompt_ids = enc.ids
    else:
        prompt_ids = []

    # Prepend BOS unless disabled
    if not args.no_bos:
        if not prompt_ids or prompt_ids[0] != bos_id:
            prompt_ids = [bos_id] + prompt_ids

    # Generation loop
    generated = list(prompt_ids)  # we include prompt in repetition penalty
    for _ in range(args.max_new_tokens):
        ids = torch.tensor(generated, dtype=torch.long, device=device)
        attn = build_causal_mask(ids.size(0), device)

        # Get logits for next token
        logits = logits_last_pos(model, ids, attn)

        # Block obviously bad specials
        if pad_id is not None:
            logits[pad_id] = float("-inf")
        if mask_id is not None:
            logits[mask_id] = float("-inf")
        # (Optionally) avoid generating BOS again
        if bos_id is not None and len(generated) > 0:
            logits[bos_id] = float("-inf")

        # Repetition penalty
        hist = generated[-args.rep_window:] if args.rep_window > 0 else generated
        apply_repetition_penalty_(logits, hist, args.repetition_penalty)

        # Temperature
        if args.temperature != 1.0:
            logits = logits / max(1e-8, args.temperature)

        # Top-k / Top-p filters (apply on logits before softmax)
        if args.top_k and args.top_k > 0:
            top_k_filter_(logits, args.top_k)
        if args.top_p and args.top_p < 1.0:
            top_p_filter_(logits, args.top_p)

        # Sample
        probs = F.softmax(logits, dim=-1)
        block_repeated_ngrams_(logits, generated, args.no_repeat_ngram_size)
        next_id = torch.multinomial(probs, num_samples=1).item()

        generated.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break

    # Decode
    # If you prepended <s>, you might want to drop it for printout.
    to_decode = generated[1:] if (not args.no_bos and generated and generated[0] == bos_id) else generated
    text = tokenizer.decode(to_decode)
    print(text)


if __name__ == "__main__":
    main()
