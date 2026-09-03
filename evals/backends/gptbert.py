"""Backend for the released checkpoints (the ``lm/gpt-bert`` hybrid encoder).

Scoring variants (``variant=``), each the scoring the corresponding released
evaluator applied:

* ``mlm_shift`` (default; the ICML 2026 paper's numbers): token *i* is read from the
  position before it while it is replaced by ``<mask>``;
* ``mlm``: the classic pseudo-log-likelihood -- token *i* is read at its own,
  masked, position, with ``</s>`` closing the sequence;
* ``causal``: one left-to-right pass, token *i* read from the position before it;
* ``prefix``: as ``causal``, but the first ``prefix_lens[i]`` tokens (and ``<s>``)
  attend to each other bidirectionally.

``--backend-arg`` keys: see ``GPTBert.argument_help()``.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from .base import LanguageModel, Layers

REPO = pathlib.Path(__file__).resolve().parents[2]
GPTBERT_DIR = REPO / "lm/gpt-bert"
DEFAULT_TOKENIZER = GPTBERT_DIR / "gpt-bert-babylm-small/tokenizer.json"
DEFAULT_CONFIG = GPTBERT_DIR / "configs/small.json"
VARIANTS = ("mlm_shift", "mlm", "causal", "prefix")


def _model_class():
    """The training-side ``Bert`` class, loaded from its file under a private module name."""
    name = "evals_backends_gptbert_model_extra"
    if name in sys.modules:
        return sys.modules[name].Bert
    spec = importlib.util.spec_from_file_location(name, GPTBERT_DIR / "pretraining/model_extra.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.Bert


class GPTBert(LanguageModel):
    def __init__(self, model, tokenizer, device: Optional[torch.device] = None, variant: str = "mlm_shift",
                 batch_size: int = 64, name: Optional[str] = None, max_seq_len: Optional[int] = None):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.variant = variant
        self.batch_size = int(batch_size)
        self.name = name or "gptbert"
        self.max_seq_len = max_seq_len
        self.n_layers = len(model.transformer.attention_layers)
        self.bos, self.eos = tokenizer.token_to_id("<s>"), tokenizer.token_to_id("</s>")
        self.pad, self.mask = tokenizer.token_to_id("<pad>"), tokenizer.token_to_id("<mask>")
        if None in (self.bos, self.eos, self.pad, self.mask):
            raise ValueError("tokenizer must define <s>, </s>, <pad> and <mask>")

    # --- construction -----------------------------------------------------------
    @classmethod
    def argument_help(cls) -> str:
        return ("checkpoint=PATH  state dict (.bin/.pt) [required]\n"
                f"config=PATH      model config json [default {DEFAULT_CONFIG.relative_to(REPO)}]\n"
                f"tokenizer=PATH   tokenizer json [default {DEFAULT_TOKENIZER.relative_to(REPO)}]\n"
                f"variant=NAME     scoring variant, one of {', '.join(VARIANTS)} [default mlm_shift]\n"
                "batch_size=N     rows per forward pass [default 64]\n"
                "name=LABEL       label used in reports [default: checkpoint stem]")

    @classmethod
    def from_args(cls, args: Dict[str, str], device: Optional[torch.device] = None) -> "GPTBert":
        from tokenizers import Tokenizer

        args = dict(args)
        checkpoint = args.pop("checkpoint", None)
        if checkpoint is None:
            raise ValueError("gptbert needs --backend-arg checkpoint=PATH")
        config_path = pathlib.Path(args.pop("config", DEFAULT_CONFIG))
        tokenizer_path = pathlib.Path(args.pop("tokenizer", DEFAULT_TOKENIZER))
        variant = args.pop("variant", "mlm_shift")
        batch_size = int(args.pop("batch_size", 64))
        name = args.pop("name", pathlib.Path(checkpoint).stem)
        if args:
            raise ValueError(f"unknown gptbert options {sorted(args)}; accepted:\n{cls.argument_help()}")
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = SimpleNamespace(**json.loads(config_path.read_text()))
        model = _model_class()(config)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        model.to(device)
        return cls(model, Tokenizer.from_file(str(tokenizer_path)), device=device, variant=variant,
                   batch_size=batch_size, name=name, max_seq_len=getattr(config, "max_position_embeddings", None))

    # --- text <-> ids -------------------------------------------------------------
    def encode(self, text: str) -> List[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=True)

    def encode_with_offsets(self, text: str):
        enc = self.tokenizer.encode(text, add_special_tokens=False)
        return list(enc.ids), [tuple(o) for o in enc.offsets]

    # --- forward ------------------------------------------------------------------
    @torch.no_grad()
    def _logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """``input_ids [B, T]``, ``attention_mask [B, T]`` (True = padding) or ``[B, T, T]`` (True = blocked)
        -> logits ``[B, T, V]`` from the label-free path (contextualise, then the output head).

        The model adds one leading axis to the mask itself, so a padding mask
        goes in as ``[B, 1, T]`` and a full pattern as ``[B, T, T]``."""
        mask = attention_mask.unsqueeze(1) if attention_mask.ndim == 2 else attention_mask
        hidden = self.model.get_contextualized(input_ids.t().contiguous(), mask.contiguous())
        return self.model.classifier.nonlinearity(hidden).transpose(0, 1)

    @staticmethod
    def _logprobs_at(logits: torch.Tensor, labels: torch.Tensor, temps: torch.Tensor) -> torch.Tensor:
        """``logits [N, V]``, ``labels [N]`` -> ``[n_temps, N]`` log-probs of the labels."""
        scaled = logits.unsqueeze(0) / temps.view(-1, 1, 1)
        log_p = F.log_softmax(scaled, dim=-1)
        return log_p.gather(-1, labels.view(1, -1, 1).expand(temps.numel(), -1, 1)).squeeze(-1)

    # --- scoring --------------------------------------------------------------------
    def token_logprobs(self, seqs, temperatures=(1.0,), prefix_lens=None, positions=None) -> List[torch.Tensor]:
        temps = torch.as_tensor(list(temperatures), dtype=torch.float32, device=self.device).clamp(min=1e-6)
        seqs = [list(s) for s in seqs]
        if any(len(s) == 0 for s in seqs):
            raise ValueError("cannot score an empty sequence")
        wanted = self._positions(seqs, positions)
        if self.variant in ("mlm_shift", "mlm"):
            per_token = self._masked_logprobs(seqs, temps, wanted)
        else:
            per_token = self._causal_logprobs(seqs, temps, prefix_lens)
            if positions is not None:
                for t, keep in zip(per_token, wanted):
                    mask = torch.zeros(t.size(1), dtype=torch.bool, device=t.device)
                    mask[keep] = True
                    t[:, ~mask] = 0.0
        return [t.detach().cpu() for t in per_token]

    @staticmethod
    def _positions(seqs, positions):
        """Per sequence the sorted token indices to score (all when no hint was given)."""
        if positions is None:
            return [list(range(len(s))) for s in seqs]
        if len(positions) != len(seqs):
            raise ValueError("positions must have one entry per sequence")
        out = []
        for s, p in zip(seqs, positions):
            p = sorted(set(int(i) for i in p))
            if p and not (0 <= p[0] and p[-1] < len(s)):
                raise ValueError(f"positions {p} out of range for a sequence of {len(s)} tokens")
            out.append(p)
        return out

    def _masked_logprobs(self, seqs, temps, wanted):
        """One masked row per scored token; all rows of a call are padded to one width."""
        shift = self.variant == "mlm_shift"
        tail = [] if shift else [self.eos]
        width = max(len(s) for s in seqs) + 1 + len(tail)
        rows, masks, gather, labels = [], [], [], []
        for s, keep in zip(seqs, wanted):
            base = torch.tensor([self.bos] + s + tail + [self.pad] * (width - 1 - len(s) - len(tail)), device=self.device)
            n = len(keep)
            if n == 0:
                continue
            keep_t = torch.tensor(keep, device=self.device)
            block = base.repeat(n, 1)
            block[torch.arange(n), keep_t + 1] = self.mask
            pad_mask = torch.zeros((n, width), dtype=torch.bool, device=self.device)
            pad_mask[:, 1 + len(s) + len(tail):] = True
            rows.append(block)
            masks.append(pad_mask)
            gather.append(keep_t if shift else keep_t + 1)
            labels.append(torch.tensor([s[i] for i in keep], device=self.device))
        result = [torch.zeros((temps.numel(), len(s)), dtype=torch.float32, device=self.device) for s in seqs]
        if not rows:
            return result
        rows, masks = torch.cat(rows), torch.cat(masks)
        gather, labels = torch.cat(gather), torch.cat(labels)
        out = torch.empty((temps.numel(), rows.size(0)), dtype=torch.float32, device=self.device)
        for start in range(0, rows.size(0), self.batch_size):
            sl = slice(start, start + self.batch_size)
            logits = self._logits(rows[sl], masks[sl])                       # [b, T, V]
            picked = logits[torch.arange(logits.size(0), device=self.device), gather[sl]]  # [b, V]
            out[:, sl] = self._logprobs_at(picked, labels[sl], temps)
        offset = 0
        for i, keep in enumerate(wanted):
            if keep:
                result[i][:, torch.tensor(keep, device=self.device)] = out[:, offset:offset + len(keep)]
                offset += len(keep)
        return result

    def _causal_logprobs(self, seqs, temps, prefix_lens):
        """One row per sequence with a causal (or prefix-bidirectional) attention pattern."""
        prefix_lens = list(prefix_lens) if prefix_lens is not None else [0] * len(seqs)
        if len(prefix_lens) != len(seqs):
            raise ValueError("prefix_lens must have one entry per sequence")
        width = max(len(s) for s in seqs) + 1
        result = []
        for start in range(0, len(seqs), self.batch_size):
            chunk, chunk_prefix = seqs[start:start + self.batch_size], prefix_lens[start:start + self.batch_size]
            ids = torch.full((len(chunk), width), self.pad, dtype=torch.long, device=self.device)
            attn = torch.ones((len(chunk), width, width), dtype=torch.bool, device=self.device).triu(diagonal=1)
            for b, (s, p) in enumerate(zip(chunk, chunk_prefix)):
                ids[b, 0], ids[b, 1:1 + len(s)] = self.bos, torch.tensor(s, device=self.device)
                if self.variant == "prefix" and p > 0:
                    attn[b, :p + 1, :p + 1] = False
            logits = self._logits(ids, attn)                                   # [b, T, V]
            for b, s in enumerate(chunk):
                picked = logits[b, :len(s)]                                    # position i predicts token i
                result.append(self._logprobs_at(picked, torch.tensor(s, device=self.device), temps))
        return result

    # --- representations -------------------------------------------------------------
    @torch.no_grad()
    def hidden_states(self, seqs, layers: Layers = "final", add_special_tokens: bool = True) -> List[torch.Tensor]:
        seqs = [list(s) for s in seqs]
        selected = self._layer_indices(layers)
        lead = [self.bos] if add_special_tokens else []
        width = max(len(s) for s in seqs) + len(lead)
        outputs: List[torch.Tensor] = []
        for start in range(0, len(seqs), self.batch_size):
            chunk = seqs[start:start + self.batch_size]
            ids = torch.full((len(chunk), width), self.pad, dtype=torch.long, device=self.device)
            pad_mask = torch.ones((len(chunk), width), dtype=torch.bool, device=self.device)
            for b, s in enumerate(chunk):
                ids[b, :len(lead) + len(s)] = torch.tensor(lead + s, device=self.device)
                pad_mask[b, :len(lead) + len(s)] = False
            captured: List[torch.Tensor] = []
            hook = self.model.transformer.dwa_modules.register_forward_hook(lambda m, i, o: captured.append(o))
            try:
                final = self.model.get_contextualized(ids.t().contiguous(), pad_mask.unsqueeze(1).contiguous())
            finally:
                hook.remove()
            per_layer = captured[1::2]  # the state after each block's feed-forward sub-layer
            assert len(per_layer) == self.n_layers and torch.equal(per_layer[-1], final)
            stack = torch.stack([per_layer[i] for i in selected], dim=0)    # [n_sel, T, b, D]
            for b, s in enumerate(chunk):
                outputs.append(stack[:, len(lead):len(lead) + len(s), b, :].detach().cpu().clone())
        return outputs

    def _layer_indices(self, layers: Layers) -> List[int]:
        if isinstance(layers, str):
            if layers != "final":
                raise ValueError("layers must be 'final' or a sequence of indices")
            return [self.n_layers - 1]
        out = []
        for i in layers:
            j = i + self.n_layers if i < 0 else i
            if not 0 <= j < self.n_layers:
                raise ValueError(f"layer {i} out of range for {self.n_layers} layers")
            out.append(j)
        return out
