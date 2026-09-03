# pretraining/lora.py
import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    """
    LoRA wrapper for nn.Linear: y = W_base(x) + (alpha/r) * B(A(x))

    - base: frozen copy of the original nn.Linear
    - lora_A, lora_B: trainable low-rank adapters
    """
    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        assert r > 0, "LoRA rank r must be > 0"

        # register base as a submodule so it stays in the graph
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r

        # LoRA adapters
        self.lora_A = nn.Linear(self.base.in_features, self.r, bias=False)
        self.lora_B = nn.Linear(self.r, self.base.out_features, bias=False)

        # initialize LoRA to do nothing at start
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # IMPORTANT: actually use the LoRA branch in the forward pass
        base_out = self.base(x)
        lora_out = self.lora_B(self.lora_A(x)) * self.scaling
        return base_out + lora_out


def _wrap_attr(parent: nn.Module, attr: str, r: int, alpha: int):
    """Replace parent.{attr} with LoRALinear(parent.{attr})."""
    mod = getattr(parent, attr)
    setattr(parent, attr, LoRALinear(mod, r=r, alpha=alpha))


def add_lora_gptbert(model: nn.Module, r: int = 8, alpha: int = 16, scope=("attn", "mlp")):
    """
    Inject LoRA into your GPT-BERT model:

      - Attention: in_proj_qk, in_proj_vg, out_proj
      - MLP:       mlp[1], mlp[-2]
    """
    if isinstance(scope, str):
        scope = tuple(s.strip() for s in scope.split(","))

    # Attention linears
    if "attn" in scope:
        for attn in model.transformer.attention_layers:
            _wrap_attr(attn, "in_proj_qk", r, alpha)
            _wrap_attr(attn, "in_proj_vg", r, alpha)
            _wrap_attr(attn, "out_proj",   r, alpha)

    # Feed-forward linears
    if "mlp" in scope:
        for ffn in model.transformer.mlp_layers:
            ffn.mlp[1]  = LoRALinear(ffn.mlp[1],  r=r, alpha=alpha)
            ffn.mlp[-2] = LoRALinear(ffn.mlp[-2], r=r, alpha=alpha)


def mark_only_lora_trainable_orig(model: nn.Module, train_layernorm: bool = False):
    """
    Freeze all base params, unfreeze only LoRA A/B (and optionally LayerNorm affine params).
    """
    # 1) freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # 2) unfreeze only lora_A and lora_B
    for m in model.modules():
        if isinstance(m, LoRALinear):
            for p in m.lora_A.parameters():
                p.requires_grad = True
            for p in m.lora_B.parameters():
                p.requires_grad = True
            # extra safety: keep base frozen
            for p in m.base.parameters():
                p.requires_grad = False

    # 3) optionally also train affine LayerNorms
    if train_layernorm:
        for m in model.modules():
            if isinstance(m, nn.LayerNorm) and m.elementwise_affine:
                for p in m.parameters():
                    p.requires_grad = True

# pretraining/lora.py

# Update the signature to accept train_embeddings
def mark_only_lora_trainable(model: nn.Module, train_layernorm: bool = False, train_embeddings: bool = False):
    """
    Freeze all base params, unfreeze only LoRA A/B, optionally LayerNorm, and optionally Embeddings.
    """
    # 1) freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # 2) unfreeze only lora_A and lora_B
    for m in model.modules():
        if isinstance(m, LoRALinear):
            for p in m.lora_A.parameters():
                p.requires_grad = True
            for p in m.lora_B.parameters():
                p.requires_grad = True
            # extra safety: keep base frozen
            for p in m.base.parameters():
                p.requires_grad = False

    # 3) optionally also train affine LayerNorms
    if train_layernorm:
        for m in model.modules():
            if isinstance(m, nn.LayerNorm) and m.elementwise_affine:
                for p in m.parameters():
                    p.requires_grad = True

    # 4) [NEW] optionally train embeddings (Input & Output Head due to weight tying)
    if train_embeddings:
        for m in model.modules():
            if isinstance(m, nn.Embedding):
                for p in m.parameters():
                    p.requires_grad = True

def lora_state_dict(model: nn.Module):
    """
    Return a state_dict containing only LoRA parameters (A/B weights).
    """
    out = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            out[f"{name}.lora_A.weight"] = module.lora_A.weight.detach().cpu()
            out[f"{name}.lora_B.weight"] = module.lora_B.weight.detach().cpu()
    return out


@torch.no_grad()
def merge_lora_(model: nn.Module):
    """
    Fold LoRA into base weights in-place:

        W_base <- W_base + (alpha/r) * (B @ A)

    After this, lora_A/B still exist but are zeroed.
    """
    for m in model.modules():
        if isinstance(m, LoRALinear):
            # B @ A : [out_dim, r] @ [r, in_dim] = [out_dim, in_dim]
            delta = (m.lora_B.weight @ m.lora_A.weight) * (m.alpha / m.r)
            m.base.weight.data.add_(delta)
            # zero out LoRA weights
            nn.init.zeros_(m.lora_A.weight)
            nn.init.zeros_(m.lora_B.weight)
