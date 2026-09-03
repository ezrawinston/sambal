# pretraining/finetune_lora.py
import argparse
import json
import math
import os
import pathlib
import random
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from model_extra import Bert
from lora import add_lora_gptbert, mark_only_lora_trainable, lora_state_dict, merge_lora_
from dataset import MaskedDataset, CausalDataset, ValidationDataset, RandomIndex
from train_v1 import evaluate_blimp, evaluate_syntaxgym

def parse_args():
    ap = argparse.ArgumentParser("LoRA finetuning for GPT-BERT (your repo)")

    # Model & data
    ap.add_argument("--config_file",     type=str, required=True)
    ap.add_argument("--tokenizer_path",  type=str, required=True)
    ap.add_argument("--checkpoint",      type=str, required=True, help="Pretrained model .bin (saved by train_10m.py::save)")
    ap.add_argument("--train_path",      type=str, required=True, help="Tokenized .bin (same format your pretraining uses)")
    ap.add_argument("--mix_train_path",  type=str, default=None,
                    help="If set, mixes --train_path with an equal number of randomly-selected segments from this dataset.")
    ap.add_argument("--mix_seed",        type=int, default=None,
                    help="Seed for selecting the mixed-in segments (defaults to --seed).")
    ap.add_argument("--mix_with_replacement", action="store_true",
                    help="Allow sampling with replacement if --mix_train_path has fewer segments than --train_path.")
    ap.add_argument("--dev_path",        type=str, required=True)
    ap.add_argument("--test_path",       type=str, default=None)
    ap.add_argument("--dataset_type",    type=str, choices=["masked","causal"], default="causal")

    # LoRA
    ap.add_argument("--use_lora",        action="store_true", default=True)
    ap.add_argument("--lora_r",          type=int, default=8)
    ap.add_argument("--lora_alpha",      type=int, default=16)
    ap.add_argument("--lora_scope",      type=str, default="attn,mlp", help="comma list over {attn,mlp}")
    ap.add_argument("--train_layernorm", action="store_true")

    # Training
    ap.add_argument("--seq_length",      type=int, default=256)
    ap.add_argument("--batch_size",      type=int, default=32)
    ap.add_argument("--epochs",          type=int, default=3)
    ap.add_argument("--lr",              type=float, default=1e-4)
    ap.add_argument("--weight_decay",    type=float, default=0.0)
    ap.add_argument("--max_grad_norm",   type=float, default=1.0)
    ap.add_argument("--mixed_precision", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seed",            type=int, default=42)

    # Masking params expected by your datasets
    ap.add_argument("--n_special_tokens", type=int, default=16)
    ap.add_argument("--mask_p_start",     type=float, default=0.3)
    ap.add_argument("--mask_p_end",       type=float, default=0.15)
    ap.add_argument("--mask_random_p",    type=float, default=0.1)
    ap.add_argument("--mask_keep_p",      type=float, default=0.1)
    ap.add_argument("--max_steps",        type=int, default=10_000, help="drives mask schedule in MaskedDataset")

    # BLiMP evaluation options
    ap.add_argument("--blimp_data_path", type=pathlib.Path,
                    default=pathlib.Path("../evaluation/blimp/blimp_really_fast"),
                    help="Path to BLiMP jsonl data.")
    ap.add_argument("--blimp_val_data_path", type=pathlib.Path, default=None,
                    help="Optional override path for BLiMP validation data.")
    ap.add_argument("--blimp_test_data_path", type=pathlib.Path, default=None,
                    help="Optional override path for BLiMP test data.")
    ap.add_argument("--blimp_batch_size", type=int, default=100, help="Batch size for BLiMP evaluation.")
    ap.add_argument("--blimp_backend", type=str, default="mlm_shift",
                    choices=["mlm", "causal", "mlm_shift", "fused"],
                    help="BLiMP evaluation backend.")
    ap.add_argument("--blimp_eval_freq", type=int, default=1, help="Evaluate BLiMP every n epochs.")
    ap.add_argument('--enable_blimp', action=argparse.BooleanOptionalAction, default=False,
                    help="Run BLiMP validation during fine-tuning.")

    # SyntaxGym evaluation options
    ap.add_argument("--syntaxgym_data_path", type=pathlib.Path,
                    default=pathlib.Path("../evaluation/syntaxgym"),
                    help="Path to SyntaxGym JSON suites.")
    ap.add_argument("--syntaxgym_val_data_path", type=pathlib.Path, default=None,
                    help="Optional override path for SyntaxGym validation suites.")
    ap.add_argument("--syntaxgym_test_data_path", type=pathlib.Path, default=None,
                    help="Optional override path for SyntaxGym test suites.")
    ap.add_argument("--syntaxgym_batch_size", type=int, default=64, help="Batch size for SyntaxGym evaluation.")
    ap.add_argument("--syntaxgym_backend", type=str, default="mlm_shift",
                    choices=["mlm_shift"],
                    help="SyntaxGym evaluation backend (currently mlm_shift only).")
    ap.add_argument("--syntaxgym_eval_freq", type=int, default=1,
                    help="Evaluate SyntaxGym every n epochs.")
    ap.add_argument("--syntaxgym_suite_set", type=str, default="core_only",
                    choices=["all", "core_only", "borderline_only"],
                    help="Which SyntaxGym suites to evaluate.")
    ap.add_argument('--enable_syntaxgym', action=argparse.BooleanOptionalAction, default=False,
                    help="Run SyntaxGym validation during fine-tuning.")
    ap.add_argument("--syntaxgym_early_stop_patience", type=int, default=10,
                    help="Stop if SyntaxGym validation doesn't improve for this many eval epochs.")

    # Validation PPL early stopping options
    ap.add_argument('--enable_val_ppl_early_stop', action=argparse.BooleanOptionalAction, default=False,
                    help="Enable early stopping based on validation perplexity.")
    ap.add_argument("--val_ppl_early_stop_patience", type=int, default=10,
                    help="Stop if validation PPL doesn't improve for this many epochs.")

    # Output
    ap.add_argument("--out_dir",          type=str, required=True)
    ap.add_argument("--save_merged",      action="store_true", help="also save full merged model (LoRA folded in)")
    ap.add_argument("--train_embeddings", action="store_true", help="Unfreeze embeddings (and tied output head)")
    ap.add_argument("--train_only_embeddings", action="store_true",
                    help="Train ONLY embeddings (no LoRA). Overrides --use_lora.")
    return ap.parse_args()


def load_config(cfg_path):
    with open(cfg_path, "r") as f:
        return json.load(f)


def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_trainable_param_state(model: nn.Module) -> dict:
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def load_trainable_param_state(model: nn.Module, trainable_state: dict) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in trainable_state:
                param.copy_(trainable_state[name].to(param.device))


def evaluate_blimp_at_path(model: nn.Module, tokenizer: Tokenizer, args, data_path: pathlib.Path) -> dict:
    original = getattr(args, "blimp_data_path", None)
    args.blimp_data_path = data_path
    try:
        return evaluate_blimp(model, tokenizer, args)
    finally:
        args.blimp_data_path = original


def evaluate_syntaxgym_at_path(model: nn.Module, tokenizer: Tokenizer, args, data_path: pathlib.Path) -> dict:
    original = getattr(args, "syntaxgym_data_path", None)
    original_suite_set = getattr(args, "syntaxgym_suite_set", None)
    args.syntaxgym_data_path = data_path
    args.syntaxgym_suite_set = "core_only"
    try:
        return evaluate_syntaxgym(model, tokenizer, args)
    finally:
        args.syntaxgym_data_path = original
        args.syntaxgym_suite_set = original_suite_set


def _subset_train_dataset_inplace(ds, keep_indices):
    if keep_indices is None:
        return ds
    ds.segments = [ds.segments[i] for i in keep_indices]
    if hasattr(ds, "counts"):
        ds.counts = [ds.counts[i] for i in keep_indices]
    if hasattr(ds, "mask_counts"):
        ds.mask_counts = [ds.mask_counts[i] for i in keep_indices]
    if hasattr(ds, "random_index"):
        ds.random_index = RandomIndex(len(ds.segments))
    return ds


def _concat_train_datasets_inplace(dst, src):
    dst.segments.extend(src.segments)
    if hasattr(dst, "counts") and hasattr(src, "counts"):
        dst.counts.extend(src.counts)
    if hasattr(dst, "mask_counts") and hasattr(src, "mask_counts"):
        dst.mask_counts.extend(src.mask_counts)
    if hasattr(dst, "random_index"):
        dst.random_index = RandomIndex(len(dst.segments))
    return dst


@torch.no_grad()
def ppl_on_loader(model: nn.Module, loader: DataLoader, device: torch.device, mp_bf16: bool):
    model.eval()
    total_loss_times_tokens = 0.0
    total_tokens = 0
    it = iter(loader)

    for _ in range(len(loader)):
        input_ids, target_ids, attention_mask, _ = next(it)
        # [B, T] -> [T, B] for your model
        input_ids  = input_ids.t().to(device, non_blocking=True)
        target_ids = target_ids.t().to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=mp_bf16 and device.type=="cuda", dtype=torch.bfloat16):
            loss, _, _, num_tokens = model(input_ids, attention_mask, target_ids)

        total_loss_times_tokens += (loss.item() * num_tokens)
        total_tokens += int(num_tokens)

    return math.exp(total_loss_times_tokens / max(1, total_tokens))


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    syntaxgym_val_path = args.syntaxgym_val_data_path or args.syntaxgym_data_path
    syntaxgym_test_path = args.syntaxgym_test_data_path or syntaxgym_val_path

    blimp_val_path = args.blimp_val_data_path or args.blimp_data_path
    blimp_test_path = args.blimp_test_data_path or blimp_val_path

    # Fail fast, before fine-tuning, on eval data the run will need later —
    # the post-fine-tuning test evals otherwise crash only after the full run.
    problems = []
    if args.enable_blimp:
        for what, p in (("BLiMP validation", blimp_val_path), ("BLiMP test", blimp_test_path)):
            pp = pathlib.Path(p)
            if not pp.is_dir() or not any(pp.glob("*.jsonl")):
                problems.append(
                    f"{what} data: no *.jsonl files under {pp} — fetch full BLiMP once "
                    "into evals/blimp/data: python evals/blimp/fetch_data.py"
                )
    if args.enable_syntaxgym:
        for what, p in (("SyntaxGym validation", syntaxgym_val_path), ("SyntaxGym test", syntaxgym_test_path)):
            pp = pathlib.Path(p)
            if not pp.is_dir() or not any(pp.glob("*.json")):
                problems.append(
                    f"{what} data: no suite *.json files under {pp} — fetch the pinned "
                    "test suites once with: python evals/syntaxgym/fetch_data.py"
                )
    if problems:
        raise RuntimeError(
            "Eval data missing (checked before fine-tuning so the run cannot fail "
            "only at test-eval time):\n  - " + "\n  - ".join(problems)
        )

    # --- tokenizer & config ---
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    cfg = load_config(args.config_file)
    cfg["vocab_size"] = tokenizer.get_vocab_size()  # match pretraining
    config = SimpleNamespace(**cfg)

    # --- datasets ---
    # Build a tiny args-like namespace to satisfy dataset fields
    dargs = SimpleNamespace(
        n_special_tokens=args.n_special_tokens,
        mask_random_p=args.mask_random_p,
        mask_keep_p=args.mask_keep_p,
        vocab_size=config.vocab_size,
        mask_p_start=args.mask_p_start,
        mask_p_end=args.mask_p_end,
        max_steps=args.max_steps,
        seed=args.seed,
        seq_length=args.seq_length,
    )

    if args.dataset_type == "masked":
        TrainDS = MaskedDataset
    else:
        TrainDS = CausalDataset

    train_ds = TrainDS(args.train_path, tokenizer, dargs, args.seq_length, rank=None, world_size=None)
    if args.mix_train_path:
        other_ds = TrainDS(args.mix_train_path, tokenizer, dargs, args.seq_length, rank=None, world_size=None)
        n_lotr = len(train_ds)
        n_other = len(other_ds)
        if n_other <= 0:
            raise ValueError(f"--mix_train_path has no usable segments: {args.mix_train_path}")

        g = torch.Generator()
        g.manual_seed(int(args.mix_seed if args.mix_seed is not None else args.seed))

        if n_other >= n_lotr:
            keep = torch.randperm(n_other, generator=g)[:n_lotr].tolist()
        else:
            if not args.mix_with_replacement:
                raise ValueError(
                    f"--mix_train_path has fewer segments ({n_other}) than --train_path ({n_lotr}). "
                    f"Pass --mix_with_replacement to allow sampling with replacement."
                )
            keep = torch.randint(low=0, high=n_other, size=(n_lotr,), generator=g).tolist()

        _subset_train_dataset_inplace(other_ds, keep)
        _concat_train_datasets_inplace(train_ds, other_ds)
        print(
            f"[data] mixed train segments: lotr={n_lotr} + other={len(other_ds)} (from {n_other}) => total={len(train_ds)}",
            flush=True,
        )

    if args.dataset_type == "masked":
        dev_ds = ValidationDataset(args.dev_path, tokenizer, dargs)
        test_ds = ValidationDataset(args.test_path, tokenizer, dargs) if args.test_path else None
    else:
        dev_ds = CausalDataset(args.dev_path, tokenizer, dargs, args.seq_length, rank=None, world_size=None)
        test_ds = CausalDataset(args.test_path, tokenizer, dargs, args.seq_length, rank=None,
                                world_size=None) if args.test_path else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True) if test_ds else None

    # --- model ---
    model = Bert(config).to(device)

    # load the *model* checkpoint your pretraining save() writes (not the full state_dict bundle)
    state = torch.load(args.checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("[warn] load_state_dict: missing:", missing, " unexpected:", unexpected, flush=True)

    # inject LoRA & freeze base weights, or train only embeddings
    if args.train_only_embeddings:
        # Train only embeddings (no LoRA)
        print("Training only embeddings (no LoRA)...", flush=True)
        for p in model.parameters():
            p.requires_grad = False
        for m in model.modules():
            if isinstance(m, nn.Embedding):
                for p in m.parameters():
                    p.requires_grad = True

        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        print("Trainable params:", len(trainable), "total elements:",
              sum(p.numel() for _, p in trainable))
        for n, _ in trainable[:10]:
            print("  ", n)

    elif args.use_lora:
        add_lora_gptbert(model, r=args.lora_r, alpha=args.lora_alpha, scope=args.lora_scope)

        # [NEW] Update this call:
        mark_only_lora_trainable(
            model,
            train_layernorm=args.train_layernorm,
            train_embeddings=args.train_embeddings
        )

        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        print("Trainable params:", len(trainable), "total elements:",
              sum(p.numel() for _, p in trainable))
        for n, _ in trainable[:10]:
            print("  ", n)
        model.to(device)

    # optimizer (only trainable params)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision and device.type=="cuda")

    # Initialize validation PPL tracking variables
    best_val_ppl = float("inf")
    best_ppl_epoch = None
    best_ppl_trainable_state_path = os.path.join(args.out_dir, "best_ppl_trainable_params.pt")
    best_ppl_meta_path = os.path.join(args.out_dir, "best_val_ppl.json")
    ppl_epochs_since_improve = 0

    # Collected across the run and written to <out_dir>/metrics.json at the
    # end, using the same field names/bases as a records/lora/lora_runs.json
    # entry (pre/post ppl at print precision; blimp unrounded).
    run_metrics = {}

    if test_ds is not None:
        test_ppl = ppl_on_loader(model, test_loader, device, args.mixed_precision)
        print(f"[test] perplexity: {test_ppl:.2f}", flush=True)
        run_metrics["pre_ft_test_ppl"] = round(test_ppl, 2)

    # --- pre-training evaluation (epoch -1 baseline) ---
    print("\n[epoch -1] Evaluating model before fine-tuning...", flush=True)
    dev_ppl = ppl_on_loader(model, dev_loader, device, args.mixed_precision)
    print(f"[dev] epoch -1 perplexity: {dev_ppl:.2f}", flush=True)

    # Initialize best PPL with baseline
    if args.enable_val_ppl_early_stop:
        best_val_ppl = dev_ppl
        best_ppl_epoch = -1
        torch.save(get_trainable_param_state(model), best_ppl_trainable_state_path)
        pathlib.Path(best_ppl_meta_path).write_text(json.dumps({
            "metric": "dev_perplexity",
            "best_epoch": best_ppl_epoch,
            "best_score": best_val_ppl,
        }, indent=2))
        print(f"[val_ppl] Baseline dev_ppl={best_val_ppl:.2f} (saved best snapshot)", flush=True)

    syntaxgym_results = {}

    if args.enable_blimp:
        print("\nRunning BLiMP evaluation (pre-finetune)...", flush=True)
        blimp_results = evaluate_blimp_at_path(model, tokenizer, args, blimp_val_path)
        if blimp_results:
            for key, value in blimp_results.items():
                print(f"  [blimp] {key}: {value}")
        print("", flush=True)

        print("\nRunning BLiMP evaluation on TEST data (pre-finetune)...", flush=True)
        blimp_test_results_pre = evaluate_blimp_at_path(model, tokenizer, args, blimp_test_path)
        if blimp_test_results_pre:
            for key, value in blimp_test_results_pre.items():
                print(f"  [blimp-test] {key}: {value}")
            run_metrics["pre_ft_blimp_test_best_temp_avg"] = blimp_test_results_pre.get(
                "blimp/best_temp_avg_uid_accuracy")
        print("", flush=True)

    if args.enable_syntaxgym:
        print("\nRunning SyntaxGym evaluation (pre-finetune)...", flush=True)
        syntaxgym_results = evaluate_syntaxgym_at_path(model, tokenizer, args, syntaxgym_val_path)
        if syntaxgym_results:
            for key, value in syntaxgym_results.items():
                print(f"  [syntaxgym] {key}: {value}")
        print("", flush=True)

        print("\nRunning SyntaxGym evaluation on TEST suites (pre-finetune)...", flush=True)
        syntaxgym_test_results_pre = evaluate_syntaxgym_at_path(model, tokenizer, args, syntaxgym_test_path)
        if syntaxgym_test_results_pre:
            for key, value in syntaxgym_test_results_pre.items():
                print(f"  [syntaxgym-test] {key}: {value}")
        print("", flush=True)

    # --- training ---
    print("Starting LoRA fine‑tune. Train iters per epoch:", len(train_loader))
    global_step = 0
    best_syntaxgym = float("-inf")
    best_epoch = None
    best_trainable_state_path = os.path.join(args.out_dir, "best_trainable_params.pt")
    best_meta_path = os.path.join(args.out_dir, "best_syntaxgym_val.json")
    epochs_since_improve = 0

    if args.enable_syntaxgym:
        baseline_score = syntaxgym_results.get("syntaxgym/best_temp_avg_accuracy")
        if baseline_score is not None:
            best_syntaxgym = float(baseline_score)
            best_epoch = -1
            torch.save(get_trainable_param_state(model), best_trainable_state_path)
            pathlib.Path(best_meta_path).write_text(json.dumps({
                "metric": "syntaxgym/best_temp_avg_accuracy",
                "best_epoch": best_epoch,
                "best_score": best_syntaxgym,
                "syntaxgym_val_path": str(syntaxgym_val_path),
            }, indent=2))
            print(f"[syntaxgym] Baseline best_temp_avg_accuracy={best_syntaxgym:.4f} (saved best snapshot)", flush=True)

    for epoch in range(args.epochs):
        model.train()
        it = iter(train_loader)
        for step in range(len(train_loader)):
            input_ids, target_ids, attention_mask, _ = next(it)

            input_ids  = input_ids.t().to(device, non_blocking=True)   # [T, B]
            target_ids = target_ids.t().to(device, non_blocking=True)  # [T, B]
            attention_mask = attention_mask.to(device, non_blocking=True)  # [B, T, T]

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.mixed_precision and device.type=="cuda", dtype=torch.bfloat16):
                loss, _, _, _ = model(input_ids, attention_mask, target_ids)

            if args.mixed_precision and device.type=="cuda":
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                if step == 0:
                    nonzero = 0
                    for n, p in model.named_parameters():
                        if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0:
                            nonzero += 1
                    print("Nonzero grad params:", nonzero)


                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()

                if step == 0:
                    nonzero = 0
                    for n, p in model.named_parameters():
                        if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0:
                            nonzero += 1
                    print("Nonzero grad params:", nonzero)

                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()

            global_step += 1
            if global_step % 50 == 0:
                print(f"epoch {epoch} step {global_step} loss {loss.item():.4f}", flush=True)

        # Dev perplexity each epoch
        dev_ppl = ppl_on_loader(model, dev_loader, device, args.mixed_precision)
        print(f"[dev] epoch {epoch} perplexity: {dev_ppl:.2f}", flush=True)

        # Track best validation PPL
        if args.enable_val_ppl_early_stop:
            # Lower PPL is better, so check if current < best (with small tolerance)
            if dev_ppl < best_val_ppl - 1e-4:
                best_val_ppl = dev_ppl
                best_ppl_epoch = epoch
                ppl_epochs_since_improve = 0
                torch.save(get_trainable_param_state(model), best_ppl_trainable_state_path)
                pathlib.Path(best_ppl_meta_path).write_text(json.dumps({
                    "metric": "dev_perplexity",
                    "best_epoch": best_ppl_epoch,
                    "best_score": best_val_ppl,
                }, indent=2))
                print(f"[val_ppl] New best={best_val_ppl:.2f} at epoch {best_ppl_epoch} (saved snapshot)", flush=True)
            else:
                ppl_epochs_since_improve += 1
                print(f"[val_ppl] No improvement for {ppl_epochs_since_improve} epoch(s)", flush=True)

            if args.val_ppl_early_stop_patience > 0 and ppl_epochs_since_improve >= args.val_ppl_early_stop_patience:
                print(f"[val_ppl] Early stopping: no improvement for {args.val_ppl_early_stop_patience} epochs.", flush=True)
                break

        if args.enable_blimp and args.blimp_eval_freq > 0 and epoch % args.blimp_eval_freq == 0:
            print("\nRunning BLiMP evaluation...", flush=True)
            blimp_results = evaluate_blimp_at_path(model, tokenizer, args, blimp_val_path)
            if blimp_results:
                for key, value in blimp_results.items():
                    print(f"  [blimp] {key}: {value}")
            print("", flush=True)

        if args.enable_syntaxgym and args.syntaxgym_eval_freq > 0 and epoch % args.syntaxgym_eval_freq == 0:
            print("\nRunning SyntaxGym evaluation...", flush=True)
            syntaxgym_results = evaluate_syntaxgym_at_path(model, tokenizer, args, syntaxgym_val_path)
            if syntaxgym_results:
                for key, value in syntaxgym_results.items():
                    print(f"  [syntaxgym] {key}: {value}")
            print("", flush=True)

            current = syntaxgym_results.get("syntaxgym/best_temp_avg_accuracy") if syntaxgym_results else None
            if current is not None:
                current = float(current)
                if current > best_syntaxgym + 1e-9:
                    best_syntaxgym = current
                    best_epoch = epoch
                    epochs_since_improve = 0
                    torch.save(get_trainable_param_state(model), best_trainable_state_path)
                    pathlib.Path(best_meta_path).write_text(json.dumps({
                        "metric": "syntaxgym/best_temp_avg_accuracy",
                        "best_epoch": best_epoch,
                        "best_score": best_syntaxgym,
                        "syntaxgym_val_path": str(syntaxgym_val_path),
                    }, indent=2))
                    print(f"[syntaxgym] New best={best_syntaxgym:.4f} at epoch {best_epoch} (saved snapshot)", flush=True)
                else:
                    epochs_since_improve += 1
                    print(f"[syntaxgym] No improvement for {epochs_since_improve} eval epoch(s)", flush=True)

                if args.syntaxgym_early_stop_patience > 0 and epochs_since_improve >= args.syntaxgym_early_stop_patience:
                    print(f"[syntaxgym] Early stopping: no improvement for {args.syntaxgym_early_stop_patience} eval epochs.", flush=True)
                    break

    # Final-state metrics, captured BEFORE any best-checkpoint restoration:
    # this is the post_ft_test_ppl basis recorded in
    # records/lora/lora_runs.json (the released adapters and the BLiMP
    # numbers are the best-validation restore below).
    if test_ds is not None:
        final_state_test_ppl = ppl_on_loader(model, test_loader, device, args.mixed_precision)
        print(f"[test] perplexity (final state): {final_state_test_ppl:.2f}", flush=True)
        run_metrics["post_ft_test_ppl"] = round(final_state_test_ppl, 2)
    run_metrics["best_dev_epoch"] = best_ppl_epoch
    run_metrics["best_dev_ppl"] = None if best_val_ppl == float("inf") else best_val_ppl

    # Restore best checkpoint based on early stopping mode
    # Priority: SyntaxGym > Val PPL > no restoration
    if args.enable_syntaxgym and os.path.exists(best_trainable_state_path):
        best_state = torch.load(best_trainable_state_path, map_location="cpu")
        load_trainable_param_state(model, best_state)
        print(f"\n[syntaxgym] Restored best params (val best={best_syntaxgym:.4f}, epoch={best_epoch})", flush=True)
        print("\nRunning SyntaxGym evaluation on TEST suites...", flush=True)
        syntaxgym_test_results = evaluate_syntaxgym_at_path(model, tokenizer, args, syntaxgym_test_path)
        if syntaxgym_test_results:
            for key, value in syntaxgym_test_results.items():
                print(f"  [syntaxgym-test] {key}: {value}")
        print("", flush=True)

        if args.enable_blimp:
            print("\nRunning BLiMP evaluation on TEST data (with best model)...", flush=True)
            blimp_test_results = evaluate_blimp_at_path(model, tokenizer, args, blimp_test_path)
            if blimp_test_results:
                for key, value in blimp_test_results.items():
                    print(f"  [blimp-test] {key}: {value}")
                run_metrics["post_ft_blimp_test_best_temp_avg"] = blimp_test_results.get(
                    "blimp/best_temp_avg_uid_accuracy")
            print("", flush=True)
    elif args.enable_val_ppl_early_stop and os.path.exists(best_ppl_trainable_state_path):
        # Restore best by validation PPL
        best_state = torch.load(best_ppl_trainable_state_path, map_location="cpu")
        load_trainable_param_state(model, best_state)
        print(f"\n[val_ppl] Restored best params (val best_ppl={best_val_ppl:.2f}, epoch={best_ppl_epoch})", flush=True)

        if test_ds is not None:
            test_ppl_best = ppl_on_loader(model, test_loader, device, args.mixed_precision)
            print(f"[test] perplexity (with best val_ppl model): {test_ppl_best:.2f}", flush=True)
            run_metrics["post_ft_test_ppl_best_val_restore"] = round(test_ppl_best, 2)

        if args.enable_blimp:
            print("\nRunning BLiMP evaluation on TEST data (with best val_ppl model)...", flush=True)
            blimp_test_results = evaluate_blimp_at_path(model, tokenizer, args, blimp_test_path)
            if blimp_test_results:
                for key, value in blimp_test_results.items():
                    print(f"  [blimp-test] {key}: {value}")
                run_metrics["post_ft_blimp_test_best_temp_avg"] = blimp_test_results.get(
                    "blimp/best_temp_avg_uid_accuracy")
            print("", flush=True)
    elif args.enable_blimp:
        # If only BLiMP is enabled (no SyntaxGym), evaluate on test with final model
        print("\nRunning BLiMP evaluation on TEST data (final model)...", flush=True)
        blimp_test_results = evaluate_blimp_at_path(model, tokenizer, args, blimp_test_path)
        if blimp_test_results:
            for key, value in blimp_test_results.items():
                print(f"  [blimp-test] {key}: {value}")
            run_metrics["post_ft_blimp_test_best_temp_avg"] = blimp_test_results.get(
                "blimp/best_temp_avg_uid_accuracy")
        print("", flush=True)

    # --- save ---
    if args.train_only_embeddings:
        # Save embeddings only
        embedding_sd = {k: v for k, v in model.state_dict().items() if "embedding" in k.lower()}
        torch.save(embedding_sd, os.path.join(args.out_dir, "embeddings.pt"))
        print(f"Saved embeddings to {os.path.join(args.out_dir, 'embeddings.pt')}", flush=True)

        # Also save full model for convenience
        torch.save(model.state_dict(), os.path.join(args.out_dir, "finetuned_model.pt"))
        print(f"Saved full model to {os.path.join(args.out_dir, 'finetuned_model.pt')}", flush=True)
    else:
        # 1) LoRA-only adapter
        lora_sd = lora_state_dict(model)
        torch.save(lora_sd, os.path.join(args.out_dir, "lora_adapter.pt"))
        print(f"Saved LoRA adapter to {os.path.join(args.out_dir, 'lora_adapter.pt')}", flush=True)

        # 2) (optional) merged full model for convenience
        if args.save_merged:
            merge_lora_(model)
            torch.save(model.state_dict(), os.path.join(args.out_dir, "merged_model.pt"))
            print(f"Saved merged full model to {os.path.join(args.out_dir, 'merged_model.pt')}", flush=True)

    # Test PPL if provided
    if test_ds is not None:
        test_ppl = ppl_on_loader(model, test_loader, device, args.mixed_precision)
        print(f"[test] perplexity: {test_ppl:.2f}", flush=True)

    # Machine-readable run metrics, in the shape the table generators and
    # records/lora/lora_runs.json use.
    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(run_metrics, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"Saved run metrics to {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
