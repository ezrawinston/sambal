# coding=utf-8
"""
Train GPT-BERT hybrid model from scratch on Lord of the Rings data.
Based on train_v1.py but configured for LOTR-only training.
"""

import os
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

import os.path
import argparse
from tqdm import tqdm
from socket import gethostname
from tokenizers import Tokenizer
from statistics import mean
import json
import math
import copy
import pathlib
from collections import Counter
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from lamb import Lamb
from model_extra import Bert
from utils import cosine_schedule_with_warmup_cooldown, seed_everything
from dataset import MaskedDataset, CausalDataset
from model_logging import ModelLogger

# The shared evaluators live at the repository root (evals/).
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[3]))

import wandb


# --------------------------------------------------------------------------
# Simple "main process" helper (single-process script)
# --------------------------------------------------------------------------
def is_main_process() -> bool:
    return True


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser()

    # LOTR-specific default paths
    parser.add_argument("--train_path", default="../../data/lotr_train.bin", type=str,
                        help="Path to the LOTR training data.")
    parser.add_argument("--valid_path", default="../../data/lotr_dev.bin", type=str,
                        help="Path to the LOTR validation data.")
    parser.add_argument("--name", default="lotr_from_scratch", type=str, help="Name of the run.")
    parser.add_argument("--config_file", default="../configs/tiny.json", type=str, help="The BERT model config")
    parser.add_argument("--tokenizer_path", default="../gpt-bert-babylm-small/tokenizer.json", type=str,
                        help="Path to the tokenizer.")
    parser.add_argument("--output_dir", default="../trained_models", type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--checkpoint_filename", default=None, type=str,
                        help="The checkpoint filename to resume training.")
    parser.add_argument("--optimizer", default="lamb", type=str, help="The optimizer to use.")
    parser.add_argument("--hybrid_numerator", default=15, type=int, help="The numerator of the hybrid ratio.")
    parser.add_argument("--hybrid_denominator", default=16, type=int, help="The denominator of the hybrid ratio.")
    parser.add_argument("--seq_length", default=128, type=int, help="Base sequence length for training.")
    parser.add_argument("--batch_size", default=256, type=int,
                        help="Batch size per optimization step (for the whole mixed batch).")
    parser.add_argument("--learning_rate", default=1.41e-2, type=float, help="The initial learning rate for Adam/LAMB.")
    parser.add_argument("--ema_decay", default=0.999, type=float, help="Exponential moving average decay.")
    parser.add_argument("--validation_steps", default=1, type=int, help="Number of validation steps.")
    parser.add_argument("--log_stats_every", default=100, type=int, help="Log stats every X steps.")
    parser.add_argument("--warmup_proportion", default=0.016, type=float, help="Proportion of training for LR warmup.")
    parser.add_argument("--cooldown_proportion", default=0.016, type=float,
                        help="Proportion of training for LR cooldown.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for initialization.")
    parser.add_argument("--mask_p_start", default=0.3, type=float, help="Initial masking probability.")
    parser.add_argument("--mask_p_end", default=0.15, type=float, help="Final masking probability.")
    parser.add_argument("--mask_random_p", default=0.1, type=float,
                        help="Probability of replacing the masked token with a random token.")
    parser.add_argument("--mask_keep_p", default=0.1, type=float, help="Probability of keeping the masked token.")
    parser.add_argument("--weight_decay", default=0.1, type=float, help="Weight decay.")
    parser.add_argument("--optimizer_eps", default=1e-8, type=float, help="Optimizer epsilon.")
    parser.add_argument("--optimizer_beta1", default=0.9, type=float, help="Optimizer beta1.")
    parser.add_argument("--optimizer_beta2", default=0.98, type=float, help="Optimizer beta2.")
    parser.add_argument("--max_gradient", default=2.0, type=float, help="Max value for gradient clipping.")
    parser.add_argument('--mixed_precision', default=True, action=argparse.BooleanOptionalAction,
                        help="Mixed precision training.")
    parser.add_argument('--n_special_tokens', default=16, type=int, help="Number of special tokens.")
    parser.add_argument('--z_loss_weight', default=1e-4, type=float, help="Weight for the z loss.")
    parser.add_argument('--seq_ramp_60_80', default=False, action=argparse.BooleanOptionalAction,
                        help="Step the sequence-length ramp (with its batch-size compensation) at 60 and 80 percent of training instead of the default 70 and 90. The 60/80 schedule follows the BabyLM 2025 GPT-BERT causal-focus baseline recipe (hf.co/BabyLM-community/babylm-baseline-10m-gpt-bert-causal-focus); the default matches the original GPT-BERT training code.")
    parser.add_argument("--epochs", default=None, type=int,
                        help="Number of epochs (required; training stops after this).")

    # BLiMP evaluation parameters
    parser.add_argument("--blimp_data_path", default="../evaluation/blimp/blimp_really_fast", type=pathlib.Path,
                        help="Path to BLiMP data for evaluation.")
    parser.add_argument("--blimp_test_data_path", default="../evaluation/blimp/data", type=pathlib.Path,
                        help="Path to BLiMP test data for final evaluation.")
    parser.add_argument("--blimp_backend", default="mlm_shift", type=str, help="BLiMP scoring variant",
                        choices=["mlm_shift", "mlm", "causal", "prefix"])
    parser.add_argument("--blimp_batch_size", default=100, type=int, help="Batch size for BLiMP evaluation.")
    parser.add_argument('--enable_blimp', default=True, action=argparse.BooleanOptionalAction,
                        help="Enable BLiMP evaluation during training.")
    parser.add_argument('--blimp_final_only', default=False, action=argparse.BooleanOptionalAction,
                        help="If enabled, skip periodic BLiMP during training and run only final BLiMP test eval.")
    parser.add_argument("--blimp_eval_freq", default=1, type=int, help="Evaluate BLiMP every n epochs.")
    parser.add_argument('--log_param_stats', default=False, action=argparse.BooleanOptionalAction,
                        help="Enable logging of parameter/activation/gradient statistics.")

    # Early stopping and best model saving
    parser.add_argument("--early_stop_patience", default=10, type=int,
                        help="Stop if validation PPL doesn't improve for this many epochs.")
    parser.add_argument("--test_path", default=None, type=str,
                        help="Path to test data for final evaluation.")

    args = parser.parse_args()

    if args.epochs is None:
        raise ValueError("--epochs must be set; training length is controlled by epochs in this script.")

    # Add datetime suffix to run name and output paths
    run_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.name_with_datetime = f"{args.name}_{run_datetime}"
    args.output_path = f"{args.output_dir}/{args.name_with_datetime}.bin"
    args.best_model_path = f"{args.output_dir}/{args.name_with_datetime}_best.bin"
    args.best_ema_model_path = f"{args.output_dir}/{args.name_with_datetime}_best_ema.bin"

    return args


# --------------------------------------------------------------------------
# Helpers for scheduling
# --------------------------------------------------------------------------
def compute_seq_length_for_epoch(args, epoch: int) -> int:
    """Epoch-based seq_length schedule. Changes only at whole epochs."""
    if args.epochs is not None and args.epochs > 0:
        epoch_progress = (epoch + 1) / args.epochs
    else:
        epoch_progress = 0.0

    if args.seq_ramp_60_80:
        # thresholds 0.6 and 0.8
        if epoch_progress >= 0.8:
            return args.seq_length * 4
        elif epoch_progress >= 0.6:
            return args.seq_length * 2
        else:
            return args.seq_length
    else:
        # thresholds 0.7 and 0.9
        if epoch_progress >= 0.9:
            return args.seq_length * 4
        elif epoch_progress >= 0.7:
            return args.seq_length * 2
        else:
            return args.seq_length


def compute_batch_size_for_epoch(args, epoch: int) -> int:
    """Epoch-based batch_size schedule. Reduces when seq_length increases to keep total tokens constant."""
    if args.epochs is not None and args.epochs > 0:
        epoch_progress = (epoch + 1) / args.epochs
    else:
        epoch_progress = 0.0

    if args.seq_ramp_60_80:
        # thresholds 0.6 and 0.8 (matching seq_length schedule)
        if epoch_progress >= 0.8:
            return args.batch_size // 4  # seq_length * 4, so batch_size / 4
        elif epoch_progress >= 0.6:
            return args.batch_size // 2  # seq_length * 2, so batch_size / 2
        else:
            return args.batch_size
    else:
        # thresholds 0.7 and 0.9 (matching seq_length schedule)
        if epoch_progress >= 0.9:
            return args.batch_size // 4  # seq_length * 4, so batch_size / 4
        elif epoch_progress >= 0.7:
            return args.batch_size // 2  # seq_length * 2, so batch_size / 2
        else:
            return args.batch_size


def estimate_total_steps_from_dataloaders(args, tokenizer) -> int:
    """
    Estimate total optimizer steps by building the train dataloaders
    for each epoch and summing their lengths.

    total_steps = sum_e min(len(masked_dataloader_e), len(causal_dataloader_e))
    """
    total_steps = 0

    for epoch in range(args.epochs):
        seq_length = compute_seq_length_for_epoch(args, epoch)
        batch_size = compute_batch_size_for_epoch(args, epoch)

        # Shard rotation (same as in load_datasets)
        mlm_shards = [
            (epoch + r) % args.hybrid_denominator
            for r in range(args.hybrid_numerator)
        ]
        clm_shards = [
            s for s in range(args.hybrid_denominator) if s not in mlm_shards
        ]

        masked_datasets = [
            MaskedDataset(args.train_path, tokenizer, args, seq_length,
                          rank=r, world_size=args.hybrid_denominator)
            for r in mlm_shards
        ]
        train_data_masked = (
            ConcatDataset(masked_datasets) if len(masked_datasets) > 1 else masked_datasets[0]
        )

        causal_datasets = [
            CausalDataset(args.train_path, tokenizer, args, seq_length,
                          rank=r, world_size=args.hybrid_denominator)
            for r in clm_shards
        ]
        train_data_causal = (
            ConcatDataset(causal_datasets) if len(causal_datasets) > 1 else causal_datasets[0]
        )

        # MLM:CLM batch split
        mlm_ratio = args.hybrid_numerator / args.hybrid_denominator
        masked_batch_size = max(1, int(batch_size * mlm_ratio + 0.5))
        causal_batch_size = batch_size - masked_batch_size

        dl_masked = DataLoader(
            train_data_masked,
            shuffle=False,
            batch_size=masked_batch_size,
            num_workers=0,
            drop_last=True,
        )
        dl_causal = DataLoader(
            train_data_causal,
            shuffle=False,
            batch_size=causal_batch_size,
            num_workers=0,
            drop_last=True,
        )

        steps_this_epoch = min(len(dl_masked), len(dl_causal))
        total_steps += steps_this_epoch

    return total_steps


# --------------------------------------------------------------------------
# Setup (single GPU, single process)
# --------------------------------------------------------------------------
def setup_training(args, tokenizer):
    assert torch.cuda.is_available(), "CUDA is required for this script."
    n_gpu = torch.cuda.device_count()
    assert n_gpu >= 1, "At least one GPU is required."
    args.device = torch.device("cuda", 0)
    torch.cuda.set_device(args.device)

    if is_main_process():
        print(f"Using 1 GPU on host {gethostname()}, device {args.device}", flush=True)

    seed_everything(args.seed)
    args.vocab_size = tokenizer.get_vocab_size()

    # Measure total training steps from actual dataloaders (for LR scheduling only)
    args.total_steps = estimate_total_steps_from_dataloaders(args, tokenizer)
    args.max_steps = args.total_steps  # for dataset.py masking schedule

    if is_main_process():
        print(f"Epochs: {args.epochs}")
        print(f"Measured total_steps from dataloaders: {args.total_steps:,}")



# --------------------------------------------------------------------------
# Config + model + optimizer
# --------------------------------------------------------------------------
def load_config(args):
    with open(args.config_file, "r") as f:
        config = json.load(f)
    for k, v in config.items():
        setattr(args, k, v)
    return args


def prepare_model_and_optimizer(args):
    # Apply JSON config to args (this is where your model hyperparams get set)
    args = load_config(args)

    if is_main_process():
        # Build a serializable config dict from args and pass it at init time
        wandb_config = {}
        for k, v in vars(args).items():
            # Make sure everything is JSON-friendly
            if isinstance(v, pathlib.Path):
                wandb_config[k] = str(v)
            elif isinstance(v, torch.device):
                wandb_config[k] = str(v)
            else:
                wandb_config[k] = v

        wandb.init(
            name=args.name_with_datetime,
            project=os.environ.get("WANDB_PROJECT", "sambal"),
            entity=os.environ.get("WANDB_ENTITY"),
            config=wandb_config,
        )

    model = Bert(args).to(args.device)

    if is_main_process():
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.config.update({"n_params": n_params})
        print(model)
        print(f"NUMBER OF PARAMETERS: {n_params:,}\n", flush=True)

    no_decay = ['bias', 'layer_norm']
    decay_params = [(n, p) for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
    no_decay_params = [(n, p) for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
    optimizer_grouped_parameters = [
        {'params': [p for _, p in decay_params], 'weight_decay': args.weight_decay},
        {'params': [p for _, p in no_decay_params], 'weight_decay': 0.0}
    ]

    if is_main_process():
        print("Parameters without weight decay:")
        for n, _ in no_decay_params:
            print(n)
        print()
        print("Parameters with weight decay:")
        for n, _ in decay_params:
            print(n)
        print(flush=True)

    if args.optimizer in ["adam", "adamw"]:
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            betas=(args.optimizer_beta1, args.optimizer_beta2),
            eps=args.optimizer_eps,
        )
    elif args.optimizer == "lamb":
        optimizer = Lamb(
            optimizer_grouped_parameters,
            args.learning_rate,
            betas=(args.optimizer_beta1, args.optimizer_beta2),
            eps=args.optimizer_eps,
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # LR scheduler based on measured total_steps
    warmup_steps = int(args.total_steps * args.warmup_proportion)
    cooldown_steps = int(args.total_steps * args.cooldown_proportion)
    scheduler = cosine_schedule_with_warmup_cooldown(
        optimizer,
        warmup_steps,
        cooldown_steps,
        args.total_steps,
        0.1
    )

    # EMA model (no DDP)
    ema_model: nn.Module = copy.deepcopy(model)
    for param in ema_model.parameters():
        param.requires_grad = False

    step, epoch = 0, 0
    if args.checkpoint_filename is not None:
        state_dict = torch.load(args.checkpoint_filename, map_location="cpu")
        model.load_state_dict(state_dict["model"])
        ema_model.load_state_dict(state_dict["ema_model"])
        optimizer.load_state_dict(state_dict["optimizer"])
        scheduler.load_state_dict(state_dict["scheduler"])
        step = state_dict.get("step", state_dict.get("global_step", 0))
        epoch = state_dict["epoch"]

    return model, ema_model, optimizer, scheduler, step, epoch


# --------------------------------------------------------------------------
# Batching helpers
# --------------------------------------------------------------------------
def get_batch(dataloader, device, step):
    if hasattr(dataloader, "_dataset") and hasattr(dataloader._dataset, "set_global_step"):
        dataloader._dataset.set_global_step(step)
    batch = next(dataloader)
    input_ids, target_ids, attention_mask, mask_p = [t.pin_memory().to(device, non_blocking=True) for t in batch]
    input_ids, target_ids = input_ids.t(), target_ids.t()
    mask_p = mask_p.mean()
    return input_ids, attention_mask, target_ids, mask_p


def get_mixed_batch(masked_iter, causal_iter, device, step):
    """Get a mixed batch combining masked and causal examples."""
    masked_batch = next(masked_iter)
    causal_batch = next(causal_iter)

    m_input, m_target, m_attn, m_mask_p = [t.pin_memory().to(device, non_blocking=True) for t in masked_batch]
    c_input, c_target, c_attn, c_mask_p = [t.pin_memory().to(device, non_blocking=True) for t in causal_batch]

    m_input, m_target = m_input.t(), m_target.t()
    c_input, c_target = c_input.t(), c_target.t()

    input_ids = torch.cat([m_input, c_input], dim=1)
    target_ids = torch.cat([m_target, c_target], dim=1)
    attention_mask = torch.cat([m_attn, c_attn], dim=0)

    mask_p = m_mask_p.mean()

    n_masked = m_input.shape[1]
    n_causal = c_input.shape[1]

    return input_ids, attention_mask, target_ids, mask_p, n_masked, n_causal


# --------------------------------------------------------------------------
# BLiMP wrapping
# --------------------------------------------------------------------------
def evaluate_blimp_at_path(model, tokenizer, args, data_path: pathlib.Path) -> dict:
    """Run BLiMP evaluation at a specific data path."""
    original = getattr(args, "blimp_data_path", None)
    args.blimp_data_path = data_path
    try:
        return evaluate_blimp(model, tokenizer, args)
    finally:
        args.blimp_data_path = original


@torch.no_grad()
def evaluate_blimp(model, tokenizer, args):
    """Run BLiMP evaluation during training (the shared minimal-pair scorer in evals/blimp)."""
    if not args.enable_blimp or not is_main_process():
        return {}

    model.eval()
    from evals.backends.gptbert import GPTBert
    from evals.blimp import blimp_eval

    backend = GPTBert(model, tokenizer, device=args.device, variant=args.blimp_backend,
                      batch_size=args.blimp_batch_size, max_seq_len=getattr(args, "max_position_embeddings", None))
    result = blimp_eval.evaluate_blimp(backend, args.blimp_data_path)
    blimp_results = blimp_eval.training_metrics(result)

    print(f"BLiMP Results:")
    print(f"  Best temperature: {blimp_results['blimp/best_temperature']:.2f} -> Avg UID: {blimp_results['blimp/best_temp_avg_uid_accuracy']:.2f}%")
    print(f"  Temperature 1.0: Avg UID: {blimp_results['blimp/temp_1_avg_uid_accuracy']:.2f}%")

    return blimp_results


# --------------------------------------------------------------------------
# Training epoch (single GPU, no accumulation)
# --------------------------------------------------------------------------
def training_epoch(model, ema_model, train_dataloader_masked, train_dataloader_causal,
                   valid_dataloader, optimizer, scheduler, step, epoch, args, tokenizer):
    """Single-GPU hybrid training epoch."""
    model.train()
    optimizer.zero_grad(set_to_none=True)

    num_steps = min(len(train_dataloader_masked), len(train_dataloader_causal))

    if is_main_process():
        print(
            f"Epoch {epoch}: steps_this_epoch = {num_steps} "
            f"(cumulative_steps {step} -> {step + num_steps} / total_steps {args.total_steps})",
            flush=True
        )

    train_iter_masked = iter(train_dataloader_masked)
    train_iter_causal = iter(train_dataloader_causal)
    train_dataloader = train_dataloader_masked  # For seq_length logging

    total_loss = 0.0
    total_accuracy = 0.0
    total_z_loss = 0.0
    total_mask_p = 0.0
    total_grad_norm = 0.0

    input_ids_, attention_mask_, target_ids_, mask_p_, n_masked_, n_causal_ = get_mixed_batch(
        train_iter_masked, train_iter_causal, args.device, step
    )

    progress_bar = tqdm(
        total=num_steps,
        initial=0,
        desc=f"Train epoch {epoch}",
        disable=not (is_main_process() and sys.stderr.isatty()),
    )

    for step_in_epoch in range(num_steps):
        input_ids, attention_mask, target_ids, mask_p = input_ids_, attention_mask_, target_ids_, mask_p_
        n_masked, n_causal = n_masked_, n_causal_

        with torch.cuda.amp.autocast(args.mixed_precision, dtype=torch.bfloat16):
            with ModelLogger(enable=(args.log_param_stats and (step + step_in_epoch) % 100 == 0), module=model):
                loss, accuracy, z_loss, num_tokens = model(input_ids, attention_mask, target_ids)

        if step_in_epoch < num_steps - 1:
            input_ids_, attention_mask_, target_ids_, mask_p_, n_masked_, n_causal_ = get_mixed_batch(
                train_iter_masked, train_iter_causal, args.device, step
            )

        weight = 1.0

        ((loss + args.z_loss_weight * z_loss) * weight).backward()

        total_loss += float(loss.detach()) * weight
        total_accuracy += float(accuracy) * weight
        total_z_loss += float(z_loss) * weight
        total_mask_p += float(mask_p) * weight

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient)
        total_grad_norm += float(grad_norm) * weight

        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            for param_q, param_k in zip(model.parameters(), ema_model.parameters()):
                param_k.data.mul_(args.ema_decay).add_((1.0 - args.ema_decay) * param_q.detach().data)

            mlm_ratio = args.hybrid_numerator / args.hybrid_denominator
            total_mlm_loss = total_loss
            total_clm_loss = total_loss
            total_mask_p_log = total_mask_p / mlm_ratio if mlm_ratio > 0 else 0.0

        if is_main_process():
            wandb.log(
                {
                    "epoch": epoch,
                    "step": step,
                    "train/loss": total_loss,
                    "train/z_loss": total_z_loss,
                    "train/perplexity": math.exp(total_loss),
                    "train/accuracy": total_accuracy * 100.0,
                    "train/mlm_loss": total_mlm_loss,
                    "train/clm_loss": total_clm_loss,
                    "stats/learning_rate": optimizer.param_groups[0]['lr'],
                    "stats/grad_norm": total_grad_norm,
                    "stats/seq_length": train_dataloader.dataset.seq_length,
                    "stats/batch_size": args.batch_size,
                    "stats/mask_p": total_mask_p_log,
                },
                commit=False
            )

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_accuracy = 0.0
        total_z_loss = 0.0
        total_mask_p = 0.0
        total_grad_norm = 0.0

        step += 1
        if is_main_process():
            wandb.log({"step": step}, commit=True)
            progress_bar.update(1)

    progress_bar.close()
    return step


# --------------------------------------------------------------------------
# Validation + BLiMP
# --------------------------------------------------------------------------
@torch.no_grad()
def validation_epoch(model, valid_dataloader, epoch, args, tokenizer, commit=False, run_blimp=True):
    """
    Run validation over the full validation set with standard CLM perplexity.

    Computes: PPL = exp(-1/N * sum_i log P(w_i | w_1...w_{i-1}))

    Uses CausalDataset (next-token prediction on all tokens) rather than
    ValidationDataset (MLM pseudo-perplexity on ~15% masked tokens).
    This matches finetune_lora.py for fair comparison.

    Returns: perplexity (float)
    """
    model.eval()

    total_loss_times_tokens = 0.0
    total_correct_times_tokens = 0.0
    total_tokens = 0

    # Iterate over the full validation set
    num_batches = len(valid_dataloader)
    valid_iter = iter(valid_dataloader)

    for local_step in tqdm(range(num_batches), desc="Valid iteration", disable=not (is_main_process() and sys.stderr.isatty())):
        batch = next(valid_iter)
        input_ids, target_ids, attention_mask, _ = [t.to(args.device) for t in batch]
        input_ids, target_ids = input_ids.t(), target_ids.t()

        with torch.cuda.amp.autocast(args.mixed_precision, dtype=torch.bfloat16):
            loss, accuracy, _, num_tokens_batch = model(input_ids, attention_mask, target_ids)

        # Token-weighted accumulation for correct perplexity
        total_loss_times_tokens += loss.item() * num_tokens_batch
        total_correct_times_tokens += accuracy.item() * num_tokens_batch
        total_tokens += num_tokens_batch

    # Compute token-weighted metrics
    avg_loss = total_loss_times_tokens / max(1, total_tokens)
    avg_accuracy = total_correct_times_tokens / max(1, total_tokens)
    perplexity = math.exp(avg_loss)

    val_metrics = {
        "epoch": epoch,
        "validation/loss": avg_loss,
        "validation/accuracy": avg_accuracy * 100.0,
        "validation/perplexity": perplexity,
        "validation/total_tokens": total_tokens,
    }

    if is_main_process():
        print(f"Validation: loss={avg_loss:.4f}, ppl={perplexity:.2f}, acc={avg_accuracy*100:.2f}%, tokens={total_tokens}")

    if run_blimp and args.enable_blimp and epoch % args.blimp_eval_freq == 0:
        if is_main_process():
            print(f"\nRunning BLiMP evaluation...")
        blimp_results = evaluate_blimp(model, tokenizer, args)
        if blimp_results:
            val_metrics.update(blimp_results)

    if is_main_process():
        wandb.log(val_metrics, commit=commit)

    return perplexity


# --------------------------------------------------------------------------
# Saving
# --------------------------------------------------------------------------
def save(model, ema_model, optimizer, scheduler, step, epoch, args):
    if is_main_process():
        model_to_save = model  # plain nn.Module
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        torch.save(model_to_save.state_dict(), args.output_path)
        torch.save(ema_model.state_dict(), args.output_path.replace(".bin", "_ema.bin"))
        torch.save(
            {
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "global_step": step,  # for backwards compatibility
                "epoch": epoch + 1,
            },
            args.output_path.replace(".bin", "_state_dict.bin")
        )


def save_best(model, ema_model, epoch, val_ppl, args):
    """Save best model checkpoint based on validation perplexity."""
    if is_main_process():
        os.makedirs(os.path.dirname(args.best_model_path), exist_ok=True)
        torch.save(model.state_dict(), args.best_model_path)
        torch.save(ema_model.state_dict(), args.best_ema_model_path)
        # Save metadata
        import json as json_module
        meta_path = args.best_model_path.replace(".bin", "_meta.json")
        with open(meta_path, "w") as f:
            json_module.dump({
                "epoch": epoch,
                "val_ppl": val_ppl,
            }, f, indent=2)
        print(f"[best] Saved best model (val_ppl={val_ppl:.2f}) at epoch {epoch}")


# --------------------------------------------------------------------------
# Dataset loading (single-GPU hybrid)
# --------------------------------------------------------------------------
def load_datasets(args, tokenizer, epoch, step,
                  train_dataloader_masked, train_dataloader_causal, valid_dataloader):
    """Load datasets with support for mixed-batch single-GPU training."""
    train_seed = args.seed + epoch

    seq_length = compute_seq_length_for_epoch(args, epoch)
    batch_size = compute_batch_size_for_epoch(args, epoch)

    def needs_reload(dl):
        if dl is None:
            return True
        ds = dl.dataset
        if hasattr(ds, 'datasets'):
            ds = ds.datasets[0]
        return getattr(ds, "seq_length", None) != seq_length

    need_reload = needs_reload(train_dataloader_masked) or needs_reload(train_dataloader_causal)

    if need_reload:
        mlm_shards = [
            (epoch + r) % args.hybrid_denominator
            for r in range(args.hybrid_numerator)
        ]
        clm_shards = [
            s for s in range(args.hybrid_denominator) if s not in mlm_shards
        ]

        masked_datasets = [
            MaskedDataset(args.train_path, tokenizer, args, seq_length,
                          rank=r, world_size=args.hybrid_denominator)
            for r in mlm_shards
        ]
        train_data_masked = ConcatDataset(masked_datasets) if len(masked_datasets) > 1 else masked_datasets[0]

        causal_datasets = [
            CausalDataset(args.train_path, tokenizer, args, seq_length,
                          rank=r, world_size=args.hybrid_denominator)
            for r in clm_shards
        ]
        train_data_causal = ConcatDataset(causal_datasets) if len(causal_datasets) > 1 else causal_datasets[0]

        if is_main_process():
            print(f"Epoch {epoch}: MLM shards {mlm_shards}, CLM shards {clm_shards}")
            print(f"Epoch {epoch}: seq_length = {seq_length}, batch_size = {batch_size}")
    else:
        train_data_masked = train_dataloader_masked.dataset if train_dataloader_masked else None
        train_data_causal = train_dataloader_causal.dataset if train_dataloader_causal else None

    mlm_ratio = args.hybrid_numerator / args.hybrid_denominator
    masked_batch_size = max(1, int(batch_size * mlm_ratio + 0.5))
    causal_batch_size = batch_size - masked_batch_size

    if is_main_process():
        print(f"Mixed batch: {masked_batch_size} masked + {causal_batch_size} causal = "
              f"{masked_batch_size + causal_batch_size} total")
        print(f"Total tokens per batch: {(masked_batch_size + causal_batch_size) * seq_length:,}")

    train_dataloader_masked = DataLoader(
        train_data_masked,
        shuffle=True,
        batch_size=masked_batch_size,
        num_workers=0,
        generator=torch.Generator().manual_seed(train_seed),
        drop_last=True,
        pin_memory=True,
    )
    train_dataloader_causal = DataLoader(
        train_data_causal,
        shuffle=True,
        batch_size=causal_batch_size,
        num_workers=0,
        generator=torch.Generator().manual_seed(train_seed + 1),
        drop_last=True,
        pin_memory=True,
    )

    if valid_dataloader is None:
        # Use CausalDataset for validation to compute standard CLM perplexity
        # (matches finetune_lora.py for fair comparison)
        valid_data = CausalDataset(
            args.valid_path, tokenizer, args, seq_length,
            rank=None, world_size=None
        )
        # Use larger batch size for validation (faster) and don't drop last batch
        # to ensure we evaluate on the full validation set
        valid_batch_size = max(args.batch_size, 128)
        valid_dataloader = DataLoader(
            valid_data,
            shuffle=False,
            batch_size=valid_batch_size,
            num_workers=0,
            generator=torch.Generator().manual_seed(42),
            drop_last=False,  # Evaluate on full validation set
            pin_memory=True,
        )

    return train_dataloader_masked, train_dataloader_causal, valid_dataloader


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def preflight_eval_data(args) -> None:
    """Verify the BLiMP data directories this run will consume, before training.

    The final BLiMP test evaluation otherwise fails with FileNotFoundError only
    after the whole training run has completed.
    """
    problems = []

    def need_jsonl(path, what):
        p = pathlib.Path(path)
        if not p.is_dir() or not any(p.glob("*.jsonl")):
            problems.append(
                f"{what}: no *.jsonl files under {p} — fetch full BLiMP once into "
                "evals/blimp/data (python evals/blimp/fetch_data.py) "
                "and point --blimp_test_data_path at it"
            )

    if args.enable_blimp:
        if not args.blimp_final_only:
            need_jsonl(args.blimp_data_path, "training-time BLiMP eval")
        need_jsonl(args.blimp_test_data_path, "final BLiMP test eval")

    if problems:
        raise RuntimeError(
            "Eval data missing (checked before training so a long run cannot fail "
            "only at final-eval time):\n  - " + "\n  - ".join(problems)
        )


if __name__ == "__main__":
    args = parse_arguments()
    preflight_eval_data(args)

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    setup_training(args, tokenizer)
    model, ema_model, optimizer, scheduler, step, start_epoch = prepare_model_and_optimizer(args)

    train_dataloader_masked, train_dataloader_causal, valid_dataloader = None, None, None
    test_dataloader = None

    # Load test data if provided
    if args.test_path is not None:
        seq_length = compute_seq_length_for_epoch(args, 0)
        test_data = CausalDataset(args.test_path, tokenizer, args, seq_length, rank=None, world_size=None)
        test_dataloader = DataLoader(
            test_data, shuffle=False, batch_size=max(args.batch_size, 128),
            num_workers=0, drop_last=False, pin_memory=True
        )

    # Initial validation before any training
    train_dataloader_masked, train_dataloader_causal, valid_dataloader = load_datasets(
        args, tokenizer, 0, step, train_dataloader_masked, train_dataloader_causal, valid_dataloader
    )
    initial_ppl = validation_epoch(model, valid_dataloader, 0, args, tokenizer, commit=False)

    # Initialize best model tracking
    best_val_ppl = initial_ppl
    best_epoch = -1
    epochs_since_improve = 0
    save_best(model, ema_model, -1, best_val_ppl, args)

    for epoch in range(start_epoch, args.epochs):
        train_dataloader_masked, train_dataloader_causal, valid_dataloader = load_datasets(
            args, tokenizer, epoch, step, train_dataloader_masked, train_dataloader_causal, valid_dataloader
        )
        step = training_epoch(
            model, ema_model, train_dataloader_masked, train_dataloader_causal, valid_dataloader,
            optimizer, scheduler, step, epoch, args, tokenizer
        )

        # Validate at the end of each epoch
        should_run_blimp = (not args.blimp_final_only) and (epoch % args.blimp_eval_freq == 0)
        val_ppl = validation_epoch(model, valid_dataloader, epoch, args, tokenizer,
                                   commit=(epoch == args.epochs - 1), run_blimp=should_run_blimp)

        # Track best model
        if val_ppl < best_val_ppl - 1e-4:
            best_val_ppl = val_ppl
            best_epoch = epoch
            epochs_since_improve = 0
            save_best(model, ema_model, epoch, val_ppl, args)
        else:
            epochs_since_improve += 1
            if is_main_process():
                print(f"[early_stop] No improvement for {epochs_since_improve} epoch(s) (best={best_val_ppl:.2f} at epoch {best_epoch})")

        # Early stopping check
        if args.early_stop_patience > 0 and epochs_since_improve >= args.early_stop_patience:
            if is_main_process():
                print(f"[early_stop] Stopping: no improvement for {args.early_stop_patience} epochs.")
            break

    # Reload best model and evaluate on test set
    if is_main_process():
        print(f"\n{'='*60}")
        print(f"Training complete. Reloading best model (epoch {best_epoch}, val_ppl={best_val_ppl:.2f})")
        print(f"{'='*60}\n")

    best_state = torch.load(args.best_model_path, map_location="cpu")
    model.load_state_dict(best_state)
    best_ema_state = torch.load(args.best_ema_model_path, map_location="cpu")
    ema_model.load_state_dict(best_ema_state)

    # Evaluate on test set
    if test_dataloader is not None:
        if is_main_process():
            print("\n[test] Evaluating best model on test set...")
        test_ppl = validation_epoch(model, test_dataloader, best_epoch, args, tokenizer,
                                    commit=False, run_blimp=False)
        if is_main_process():
            print(f"[test] Test perplexity: {test_ppl:.2f}")
            wandb.log({"test/perplexity": test_ppl})

    # Evaluate on BLiMP test set
    if args.enable_blimp:
        if is_main_process():
            print("\n[blimp] Evaluating best model on BLiMP test set...")
        blimp_test_results = evaluate_blimp_at_path(model, tokenizer, args, args.blimp_test_data_path)
        if blimp_test_results and is_main_process():
            for key, value in blimp_test_results.items():
                print(f"  [blimp-test] {key}: {value}")
                wandb.log({f"blimp_test/{key.replace('blimp/', '')}": value})

    wandb.finish()
