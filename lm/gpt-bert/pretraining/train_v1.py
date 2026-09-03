# coding=utf-8

import os
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

import os.path
import argparse
import re
import ast
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from lamb import Lamb
from model_extra import Bert
from utils import cosine_schedule_with_warmup_cooldown, seed_everything
from dataset import MaskedDataset, CausalDataset, ValidationDataset
from model_logging import ModelLogger

# The shared evaluators live at the repository root (evals/).
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[3]))

import wandb


from evals.syntaxgym.syntaxgym_eval import ALL_HU2020_SYNTAXGYM, BORDERLINE_LEXICOSYNTACTIC, CORE_IN_SCOPE_SUITES


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

    parser.add_argument("--train_path", default="../data/train_10M_tokenized.bin", type=str,
                        help="Path to the training data.")
    parser.add_argument("--valid_path", default="../data/dev_10M_tokenized.bin", type=str,
                        help="Path to the validation data.")
    parser.add_argument("--name", default="small_15-16_babylm_10M", type=str, help="Name of the run.")
    parser.add_argument("--config_file", default="../configs/small.json", type=str, help="The BERT model config")
    parser.add_argument("--tokenizer_path", default="../tokenizers/tokenizer_10M.json", type=str,
                        help="Path to the tokenizer.")
    parser.add_argument("--output_dir", default="../model_checkpoints", type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--checkpoint_filename", default=None, type=str,
                        help="The checkpoint filename to resume training.")
    parser.add_argument("--no_save", action="store_true",
                        help="Skip saving model checkpoints.")
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
    parser.add_argument("--data_fraction", default=1.0, type=float,
                        help="Fraction of training data to use (0.0-1.0). Useful for ablations.")

    # BLiMP evaluation parameters
    parser.add_argument("--blimp_data_path", default="../evaluation/blimp/blimp_really_fast", type=pathlib.Path,
                        help="Path to BLiMP data for evaluation.")
    parser.add_argument("--blimp_backend", default="mlm_shift", type=str, help="BLiMP scoring variant",
                        choices=["mlm_shift", "mlm", "causal", "prefix"])
    parser.add_argument("--blimp_batch_size", default=100, type=int, help="Batch size for BLiMP evaluation.")
    parser.add_argument('--enable_blimp', default=True, action=argparse.BooleanOptionalAction,
                        help="Enable BLiMP evaluation during training.")
    parser.add_argument("--blimp_eval_freq", default=1, type=int, help="Evaluate BLiMP every n epochs.")

    # SyntaxGym evaluation parameters
    parser.add_argument('--enable_syntaxgym', default=True, action=argparse.BooleanOptionalAction,
                        help="Enable SyntaxGym evaluation during training.")
    parser.add_argument("--syntaxgym_data_path", default="../evaluation/syntaxgym", type=pathlib.Path,
                        help="Path to SyntaxGym suite JSON files.")
    parser.add_argument("--syntaxgym_backend", default="mlm_shift", type=str, help="SyntaxGym evaluation backend",
                        choices=["mlm_shift"])
    parser.add_argument("--syntaxgym_batch_size", default=64, type=int, help="Batch size for SyntaxGym evaluation.")
    parser.add_argument("--syntaxgym_eval_freq", default=1, type=int, help="Evaluate SyntaxGym every n epochs.")
    parser.add_argument("--syntaxgym_suite_set", default="all", type=str,
                        choices=["all", "core_only", "borderline_only"],
                        help="Which SyntaxGym suites to evaluate.")

    # Test set evaluation at end of training
    parser.add_argument("--blimp_test_data_path", default=None, type=pathlib.Path,
                        help="Path to BLiMP test data for final evaluation.")
    parser.add_argument("--syntaxgym_test_data_path", default=None, type=pathlib.Path,
                        help="Path to SyntaxGym test suites for final evaluation.")
    parser.add_argument('--enable_final_eval', default=False, action=argparse.BooleanOptionalAction,
                        help="Run BLiMP/SyntaxGym on test sets at end of training using EMA model.")
    parser.add_argument('--eval_only_at_end', default=False, action=argparse.BooleanOptionalAction,
                        help="Skip BLiMP/SyntaxGym during training, only run at end on validation paths using EMA model.")

    parser.add_argument('--log_param_stats', default=False, action=argparse.BooleanOptionalAction,
                        help="Enable logging of parameter/activation/gradient statistics.")

    args = parser.parse_args()

    if args.epochs is None:
        raise ValueError("--epochs must be set; training length is controlled by epochs in this script.")

    # Add datetime suffix to run name and output paths
    run_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.name_with_datetime = f"{args.name}_{run_datetime}"
    args.output_path = f"{args.output_dir}/{args.name_with_datetime}.bin"

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
# SyntaxGym evaluation
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate_syntaxgym(model, tokenizer, args):
    """Run SyntaxGym evaluation during training (the shared scorer in evals/syntaxgym)."""
    if not args.enable_syntaxgym or not is_main_process():
        return {}

    model.eval()
    from evals.backends.gptbert import GPTBert
    from evals.syntaxgym import syntaxgym_eval

    backend = GPTBert(model, tokenizer, device=args.device, batch_size=args.syntaxgym_batch_size,
                      max_seq_len=getattr(args, "max_position_embeddings", None))
    result = syntaxgym_eval.evaluate_syntaxgym(backend, args.syntaxgym_data_path, suites=args.syntaxgym_suite_set)
    results = syntaxgym_eval.training_metrics(result)

    print(f"SyntaxGym Results:")
    print(f"  Best temperature: {results['syntaxgym/best_temperature']:.2f} -> Avg: {results['syntaxgym/best_temp_avg_accuracy']:.2f}%")
    print(f"  Temperature 1.0: Avg: {results['syntaxgym/temp_1_avg_accuracy']:.2f}%")

    return results


def evaluate_blimp_at_path(model, tokenizer, args, data_path):
    """Evaluate BLiMP at a specific data path."""
    original = getattr(args, "blimp_data_path", None)
    args.blimp_data_path = data_path
    try:
        return evaluate_blimp(model, tokenizer, args)
    finally:
        args.blimp_data_path = original


def evaluate_syntaxgym_at_path(model, tokenizer, args, data_path):
    """Evaluate SyntaxGym at a specific data path."""
    original = getattr(args, "syntaxgym_data_path", None)
    original_suite_set = getattr(args, "syntaxgym_suite_set", None)
    args.syntaxgym_data_path = data_path
    args.syntaxgym_suite_set = "core_only"
    try:
        return evaluate_syntaxgym(model, tokenizer, args)
    finally:
        args.syntaxgym_data_path = original
        args.syntaxgym_suite_set = original_suite_set


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
def validation_epoch(model, valid_dataloader, epoch, args, tokenizer, commit=False):
    model.eval()

    losses, accuracies = [], []
    valid_dataloader = iter(valid_dataloader)
    input_ids, attention_mask, target_ids, _ = get_batch(valid_dataloader, args.device, 0)

    for local_step in tqdm(range(args.validation_steps), desc="Valid iteration", disable=not (is_main_process() and sys.stderr.isatty())):

        with torch.cuda.amp.autocast(args.mixed_precision, dtype=torch.bfloat16):
            loss, accuracy, _, num_tokens = model(input_ids, attention_mask, target_ids)

        if local_step < args.validation_steps - 1:
            input_ids, attention_mask, target_ids, _ = get_batch(valid_dataloader, args.device, 0)

        loss_val = float(loss)
        acc_val = float(accuracy)

        losses.append(loss_val)
        accuracies.append(acc_val)

    val_metrics = {
        "epoch": epoch,
        "validation/loss": mean(losses),
        "validation/accuracy": mean(accuracies) * 100.0,
        "validation/perplexity": math.exp(mean(losses))
    }

    if args.enable_blimp and epoch % args.blimp_eval_freq == 0 and not args.eval_only_at_end:
        if is_main_process():
            print(f"\nRunning BLiMP evaluation...")
        blimp_results = evaluate_blimp(model, tokenizer, args)
        if blimp_results:
            val_metrics.update(blimp_results)

    if args.enable_syntaxgym and epoch % args.syntaxgym_eval_freq == 0 and not args.eval_only_at_end:
        if is_main_process():
            print(f"\nRunning SyntaxGym evaluation...")
        syntaxgym_results = evaluate_syntaxgym(model, tokenizer, args)
        if syntaxgym_results:
            val_metrics.update(syntaxgym_results)

    if is_main_process():
        wandb.log(val_metrics, commit=commit)


# --------------------------------------------------------------------------
# Saving
# --------------------------------------------------------------------------
def save(model, ema_model, optimizer, scheduler, step, epoch, args):
    if is_main_process():
        model_to_save = model  # plain nn.Module
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        torch.save(model_to_save.state_dict(), args.output_path)
        torch.save(ema_model.state_dict(), args.output_path.replace(".bin", "_ema.bin"))
        # torch.save(
        #     {
        #         "model": model.state_dict(),
        #         "ema_model": ema_model.state_dict(),
        #         "optimizer": optimizer.state_dict(),
        #         "scheduler": scheduler.state_dict(),
        #         "step": step,
        #         "global_step": step,  # for backwards compatibility
        #         "epoch": epoch + 1,
        #     },
        #     args.output_path.replace(".bin", "_state_dict.bin")
        # )


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
        valid_data = ValidationDataset(args.valid_path, tokenizer, args)
        valid_dataloader = DataLoader(
            valid_data,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=0,
            generator=torch.Generator().manual_seed(42),
            drop_last=True,
            pin_memory=True,
        )

    return train_dataloader_masked, train_dataloader_causal, valid_dataloader


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def _select_suite_names(args) -> List[str]:
    """The SyntaxGym suite set an evaluation with these args will score."""
    if args.syntaxgym_suite_set == "core_only":
        return CORE_IN_SCOPE_SUITES
    if args.syntaxgym_suite_set == "borderline_only":
        return BORDERLINE_LEXICOSYNTACTIC
    return ALL_HU2020_SYNTAXGYM


def preflight_eval_data(args) -> None:
    """Verify every eval-data directory this run will consume, before training.

    A missing benchmark directory otherwise surfaces only when the eval runs —
    for the final test evaluations that is after the whole training run has
    completed.
    """
    problems = []

    def need_jsonl(path, what):
        if path is None:
            return
        p = pathlib.Path(path)
        if not p.is_dir() or not any(p.glob("*.jsonl")):
            problems.append(
                f"{what}: no *.jsonl files under {p} — fetch full BLiMP once into "
                "evals/blimp/data: python evals/blimp/fetch_data.py"
            )

    def need_suites(path, what):
        if path is None:
            return
        p = pathlib.Path(path)
        missing = [s for s in _select_suite_names(args) if not (p / f"{s}.json").exists()]
        if missing:
            problems.append(
                f"{what}: SyntaxGym suite files missing under {p} ({', '.join(missing)}) "
                "— fetch the pinned test suites once with: python evals/syntaxgym/fetch_data.py"
            )

    if args.enable_blimp:
        need_jsonl(args.blimp_data_path, "training-time BLiMP eval")
    if args.enable_syntaxgym:
        need_suites(args.syntaxgym_data_path, "training-time SyntaxGym eval")
    if args.enable_final_eval:
        if args.enable_blimp:
            need_jsonl(args.blimp_test_data_path, "final BLiMP test eval")
        if args.enable_syntaxgym:
            need_suites(args.syntaxgym_test_data_path, "final SyntaxGym test eval")

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

    # Initial validation before any training
    train_dataloader_masked, train_dataloader_causal, valid_dataloader = load_datasets(
        args, tokenizer, 0, step, train_dataloader_masked, train_dataloader_causal, valid_dataloader
    )
    validation_epoch(model, valid_dataloader, 0, args, tokenizer, commit=False)

    for epoch in range(start_epoch, args.epochs):
        train_dataloader_masked, train_dataloader_causal, valid_dataloader = load_datasets(
            args, tokenizer, epoch, step, train_dataloader_masked, train_dataloader_causal, valid_dataloader
        )
        step = training_epoch(
            model, ema_model, train_dataloader_masked, train_dataloader_causal, valid_dataloader,
            optimizer, scheduler, step, epoch, args, tokenizer
        )

        # Save + validate at the end of each epoch
        if not args.no_save:
            save(model, ema_model, optimizer, scheduler, step, epoch, args)
        validation_epoch(model, valid_dataloader, epoch, args, tokenizer, commit=(epoch == args.epochs - 1))

    # Final evaluation on test sets using EMA model
    if args.enable_final_eval:
        print("\n" + "=" * 60)
        print("Running final evaluation on test sets using EMA model...")
        print("=" * 60 + "\n")

        if args.syntaxgym_test_data_path and args.enable_syntaxgym:
            print(f"Running SyntaxGym evaluation on TEST suites at {args.syntaxgym_test_data_path}...")
            syntaxgym_test_results = evaluate_syntaxgym_at_path(
                ema_model, tokenizer, args, args.syntaxgym_test_data_path
            )
            if syntaxgym_test_results:
                for key, value in syntaxgym_test_results.items():
                    print(f"  [syntaxgym-test] {key}: {value}")
                wandb.log({f"final_test/{k}": v for k, v in syntaxgym_test_results.items()})
            print()

        if args.blimp_test_data_path and args.enable_blimp:
            print(f"Running BLiMP evaluation on TEST data at {args.blimp_test_data_path}...")
            blimp_test_results = evaluate_blimp_at_path(
                ema_model, tokenizer, args, args.blimp_test_data_path
            )
            if blimp_test_results:
                for key, value in blimp_test_results.items():
                    print(f"  [blimp-test] {key}: {value}")
                wandb.log({f"final_test/{k}": v for k, v in blimp_test_results.items()})
            print()

    # End-of-training evaluation on validation paths (when skipped during training)
    if args.eval_only_at_end:
        print("\n" + "=" * 60)
        print("Running end-of-training evaluation on validation sets using EMA model...")
        print("=" * 60 + "\n")

        if args.enable_syntaxgym:
            print(f"Running SyntaxGym evaluation at {args.syntaxgym_data_path}...")
            syntaxgym_val_results = evaluate_syntaxgym(ema_model, tokenizer, args)
            if syntaxgym_val_results:
                for key, value in syntaxgym_val_results.items():
                    print(f"  [syntaxgym-final] {key}: {value}")
                wandb.log({f"final/{k}": v for k, v in syntaxgym_val_results.items()})
            print()

        if args.enable_blimp:
            print(f"Running BLiMP evaluation at {args.blimp_data_path}...")
            blimp_val_results = evaluate_blimp(ema_model, tokenizer, args)
            if blimp_val_results:
                for key, value in blimp_val_results.items():
                    print(f"  [blimp-final] {key}: {value}")
                wandb.log({f"final/{k}": v for k, v in blimp_val_results.items()})
            print()

    wandb.finish()