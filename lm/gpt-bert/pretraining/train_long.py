# coding=utf-8

import os
import os.path
import argparse
import sys
import re
import ast
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
from tqdm import tqdm
from itertools import count
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
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from lamb import Lamb
from model_extra import Bert
from utils import cosine_schedule_with_warmup_cooldown, is_main_process, get_rank, seed_everything, get_world_size
from dataset import MaskedDataset, CausalDataset, ValidationDataset
from model_logging import ModelLogger

# The shared evaluators live at the repository root (evals/).
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3]))


from evals.syntaxgym.syntaxgym_eval import ALL_HU2020_SYNTAXGYM, BORDERLINE_LEXICOSYNTACTIC, CORE_IN_SCOPE_SUITES


def get_torchrun_dist_env():
    """
    Read distributed env assuming we are launched with torchrun / torch.distributed.run.

    Required env vars (set by torchrun):
        - RANK
        - LOCAL_RANK
        - WORLD_SIZE
    """
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except KeyError as e:
        raise RuntimeError(
            f"Expected torchrun env variables RANK, LOCAL_RANK, WORLD_SIZE, but {e} is missing. "
            "Make sure to launch with 'python -m torch.distributed.run ...' or 'torchrun ...'."
        )

    gpus_per_node = torch.cuda.device_count()
    if gpus_per_node == 0:
        raise RuntimeError("torchrun distributed mode requires at least one CUDA device per process.")

    return rank, local_rank, world_size, gpus_per_node


if int(os.environ["RANK"]) == 0:
    import wandb


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
    parser.add_argument("--optimizer", default="lamb", type=str, help="The optimizer to use.")
    parser.add_argument("--hybrid_numerator", default=15, type=int, help="The numerator of the hybrid ratio.")
    parser.add_argument("--hybrid_denominator", default=16, type=int,
                        help="The denominator of the hybrid ratio (the number of GPUs should be divisible by this number).")
    parser.add_argument("--seq_length", default=128, type=int, help="Sequence length for training.")
    parser.add_argument("--local_batch_size", default=256, type=int, help="Batch size for training per GPU.")
    parser.add_argument("--global_batch_size", default=32768, type=int,
                        help="Total batch size for training per GPUs and per grad accumulation step.")
    parser.add_argument("--batch_reduction", default=4, type=int, help="The initial batch size reduction factor.")
    parser.add_argument("--learning_rate", default=1.41e-2, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--schedule_horizon", default=7812, type=int,
                        help="The horizon, in optimizer steps, over which the learning-rate decay and the "
                             "batch-size ramp are defined. This is a schedule-shape parameter, deliberately "
                             "longer than the run itself, so that training ends inside the batch ramp with the "
                             "learning rate still decaying.")
    parser.add_argument("--steps", default=5400, type=int,
                        help="Number of optimizer steps to train, with a final save at the end. The LR, masking "
                             "and batch-size schedules stay defined by --schedule_horizon.")
    parser.add_argument("--ema_decay", default=0.999, type=float, help="Exponential moving average decay.")
    parser.add_argument("--validate_every", default=1_000, type=int,
                        help="Run validation after every X training shards.")
    parser.add_argument("--validation_steps", default=1, type=int, help="Number of validation steps.")
    parser.add_argument("--log_stats_every", default=100, type=int, help="Log stats every X steps.")
    parser.add_argument("--warmup_proportion", default=0.016, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument("--cooldown_proportion", default=0.016, type=float,
                        help="Proportion of training to perform linear learning rate cooldown for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--save_every', type=int, default=1_000, help="save every X steps")
    parser.add_argument("--mask_p_start", default=0.3, type=float, help="Initial masking probability.")
    parser.add_argument("--mask_p_end", default=0.15, type=float, help="Final masking probability.")
    parser.add_argument("--mask_random_p", default=0.1, type=float,
                        help="Probability of replacing the masked token with a random token.")
    parser.add_argument("--mask_keep_p", default=0.1, type=float, help="Probability of keeping the masked token.")
    parser.add_argument("--weight_decay", default=0.1, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--optimizer_eps", default=1e-8, type=float, help="Optimizer epsilon.")
    parser.add_argument("--optimizer_beta1", default=0.9, type=float, help="Optimizer beta1.")
    parser.add_argument("--optimizer_beta2", default=0.98, type=float, help="Optimizer beta2.")
    parser.add_argument("--max_gradient", default=2.0, type=float, help="Max value for gradient clipping.")
    parser.add_argument('--mixed_precision', default=True, action=argparse.BooleanOptionalAction,
                        help="Mixed precision training.")
    parser.add_argument('--n_special_tokens', default=16, type=int, help="Number of special tokens.")
    parser.add_argument('--z_loss_weight', default=1e-4, type=float, help="Weight for the z loss.")
    parser.add_argument('--token_weighted_loss', default=False, action=argparse.BooleanOptionalAction,
                        help="Use token weighted loss.")

    # BLiMP evaluation parameters
    parser.add_argument("--blimp_data_path", default="../evaluation/blimp/blimp_really_fast", type=pathlib.Path,
                        help="Path to BLiMP data for evaluation.")
    parser.add_argument("--blimp_backend", default="mlm_shift", type=str, help="BLiMP scoring variant",
                        choices=["mlm_shift", "mlm", "causal", "prefix"])
    parser.add_argument("--blimp_batch_size", default=100, type=int, help="Batch size for BLiMP evaluation.")
    parser.add_argument('--enable_blimp', default=True, action=argparse.BooleanOptionalAction,
                        help="Enable BLiMP evaluation during training.")
    parser.add_argument("--blimp_eval_freq", default=1, type=int, help="Evaluate BLiMP every n validations.")

    # SyntaxGym evaluation parameters
    parser.add_argument('--enable_syntaxgym', default=True, action=argparse.BooleanOptionalAction,
                        help="Enable SyntaxGym evaluation during training.")
    parser.add_argument("--syntaxgym_data_path", default="../evaluation/syntaxgym", type=pathlib.Path,
                        help="Path to SyntaxGym suite JSON files.")
    parser.add_argument("--syntaxgym_backend", default="mlm_shift", type=str, help="SyntaxGym evaluation backend",
                        choices=["mlm_shift"])
    parser.add_argument("--syntaxgym_batch_size", default=64, type=int, help="Batch size for SyntaxGym evaluation.")
    parser.add_argument("--syntaxgym_eval_freq", default=1, type=int, help="Evaluate SyntaxGym every n validations.")
    parser.add_argument("--syntaxgym_suite_set", default="all", type=str,
                        choices=["all", "core_only", "borderline_only"],
                        help="Which SyntaxGym suites to evaluate.")

    parser.add_argument('--log_param_stats', default=False, action=argparse.BooleanOptionalAction,
                        help="Enable logging of parameter/activation/gradient statistics.")
    parser.add_argument('--auto_resume', default=False, action=argparse.BooleanOptionalAction,
                        help="Automatically find and resume from the latest checkpoint matching --name.")
    parser.add_argument('--adaptive_validation', default=False, action=argparse.BooleanOptionalAction,
                        help="Use adaptive validation frequency: every 50 steps for first 500, every 100 for next 1000, every 500 after.")

    args = parser.parse_args()

    args.max_steps = args.schedule_horizon  # for dataset.py masking schedule

    # Auto-resume: the latest of this run name's own stamped saves
    # (<name>_YYYYMMDD_HHMMSS_state_dict.bin). A run named <name>_<suffix>, such as
    # the same launcher on another corpus, is not a resume point of <name>.
    if args.auto_resume and args.checkpoint_filename is None:
        import glob
        import re
        stamped = re.compile(re.escape(args.name) + r"_\d{8}_\d{6}_state_dict\.bin")
        checkpoints = [path for path in glob.glob(f"{args.output_dir}/{args.name}_*_state_dict.bin")
                       if stamped.fullmatch(os.path.basename(path))]
        if checkpoints:
            # Sort by modification time, get the most recent
            latest_checkpoint = max(checkpoints, key=os.path.getmtime)
            args.checkpoint_filename = latest_checkpoint
            print(f"Auto-resume: found checkpoint {latest_checkpoint}")

    # Handle output path naming
    if args.checkpoint_filename is not None:
        # Resuming: extract the run name from the checkpoint path
        # Expected format: .../run_name_YYYYMMDD_HHMMSS_state_dict.bin
        checkpoint_basename = os.path.basename(args.checkpoint_filename)
        if checkpoint_basename.endswith("_state_dict.bin"):
            args.name_with_datetime = checkpoint_basename.replace("_state_dict.bin", "")
        else:
            # Fallback: use the checkpoint name without extension
            args.name_with_datetime = os.path.splitext(checkpoint_basename)[0]
        print(f"Resuming run: {args.name_with_datetime}")
    else:
        # New run: add datetime suffix
        run_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.name_with_datetime = f"{args.name}_{run_datetime}"

    args.output_path = f"{args.output_dir}/{args.name_with_datetime}.bin"

    return args


def setup_training(args, tokenizer):
    assert torch.cuda.is_available()
    args.n_gpu = torch.cuda.device_count()

    rank, local_rank, world_size, gpus_per_node = get_torchrun_dist_env()
    args.rank = rank
    args.local_rank = local_rank
    args.world_size = world_size
    args.gpus_per_node = gpus_per_node
    assert args.gpus_per_node == torch.cuda.device_count()
    print(
        f"Hello from rank {args.rank} of {args.world_size} on {gethostname()} where there are {args.gpus_per_node} allocated GPUs per node.",
        flush=True)

    assert args.world_size % args.hybrid_denominator == 0

    # if args.rank / args.world_size < args.hybrid_numerator / args.hybrid_denominator:
    if args.rank * args.hybrid_denominator < args.hybrid_numerator * args.world_size:
        args.dataset_type = "masked"
    else:
        args.dataset_type = "causal"

    print(f"Dataset type: {args.dataset_type}", flush=True)

    seed_everything(args.seed + args.rank)

    torch.distributed.init_process_group(backend="nccl", rank=args.rank, world_size=args.world_size)
    if args.rank == 0:
        print(f"Group initialized? {torch.distributed.is_initialized()}", flush=True)

    args.local_rank = args.rank - args.gpus_per_node * (args.rank // args.gpus_per_node)
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device("cuda", args.local_rank)
    print(f"RCCL started on device {args.device}", flush=True)
    print(f"host: {gethostname()}, rank: {args.rank}, local_rank: {args.local_rank}")

    if is_main_process():
        print(f"Training for {args.steps:,} steps with {get_world_size()} GPUs")
        print(
            f"Schedule shape: the learning rate decay and the batch size ramp are defined over a horizon of {args.schedule_horizon:,} steps")
        if args.adaptive_validation:
            print(f"Validation schedule: every 50 steps (0-500), every 100 steps (500-1500), every 500 steps (1500+)")
        else:
            print(f"Validation schedule: every {args.validate_every} steps")

    args.vocab_size = tokenizer.get_vocab_size()

    if is_main_process():
        # Build a serializable config dict from args
        wandb_config = {}
        for k, v in vars(args).items():
            if isinstance(v, pathlib.Path):
                wandb_config[k] = str(v)
            elif isinstance(v, torch.device):
                wandb_config[k] = str(v)
            else:
                wandb_config[k] = v

        # Check if we're resuming and have a wandb run ID
        wandb_run_id = None
        if args.checkpoint_filename is not None:
            state_dict = torch.load(args.checkpoint_filename, map_location="cpu")
            wandb_run_id = state_dict.get("wandb_run_id", None)
            if wandb_run_id:
                print(f"Found wandb_run_id in checkpoint: {wandb_run_id}")
            else:
                print(f"WARNING: Checkpoint has no wandb_run_id - will create new wandb run")
                print(f"  (This is expected if checkpoint was created before wandb resume support was added)")

        wandb.init(
            name=args.name_with_datetime,
            project=os.environ.get("WANDB_PROJECT", "sambal"),
            entity=os.environ.get("WANDB_ENTITY"),
            config=wandb_config,
            id=wandb_run_id,
            resume="allow",  # "allow" works in offline mode; "must" may fail if local run data missing
        )

        # Define global_step as the x-axis for all metrics
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")

        # Store run ID for future resumes
        args.wandb_run_id = wandb.run.id
        if wandb_run_id and wandb.run.id == wandb_run_id:
            print(f"Successfully resumed wandb run: {args.wandb_run_id}")
        elif wandb_run_id:
            print(f"WARNING: Could not resume run {wandb_run_id}, created new run: {args.wandb_run_id}")
        else:
            print(f"Created new wandb run: {args.wandb_run_id}")


def load_config(args):
    with open(args.config_file, "r") as f:
        config = json.load(f)
    for k, v in config.items():
        setattr(args, k, v)
    return args


def prepare_model_and_optimizer(args):
    args = load_config(args)
    model = Bert(args)

    if is_main_process():
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.config.update(args)
        wandb.config.update({"n_params": n_params})
        print(model)
        print(f"NUMBER OF PARAMETERS: {n_params}\n", flush=True)

    model.to(args.device)

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

    if args.optimizer == "adam" or args.optimizer == "adamw":
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

    scheduler = cosine_schedule_with_warmup_cooldown(
        optimizer,
        int(args.schedule_horizon * args.warmup_proportion),
        int(args.schedule_horizon * args.cooldown_proportion),
        args.schedule_horizon,
        0.1
    )

    model = DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        bucket_cap_mb=torch.cuda.get_device_properties(args.device).total_memory,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
        static_graph=True
    )

    ema_model: nn.Module = copy.deepcopy(model.module)
    for param in ema_model.parameters():
        param.requires_grad = False

    global_step, epoch, validation_count = 0, 0, 0
    if args.checkpoint_filename is not None:
        state_dict = torch.load(args.checkpoint_filename, map_location="cpu")
        model.load_state_dict(state_dict["model"])
        ema_model.load_state_dict(state_dict["ema_model"])
        optimizer.load_state_dict(state_dict["optimizer"])
        scheduler.load_state_dict(state_dict["scheduler"])
        global_step = state_dict["global_step"]
        epoch = state_dict["epoch"]
        validation_count = state_dict.get("validation_count", 0)
        if is_main_process():
            print(f"=== CHECKPOINT LOADED ===")
            print(f"  global_step: {global_step}")
            print(f"  epoch: {epoch}")
            print(f"  validation_count: {validation_count}")
            print(f"  scheduler.last_epoch: {scheduler.last_epoch}")
            print(f"  LR after restore: {scheduler.get_last_lr()[0]:.6e}")
            progress_frac = (global_step + 1) / args.steps
            print(f"  progress: {progress_frac:.3f} ({global_step + 1}/{args.steps})")
            print(f"  checkpoint wandb_run_id: {state_dict.get('wandb_run_id', 'NOT FOUND')}")
            print(f"=========================")

    return model, ema_model, optimizer, scheduler, global_step, epoch, validation_count


def get_batch(dataloader, device, global_step):
    if hasattr(dataloader._dataset, "set_global_step"):
        dataloader._dataset.set_global_step(global_step)
    batch = next(dataloader)
    input_ids, target_ids, attention_mask, mask_p = [t.pin_memory().to(device, non_blocking=True) for t in batch]
    input_ids, target_ids = input_ids.t(), target_ids.t()
    mask_p = mask_p.mean()

    return input_ids, attention_mask, target_ids, mask_p


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
    syntaxgym_results = syntaxgym_eval.training_metrics_weighted(result)

    print(f"SyntaxGym Results:")
    print(f"  Best temperature: {syntaxgym_results['syntaxgym/best_temperature']:.2f} -> Weighted Acc: {syntaxgym_results['syntaxgym/best_temp_weighted_accuracy']:.2f}%")
    print(f"  Temperature 1.0: Weighted Acc: {syntaxgym_results['syntaxgym/temp_1_weighted_accuracy']:.2f}%")

    return syntaxgym_results


def should_validate(global_step, args):
    """Determine if we should validate after this step.

    Called with current global_step before increment, so we check global_step + 1
    to determine if we should validate at the upcoming step number.
    """
    next_step = global_step + 1
    if args.adaptive_validation:
        # Adaptive schedule:
        # - Every 50 steps for first 500 steps (validate at 50, 100, ..., 500)
        # - Every 100 steps for steps 501-1500 (validate at 600, 700, ..., 1500)
        # - Every 500 steps after step 1500 (validate at 2000, 2500, ...)
        if next_step <= 500:
            return next_step % 50 == 0
        elif next_step <= 1500:
            return next_step % 100 == 0
        else:
            return next_step % 500 == 0
    else:
        # Fixed frequency
        return next_step % args.validate_every == 0


def training_epoch(model, ema_model, train_dataloader, valid_dataloader, optimizer, scheduler, global_step, epoch, args,
                   tokenizer, validation_count):
    model = model.train()
    optimizer.zero_grad(set_to_none=True)

    if is_main_process():
        print(f"\n=== Starting epoch {epoch} at global_step {global_step} ===")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6e}")

    # calculate the number of steps to perform in this epoch
    # num_steps = min(len(train_dataloader), (args.schedule_horizon - global_step) * args.accumulate_steps)
    # Compute local steps
    local_steps = torch.tensor([len(train_dataloader)], device=args.device)

    # Take the MIN across all ranks (safest)
    torch.distributed.all_reduce(local_steps, op=torch.distributed.ReduceOp.MIN)

    num_steps = min(
        int(local_steps.item()),
        (args.schedule_horizon - global_step) * args.accumulate_steps
    )

    # initialize the dataloader and the metrics
    train_iter = iter(train_dataloader)
    total_loss, total_accuracy, total_z_loss, total_mask_p, total_grad_norm = 0.0, 0.0, 0.0, 0.0, 0.0

    # get the first batch
    input_ids_, attention_mask_, target_ids_, mask_p_ = get_batch(train_iter, args.device, global_step)

    progress_bar = tqdm(
        total=args.steps,
        initial=global_step,
        desc="Train step",
        disable=not (is_main_process() and sys.stderr.isatty()),
    )

    # iterate over the steps
    for local_step in range(num_steps):
        input_ids, attention_mask, target_ids, mask_p = input_ids_, attention_mask_, target_ids_, mask_p_
        # print(f"[rank {args.rank}] before forward, local step {local_step} global step {global_step}", flush=True)
        # forward pass, do a more detailed check of the model every 100 steps (if enabled)
        with torch.cuda.amp.autocast(args.mixed_precision, dtype=torch.bfloat16):
            with ModelLogger(enable=(args.log_param_stats and global_step % 100 == 0), module=model):
                loss, accuracy, z_loss, num_tokens = model(input_ids, attention_mask, target_ids)

        # get the next batch
        if local_step < num_steps - 1:
            input_ids_, attention_mask_, target_ids_, mask_p_ = get_batch(train_iter, args.device, global_step)

        # calculate the weight for the loss (either token-weighted or not)
        if args.token_weighted_loss:
            total_tokens = torch.tensor(num_tokens, device=args.device, dtype=torch.long)
            torch.distributed.all_reduce(total_tokens, torch.distributed.ReduceOp.SUM)
            weight = args.world_size * num_tokens / total_tokens / args.accumulate_steps
        else:
            weight = 1.0 / args.accumulate_steps
        # print(f"[rank {args.rank}] after forward, before backward, local step {local_step} global step {global_step}", flush=True)
        # backward pass through both losses
        ((loss + args.z_loss_weight * z_loss) * weight).backward()
        # print(f"[rank {args.rank}] after backward, before optimizer step, local step {local_step} global step {global_step}", flush=True)
        # add the tracked metrics (for gradient accumulation)
        total_loss += loss.detach() * weight
        total_accuracy += accuracy * weight
        total_z_loss += z_loss * weight
        total_mask_p += mask_p * weight

        # gradient accumulation -- if we have accumulated enough gradients, we can perform the optimizer step; otherwise, we just continue and backpropagate through the next batch
        if (local_step + 1) % args.accumulate_steps != 0:
            continue

        # clip the gradients
        total_grad_norm += nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient) * weight

        # optimizer step
        optimizer.step()
        scheduler.step()
        # print(f"[rank {args.rank}] after optimizer step, local step {local_step} global step {global_step}", flush=True)

        with torch.no_grad():

            # EMA update
            for param_q, param_k in zip(model.module.parameters(), ema_model.parameters()):
                param_k.data.mul_(args.ema_decay).add_((1.0 - args.ema_decay) * param_q.detach().data)

            # be careful here, not all GPUs work with the same training objective
            if args.dataset_type == "masked":
                total_mlm_loss = total_loss / (args.hybrid_numerator / args.hybrid_denominator)
                total_clm_loss = torch.zeros_like(total_mlm_loss)
                total_mask_p = total_mask_p / (args.hybrid_numerator / args.hybrid_denominator)
            else:
                total_clm_loss = total_loss / (1 - args.hybrid_numerator / args.hybrid_denominator)
                total_mlm_loss = torch.zeros_like(total_clm_loss)
                total_mask_p = torch.zeros_like(total_mask_p)

            # accumulate the metrics across GPUs
            metrics = torch.stack(
                [total_loss, total_accuracy, total_z_loss, total_mask_p, total_mlm_loss, total_clm_loss])
            torch.distributed.all_reduce(metrics, torch.distributed.ReduceOp.AVG)
            total_loss, total_accuracy, total_z_loss, total_mask_p, total_mlm_loss, total_clm_loss = metrics.tolist()

        # log the metrics
        if is_main_process():
            wandb.log(
                {
                    "global_step": global_step,
                    "epoch": epoch,
                    "train/loss": total_loss,
                    "train/z_loss": total_z_loss,
                    "train/perplexity": math.exp(total_loss),
                    "train/accuracy": total_accuracy * 100.0,
                    "train/mlm_loss": total_mlm_loss,
                    "train/clm_loss": total_clm_loss,
                    "stats/learning_rate": optimizer.param_groups[0]['lr'],
                    "stats/grad_norm": total_grad_norm,
                    "stats/seq_length": train_dataloader.dataset.seq_length,
                    "stats/global_batch_size": args.current_global_batch_size,
                    "stats/local_batch_size": args.current_local_batch_size,
                    "stats/accumulate_steps": args.accumulate_steps,
                    "stats/mask_p": total_mask_p,
                },
                commit=False,
            )
            wandb.log({"global_step": global_step}, commit=True)

        # zero the accumulated gradients and the metrics
        optimizer.zero_grad(set_to_none=True)
        total_loss, total_accuracy, total_z_loss, total_mask_p, total_grad_norm = 0.0, 0.0, 0.0, 0.0, 0.0

        # checkpoint the model and the full training state
        if global_step % args.save_every == 0:
            save(model, ema_model, optimizer, scheduler, global_step, epoch, args, validation_count)

        # validate the model
        if should_validate(global_step, args):
            validation_epoch(model, valid_dataloader, epoch, args, tokenizer, global_step, validation_count)
            validation_count += 1
            model.train()

        # log the stats and commit
        if is_main_process():
            progress_bar.update(1)

        global_step += 1

        # Exiting the training: the protocol's step count is done
        if global_step >= args.schedule_horizon or global_step > args.steps:
            save(model, ema_model, optimizer, scheduler, global_step, epoch, args, validation_count)
            validation_epoch(model, valid_dataloader, epoch, args, tokenizer, global_step, validation_count)
            progress_bar.close()
            return global_step, validation_count

    progress_bar.close()
    return global_step, validation_count


@torch.no_grad()
def validation_epoch(model, valid_dataloader, epoch, args, tokenizer, global_step, validation_count=0):
    model = model.eval()

    losses, accuracies = [], []
    valid_dataloader = iter(valid_dataloader)
    input_ids, attention_mask, target_ids, _ = get_batch(valid_dataloader, args.device, 0)
    for local_step in tqdm(range(args.validation_steps), desc="Valid iteration", disable=not (is_main_process() and sys.stderr.isatty())):

        with torch.cuda.amp.autocast(args.mixed_precision, dtype=torch.bfloat16):
            loss, accuracy, _, num_tokens = model(input_ids, attention_mask, target_ids)

        if local_step < args.validation_steps - 1:
            input_ids, attention_mask, target_ids, _ = get_batch(valid_dataloader, args.device, 0)

        total_tokens = torch.tensor(num_tokens, device=args.device, dtype=torch.long)
        torch.distributed.all_reduce(total_tokens, torch.distributed.ReduceOp.SUM)
        weight = args.world_size * num_tokens / total_tokens

        metrics = torch.stack([loss * weight, accuracy * weight])
        torch.distributed.all_reduce(metrics, torch.distributed.ReduceOp.AVG)
        loss, accuracy = metrics.tolist()

        losses.append(loss)
        accuracies.append(accuracy)

    val_metrics = {
        "epoch": epoch,
        "validation/loss": mean(losses),
        "validation/accuracy": mean(accuracies) * 100.0,
        "validation/perplexity": math.exp(mean(losses))
    }

    # Run BLiMP evaluation periodically
    if args.enable_blimp and validation_count % args.blimp_eval_freq == 0:
        if is_main_process():
            print(f"\nRunning BLiMP evaluation...")
        blimp_results = evaluate_blimp(model, tokenizer, args)
        if blimp_results:
            if is_main_process():
                print(f"DEBUG: blimp_results keys: {list(blimp_results.keys())[:5]}...")
            val_metrics.update(blimp_results)
        elif is_main_process():
            print(f"DEBUG: blimp_results is empty/None")

    # Run SyntaxGym evaluation periodically
    if args.enable_syntaxgym and validation_count % args.syntaxgym_eval_freq == 0:
        if is_main_process():
            print(f"\nRunning SyntaxGym evaluation...")
        syntaxgym_results = evaluate_syntaxgym(model, tokenizer, args)
        if syntaxgym_results:
            if is_main_process():
                print(f"DEBUG: syntaxgym_results keys: {list(syntaxgym_results.keys())[:5]}...")
            val_metrics.update(syntaxgym_results)
        elif is_main_process():
            print(f"DEBUG: syntaxgym_results is empty/None")

    if is_main_process():
        val_metrics["global_step"] = global_step
        print(f"DEBUG: Final val_metrics keys: {list(val_metrics.keys())}")
        print(f"DEBUG: Any blimp keys? {any('blimp' in k for k in val_metrics.keys())}")
        print(f"DEBUG: Any syntaxgym keys? {any('syntaxgym' in k for k in val_metrics.keys())}")
        try:
            wandb.log(val_metrics, commit=True)
            print(f"DEBUG: wandb.log completed successfully")
        except Exception as e:
            print(f"DEBUG: wandb.log failed with error: {e}")


def save(model, ema_model, optimizer, scheduler, global_step, epoch, args, validation_count=0):
    if is_main_process():
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model itself
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        torch.save(model_to_save.state_dict(), args.output_path)
        torch.save(ema_model.state_dict(), args.output_path.replace(".bin", "_ema.bin"))
        torch.save(
            {
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "global_step": global_step,
                "epoch": epoch + 1,
                "validation_count": validation_count,
                "wandb_run_id": getattr(args, 'wandb_run_id', None),
            },
            args.output_path.replace(".bin", "_state_dict.bin")
        )


def load_datasets(args, tokenizer, epoch, global_step, train_dataloader, valid_dataloader):
    train_seed = args.seed + get_rank() + epoch * get_world_size()

    # The run trains at a single sequence length throughout.
    seq_length = args.seq_length
    global_batch_size = args.global_batch_size

    if train_dataloader is None or train_dataloader.dataset.seq_length != seq_length:
        if is_main_process():
            progress_frac = (global_step + 1) / args.steps
            print(
                f"load_datasets: global_step={global_step}, progress={progress_frac:.3f}, seq_length={seq_length}, global_batch_size={global_batch_size}")

        if args.dataset_type == "masked":
            rank = args.rank
            world_size = args.world_size * args.hybrid_numerator // args.hybrid_denominator
            train_data = MaskedDataset(args.train_path, tokenizer, args, seq_length, rank, world_size)
        else:
            rank = args.rank - args.world_size * args.hybrid_numerator // args.hybrid_denominator
            world_size = args.world_size * (args.hybrid_denominator - args.hybrid_numerator) // args.hybrid_denominator
            train_data = CausalDataset(args.train_path, tokenizer, args, seq_length, rank, world_size)

        if is_main_process():
            train_data.show_random_item(tokenizer)
    else:
        train_data = train_dataloader.dataset

    # linear batch size scaling
    args.current_global_batch_size = int(
        global_batch_size / args.batch_reduction * (1 - global_step / args.schedule_horizon) + global_batch_size * (
                    global_step / args.schedule_horizon) + 0.5)
    total_local_batch_size = int(args.current_global_batch_size / args.world_size + 0.5)
    args.accumulate_steps = int(math.ceil(total_local_batch_size / args.local_batch_size))
    args.current_local_batch_size = total_local_batch_size // args.accumulate_steps

    train_dataloader = DataLoader(
        train_data,
        shuffle=True,
        batch_size=args.current_local_batch_size,
        num_workers=0,  # non-zero num_workers causes segmenation fault
        generator=torch.Generator().manual_seed(train_seed),
        drop_last=True,
        pin_memory=True,
    )

    if valid_dataloader is None:
        valid_data = ValidationDataset(args.valid_path, tokenizer, args)

        valid_dataloader = DataLoader(
            valid_data,
            shuffle=False,
            batch_size=args.local_batch_size,
            num_workers=0,  # non-zero num_workers causes segmenation fault
            generator=torch.Generator().manual_seed(42),
            drop_last=True,
            pin_memory=True,
        )

    return train_dataloader, valid_dataloader


if __name__ == "__main__":
    args = parse_arguments()

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    setup_training(args, tokenizer)
    model, ema_model, optimizer, scheduler, global_step, start_epoch, validation_count = prepare_model_and_optimizer(
        args)

    train_dataloader, valid_dataloader = None, None

    for epoch in count(start=start_epoch):
        train_dataloader, valid_dataloader = load_datasets(args, tokenizer, epoch, global_step, train_dataloader,
                                                           valid_dataloader)
        global_step, validation_count = training_epoch(model, ema_model, train_dataloader, valid_dataloader, optimizer,
                                                       scheduler, global_step, epoch, args, tokenizer, validation_count)

        if global_step >= args.schedule_horizon or global_step > args.steps:
            break

    save(model, ema_model, optimizer, scheduler, global_step, epoch, args, validation_count)
    validation_epoch(model, valid_dataloader, epoch, args, tokenizer, global_step, validation_count)