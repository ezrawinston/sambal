"""Pieces shared by the evaluators: the temperature grid, backend command-line options, sharding."""

from __future__ import annotations

import argparse
from typing import Optional

import torch

from .backends import BACKEND_ALIASES, LanguageModel, load_backend, parse_backend_args, resolve_backend_class

#: the grid every reported sweep uses: 0.00 .. 3.00 in steps of 0.05, floored away from zero
TEMPERATURE_STEP = 0.05
TEMPERATURE_INDEX_1 = 20


def temperature_grid(device=None) -> torch.Tensor:
    return torch.arange(0.0, 3.05, TEMPERATURE_STEP, device=device).clamp(min=1e-6)


def add_backend_arguments(parser: argparse.ArgumentParser, prefix: str = "", role: str = "model",
                          with_device: bool = True) -> None:
    """Add ``--[prefix-]backend`` and ``--[prefix-]backend-arg`` (and ``--device`` once).

    ``prefix`` distinguishes several models in one command, e.g. ``prefix="other-"``
    gives ``--other-backend`` / ``--other-backend-arg`` (namespace ``other_backend``).
    """
    parser.add_argument(f"--{prefix}backend", default="gptbert",
                        help=f"{role}: a backend alias ({', '.join(sorted(BACKEND_ALIASES))}) "
                             "or module:Class implementing evals.backends.LanguageModel")
    parser.add_argument(f"--{prefix}backend-arg", action="append", default=[], metavar="KEY=VALUE",
                        help=f"{role} backend option; repeatable (gptbert: checkpoint=, config=, tokenizer=, "
                             "variant=, batch_size=, name=)")
    if with_device:
        parser.add_argument("--device", default=None, help="cpu, cuda, cuda:0 ... (default: cuda if available)")


def backend_from_args(args: argparse.Namespace, prefix: str = "") -> LanguageModel:
    """Build the backend named by ``--[prefix-]backend`` / ``--[prefix-]backend-arg``."""
    key = prefix.replace("-", "_")
    spec = getattr(args, key + "backend")
    pairs = getattr(args, key + "backend_arg")
    device = torch.device(args.device) if getattr(args, "device", None) else None
    return load_backend(spec, parse_backend_args(pairs), device=device)


def backend_help(spec: str) -> str:
    return resolve_backend_class(spec).argument_help()


def shard(items, rank: int = 0, world_size: int = 1):
    """The slice of ``items`` this process scores when work is split across processes."""
    return list(items)[rank::world_size]
