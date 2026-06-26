"""Reproducibility utilities for deterministic ML experiments."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class SeedBundle:
    """Documented random seeds for an experiment run."""

    python: int = 42
    numpy: int = 42
    torch: int = 42
    cuda: int = 42
    dataloader: int = 42
    hash: int = 42

    def as_dict(self) -> dict[str, int]:
        return {
            "python": self.python,
            "numpy": self.numpy,
            "torch": self.torch,
            "cuda": self.cuda,
            "dataloader": self.dataloader,
            "hash": self.hash,
        }


def load_ml_config(config_path: str) -> dict[str, Any]:
    """Load a YAML ML configuration file."""
    with open(config_path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def seeds_from_config(config: dict[str, Any]) -> SeedBundle:
    """Extract seed bundle from experiment configuration."""
    training = config.get("training", {})
    repro = config.get("reproducibility", {})
    base = int(training.get("seed", 42))
    return SeedBundle(
        python=base,
        numpy=base,
        torch=base,
        cuda=base,
        dataloader=base,
        hash=base if repro.get("deterministic", True) else base + 1,
    )


def set_global_seeds(seeds: SeedBundle, deterministic: bool = True) -> None:
    """Apply seeds across Python, NumPy, and PyTorch when available.

    Parameters:
        seeds: Seed values to apply.
        deterministic: Enable strict CUDA deterministic mode when True.
    """
    random.seed(seeds.python)
    os.environ["PYTHONHASHSEED"] = str(seeds.hash)

    try:
        import numpy as np

        np.random.seed(seeds.numpy)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seeds.torch)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seeds.cuda)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if hasattr(torch, "use_deterministic_algorithms"):
                torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def dataloader_generator(seed: int):
    """Return a PyTorch generator for DataLoader workers."""
    try:
        import torch

        generator = torch.Generator()
        generator.manual_seed(seed)
        return generator
    except ImportError:
        return None
