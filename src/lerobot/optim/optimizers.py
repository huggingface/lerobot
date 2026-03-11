#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import abc
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import draccus
import torch
from safetensors.torch import load_file, save_file

from lerobot.datasets.utils import flatten_dict, unflatten_dict, write_json
from lerobot.utils.constants import (
    OPTIMIZER_PARAM_GROUPS,
    OPTIMIZER_STATE,
)
from lerobot.utils.io_utils import deserialize_json_into_object

# Type alias for parameters accepted by optimizer build() methods.
# This matches PyTorch's optimizer signature while also supporting:
# - dict[str, Parameter]: Named parameters for differential LR by name (e.g., XVLA)
# - dict[str, Iterable]: Multiple parameter groups for multi-optimizer configs (e.g., SAC)
OptimizerParams = (
    Iterable[torch.nn.Parameter]  # From model.parameters()
    | Iterable[dict[str, Any]]  # List of param groups with lr/weight_decay overrides
    | dict[str, torch.nn.Parameter]  # From dict(model.named_parameters()) for name-based LR
    | dict[str, Any]  # For multi-optimizer configs (SAC) with multiple param groups
)


@dataclass
class OptimizerConfig(draccus.ChoiceRegistry, abc.ABC):
    lr: float
    weight_decay: float
    grad_clip_norm: float

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @classmethod
    def default_choice_name(cls) -> str | None:
        return "adam"

    @abc.abstractmethod
    def build(self, params: OptimizerParams) -> torch.optim.Optimizer | dict[str, torch.optim.Optimizer]:
        """
        Build the optimizer. It can be a single optimizer or a dictionary of optimizers.

        NOTE: Multiple optimizers are useful when you have different models to optimize.
        For example, you can have one optimizer for the policy and another one for the value function
        in reinforcement learning settings.

        Args:
            params: Parameters to optimize. Accepts multiple formats depending on the optimizer:
                - Iterable[Parameter]: From model.parameters() - standard PyTorch usage
                - Iterable[dict]: List of param groups with 'params' key and optional
                  'lr', 'weight_decay' overrides (e.g., ACT, VQBeT policies)
                - dict[str, Parameter]: From dict(model.named_parameters()) for optimizers
                  that apply differential learning rates by parameter name (e.g., XVLA)
                - dict[str, Iterable]: For multi-optimizer configs where each key maps to
                  a separate optimizer's parameters (e.g., SAC with actor/critic/temperature)

        Returns:
            The optimizer or a dictionary of optimizers.
        """
        raise NotImplementedError


@OptimizerConfig.register_subclass("adam")
@dataclass
class AdamConfig(OptimizerConfig):
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 10.0

    def build(self, params: OptimizerParams) -> torch.optim.Optimizer:
        kwargs = asdict(self)
        kwargs.pop("grad_clip_norm")
        return torch.optim.Adam(params, **kwargs)


@OptimizerConfig.register_subclass("adamw")
@dataclass
class AdamWConfig(OptimizerConfig):
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    grad_clip_norm: float = 10.0

    def build(self, params: OptimizerParams) -> torch.optim.Optimizer:
        kwargs = asdict(self)
        kwargs.pop("grad_clip_norm")
        return torch.optim.AdamW(params, **kwargs)


@OptimizerConfig.register_subclass("sgd")
@dataclass
class SGDConfig(OptimizerConfig):
    lr: float = 1e-3
    momentum: float = 0.0
    dampening: float = 0.0
    nesterov: bool = False
    weight_decay: float = 0.0
    grad_clip_norm: float = 10.0

    def build(self, params: OptimizerParams) -> torch.optim.Optimizer:
        kwargs = asdict(self)
        kwargs.pop("grad_clip_norm")
        return torch.optim.SGD(params, **kwargs)


@OptimizerConfig.register_subclass("xvla-adamw")
@dataclass
class XVLAAdamWConfig(OptimizerConfig):
    """Custom AdamW optimizer for XVLA with differential learning rates.

    The Vision-Language Model (VLM) is trained with 1/10 of the base learning rate
    for stable optimization, while all other components use the full LR.

    This LR ratio is crucial for achieving strong and stable finetuning performance.

    Soft-prompts can optionally use a separate learning rate with warm-up support.
    Set `soft_prompt_lr_scale` to a value < 1.0 (e.g., 0.1) to start soft-prompts
    at a lower LR. Combine with a warmup scheduler for optimal results.

    Note:
        Completely matching official reported performance may require an additional
        warm-up LR schedule for soft-prompts, which can bring minor improvements.
        When `soft_prompt_warmup_lr_scale` is set, soft-prompts start at
        `lr * soft_prompt_warmup_lr_scale` and should be warmed up via the scheduler.

    Parameter Groups:
        - Group 0 (vlm): VLM parameters at lr * 0.1, weight_decay * 0.1
        - Group 1 (soft_prompts): Soft-prompt parameters at lr * soft_prompt_lr_scale
        - Group 2 (other): All other parameters at full lr
    """

    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.99)
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 10.0
    # Soft-prompt specific settings
    soft_prompt_lr_scale: float = 1.0  # Scale factor for soft-prompt LR (1.0 = same as base LR)
    soft_prompt_warmup_lr_scale: float | None = None  # If set, start soft-prompts at this scale (e.g., 0.01)

    def build(self, params: OptimizerParams) -> torch.optim.Optimizer:
        """
        Build AdamW optimizer with differential learning rates.

        Args:
            params: Must be a dict[str, Parameter] from dict(model.named_parameters())
                or equivalent.

        Returns:
            AdamW optimizer with parameter groups for VLM, soft-prompts, and other components

        Raises:
            AssertionError: If params is not a dict (e.g., from model.parameters())
        """
        assert isinstance(params, dict), "Custom LR optimizer requires `named_parameters()` as inputs."

        vlm_group, soft_prompt_group, other_group = [], [], []
        for name, p in params.items():
            if not p.requires_grad:
                continue
            if "vlm" in name.lower():
                vlm_group.append(p)
            elif "soft_prompt" in name.lower():
                soft_prompt_group.append(p)
            else:
                other_group.append(p)

        # Determine soft-prompt LR
        soft_prompt_lr = self.lr * self.soft_prompt_lr_scale
        if self.soft_prompt_warmup_lr_scale is not None:
            # Start at warmup scale, scheduler will warm up to soft_prompt_lr
            soft_prompt_lr = self.lr * self.soft_prompt_warmup_lr_scale

        param_groups: list[dict[str, Any]] = [
            {
                "params": vlm_group,
                "lr": self.lr * 0.1,
                "weight_decay": self.weight_decay * 0.1,
                "name": "vlm",
            },
            {
                "params": soft_prompt_group,
                "lr": soft_prompt_lr,
                "weight_decay": self.weight_decay,
                "name": "soft_prompts",
            },
            {
                "params": other_group,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "name": "other",
            },
        ]

        # Filter out empty groups
        param_groups = [g for g in param_groups if len(g["params"]) > 0]

        return torch.optim.AdamW(
            param_groups,
            betas=self.betas,
            eps=self.eps,
        )


@OptimizerConfig.register_subclass("multi_adam")
@dataclass
class MultiAdamConfig(OptimizerConfig):
    """Configuration for multiple Adam optimizers with different parameter groups.

    This creates a dictionary of Adam optimizers, each with its own hyperparameters.

    Args:
        lr: Default learning rate (used if not specified for a group)
        weight_decay: Default weight decay (used if not specified for a group)
        optimizer_groups: Dictionary mapping parameter group names to their hyperparameters
        grad_clip_norm: Gradient clipping norm
    """

    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float = 10.0
    optimizer_groups: dict[str, dict[str, Any]] = field(default_factory=dict)

    def build(self, params: OptimizerParams) -> dict[str, torch.optim.Optimizer]:
        """Build multiple Adam optimizers.

        Args:
            params: Must be a dict[str, Iterable[Parameter]] mapping parameter group names
                to iterables of parameters. The keys should match the keys in optimizer_groups.
                Typically from policies that need separate optimizers (e.g., SAC with
                actor/critic/temperature).

        Returns:
            Dictionary mapping parameter group names to their optimizers

        Raises:
            AssertionError: If params is not a dict
        """
        assert isinstance(params, dict), "MultiAdamConfig requires a dict of parameter groups as inputs."
        optimizers = {}

        for name, group_params in params.items():
            # Get group-specific hyperparameters or use defaults
            group_config = self.optimizer_groups.get(name, {})

            # Create optimizer with merged parameters (defaults + group-specific)
            optimizer_kwargs = {
                "lr": group_config.get("lr", self.lr),
                "betas": group_config.get("betas", (0.9, 0.999)),
                "eps": group_config.get("eps", 1e-5),
                "weight_decay": group_config.get("weight_decay", self.weight_decay),
            }

            optimizers[name] = torch.optim.Adam(group_params, **optimizer_kwargs)

        return optimizers


def save_optimizer_state(
    optimizer: torch.optim.Optimizer | dict[str, torch.optim.Optimizer], save_dir: Path
) -> None:
    """Save optimizer state to disk.

    Args:
        optimizer: Either a single optimizer or a dictionary of optimizers.
        save_dir: Directory to save the optimizer state.
    """
    if isinstance(optimizer, dict):
        # Handle dictionary of optimizers
        for name, opt in optimizer.items():
            optimizer_dir = save_dir / name
            optimizer_dir.mkdir(exist_ok=True, parents=True)
            _save_single_optimizer_state(opt, optimizer_dir)
    else:
        # Handle single optimizer
        _save_single_optimizer_state(optimizer, save_dir)


def _save_single_optimizer_state(optimizer: torch.optim.Optimizer, save_dir: Path) -> None:
    """Save a single optimizer's state to disk."""
    state = optimizer.state_dict()
    param_groups = state.pop("param_groups")
    flat_state = flatten_dict(state)
    save_file(flat_state, save_dir / OPTIMIZER_STATE)
    write_json(param_groups, save_dir / OPTIMIZER_PARAM_GROUPS)


def load_optimizer_state(
    optimizer: torch.optim.Optimizer | dict[str, torch.optim.Optimizer], save_dir: Path
) -> torch.optim.Optimizer | dict[str, torch.optim.Optimizer]:
    """Load optimizer state from disk.

    Args:
        optimizer: Either a single optimizer or a dictionary of optimizers.
        save_dir: Directory to load the optimizer state from.

    Returns:
        The updated optimizer(s) with loaded state.
    """
    if isinstance(optimizer, dict):
        # Handle dictionary of optimizers
        loaded_optimizers = {}
        for name, opt in optimizer.items():
            optimizer_dir = save_dir / name
            if optimizer_dir.exists():
                loaded_optimizers[name] = _load_single_optimizer_state(opt, optimizer_dir)
            else:
                loaded_optimizers[name] = opt
        return loaded_optimizers
    else:
        # Handle single optimizer
        return _load_single_optimizer_state(optimizer, save_dir)


def _load_single_optimizer_state(optimizer: torch.optim.Optimizer, save_dir: Path) -> torch.optim.Optimizer:
    """Load a single optimizer's state from disk."""
    current_state_dict = optimizer.state_dict()
    flat_state = load_file(save_dir / OPTIMIZER_STATE)
    state = unflatten_dict(flat_state)

    # Handle case where 'state' key might not exist (for newly created optimizers)
    if "state" in state:
        loaded_state_dict = {"state": {int(k): v for k, v in state["state"].items()}}
    else:
        loaded_state_dict = {"state": {}}

    if "param_groups" in current_state_dict:
        param_groups = deserialize_json_into_object(
            save_dir / OPTIMIZER_PARAM_GROUPS, current_state_dict["param_groups"]
        )
        loaded_state_dict["param_groups"] = param_groups

    optimizer.load_state_dict(loaded_state_dict)
    return optimizer
