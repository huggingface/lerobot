"""LingBot optimizers with named expert groups and standard checkpoint state."""

import re
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from .optimizers import OptimizerConfig, OptimizerParams

_EXPERT = re.compile(r"\.layers\.\d+\.mlp\.experts\.")
_ADAMW_NAMES = ("embed_tokens", "embedding", "lm_head", "output_layer")


def _groups(params: OptimizerParams, lr: float, expert_lr_scale: float) -> list[dict]:
    if not isinstance(params, Mapping):
        raise TypeError("LingBot optimizers require the name-keyed policy.get_optim_params() mapping")
    groups: dict[tuple[str, float], dict] = {}
    seen: set[int] = set()
    for name, param in params.items():
        if not isinstance(param, torch.nn.Parameter):
            raise TypeError(f"Expected Parameter for {name}, got {type(param).__name__}")
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        group_lr = lr * expert_lr_scale if _EXPERT.search(name) else lr
        key = ("adamw", group_lr)
        group = groups.setdefault(key, {"params": [], "param_names": [], "lr": group_lr, "kind": "adamw"})
        group["params"].append(param)
        group["param_names"].append(name)
    if not groups:
        raise ValueError("No trainable LingBot parameters")
    # Base-LR groups first: logger's first group remains the nominal LR.
    return [groups[k] for k in sorted(groups, key=lambda k: (k[1], k[0]))]


@OptimizerConfig.register_subclass("lingbot_adamw")
@dataclass
class LingbotAdamWConfig(OptimizerConfig):
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    expert_lr_scale: float = 1.0
    fused: bool = False

    def build(self, params: OptimizerParams) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            _groups(params, self.lr, self.expert_lr_scale),
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            fused=self.fused,
        )
