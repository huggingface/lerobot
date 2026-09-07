"""LingBot optimizers with named expert groups and standard checkpoint state."""

import re
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from .muon import DistributedMuon
from .optimizers import OptimizerConfig, OptimizerParams

_EXPERT = re.compile(r"\.layers\.\d+\.mlp\.experts\.")
_ADAMW_NAMES = ("embed_tokens", "embedding", "lm_head", "output_layer")


def _groups(params: OptimizerParams, lr: float, expert_lr_scale: float, muon: bool) -> list[dict]:
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
        kind = (
            "muon"
            if muon and param.ndim in (2, 3) and not any(token in name.lower() for token in _ADAMW_NAMES)
            else "adamw"
        )
        key = (kind, group_lr)
        group = groups.setdefault(key, {"params": [], "param_names": [], "lr": group_lr, "kind": kind})
        group["params"].append(param)
        group["param_names"].append(name)
    if not groups:
        raise ValueError("No trainable LingBot parameters")
    # Base-LR groups first: logger's first group remains the nominal LR.
    return [groups[k] for k in sorted(groups, key=lambda k: (k[1], k[0]))]


class LingbotMuon(torch.optim.Optimizer):
    """Muon + AdamW sharing one ordinary Optimizer state/parameter namespace.

    Inner optimizers share the *same dictionaries*, including after load_state_dict.
    Accelerate's FSDP2 parameter replacement therefore reaches both inner optimizers.
    Only the outer optimizer is serialized or passed to Accelerate/DCP.
    """

    def __init__(self, groups: list[dict], **defaults):
        super().__init__(groups, defaults)
        for group in self.param_groups:
            if group["kind"] == "muon":
                group["eps"] = 1e-7  # Upstream NS epsilon, distinct from AdamW epsilon.
        self._bind_inner_optimizers()

    def _bind_inner_optimizers(self) -> None:
        muon_groups = [g for g in self.param_groups if g["kind"] == "muon"]
        adamw_groups = [g for g in self.param_groups if g["kind"] == "adamw"]
        self._inner: list[torch.optim.Optimizer] = []
        if muon_groups:
            self._inner.append(DistributedMuon(muon_groups))
        if adamw_groups:
            # foreach=False is DTensor-safe and avoids fused-AdamW DTensor limitations.
            self._inner.append(torch.optim.AdamW(adamw_groups, foreach=False, fused=False))
        for optimizer in self._inner:
            optimizer.state = self.state

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self._bind_inner_optimizers()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for optimizer in self._inner:
            optimizer.state = self.state
            optimizer.step()
        return loss


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
            _groups(params, self.lr, self.expert_lr_scale, False),
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            fused=self.fused,
        )


@OptimizerConfig.register_subclass("lingbot_muon")
@dataclass
class LingbotMuonConfig(OptimizerConfig):
    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    adamw_betas: tuple[float, float] = (0.9, 0.95)
    adamw_eps: float = 1e-8
    expert_lr_scale: float = 1.0

    def build(self, params: OptimizerParams) -> torch.optim.Optimizer:
        return LingbotMuon(
            _groups(params, self.lr, self.expert_lr_scale, True),
            lr=self.lr,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            nesterov=self.nesterov,
            ns_steps=self.ns_steps,
            adjust_lr_fn="match_rms_adamw",
            betas=self.adamw_betas,
            eps=self.adamw_eps,
        )
