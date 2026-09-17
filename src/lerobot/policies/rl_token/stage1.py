#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor

from .modeling_rl_token import RLTokenModel


class RLTokenStage1Trainer:
    """Optimize RL-token reconstruction and an optional independent VLA objective."""

    def __init__(
        self,
        model: RLTokenModel,
        *,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        grad_clip: float = 1.0,
        vla_parameters: Iterable[torch.nn.Parameter] | None = None,
        vla_lr: float = 1e-5,
        vla_alpha: float = 0.0,
    ) -> None:
        self.model = model
        self.grad_clip = grad_clip
        self.vla_alpha = vla_alpha
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.vla_parameters = list(vla_parameters or [])
        self.vla_optimizer = (
            torch.optim.AdamW(self.vla_parameters, lr=vla_lr, weight_decay=weight_decay)
            if self.vla_parameters and vla_alpha > 0.0
            else None
        )
        if vla_alpha > 0.0 and self.vla_optimizer is None:
            raise ValueError("vla_parameters are required when vla_alpha > 0")
        self.steps = 0

    def step(self, embeddings: Tensor, mask: Tensor, vla_loss: Tensor | None = None) -> dict[str, float]:
        self.model.train()
        reconstruction_loss, _ = self.model.reconstruction_loss(embeddings, mask)
        total_loss = reconstruction_loss
        if self.vla_alpha > 0.0:
            if vla_loss is None:
                raise ValueError("vla_loss is required for joint Stage 1 training")
            total_loss = total_loss + self.vla_alpha * vla_loss

        self.optimizer.zero_grad(set_to_none=True)
        if self.vla_optimizer is not None:
            self.vla_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        token_grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        metrics = {
            "loss": total_loss.item(),
            "reconstruction_loss": reconstruction_loss.item(),
            "token_grad_norm": float(token_grad),
        }
        if self.vla_optimizer is not None:
            vla_grad = torch.nn.utils.clip_grad_norm_(self.vla_parameters, self.grad_clip)
            self.vla_optimizer.step()
            metrics["vla_loss"] = float(vla_loss.detach())
            metrics["vla_grad_norm"] = float(vla_grad)
        self.steps += 1
        return metrics

    def state_dict(self) -> dict[str, object]:
        state: dict[str, object] = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
        }
        if self.vla_optimizer is not None:
            state["vla_optimizer"] = self.vla_optimizer.state_dict()
        return state

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.vla_optimizer is not None and "vla_optimizer" in state:
            self.vla_optimizer.load_state_dict(state["vla_optimizer"])
        self.steps = int(state["steps"])
