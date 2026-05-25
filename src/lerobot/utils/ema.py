# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Exponential Moving Average of model parameters for training stability.

Maintains a shadow copy of every trainable parameter, updated after each
optimizer step::

    θ_ema  ←  β · θ_ema  +  (1 - β) · θ_live

At eval / inference / final checkpoint, use ``θ_ema`` instead of
``θ_live``. For diffusion / flow-matching policies, averaging late-
training oscillations yields a smoother model that generalises
substantially better at inference — see Chi et al. 2023 (Diffusion
Policy §V.D, β=0.75), Ho et al. 2020 (DDPM appendix). For VLAs with a
flow-matching action expert the same logic applies: flow gradients have
high variance per sample (different noise levels in the same batch),
so EMA smooths over that variance.

Cost: 1× model parameters in fp32 (~13 GB for pi052's 3.3B params),
plus one elementwise update per training step (~1% of step time).
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file


class ModelEMA:
    """Exponential moving average of trainable model parameters.

    Args:
        model: The live model whose parameter shapes/names define the
            shadow. Only parameters with ``requires_grad=True`` are
            tracked. Buffers are intentionally NOT tracked (LayerNorm
            running stats, RoPE caches, etc.) — they are updated in
            ``train()`` mode regardless of which weights we apply.
        decay: Target EMA decay (``β`` in
            ``θ_ema ← β·θ_ema + (1-β)·θ_live``). Typical values:

            * ``0.999`` — averages roughly the last 1000 steps. Standard
              for diffusion-style policies and the default here.
            * ``0.75`` — very fast EMA, used by Diffusion Policy (Chi
              et al. 2023). Useful when training is short or noisy.
            * ``0.9999`` — very slow EMA, used in image classification
              for very long runs.
        warmup_steps: If > 0, ramp the effective decay from a low value
            up to ``decay`` over the first ``warmup_steps`` updates as
            ``min(decay, (1 + n) / (10 + n))``. Lets the EMA track
            rapid early-training changes before settling on the target.
        device: Where to keep the shadow parameters. ``None`` keeps each
            shadow on its parameter's device (good for FSDP / multi-GPU).
            Pass an explicit device to relocate everything (e.g. ``"cpu"``
            to free GPU memory at the cost of slower updates).
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        decay: float = 0.999,
        warmup_steps: int = 0,
        device: torch.device | str | None = None,
    ) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.warmup_steps = int(warmup_steps)
        self.num_updates = 0
        self.device = torch.device(device) if device is not None else None

        # fp32 shadow — small EMA updates lose precision in bf16.
        self.shadow: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            shadow_p = p.detach().clone().float()
            if self.device is not None:
                shadow_p = shadow_p.to(self.device)
            self.shadow[name] = shadow_p

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def _effective_decay(self) -> float:
        if self.warmup_steps <= 0 or self.num_updates >= self.warmup_steps:
            return self.decay
        # Standard EMA warmup (timm / diffusers convention): grows
        # 0.09, 0.16, 0.23, ... and saturates at ``decay``.
        return min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))

    @torch.no_grad()
    def update(self, model: nn.Module) -> float:
        """Pull one update from the live model into the shadow.

        Returns the effective decay used this step (useful to log during
        warmup, when the value differs from ``self.decay``).
        """
        self.num_updates += 1
        beta = self._effective_decay()
        one_minus_beta = 1.0 - beta
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            shadow = self.shadow.get(name)
            if shadow is None:
                # New parameter appeared mid-training — seed it.
                shadow = p.detach().clone().float()
                if self.device is not None:
                    shadow = shadow.to(self.device)
                self.shadow[name] = shadow
                continue
            # In-place fused: shadow ← β · shadow + (1 - β) · p
            shadow.mul_(beta).add_(
                p.detach().to(shadow.device, dtype=torch.float32),
                alpha=one_minus_beta,
            )
        return beta

    # ------------------------------------------------------------------
    # Applying the EMA to the live model
    # ------------------------------------------------------------------

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Overwrite the live model's parameters with the EMA shadow.

        In-place and **irreversible** — the previous live weights are
        lost. Use this only at the very end of training when you want
        the EMA to *be* the final saved policy. For temporary swaps
        (e.g. during eval), use :meth:`apply_to`.
        """
        for name, p in model.named_parameters():
            shadow = self.shadow.get(name)
            if shadow is not None:
                p.data.copy_(shadow.to(p.device, dtype=p.dtype))

    @contextmanager
    def apply_to(self, model: nn.Module) -> Iterator[None]:
        """Temporarily swap the live model's weights with the EMA copy.

        On exit, the original live weights are restored byte-for-byte
        (we keep a backup clone of every tracked parameter inside the
        context). Use this around eval / sample-logging without
        disturbing the live training state::

            with ema.apply_to(policy):
                eval_metrics = evaluate(policy)
            # policy is back to its pre-eval state here.
        """
        backup: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                backup[name] = p.detach().clone()
                p.data.copy_(self.shadow[name].to(p.device, dtype=p.dtype))
        try:
            yield
        finally:
            for name, p in model.named_parameters():
                if name in backup:
                    p.data.copy_(backup[name].to(p.device, dtype=p.dtype))

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "shadow": {k: v.detach().cpu() for k, v in self.shadow.items()},
            "num_updates": self.num_updates,
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.decay = float(state["decay"])
        self.warmup_steps = int(state["warmup_steps"])
        self.num_updates = int(state["num_updates"])
        new_shadow: dict[str, torch.Tensor] = {}
        for k, v in state["shadow"].items():
            t = v.detach()
            if self.device is not None:
                t = t.to(self.device)
            new_shadow[k] = t.float()
        self.shadow = new_shadow

    def save(self, path: Path | str) -> None:
        """Save the shadow as safetensors + a tiny JSON sidecar with metadata.

        Sidecar lives at ``<path>.json`` and stores ``num_updates``,
        ``decay``, ``warmup_steps`` — enough to resume exact EMA state.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in self.shadow.items()},
            str(path),
        )
        meta = {
            "num_updates": self.num_updates,
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
        }
        path.with_suffix(path.suffix + ".json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load_from_file(
        cls,
        model: nn.Module,
        path: Path | str,
        *,
        device: torch.device | str | None = None,
    ) -> "ModelEMA":
        """Reconstruct a ``ModelEMA`` from a previously-saved safetensors + sidecar pair."""
        path = Path(path)
        meta_path = path.with_suffix(path.suffix + ".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        ema = cls(
            model,
            decay=float(meta.get("decay", 0.999)),
            warmup_steps=int(meta.get("warmup_steps", 0)),
            device=device,
        )
        shadow = load_file(str(path))
        target_device = ema.device
        ema.shadow = {
            k: (v.to(target_device) if target_device is not None else v).float()
            for k, v in shadow.items()
        }
        ema.num_updates = int(meta.get("num_updates", 0))
        return ema
