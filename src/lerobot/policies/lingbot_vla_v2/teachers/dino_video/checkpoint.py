# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
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

"""Checkpoint loading for the first-party DINO-video runtime.

Contract with the published ``teacher_step_10000.pth`` (verified locally):

- loads with ``torch.load(..., weights_only=True)`` — plain tensors only;
- top-level ``teacher`` entry holds the state dict: 359 tensors = 345 backbone
  + 14 ``dino_head``/``ibot_head`` distillation-head tensors;
- backbone keys carry a leading ``backbone.`` prefix which is stripped here;
- unlike the depth teachers, no strict=False quirk is tolerated here: every
  backbone tensor must load exactly, and anything outside
  ``ALLOWED_UNUSED_PREFIXES`` that is missing, unexpected, or shape-mismatched
  is a hard error.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

#: Distillation heads present in the checkpoint but unused by
#: ``get_future_feature``; they must never enter the first-party runtime.
ALLOWED_UNUSED_PREFIXES: tuple[str, ...] = ("dino_head.", "ibot_head.")

_BACKBONE_PREFIX = "backbone."


@dataclass(frozen=True)
class LoadReport:
    """Outcome of a checkpoint restore, for tests and logs."""

    loaded_tensors: int
    unused_tensors: tuple[str, ...]
    checksum: str = ""


def load_dino_video_checkpoint(path: str | Path) -> dict[str, torch.Tensor]:
    """Read ``teacher_step_10000.pth`` and return the ``teacher`` state dict."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"DINO-video teacher checkpoint not found at {str(file_path)!r}.")
    raw = torch.load(file_path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict) or "teacher" not in raw:
        keys = list(raw)[:5] if isinstance(raw, dict) else type(raw).__name__
        raise ValueError(f"expected a top-level 'teacher' state dict in {str(file_path)!r}, got: {keys}")
    state = raw["teacher"]
    if not isinstance(state, dict):
        raise ValueError(f"'teacher' entry must be a state dict, got {type(state).__name__}.")
    non_tensors = sorted(name for name, tensor in state.items() if not isinstance(tensor, torch.Tensor))
    if non_tensors:
        raise ValueError(f"checkpoint 'teacher' entry holds non-tensor values: {non_tensors[:5]}.")
    return state


def load_backbone_strict(
    model: nn.Module, state: dict[str, torch.Tensor], *, strict: bool = True
) -> LoadReport:
    """Restore ``model`` from ``state`` with strict, total coverage.

    Every tensor of ``model.state_dict()`` must be provided by ``state`` (only
    the ``dino_head.*`` / ``ibot_head.*`` distillation heads may sit unused in
    the file). With ``strict=True`` (default) any other unexpected key is a
    hard error; with ``strict=False`` unexpected keys are reported as unused
    instead — missing or shape-mismatched backbone tensors always fail, because
    the runtime cannot execute without them.
    """
    model_state = model.state_dict()

    head_names = {name for name in state if name.startswith(ALLOWED_UNUSED_PREFIXES)}
    normalized: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        if name in head_names:
            continue
        bare = name.removeprefix(_BACKBONE_PREFIX) if name.startswith(_BACKBONE_PREFIX) else name
        if bare in normalized:
            raise ValueError(
                f"checkpoint provides both {bare!r} and {name!r}; refusing to guess the mapping."
            )
        normalized[bare] = tensor
    unused = sorted(head_names)

    unexpected = sorted(name for name in normalized if name not in model_state)
    if unexpected:
        if strict:
            raise ValueError(
                f"checkpoint holds {len(unexpected)} tensors that do not belong to the first-party "
                f"backbone (e.g. {unexpected[:5]}); only {list(ALLOWED_UNUSED_PREFIXES)} may be unused."
            )
        for name in unexpected:
            del normalized[name]
        unused = sorted(unused + unexpected)

    missing = sorted(name for name in model_state if name not in normalized)
    if missing:
        raise ValueError(
            f"checkpoint is missing {len(missing)} required backbone tensors (e.g. {missing[:5]}); "
            "the first-party runtime requires total coverage."
        )

    mismatched = [
        (name, tuple(normalized[name].shape), tuple(model_state[name].shape))
        for name in normalized
        if normalized[name].shape != model_state[name].shape
    ]
    if mismatched:
        details = "; ".join(
            f"{name}: ckpt {ckpt_shape} vs model {model_shape}"
            for name, ckpt_shape, model_shape in mismatched[:5]
        )
        raise ValueError(f"checkpoint/model shape mismatches: {details} ({len(mismatched)} total).")

    model.load_state_dict(normalized, strict=True)
    return LoadReport(
        loaded_tensors=len(normalized),
        unused_tensors=tuple(unused),
        checksum=_checksum(normalized),
    )


def _checksum(tensors: dict[str, torch.Tensor]) -> str:
    """Stable sha256 over tensor names and raw bytes (for logs and tests)."""
    digest = hashlib.sha256()
    for name in sorted(tensors):
        digest.update(name.encode("utf-8"))
        flat = tensors[name].detach().cpu().contiguous().reshape(-1)
        digest.update(flat.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
