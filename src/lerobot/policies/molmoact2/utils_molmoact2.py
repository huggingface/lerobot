# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team. All rights reserved.
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

"""Shared dependency-free helpers for the MolmoAct2 policy package."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import Tensor


def hf_token() -> str | None:
    """Return the Hugging Face access token configured for this process."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HF_ACCESS_TOKEN")


def resolve_checkpoint_location(
    checkpoint_path: str,
    *,
    revision: str | None = None,
    force_download: bool = False,
) -> str:
    """Resolve a local or Hub checkpoint without downloading remote Python code."""
    checkpoint_path = str(checkpoint_path or "").strip()
    if not checkpoint_path:
        raise ValueError("MolmoAct2 policy requires `checkpoint_path`.")

    local_path = Path(checkpoint_path).expanduser()
    if local_path.exists():
        return str(local_path)

    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=checkpoint_path,
        repo_type="model",
        revision=revision,
        force_download=force_download,
        ignore_patterns=["*.py", "*.pyc", "__pycache__/*"],
        token=hf_token(),
    )


def position_ids_from_attention_mask(attention_mask: Tensor) -> Tensor:
    """Build padding-invariant positions matching native MolmoAct2 training."""
    if attention_mask.ndim != 2:
        raise ValueError(
            "MolmoAct2 position ids require a 2D attention mask, "
            f"got shape {tuple(attention_mask.shape)}."
        )
    return torch.clamp(torch.cumsum(attention_mask.to(torch.long), dim=-1) - 1, min=0)
