# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Checkpoint helpers for LingBot-VLA 2.0.

The released upstream checkpoint (`robbyant/lingbot-vla-v2-6b`) is not saved in
LeRobot's `config.json` + single `model.safetensors` format. It is a sharded
safetensors checkpoint with only a minimal upstream config. These helpers keep
raw-upstream detection and load-key validation separate from the heavy modeling
module so CI can test them without importing Transformers or downloading weights.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

LINGBOT_VLA_V2_UPSTREAM_REPO_ID = "robbyant/lingbot-vla-v2-6b"
LINGBOT_VLA_V2_SAFE_WEIGHTS_INDEX = "model.safetensors.index.json"
LINGBOT_VLA_V2_SAFE_WEIGHTS = "model.safetensors"

LINGBOT_VLA_V2_UPSTREAM_CONFIG_OVERRIDES = {
    "use_moe": True,
    "token_moe_layers": list(range(36)),
    "token_num_experts": 32,
    "token_top_k": 4,
    "token_moe_intermediate_size": 512,
    "token_shared_intermediate_size": 704,
    "expert_hidden_size": 768,
    "router_activation": "sigmoid",
    "routed_scaling_factor": 4.0,
    "use_shared_expert_gate": False,
    "moe_implementation": "fused",
    "use_depth": False,
    "max_action_dim": 55,
    "max_state_dim": 55,
}

_ALLOWED_UPSTREAM_UNEXPECTED_PREFIXES = (
    "model.current_video_align_",
    "model.future_video_align_",
    "model.depth_align_",
    "model.future_depth_align_",
    "model.current_shared_task_proj.",
    "model.future_shared_task_proj.",
)


def is_raw_lingbot_vla_v2_checkpoint(model_path: str | Path | None) -> bool:
    """Return whether a path/repo id looks like the raw upstream v2 checkpoint."""
    if model_path is None:
        return False

    model_id = str(model_path)
    if model_id.rstrip("/") == LINGBOT_VLA_V2_UPSTREAM_REPO_ID:
        return True

    path = Path(model_path).expanduser()
    if not path.is_dir():
        return False

    index_path = path / LINGBOT_VLA_V2_SAFE_WEIGHTS_INDEX
    if not index_path.exists():
        return False

    config_path = path / "config.json"
    if not config_path.exists():
        return True

    try:
        with config_path.open() as f:
            config = json.load(f)
    except json.JSONDecodeError:
        return False

    # LeRobot checkpoints include `type`; the raw upstream config currently only
    # carries `vlm_family`. Keep this conservative so a LeRobot sharded checkpoint
    # is not mistaken for raw upstream.
    return config.get("type") is None and config.get("vlm_family") == "qwen3_vl"


def apply_lingbot_vla_v2_upstream_config(config):
    """Mutate `config` so its architecture matches the released upstream 6B weights."""
    for key, value in LINGBOT_VLA_V2_UPSTREAM_CONFIG_OVERRIDES.items():
        setattr(config, key, value)
    config._moe_implementation = config.moe_implementation
    return config


def remap_lingbot_vla_v2_upstream_key(key: str) -> str:
    """Map an upstream tensor key to the LeRobot policy key.

    The released checkpoint already stores tensors under the LeRobot wrapper's
    `model.` prefix. This function intentionally remains explicit so future raw
    checkpoints can add mappings without changing the loader contract.
    """
    return key


def is_allowed_lingbot_vla_v2_upstream_unexpected_key(key: str, *, use_depth: bool = False) -> bool:
    """Return whether an upstream-only tensor may be skipped by the LeRobot action path."""
    if use_depth:
        return False
    return key.startswith(_ALLOWED_UPSTREAM_UNEXPECTED_PREFIXES)


def filter_lingbot_vla_v2_upstream_unexpected_keys(
    keys: Iterable[str], *, use_depth: bool = False
) -> tuple[list[str], list[str]]:
    """Split unexpected raw-upstream keys into allowed-skipped and hard unexpected lists."""
    allowed: list[str] = []
    unexpected: list[str] = []
    for key in keys:
        if is_allowed_lingbot_vla_v2_upstream_unexpected_key(key, use_depth=use_depth):
            allowed.append(key)
        else:
            unexpected.append(key)
    return allowed, unexpected


def validate_lingbot_vla_v2_upstream_loading_keys(
    missing_keys: Iterable[str],
    unexpected_keys: Iterable[str],
    *,
    use_depth: bool = False,
) -> tuple[list[str], list[str]]:
    """Validate load-state keys and return `(missing, allowed_skipped)` for logging."""
    missing = sorted(missing_keys)
    allowed_skipped, hard_unexpected = filter_lingbot_vla_v2_upstream_unexpected_keys(
        sorted(unexpected_keys), use_depth=use_depth
    )
    if missing or hard_unexpected:
        message_parts = []
        if missing:
            preview = ", ".join(missing[:5])
            message_parts.append(f"missing required keys ({len(missing)}): {preview}")
        if hard_unexpected:
            preview = ", ".join(hard_unexpected[:5])
            message_parts.append(f"unexpected non-whitelisted keys ({len(hard_unexpected)}): {preview}")
        raise RuntimeError("LingBot-VLA 2.0 raw checkpoint load failed: " + "; ".join(message_parts))
    return missing, allowed_skipped
