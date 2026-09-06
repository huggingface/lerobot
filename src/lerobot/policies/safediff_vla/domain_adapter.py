"""Normalization and controller-domain bridge for a pretrained LIBERO backbone."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_STATE

from lerobot.action_semantics import (
    LIBERO_ACTION_CONTRACT,
    LIBERO_SAFETY_ENV_ACTION_CONTRACT,
    convert_action,
)


def load_processor_normalization_stats(model_id: str | Path) -> dict[str, dict[str, Tensor]]:
    """Load mean/std tensors from a policy's serialized normalizer step."""
    model_id = str(model_id)
    config_name = "policy_preprocessor.json"
    if os.path.isdir(model_id):
        config_path = Path(model_id) / config_name
    else:
        config_path = Path(hf_hub_download(model_id, config_name))
    config = json.loads(config_path.read_text())
    normalizer = next(
        (step for step in config["steps"] if step.get("registry_name") == "normalizer_processor"), None
    )
    if normalizer is None or "state_file" not in normalizer:
        raise ValueError(f"No stateful normalizer_processor found in {model_id}/{config_name}")
    state_file = normalizer["state_file"]
    if os.path.isdir(model_id):
        state_path = Path(model_id) / state_file
    else:
        state_path = Path(hf_hub_download(model_id, state_file))
    flat = load_file(state_path)
    result: dict[str, dict[str, Tensor]] = {}
    for flat_key, value in flat.items():
        feature, statistic = flat_key.rsplit(".", 1)
        result.setdefault(feature, {})[statistic] = value.float()
    return result


def _stat(stats: dict[str, dict[str, Any]], feature: str, statistic: str, size: int) -> Tensor:
    try:
        value = torch.as_tensor(stats[feature][statistic], dtype=torch.float32).reshape(-1)
    except KeyError as error:
        raise ValueError(f"Missing normalization statistic {feature}.{statistic}") from error
    if value.numel() != size:
        raise ValueError(f"Expected {feature}.{statistic} to contain {size} values, got {value.numel()}")
    return value


class LiberoBackboneDomainAdapter(nn.Module):
    """Bridge Safety-normalized tensors and a regular-LIBERO SmolVLA backbone."""

    def __init__(
        self,
        source_stats: dict[str, dict[str, Any]],
        target_stats: dict[str, dict[str, Any]],
        *,
        state_dim: int,
        action_dim: int,
        semantics: ConversionSemantics = "per_step",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if action_dim != 7:
            raise ValueError("LIBERO domain adaptation requires a 7-D action")
        self.semantics = semantics
        self.eps = eps
        for prefix, stats in (("source", source_stats), ("target", target_stats)):
            for feature, size in ((OBS_STATE, state_dim), (ACTION, action_dim)):
                for statistic in ("mean", "std"):
                    self.register_buffer(
                        f"{prefix}_{'state' if feature == OBS_STATE else 'action'}_{statistic}",
                        _stat(stats, feature, statistic, size),
                    )

    @staticmethod
    def _like(value: Tensor, reference: Tensor) -> Tensor:
        return value.to(device=reference.device, dtype=reference.dtype)

    def observation_for_backbone(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Undo target normalization and apply the backbone's source normalization."""
        state = batch[OBS_STATE]
        target_mean = self._like(self.target_state_mean, state)
        target_std = self._like(self.target_state_std, state)
        source_mean = self._like(self.source_state_mean, state)
        source_std = self._like(self.source_state_std, state)
        raw_state = state * target_std + target_mean
        result = dict(batch)
        result[OBS_STATE] = (raw_state - source_mean) / (source_std + self.eps)
        return result

    def nominal_for_target(self, source_normalized_action: Tensor) -> Tensor:
        """Map backbone-normalized actions into target-normalized Safety actions."""
        source_mean = self._like(self.source_action_mean, source_normalized_action)
        source_std = self._like(self.source_action_std, source_normalized_action)
        target_mean = self._like(self.target_action_mean, source_normalized_action)
        target_std = self._like(self.target_action_std, source_normalized_action)
        source_command = source_normalized_action * source_std + source_mean
        target_command = convert_action(
            source_command,
            LIBERO_ACTION_CONTRACT,
            LIBERO_SAFETY_ENV_ACTION_CONTRACT,
            semantics=self.semantics,
        )
        return (target_command - target_mean) / (target_std + self.eps)
