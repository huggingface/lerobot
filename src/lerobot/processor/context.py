#!/usr/bin/env python

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

"""Build-time inputs for policy processor factories."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets import LeRobotDatasetMetadata

# Legacy step-override keys whose value now lives on the policy config, and the config field each
# one sets. These reach every policy's factory because every factory already receives the config.
_CONFIG_OWNED_OVERRIDES: dict[tuple[str, str], str] = {
    ("device_processor", "device"): "device",
    ("rename_observations_processor", "rename_map"): "rename_map",
}


@dataclass(frozen=True, kw_only=True)
class ProcessorBuildContext:
    """Per-run inputs a policy's processor factory needs that its config does not carry.

    This deliberately does **not** include `features`, `norm_map`, or anything else describing the
    *shape* of the pipeline: those belong to the policy's factory, the only thing that knows whether a
    policy reshapes tensors internally (EVO1 padding state to `max_state_dim`, for instance).
    Force-feeding dataset-derived shapes past a factory that pads is what made issue #4006 possible.

    Values that every policy's factory must see, like the device and the rename map, live on
    `PreTrainedConfig` instead — a factory already receives the config, so putting them there reaches
    all policies without changing 14 factory signatures.

    Frozen on purpose: a mistyped field is a `TypeError` at the call site, replacing the load-time
    `KeyError` that the step-override dicts used to raise.

    Attributes:
        dataset_stats: Statistics to normalize against. Supplying them makes the *dataset*
            authoritative over the checkpoint's saved stats; leaving them `None` keeps the
            checkpoint's. That is the whole finetune-vs-eval distinction — see
            `NormalizerProcessorStep._stats_explicitly_provided`.
        dataset_meta: Dataset metadata, for factories that derive steps from it.
        training: Whether the pipeline is being built for training. Explicit, rather than inferred
            from `dataset_meta is not None`.
        pretrained_path: Checkpoint the pipeline's state will be loaded from, if any.
        pretrained_revision: Hub revision of `pretrained_path`.
    """

    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None
    dataset_meta: LeRobotDatasetMetadata | None = None
    training: bool = False
    pretrained_path: str | None = None
    pretrained_revision: str | None = None

    @classmethod
    def from_legacy_kwargs(
        cls, kwargs: dict[str, Any], policy_cfg: PreTrainedConfig
    ) -> ProcessorBuildContext:
        """Build a context from the older `ProcessorConfigKwargs` shape, applying config-owned values.

        Kept so out-of-tree callers passing `preprocessor_overrides`/`postprocessor_overrides` keep
        working. Keys whose value now lives on the config are written to `policy_cfg`; keys describing
        pipeline shape are reported once and ignored, because the policy factory now decides them.
        """
        overrides: dict[str, Any] = {}
        for key in ("preprocessor_overrides", "postprocessor_overrides"):
            overrides.update(kwargs.get(key) or {})

        ignored: list[str] = []
        for step_key, step_overrides in overrides.items():
            for param, value in (step_overrides or {}).items():
                config_field = _CONFIG_OWNED_OVERRIDES.get((step_key, param))
                if config_field is not None:
                    setattr(policy_cfg, config_field, value)
                else:
                    ignored.append(f"{step_key}.{param}")

        dataset_stats = kwargs.get("dataset_stats")
        if dataset_stats is None:
            # A caller that passed stats only as a step override still means "normalize against
            # these", so honour it rather than silently dropping normalization.
            stats_override = (overrides.get("normalizer_processor") or {}).get("stats")
            if stats_override is not None:
                dataset_stats = stats_override
                ignored = [key for key in ignored if key != "normalizer_processor.stats"]

        if ignored:
            warnings.warn(
                "Step-level processor overrides are deprecated and were ignored: "
                f"{sorted(ignored)}. Pipeline structure and shape now come from the policy config "
                "and its processor factory; pass per-run values via ProcessorBuildContext.",
                DeprecationWarning,
                stacklevel=3,
            )

        return cls(
            dataset_stats=dataset_stats,
            dataset_meta=kwargs.get("dataset_meta"),
        )


def apply_checkpoint_rename_map(
    policy_cfg: PreTrainedConfig, preprocessor_config: dict[str, Any] | None
) -> None:
    """Fill an unset `policy_cfg.rename_map` from a checkpoint's serialized preprocessor.

    A rename map is a dataset-to-policy key binding that exists nowhere but the CLI, so before this
    refactor an eval run that passed none silently inherited the one baked into the saved pipeline.
    Rebuilding from the config would drop it, so it is carried over here — and only here. Everything
    else about the pipeline is rebuilt from the config on purpose.
    """
    if getattr(policy_cfg, "rename_map", None) or not preprocessor_config:
        return

    for step_entry in preprocessor_config.get("steps", []):
        if step_entry.get("registry_name") != "rename_observations_processor":
            continue
        rename_map = (step_entry.get("config") or {}).get("rename_map")
        if rename_map:
            logging.info("Using the rename map recorded in the checkpoint's preprocessor: %s", rename_map)
            policy_cfg.rename_map = dict(rename_map)
        return
