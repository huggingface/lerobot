# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LeRobot processor persistence with the released Long-WAM normalization rules."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from lerobot.configs import FeatureType, NormalizationMode, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    ActionProcessorStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    UnnormalizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)

from .configuration_longwam import LongWAMConfig

FeatureMap = dict[PipelineFeatureType, dict[str, PolicyFeature]]
Stats = dict[str, dict[str, torch.Tensor]]


class _LongWAMNormalization:
    norm_map: dict[FeatureType, NormalizationMode] | dict[str, NormalizationMode]
    _tensor_stats: Stats

    def _apply_transform(
        self, tensor: torch.Tensor, key: str, feature_type: FeatureType, *, inverse: bool = False
    ) -> torch.Tensor:
        mode = self.norm_map.get(feature_type, NormalizationMode.IDENTITY)
        if mode == NormalizationMode.IDENTITY:
            return tensor
        if mode != NormalizationMode.MIN_MAX:
            raise ValueError("The first Long-WAM port supports checkpoint MIN_MAX normalization only")
        if key not in self._tensor_stats:
            raise ValueError(f"Missing Long-WAM normalization statistics for {key}")
        stats = self._tensor_stats[key]
        if "min" not in stats or "max" not in stats:
            raise ValueError(f"Missing min/max statistics for {key}")
        low, high = (stats[name].to(tensor) for name in ("min", "max"))
        width = high - low
        constant = width < 1e-4
        scale = 2 / torch.where(constant, 2.0, width)
        offset = torch.where(constant, -low, -1 - scale * low)
        # Match the release: constant channels map to zero, and only the forward
        # transformation clips to [-5,5]. The generic MIN_MAX step differs here.
        return (tensor - offset) / scale if inverse else (tensor * scale + offset).clamp(-5, 5)


@ProcessorStepRegistry.register(name="longwam_normalizer")
@dataclass
class LongWAMNormalizer(_LongWAMNormalization, NormalizerProcessorStep):
    pass


@ProcessorStepRegistry.register(name="longwam_unnormalizer")
@dataclass
class LongWAMUnnormalizer(_LongWAMNormalization, UnnormalizerProcessorStep):
    pass


@ProcessorStepRegistry.register(name="longwam_action_toggle")
@dataclass
class LongWAMActionToggle(ActionProcessorStep):
    dimensions: list[int]

    def action(self, action: torch.Tensor) -> torch.Tensor:
        result = action.clone()
        for dim in self.dimensions:
            if not -result.shape[-1] <= dim < result.shape[-1]:
                raise ValueError("Invalid gripper action dimension")
            result[..., dim] = torch.sign(1 - 2 * result[..., dim])
        return result

    def get_config(self) -> dict[str, Any]:
        return {"dimensions": self.dimensions}

    def transform_features(self, features: FeatureMap) -> FeatureMap:
        return features


def make_longwam_pre_post_processors(
    config: LongWAMConfig, dataset_stats: Stats | None = None
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    steps = make_default_policy_processor_steps(config, dataset_stats, normalizer_device=config.device)
    pre = [
        steps.rename_observations,
        steps.add_batch_dim,
        steps.to_device,
        LongWAMNormalizer(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            device=config.device,
        ),
    ]
    post: list[ProcessorStep] = [
        LongWAMUnnormalizer(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        )
    ]
    if config.toggle_action_dimensions:
        post.append(LongWAMActionToggle(config.toggle_action_dimensions))
    post.append(steps.to_cpu)
    return make_policy_processor_pipelines(input_steps=pre, output_steps=post)


def make_longwam_pre_post_processors_from_pretrained(
    config: LongWAMConfig,
    pretrained_path: str | Path,
    *,
    revision: str | None = None,
    dataset_stats: Stats | None = None,
    dataset_meta: Any = None,
    preprocessor_overrides: dict[str, Any] | None = None,
    postprocessor_overrides: dict[str, Any] | None = None,
    preprocessor_config_filename: str = "policy_preprocessor.json",
    postprocessor_config_filename: str = "policy_postprocessor.json",
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Translate LeRobot's standard train/resume overrides to our registered steps.

    Resume does not supply new statistics: the saved processor state remains
    authoritative. Fine-tuning supplies dataset_stats and updates both sides.
    """
    from lerobot.processor import load_pretrained_policy_processors

    def overrides(values: dict[str, Any] | None, standard: str, custom: str) -> dict[str, Any]:
        result = {key: dict(value) for key, value in (values or {}).items()}
        if standard in result:
            if custom in result:
                raise ValueError("Conflicting standard and Long-WAM normalizer overrides")
            result[custom] = dict(result.pop(standard))
        if dataset_stats is not None:
            result.setdefault(custom, {})["stats"] = dataset_stats
        return result

    return load_pretrained_policy_processors(
        pretrained_path,
        revision=revision,
        preprocessor_overrides=overrides(
            preprocessor_overrides, "normalizer_processor", "longwam_normalizer"
        ),
        postprocessor_overrides=overrides(
            postprocessor_overrides, "unnormalizer_processor", "longwam_unnormalizer"
        ),
        preprocessor_config_filename=preprocessor_config_filename,
        postprocessor_config_filename=postprocessor_config_filename,
    )
