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

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    ComplementaryDataProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)

from .configuration_imprint import ImprintConfig


@dataclass
@ProcessorStepRegistry.register(name="imprint_episode_progress")
class ImprintEpisodeProgressStep(ComplementaryDataProcessorStep):
    """Fill ``episode_progress`` from the current frame and the episode length.

    Progress is the current ``frame_index`` divided by the last frame index of
    that episode. A batch that already has ``episode_progress`` is left as it is.
    """

    episode_indices: list[int]
    episode_lengths: list[int]

    def get_config(self) -> dict[str, Any]:
        return {
            "episode_indices": list(self.episode_indices),
            "episode_lengths": list(self.episode_lengths),
        }

    def complementary_data(self, complementary_data: dict[str, Any]) -> dict[str, Any]:
        if "episode_progress" in complementary_data:
            return complementary_data
        frame = complementary_data.get("frame_index")
        episode = complementary_data.get("episode_index")
        if not isinstance(frame, Tensor) or not isinstance(episode, Tensor):
            return complementary_data
        lookup = {
            int(index): int(length)
            for index, length in zip(self.episode_indices, self.episode_lengths, strict=True)
        }
        episode_ids = [int(value) for value in episode.detach().cpu().reshape(-1).tolist()]
        missing = sorted({ep for ep in episode_ids if ep not in lookup})
        if missing:
            raise ValueError(
                "IMPRINT stage 1 has no episode length for episode_index "
                f"{missing}. The dataset metadata passed to the processor is incomplete."
            )
        lengths = torch.tensor(
            [lookup[ep] for ep in episode_ids], dtype=torch.float32, device=frame.device
        ).reshape(frame.shape)
        denom = torch.clamp(lengths - 1.0, min=1.0)
        progress = frame.to(dtype=torch.float32) / denom
        complementary_data["episode_progress"] = progress.clamp(0.0, 1.0)
        return complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_imprint_pre_post_processors(
    config: ImprintConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    dataset_meta: Any | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Normalize, batch, and place IMPRINT observations. Unnormalize actions back to CPU.

    Stage 1 also writes ``episode_progress`` when episode lengths are available.
    ``lerobot-train`` stores those lengths on ``config._runtime_dataset_meta``.
    """
    steps = make_default_policy_processor_steps(config, dataset_stats, normalizer_device=config.device)
    input_steps: list[Any] = [
        steps.rename_observations,
        steps.add_batch_dim,
        steps.to_device,
        steps.normalize,
    ]
    progress = _episode_progress_step(config, dataset_meta)
    if progress is not None:
        input_steps.append(progress)
    return make_policy_processor_pipelines(
        input_steps=input_steps,
        output_steps=[steps.unnormalize, steps.to_cpu],
    )


def _episode_progress_step(
    config: ImprintConfig, dataset_meta: Any | None
) -> ImprintEpisodeProgressStep | None:
    if config.training_stage != "memory":
        return None
    if dataset_meta is None:
        dataset_meta = getattr(config, "_runtime_dataset_meta", None)
    if dataset_meta is None:
        return None
    episodes = getattr(dataset_meta, "episodes", None)
    if episodes is None:
        return None
    indices: list[int] = []
    lengths: list[int] = []
    for index in range(len(episodes)):
        row = episodes[index]
        indices.append(int(row["episode_index"]))
        lengths.append(int(row["length"]))
    if not indices:
        return None
    return ImprintEpisodeProgressStep(episode_indices=indices, episode_lengths=lengths)
