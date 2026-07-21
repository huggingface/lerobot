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

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.policies.vla_jepa.configuration_vla_jepa import VLAJEPAConfig
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    EnvTransition,
    ObservationProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    TransitionKey,
    UnnormalizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)


@ProcessorStepRegistry.register(name="vla_jepa_image_prep")
class ImagePrepProcessorStep(ObservationProcessorStep):
    """Prepares image observations for the VLA-JEPA model: float cast, 1->3 channel expand, resize.

    This makes explicit (in the serialized pipeline) the image prep the model used to do
    internally. The model keeps the same operations as idempotent guards, so:
      - checkpoints saved WITHOUT this step (older uploads) are unaffected — the model still
        does the prep;
      - checkpoints saved WITH this step get it done here, and the model-side guards no-op.

    Mirrors `Qwen3VLInterface.to_pixel_values` + the `F.interpolate(mode="area")` resize in
    `VLAJEPAPolicy._prepare_model_inputs`/`predict_action`. Deliberately does NOT clamp (the
    model path doesn't), so values stay bit-identical. Handles [C,H,W], [B,C,H,W]/[T,C,H,W]
    and [B,T,C,H,W] image tensors.
    """

    def __init__(self, resize_to: tuple[int, int] | None = None, expand_channels: bool = True):
        self.resize_to = tuple(resize_to) if resize_to is not None else None
        self.expand_channels = expand_channels

    def observation(self, observation: dict) -> dict:
        new_observation = dict(observation)
        for key in observation:
            if "image" not in key:
                continue
            image = observation[key].float()
            if self.expand_channels and image.shape[-3] == 1:
                repeats = [1] * image.ndim
                repeats[-3] = 3
                image = image.repeat(*repeats)
            if self.resize_to is not None and tuple(image.shape[-2:]) != self.resize_to:
                device = image.device
                # NOTE: no "area" kernel on mps; resize on cpu then move back.
                if device.type == "mps":
                    image = image.cpu()
                lead = image.shape[:-3]
                c, h, w = image.shape[-3:]
                flat = image.reshape(-1, c, h, w)
                flat = F.interpolate(flat, size=self.resize_to, mode="area")
                image = flat.reshape(*lead, c, *self.resize_to).to(device)
            new_observation[key] = image
        return new_observation

    def get_config(self) -> dict[str, Any]:
        return {
            "resize_to": list(self.resize_to) if self.resize_to is not None else None,
            "expand_channels": self.expand_channels,
        }

    def transform_features(self, features):
        for key in features[PipelineFeatureType.OBSERVATION]:
            if "image" not in key:
                continue
            feat = features[PipelineFeatureType.OBSERVATION][key]
            # Match `to_pixel_values`: only a single channel is expanded to 3.
            nb_channel = 3 if (self.expand_channels and feat.shape[0] == 1) else feat.shape[0]
            spatial = self.resize_to if self.resize_to is not None else tuple(feat.shape[1:])
            features[PipelineFeatureType.OBSERVATION][key] = PolicyFeature(
                type=feat.type, shape=(nb_channel, *spatial)
            )
        return features


@ProcessorStepRegistry.register(name="vla_jepa_clip_actions")
class ClipActionsProcessorStep(ProcessorStep):
    """Clips action tensor to [-1, 1] before unnormalization."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is not None:
            transition = dict(transition)
            transition[TransitionKey.ACTION] = action.clamp(-1.0, 1.0)
        return transition

    def transform_features(self, features):
        return features


@ProcessorStepRegistry.register(name="vla_jepa_pre_snap_gripper")
class PreSnapGripperProcessorStep(ProcessorStep):
    """Snaps a gripper dimension to {0, 1} BEFORE unnormalization.

    Mirrors the original starVLA LIBERO eval:
      normalized[:, gripper_dim] = np.where(normalized[:, gripper_dim] < threshold, 0, 1)
    This ensures the unnormalizer receives an exact binary value, which is
    required when the model was trained with gripper in identity (mask=False)
    space where 0=open and 1=close.
    """

    def __init__(self, gripper_dim: int = 6, threshold: float = 0.5):
        self.gripper_dim = gripper_dim
        self.threshold = threshold

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is not None and action.shape[-1] > self.gripper_dim:
            transition = dict(transition)
            a = action.clone()
            a[..., self.gripper_dim] = (a[..., self.gripper_dim] >= self.threshold).float()
            transition[TransitionKey.ACTION] = a
        return transition

    def transform_features(self, features):
        return features


@ProcessorStepRegistry.register(name="vla_jepa_binarize_gripper")
class BinarizeGripperProcessorStep(ProcessorStep):
    """Binarizes a gripper dimension after unnormalization.

    Maps continuous value to {-1, 1}: > threshold → -1, <= threshold → 1 (matches starVLA convention).
    Only applied when action has more dimensions than gripper_dim.
    """

    def __init__(self, gripper_dim: int = 6, threshold: float = 0.5):
        self.gripper_dim = gripper_dim
        self.threshold = threshold

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is not None and action.shape[-1] > self.gripper_dim:
            transition = dict(transition)
            a = action.clone()
            a[..., self.gripper_dim] = 1.0 - 2.0 * (a[..., self.gripper_dim] > self.threshold).float()
            transition[TransitionKey.ACTION] = a
        return transition

    def transform_features(self, features):
        return features


def make_vla_jepa_pre_post_processors(
    config: VLAJEPAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    features = {**config.input_features, **config.output_features}
    steps = make_default_policy_processor_steps(config, dataset_stats)

    # Shared relative-action step (OpenPI order: raw -> relative -> normalize -> model ->
    # unnormalize -> absolute). The SAME instance is passed to AbsoluteActionsProcessorStep
    # below so its cached raw state (set during preprocessing) flows to postprocessing.
    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=getattr(config, "relative_exclude_joints", []),
        action_names=getattr(config, "action_feature_names", None),
    )

    input_steps = [
        steps.rename_observations,
        steps.add_batch_dim,
        steps.to_device,
        ImagePrepProcessorStep(
            resize_to=tuple(config.resize_images_to) if config.resize_images_to else None,
        ),
        relative_step,
        steps.normalize,
    ]
    output_steps: list[ProcessorStep] = []
    if config.clip_normalized_actions:
        output_steps.append(ClipActionsProcessorStep())
    if config.pre_snap_gripper_action:
        output_steps.append(
            PreSnapGripperProcessorStep(gripper_dim=config.gripper_dim, threshold=config.gripper_threshold)
        )
    # NOTE: unlike the default policy unnormalizer (output features only), VLA-JEPA
    # unnormalizes over BOTH input and output features.
    output_steps.append(
        UnnormalizerProcessorStep(
            features=features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        )
    )
    # Reverse the relative conversion on the unnormalized action, before gripper binarization.
    # gripper is kept absolute by relative_exclude_joints, so the two steps touch disjoint dims.
    output_steps.append(
        AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step)
    )
    if config.binarize_gripper_action:
        output_steps.append(
            BinarizeGripperProcessorStep(gripper_dim=config.gripper_dim, threshold=config.gripper_threshold)
        )
    output_steps.append(steps.to_cpu)
    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
