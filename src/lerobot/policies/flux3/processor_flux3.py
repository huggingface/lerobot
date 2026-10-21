# Copyright 2026 Black Forest Labs. All rights reserved.
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
"""Pre/post-processing pipelines of the ``flux3`` policy.

Order (the OpenPI / pi05 convention): raw -> relative actions (optional) -> normalize -> model ->
unnormalize -> absolute actions (optional). Grid cameras are resized to a common resolution here;
the model applies training augmentation, composites its canvas and maps ``[0, 1]`` to the VAE's
``[-1, 1]``. Normalization statistics (``dataset_stats``) are computed on relative actions when
``use_relative_actions`` is set, exactly as pi05 does.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import replace
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import TransitionKey
from lerobot.policies.flux3.configuration_flux3 import Flux3Config
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.processor.factory import make_default_policy_processor_steps, make_policy_processor_pipelines
from lerobot.processor.pipeline import (
    ActionProcessorStep,
    ObservationProcessorStep,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
)
from lerobot.processor.relative_action_processor import (
    AbsoluteActionsProcessorStep,
    RelativeActionsProcessorStep,
)
from lerobot.utils.constants import (
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


def make_flux3_pre_post_processors_from_pretrained(
    config: Flux3Config,
    pretrained_path: str,
    *,
    revision: str | None = None,
    preprocessor_overrides: dict[str, Any] | None = None,
    postprocessor_overrides: dict[str, Any] | None = None,
    preprocessor_config_filename: str = f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
    postprocessor_config_filename: str = f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
    **_: Any,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Load checkpoint-owned normalization and reconnect the policy's processor state."""
    preprocessor_overrides = dict(preprocessor_overrides or {})
    postprocessor_overrides = dict(postprocessor_overrides or {})
    if config.conditioning == "history":
        # History steps load their own quantiles; standard dataset-stat overrides do not apply.
        preprocessor_overrides.pop("normalizer_processor", None)
        postprocessor_overrides.pop("unnormalizer_processor", None)
    pre = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=pretrained_path,
        config_filename=preprocessor_config_filename,
        overrides=preprocessor_overrides,
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
        revision=revision,
    )
    post = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=pretrained_path,
        config_filename=postprocessor_config_filename,
        overrides=postprocessor_overrides,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
        revision=revision,
    )
    return connect_flux3_processors(config, pre, post)


def make_flux3_pre_post_processors(
    config: Flux3Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    normalization_stats: dict[str, dict[str, Any]] = dict(dataset_stats or {})
    steps = make_default_policy_processor_steps(config, normalization_stats, normalizer_device=config.device)
    resize = CameraResizeProcessorStep(config.camera_order, config.camera_layout)
    if config.conditioning == "history":
        shared = {
            "action_dim": config.action_dim,
            "action_representation": config.action_representation,
            "absolute_dims": config.delta_absolute_dims,
            "normalization_clip": config.normalization_clip,
        }
        normalizer = ObservationHistoryNormalizerProcessorStep(
            **shared,
            n_obs_steps=config.n_obs_steps,
            chunk_size=config.chunk_size,
            camera_keys=config.camera_order,
            condition_on_past_actions=config.condition_on_past_actions,
        )
        action_normalizer = ActionTargetNormalizerProcessorStep(
            **shared, n_obs_steps=config.n_obs_steps, chunk_size=config.chunk_size
        )
        unnormalizer = ActionHistoryUnnormalizerProcessorStep(**shared)
        if config.normalization_stats is None:
            raise ValueError(
                "Load saved history processors or export with explicit action/state quantiles; "
                "raw dataset stats cannot initialize command-delta normalization"
            )
        values = {
            f"{stream}.{q}": torch.tensor(config.normalization_stats[stream][q])
            for stream in ("state", "action")
            for q in ("q01", "q99")
        }
        normalizer.load_state_dict(values)
        action_values = {key: value for key, value in values.items() if key.startswith("action.")}
        action_normalizer.load_state_dict(action_values)
        unnormalizer.load_state_dict(action_values)
        pre, post = make_policy_processor_pipelines(
            input_steps=[
                steps.rename_observations,
                steps.add_batch_dim,
                steps.to_device,
                normalizer,
                action_normalizer,
                resize,
            ],
            output_steps=[unnormalizer, steps.to_cpu],
        )
        return connect_history_processors(config, pre, post)
    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=list(config.relative_exclude_joints),
        action_names=config.action_feature_names,
    )
    input_steps: list[ProcessorStep] = [
        steps.rename_observations,
        steps.add_batch_dim,
        relative_step,
        steps.normalize,
        steps.to_device,
        resize,
    ]
    output_steps: list[ProcessorStep] = [
        steps.unnormalize,
        AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step),
        steps.to_cpu,
    ]
    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)


PAST_ACTIONS = "observation.past_actions"


@ProcessorStepRegistry.register("flux3_camera_resize")
class CameraResizeProcessorStep(ObservationProcessorStep):
    """Resize mixed-resolution grid cameras to the largest height and width in the observation."""

    def __init__(self, camera_keys: list[str], camera_layout: str):
        self.camera_keys = list(camera_keys)
        self.camera_layout = camera_layout

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self.camera_layout != "grid":
            return observation
        images = [observation[key] for key in self.camera_keys]
        for key, image in zip(self.camera_keys, images, strict=True):
            if image.ndim not in (4, 5):
                raise ValueError(f"{key}: expected (B, C, H, W) or (B, T, C, H, W), got {tuple(image.shape)}")
        shapes = [(1, *image.shape[1:]) if image.ndim == 4 else image.shape[1:] for image in images]
        if len(set(shapes)) == 1:
            return observation
        if len({shape[:2] for shape in shapes}) != 1:
            raise ValueError(f"cameras must share (T, C), got {shapes}")
        hw = (max(image.shape[-2] for image in images), max(image.shape[-1] for image in images))
        for key, image in zip(self.camera_keys, images, strict=True):
            resized = F.interpolate(
                image.reshape(-1, *image.shape[-3:]).float(),
                size=hw,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).reshape(*image.shape[:-2], *hw)
            if image.dtype == torch.uint8:
                resized = resized.round_().clamp_(0, 255).to(torch.uint8)
            observation[key] = resized
        return observation

    def get_config(self) -> dict[str, Any]:
        return {"camera_keys": self.camera_keys, "camera_layout": self.camera_layout}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        if self.camera_layout != "grid":
            return features
        observations = dict(features[PipelineFeatureType.OBSERVATION])
        hw = tuple(max(observations[key].shape[axis] for key in self.camera_keys) for axis in (-2, -1))
        for key in self.camera_keys:
            feature = observations[key]
            observations[key] = replace(feature, shape=(*feature.shape[:-2], *hw))
        return {**features, PipelineFeatureType.OBSERVATION: observations}


class _HistoryQuantiles(ProcessorStep):
    """Checkpoint-owned quantiles in the configured action representation."""

    def __init__(
        self, action_dim, action_representation="absolute", absolute_dims=None, normalization_clip=6.0
    ):
        self.action_dim = action_dim
        self.action_representation = action_representation
        self.absolute_dims = list(absolute_dims or [])
        self.normalization_clip = normalization_clip
        self._quantiles = {}
        if action_dim < 1:
            raise ValueError("action_dim must be positive")
        if action_representation not in ("absolute", "delta"):
            raise ValueError("action_representation must be absolute or delta")
        if any(not -action_dim <= d < action_dim for d in self.absolute_dims):
            raise ValueError("absolute_dims must be within the action dimension")
        if len({d % action_dim for d in self.absolute_dims}) != len(self.absolute_dims):
            raise ValueError("absolute_dims must identify distinct channels")
        if action_representation != "delta" and self.absolute_dims:
            raise ValueError("absolute_dims requires delta actions")
        if not math.isfinite(normalization_clip) or normalization_clip <= 0:
            raise ValueError("normalization_clip must be finite and positive")

    def get_config(self):
        return {
            "action_dim": self.action_dim,
            "action_representation": self.action_representation,
            "absolute_dims": self.absolute_dims,
            "normalization_clip": self.normalization_clip,
        }

    def state_dict(self):
        self.validate_statistics()
        return {key: value.clone() for key, value in self._quantiles.items()}

    def load_state_dict(self, state):
        required = {f"{stream}.{q}" for stream in self.streams for q in ("q01", "q99")}
        if set(state) != required:
            raise ValueError(f"History processor requires saved quantiles: {sorted(required)}")
        quantiles = {key: value.detach().cpu().float().clone() for key, value in state.items()}
        for key, value in quantiles.items():
            if value.shape != (self.action_dim,) or not torch.isfinite(value).all():
                raise ValueError(f"{key} must contain {self.action_dim} finite quantiles")
        for stream in self.streams:
            if (quantiles[f"{stream}.q99"] < quantiles[f"{stream}.q01"]).any():
                raise ValueError(f"{stream} quantiles are inverted")
        self._quantiles = quantiles

    def validate_statistics(self):
        if not self._quantiles:
            raise ValueError("History processor is missing saved quantiles; re-export the model package")

    def scale(self, value, stream, *, inverse=False):
        self.validate_statistics()
        lo = self._quantiles[f"{stream}.q01"].to(value)
        hi = self._quantiles[f"{stream}.q99"].to(value)
        span = torch.where(hi - lo > 1e-6, hi - lo, torch.ones_like(lo))
        if inverse:
            return (value + 1) * span / 2 + lo
        return (2 * (value - lo) / span - 1).clamp(-self.normalization_clip, self.normalization_clip)

    def _normalized_commands(self, commands):
        values = commands[:, 1:]
        if self.action_representation == "delta":
            values = values - commands[:, :-1]
            values[..., self.absolute_dims] = commands[:, 1:, self.absolute_dims]
        return self.scale(values, "action")

    def _validate_command_window(self, commands):
        if commands.ndim != 3 or commands.shape[1:] != (
            self.n_obs_steps + self.chunk_size,
            self.action_dim,
        ):
            raise ValueError("History training requires the full absolute command window")

    def transform_features(self, features):
        return features


@ProcessorStepRegistry.register("flux3_observation_history_normalizer")
class ObservationHistoryNormalizerProcessorStep(_HistoryQuantiles, ObservationProcessorStep):
    """Prepare training windows or synchronous image/state/command history.

    Commands are absolute at the pipeline boundary. Quantiles describe the selected
    representation: consecutive command deltas (with explicit absolute channels) or
    absolute commands. Reset both pipelines and the policy between episodes.
    """

    streams = ("state", "action")

    def __init__(
        self,
        action_dim,
        n_obs_steps=8,
        chunk_size=32,
        camera_keys=None,
        action_representation="absolute",
        absolute_dims=None,
        normalization_clip=6.0,
        condition_on_past_actions=False,
    ):
        super().__init__(action_dim, action_representation, absolute_dims, normalization_clip)
        self.n_obs_steps = n_obs_steps
        self.chunk_size = chunk_size
        self.camera_keys = list(camera_keys or [])
        self.condition_on_past_actions = condition_on_past_actions
        if n_obs_steps < 1 or chunk_size < 1:
            raise ValueError("Invalid history/chunk length")
        self.reset()

    def reset(self):
        self.observations = deque(maxlen=self.n_obs_steps)
        self.commands = deque(maxlen=self.n_obs_steps)
        self.last_command = None
        self.output_anchor = None

    def get_config(self):
        return {
            **super().get_config(),
            "n_obs_steps": self.n_obs_steps,
            "chunk_size": self.chunk_size,
            "camera_keys": self.camera_keys,
            "condition_on_past_actions": self.condition_on_past_actions,
        }

    def _past(self, commands):
        if commands.ndim != 3 or commands.shape[1:] != (self.n_obs_steps, self.action_dim):
            raise ValueError(f"Expected (B, {self.n_obs_steps}, {self.action_dim}) absolute command history")
        return torch.cat([torch.zeros_like(commands[:, :1]), self._normalized_commands(commands)], 1)

    def observation(self, obs):
        self.validate_statistics()
        if PAST_ACTIONS in obs:
            raise ValueError("History inputs have already been processed")
        state = obs[OBS_STATE].float()
        actions = self.transition.get(TransitionKey.ACTION)
        commands = obs.get("observation.command_history")
        past = None
        if actions is not None:
            self._validate_command_window(actions)
            past = self._past(actions.float()[:, : self.n_obs_steps])
        else:
            if commands is None and (state.ndim == 2 or (state.ndim == 3 and state.shape[1] == 1)):
                state = state[:, -1] if state.ndim == 3 else state
                if state.shape[-1] != self.action_dim:
                    raise ValueError(f"Expected {self.action_dim} measured state channels")
                if self.last_command is None:
                    self.last_command = state.clone()
                current = {OBS_STATE: state.clone()}
                for key in self.camera_keys:
                    value = obs[key]
                    if value.ndim == 5:
                        if value.shape[1] != 1:
                            raise ValueError("Synchronous input needs one image per tick")
                        value = value[:, 0]
                    current[key] = value.clone()
                repeats = self.n_obs_steps if not self.observations else 1
                for _ in range(repeats):
                    self.observations.append(current)
                    self.commands.append(self.last_command.clone())
                for key in current:
                    obs[key] = torch.stack([item[key] for item in self.observations], 1)
                state = obs[OBS_STATE]
                commands = torch.stack(list(self.commands), 1)
            if commands is None:
                if self.condition_on_past_actions or self.action_representation == "delta":
                    raise ValueError("Offline prediction requires observation.command_history")
            else:
                commands = commands.float()
                past = self._past(commands)
                self.output_anchor = commands[:, -1].clone()
        if state.ndim != 3 or state.shape[1:] != (self.n_obs_steps, self.action_dim):
            raise ValueError("History conditioning requires a complete state history")
        obs[OBS_STATE] = self.scale(state, "state")
        if self.condition_on_past_actions:
            obs[PAST_ACTIONS] = past
        obs.pop("observation.command_history", None)
        return obs


@ProcessorStepRegistry.register("flux3_action_target_normalizer")
class ActionTargetNormalizerProcessorStep(_HistoryQuantiles, ActionProcessorStep):
    """Convert an absolute training command window into normalized action targets."""

    streams = ("action",)

    def __init__(
        self,
        action_dim,
        n_obs_steps=8,
        chunk_size=32,
        action_representation="absolute",
        absolute_dims=None,
        normalization_clip=6.0,
    ):
        super().__init__(action_dim, action_representation, absolute_dims, normalization_clip)
        self.n_obs_steps = n_obs_steps
        self.chunk_size = chunk_size
        if n_obs_steps < 1 or chunk_size < 1:
            raise ValueError("Invalid history/chunk length")

    def get_config(self):
        return {**super().get_config(), "n_obs_steps": self.n_obs_steps, "chunk_size": self.chunk_size}

    def __call__(self, transition):
        # Inference supplies observations without target actions.
        if transition.get(TransitionKey.ACTION) is None:
            return transition.copy()
        return super().__call__(transition)

    def action(self, action):
        self._validate_command_window(action)
        return self._normalized_commands(action.float())[:, self.n_obs_steps - 1 :]


@ProcessorStepRegistry.register("flux3_action_history_unnormalizer")
class ActionHistoryUnnormalizerProcessorStep(_HistoryQuantiles, ActionProcessorStep):
    """Undo normalization and, when configured, integrate consecutive command deltas."""

    streams = ("action",)

    def __init__(
        self, action_dim, action_representation="absolute", absolute_dims=None, normalization_clip=6.0
    ):
        super().__init__(action_dim, action_representation, absolute_dims, normalization_clip)
        self.history = None

    def reset(self):
        if self.history is not None:
            self.history.reset()

    def action(self, action):
        self.validate_statistics()
        if self.history is None:
            raise ValueError("History postprocessing requires the paired preprocessor")
        if action.ndim not in (2, 3) or action.shape[-1] != self.action_dim:
            raise ValueError(f"Predictions must be (B, {self.action_dim}) or (B, T, {self.action_dim})")
        values = self.scale(action.float(), "action", inverse=True)
        commands = values.clone()
        if self.action_representation == "delta":
            if self.history.output_anchor is None:
                raise ValueError("Delta postprocessing requires the preprocessor's command history")
            anchor = self.history.output_anchor.to(values)
            absolute = {d % self.action_dim for d in self.absolute_dims}
            dims = [d for d in range(self.action_dim) if d not in absolute]
            if values.ndim == 3:
                commands[..., dims] = values[..., dims].cumsum(1) + anchor[:, None, dims]
            else:
                commands[..., dims] += anchor[..., dims]
        if values.ndim == 2:
            self.history.last_command = commands.clone()
        return commands


def connect_history_processors(config, pre, post):
    observations = [step for step in pre.steps if isinstance(step, ObservationHistoryNormalizerProcessorStep)]
    actions = [step for step in pre.steps if isinstance(step, ActionTargetNormalizerProcessorStep)]
    outputs = [step for step in post.steps if isinstance(step, ActionHistoryUnnormalizerProcessorStep)]
    if len(observations) != 1 or len(actions) != 1 or len(outputs) != 1:
        raise ValueError("History checkpoint requires its saved pre/postprocessor steps")
    history, action, output = observations[0], actions[0], outputs[0]
    if pre.steps.index(history) > pre.steps.index(action):
        raise ValueError("Observation history must be processed before action targets")
    for step in (history, action, output):
        step.validate_statistics()
        if (
            step.action_dim != config.action_dim
            or step.action_representation != config.action_representation
            or step.absolute_dims != config.delta_absolute_dims
            or step.normalization_clip != config.normalization_clip
        ):
            raise ValueError("History processor settings disagree with the policy")
    for step in (history, action):
        if step.n_obs_steps != config.n_obs_steps or step.chunk_size != config.chunk_size:
            raise ValueError("History processor settings disagree with the policy")
    if history.condition_on_past_actions != config.condition_on_past_actions:
        raise ValueError("History processor settings disagree with the policy")
    for step in (action, output):
        for key, value in step.state_dict().items():
            if not torch.equal(value, history.state_dict()[key]):
                raise ValueError("Pre/postprocessor action quantiles disagree")
    # Camera selection follows the policy, including overrides when adapting a saved base.
    history.camera_keys = list(config.camera_order)
    output.history = history
    return pre, post


def connect_flux3_processors(config, pre, post):
    """Reconnect saved history state and apply the policy's current camera selection."""
    if config.conditioning == "history":
        connect_history_processors(config, pre, post)
    else:
        relative = next((step for step in pre.steps if isinstance(step, RelativeActionsProcessorStep)), None)
        for step in post.steps:
            if isinstance(step, AbsoluteActionsProcessorStep) and step.relative_step is None:
                step.relative_step = relative
    resizers = [step for step in pre.steps if isinstance(step, CameraResizeProcessorStep)]
    if len(resizers) != 1:
        raise ValueError("FLUX3 requires its saved camera processor; re-export the model package")
    resizers[0].camera_keys = list(config.camera_order)
    resizers[0].camera_layout = config.camera_layout
    return pre, post
