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

"""LeRobot preprocessing and postprocessing pipelines for LaWAM."""

from __future__ import annotations

from typing import Any

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    EnvTransition,
    ImageCropResizeProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    TransitionKey,
    UnnormalizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.utils.constants import OBS_STATE

from .configuration_lawam import LaWAMConfig
from .latent_world.batch_utils import (
    build_placeholder_masks,
    imagenet_normalize_video_,
)
from .latent_world.processor_utils import LatentWorldProcessorSpec, load_latent_world_processor
from .latent_world.vlm_adapter import (
    DEFAULT_LATENT_WORLD_POLICY_COT_PROMPT,
    DEFAULT_LATENT_WORLD_TEMPORAL_COT_PROMPT,
    build_qwenvl_messages,
)


@ProcessorStepRegistry.register(name="lawam_clip_actions")
class LaWAMClipActionsProcessorStep(ProcessorStep):
    """Clamp normalized actions to the range expected by LaWAM."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Clamp an action transition to the normalized interval."""
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        transition = dict(transition)
        transition[TransitionKey.ACTION] = action.clamp(-1.0, 1.0)
        return transition

    def transform_features(self, features):
        """Preserve feature declarations because clipping does not change shape."""
        return features

    def get_config(self) -> dict[str, Any]:
        """Return the serializable processor configuration."""
        return {}


@ProcessorStepRegistry.register(name="lawam_pre_snap_gripper")
class LaWAMPreSnapGripperProcessorStep(ProcessorStep):
    """Snap the normalized gripper channel to binary values before unnormalizing."""

    def __init__(self, gripper_dim: int = 6, threshold: float = 0.5):
        self.gripper_dim = gripper_dim
        self.threshold = threshold

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Snap the configured gripper channel when it is present."""
        action = transition.get(TransitionKey.ACTION)
        if action is None or action.shape[-1] <= self.gripper_dim:
            return transition
        transition = dict(transition)
        snapped = action.clone()
        snapped[..., self.gripper_dim] = (snapped[..., self.gripper_dim] >= self.threshold).float()
        transition[TransitionKey.ACTION] = snapped
        return transition

    def transform_features(self, features):
        """Preserve feature declarations because snapping does not change shape."""
        return features

    def get_config(self) -> dict[str, Any]:
        """Return the gripper index and threshold for serialization."""
        return {"gripper_dim": self.gripper_dim, "threshold": self.threshold}


@ProcessorStepRegistry.register(name="lawam_binarize_gripper")
class LaWAMBinarizeGripperProcessorStep(ProcessorStep):
    """Map the emitted gripper channel to the LIBERO minus-one/plus-one convention."""

    def __init__(self, gripper_dim: int = 6, threshold: float = 0.5):
        self.gripper_dim = gripper_dim
        self.threshold = threshold

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Binarize the configured gripper channel when it is present."""
        action = transition.get(TransitionKey.ACTION)
        if action is None or action.shape[-1] <= self.gripper_dim:
            return transition
        transition = dict(transition)
        binarized = action.clone()
        binarized[..., self.gripper_dim] = (
            2.0 * (binarized[..., self.gripper_dim] > self.threshold).float() - 1.0
        )
        transition[TransitionKey.ACTION] = binarized
        return transition

    def transform_features(self, features):
        """Preserve feature declarations because binarization does not change shape."""
        return features

    def get_config(self) -> dict[str, Any]:
        """Return the gripper index and threshold for serialization."""
        return {"gripper_dim": self.gripper_dim, "threshold": self.threshold}


def _as_video_batch(value: Any, *, key: str, image_hw: tuple[int, int]) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"LaWAM image feature `{key}` must be a torch.Tensor.")
    if not value.is_floating_point():
        raise TypeError(f"LaWAM expects canonical LeRobot float images for `{key}`, got {value.dtype}.")
    if value.ndim == 4:
        value = value.unsqueeze(1)
    if value.ndim != 5 or int(value.shape[2]) != 3:
        raise ValueError(
            f"LaWAM image feature `{key}` must have shape [B,3,H,W] or [B,T,3,H,W], got {tuple(value.shape)}."
        )
    if tuple(value.shape[-2:]) != image_hw:
        raise ValueError(
            f"LaWAM image feature `{key}` was not resized by the processor pipeline: "
            f"expected {image_hw}, got {tuple(value.shape[-2:])}."
        )
    return value.contiguous()


def _task_batch(tasks: Any, *, batch_size: int, default_task: str) -> list[str]:
    if tasks is None:
        return [default_task] * batch_size
    if isinstance(tasks, str):
        return [tasks] * batch_size
    task_batch = [str(task) for task in tasks]
    if len(task_batch) != batch_size:
        raise ValueError(f"LaWAM received {len(task_batch)} tasks for batch size {batch_size}.")
    return task_batch


@ProcessorStepRegistry.register(name="lawam_resize_images")
class LaWAMResizeImagesProcessorStep(ProcessorStep):
    """Resize current and temporal image batches to the LaWAM input grid."""

    def __init__(
        self,
        *,
        image_features: list[str],
        image_hw: list[int] | tuple[int, int],
    ) -> None:
        self.image_features = list(image_features)
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self._resize_step = ImageCropResizeProcessorStep(resize_size=self.image_hw)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION) or {}
        missing = [key for key in self.image_features if key not in observation]
        if missing:
            raise KeyError(f"LaWAM input batch is missing configured camera features: {missing}.")

        leading_shapes: dict[str, torch.Size] = {}
        flat_observation: dict[str, torch.Tensor] = {}
        for key in self.image_features:
            image = observation[key]
            if not torch.is_tensor(image) or image.ndim not in (4, 5) or int(image.shape[-3]) != 3:
                shape = tuple(image.shape) if torch.is_tensor(image) else type(image).__name__
                raise ValueError(
                    f"LaWAM image feature `{key}` must have shape [B,3,H,W] or [B,T,3,H,W], got {shape}."
                )
            leading_shapes[key] = image.shape[:-3]
            flat_observation[key] = image.reshape(-1, *image.shape[-3:])

        resized_transition = self._resize_step({TransitionKey.OBSERVATION: flat_observation})
        resized_observation = dict(observation)
        resized_images = resized_transition[TransitionKey.OBSERVATION]
        for key in self.image_features:
            resized = resized_images[key]
            resized_observation[key] = resized.reshape(*leading_shapes[key], *resized.shape[-3:])

        result = dict(transition)
        result[TransitionKey.OBSERVATION] = resized_observation
        return result

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        observations = features[PipelineFeatureType.OBSERVATION]
        for key in self.image_features:
            if key not in observations:
                continue
            observations[key] = PolicyFeature(
                type=observations[key].type,
                shape=(observations[key].shape[0], *self.image_hw),
            )
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "image_features": self.image_features,
            "image_hw": list(self.image_hw),
        }


@ProcessorStepRegistry.register(name="lawam_qwen_inputs")
class LaWAMQwenInputsProcessorStep(ProcessorStep):
    """Create Qwen image/text tensors and LaWAM placeholder masks."""

    def __init__(
        self,
        *,
        model_id: str,
        placeholder_token: str,
        act_queries: int,
        flow_queries: int,
        primary_image_features: list[str],
        wrist_image_features: list[str],
        image_hw: list[int] | tuple[int, int],
        default_task: str,
        cot_prompt_before_wrist: str = DEFAULT_LATENT_WORLD_TEMPORAL_COT_PROMPT,
        cot_prompt_after_wrist: str = DEFAULT_LATENT_WORLD_POLICY_COT_PROMPT,
    ) -> None:
        self.model_id = str(model_id)
        self.placeholder_token = str(placeholder_token)
        self.act_queries = int(act_queries)
        self.flow_queries = int(flow_queries)
        self.primary_image_features = list(primary_image_features)
        self.wrist_image_features = list(wrist_image_features)
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.default_task = str(default_task)
        self.cot_prompt_before_wrist = str(cot_prompt_before_wrist)
        self.cot_prompt_after_wrist = str(cot_prompt_after_wrist)
        self._processor: Any | None = None
        self._placeholder_token_id: int | None = None

    def _ensure_processor(self) -> tuple[Any, int]:
        if self._processor is None:
            processor, _, placeholder_token_id = load_latent_world_processor(
                LatentWorldProcessorSpec(
                    model_id=self.model_id,
                    placeholder_token=self.placeholder_token,
                )
            )
            self._processor = processor
            self._placeholder_token_id = placeholder_token_id
        if self._placeholder_token_id is None:
            raise RuntimeError("LaWAM Qwen processor did not initialize its placeholder token.")
        return self._processor, self._placeholder_token_id

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION) or {}
        camera_features = self.primary_image_features + self.wrist_image_features
        missing = [key for key in camera_features if key not in observation]
        if missing:
            raise KeyError(f"LaWAM input batch is missing configured camera features: {missing}.")

        videos = {
            key: _as_video_batch(observation[key], key=key, image_hw=self.image_hw) for key in camera_features
        }
        batch_size = int(videos[self.primary_image_features[0]].shape[0])
        if any(int(video.shape[0]) != batch_size for video in videos.values()):
            raise ValueError("LaWAM camera features must share the same batch size.")

        comp = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        tasks = _task_batch(comp.get("task"), batch_size=batch_size, default_task=self.default_task)
        image_views = [
            [videos[key][batch_idx, 0] for key in self.primary_image_features]
            for batch_idx in range(batch_size)
        ]
        wrist_image_views = [
            [videos[key][batch_idx, 0] for key in self.wrist_image_features]
            for batch_idx in range(batch_size)
        ]
        device = videos[self.primary_image_features[0]].device
        processor, placeholder_token_id = self._ensure_processor()
        qwen_inputs = processor.apply_chat_template(
            build_qwenvl_messages(
                images=image_views,
                wrist_images=wrist_image_views,
                instructions=tasks,
                placeholder_token=self.placeholder_token,
                act_queries=self.act_queries,
                flow_queries=self.flow_queries,
                cot_prompt_before_wrist=self.cot_prompt_before_wrist,
                cot_prompt_after_wrist=self.cot_prompt_after_wrist,
            ),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            processor_kwargs={
                "padding": True,
                "return_tensors": "pt",
                "device": device,
                "do_rescale": False,
            },
        )
        qwen_inputs = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in qwen_inputs.items()
        }
        input_ids = qwen_inputs["input_ids"]
        act_mask, flow_mask = build_placeholder_masks(
            input_ids,
            act_queries=self.act_queries,
            flow_queries=self.flow_queries,
            placeholder_id=placeholder_token_id,
        )
        comp.update(
            {
                "pixel_values": qwen_inputs["pixel_values"],
                "input_ids": input_ids,
                "attention_mask": qwen_inputs["attention_mask"],
                "act_placeholder_mask": act_mask,
                "flow_placeholder_mask": flow_mask,
                "image_grid_thw": qwen_inputs.get("image_grid_thw"),
            }
        )
        result = dict(transition)
        result[TransitionKey.COMPLEMENTARY_DATA] = comp
        return result

    def transform_features(self, features):
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "placeholder_token": self.placeholder_token,
            "act_queries": self.act_queries,
            "flow_queries": self.flow_queries,
            "primary_image_features": self.primary_image_features,
            "wrist_image_features": self.wrist_image_features,
            "image_hw": list(self.image_hw),
            "default_task": self.default_task,
            "cot_prompt_before_wrist": self.cot_prompt_before_wrist,
            "cot_prompt_after_wrist": self.cot_prompt_after_wrist,
        }


@ProcessorStepRegistry.register(name="lawam_prepare_batch")
class LaWAMPrepareBatchProcessorStep(ProcessorStep):
    """Align state/action tensors and expose the final LaWAM backend batch."""

    def __init__(
        self,
        *,
        lam_image_feature: str,
        image_hw: list[int] | tuple[int, int],
        action_horizon: int,
        action_dim: int,
        state_dim: int,
        use_state: bool,
        action_hz: float,
        embodiment_id: int,
        chunk_size: int | None = None,
    ) -> None:
        self.lam_image_feature = str(lam_image_feature)
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.action_horizon = int(action_horizon)
        self.chunk_size = self.action_horizon if chunk_size is None else int(chunk_size)
        if not 1 <= self.action_horizon <= self.chunk_size:
            raise ValueError("`action_horizon` must be in [1, chunk_size].")
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.use_state = bool(use_state)
        self.action_hz = float(action_hz)
        self.embodiment_id = int(embodiment_id)

    @staticmethod
    def _align_feature_batch(values: torch.Tensor, *, target_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        if values.ndim != 2:
            raise ValueError(f"Expected feature batch [B,D], got {tuple(values.shape)}.")
        source_dim = int(values.shape[-1])
        if source_dim > target_dim:
            raise ValueError(f"Feature width {source_dim} exceeds LaWAM padded width {target_dim}.")
        mask = torch.zeros((int(values.shape[0]), target_dim), dtype=torch.bool, device=values.device)
        mask[:, :source_dim] = True
        if source_dim == target_dim:
            return values, mask
        pad = values.new_zeros((int(values.shape[0]), target_dim - source_dim))
        return torch.cat((values, pad), dim=-1), mask

    def _prepare_state(
        self, state: Any, *, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            return (
                torch.zeros((batch_size, self.state_dim), dtype=torch.float32, device=device),
                torch.zeros((batch_size, self.state_dim), dtype=torch.bool, device=device),
            )
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
        if state_tensor.ndim == 3:
            state_tensor = state_tensor[:, 0]
        if state_tensor.ndim != 2 or int(state_tensor.shape[0]) != batch_size:
            raise ValueError(
                f"LaWAM state must have shape [B,D] or [B,T,D], got {tuple(state_tensor.shape)}."
            )
        return self._align_feature_batch(state_tensor, target_dim=self.state_dim)

    def _prepare_actions(
        self, action: Any, *, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_tensor = torch.as_tensor(action, dtype=torch.float32, device=device)
        if action_tensor.ndim == 2:
            action_tensor = action_tensor.unsqueeze(1)
        if action_tensor.ndim != 3 or int(action_tensor.shape[0]) != batch_size:
            raise ValueError(
                f"LaWAM actions must have shape [B,D] or [B,T,D], got {tuple(action_tensor.shape)}."
            )
        source_steps = int(action_tensor.shape[1])
        source_dim = int(action_tensor.shape[2])
        if source_steps > self.action_horizon:
            raise ValueError(
                f"Action sequence length {source_steps} exceeds LaWAM horizon {self.action_horizon}."
            )
        if source_dim > self.action_dim:
            raise ValueError(f"Action width {source_dim} exceeds LaWAM padded width {self.action_dim}.")
        actions = action_tensor.new_zeros((batch_size, self.chunk_size, self.action_dim))
        actions[:, :source_steps, :source_dim] = action_tensor
        mask = torch.zeros_like(actions, dtype=torch.bool)
        mask[:, :source_steps, :source_dim] = True
        return actions, mask

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION) or {}
        if self.lam_image_feature not in observation:
            raise KeyError(f"LaWAM input batch is missing LAM camera feature `{self.lam_image_feature}`.")
        primary_video = _as_video_batch(
            observation[self.lam_image_feature],
            key=self.lam_image_feature,
            image_hw=self.image_hw,
        ).detach()
        primary_video = primary_video.to(dtype=torch.float32).clone()
        imagenet_normalize_video_(primary_video)
        batch_size = int(primary_video.shape[0])
        state_input = observation.get(OBS_STATE) if self.use_state else None
        state, state_mask = self._prepare_state(
            state_input, batch_size=batch_size, device=primary_video.device
        )

        comp = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        required = (
            "pixel_values",
            "input_ids",
            "attention_mask",
            "act_placeholder_mask",
            "flow_placeholder_mask",
        )
        missing = [key for key in required if key not in comp]
        if missing:
            raise KeyError(f"LaWAM Qwen processor outputs are missing: {missing}.")
        prepared: dict[str, Any] = {key: comp[key] for key in required}
        prepared.update(
            {
                "primary_video": primary_video,
                "primary_image": primary_video[:, 0],
                "state": state,
                "state_mask": state_mask,
                "embodiment_id": torch.full(
                    (batch_size,), self.embodiment_id, dtype=torch.long, device=primary_video.device
                ),
                "action_hz": torch.full(
                    (batch_size,), self.action_hz, dtype=torch.float32, device=primary_video.device
                ),
                "image_grid_thw": comp.get("image_grid_thw"),
            }
        )
        action = transition.get(TransitionKey.ACTION)
        if action is not None:
            prepared["actions"], prepared["actions_mask"] = self._prepare_actions(
                action,
                batch_size=batch_size,
                device=primary_video.device,
            )

        result = dict(transition)
        result[TransitionKey.OBSERVATION] = {}
        result[TransitionKey.ACTION] = None
        result[TransitionKey.COMPLEMENTARY_DATA] = prepared
        return result

    def transform_features(self, features):
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "lam_image_feature": self.lam_image_feature,
            "image_hw": list(self.image_hw),
            "action_horizon": self.action_horizon,
            "chunk_size": self.chunk_size,
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "use_state": self.use_state,
            "action_hz": self.action_hz,
            "embodiment_id": self.embodiment_id,
        }


def _make_lawam_model_input_steps(config: LaWAMConfig, *, action_hz: float) -> list[ProcessorStep]:
    return [
        LaWAMResizeImagesProcessorStep(
            image_features=config.primary_image_features + config.wrist_image_features,
            image_hw=config.lam_image_hw,
        ),
        LaWAMQwenInputsProcessorStep(
            model_id=config.base_vlm,
            placeholder_token=config.latent_action_placeholder_token,
            act_queries=config.num_action_queries,
            flow_queries=config.flow_action_num_queries,
            primary_image_features=config.primary_image_features,
            wrist_image_features=config.wrist_image_features,
            image_hw=config.lam_image_hw,
            default_task=config.default_task,
        ),
        LaWAMPrepareBatchProcessorStep(
            lam_image_feature=config.lam_image_feature,
            image_hw=config.lam_image_hw,
            action_horizon=config.action_horizon,
            chunk_size=config.chunk_size,
            action_dim=config.flow_action_dim,
            state_dim=config.flow_state_dim,
            use_state=config.flow_use_state,
            action_hz=action_hz,
            embodiment_id=config.embodiment_id,
        ),
    ]


def make_lawam_pre_post_processors(
    config: LaWAMConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    dataset_meta=None,
    rename_map: dict[str, str] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build LaWAM input normalization and action postprocessing pipelines."""
    config.validate_features()
    if dataset_meta is None:
        dataset_meta = getattr(config, "_runtime_dataset_meta", None)
    dataset_fps = getattr(dataset_meta, "fps", None)
    if dataset_fps is None or float(dataset_fps) <= 0:
        raise ValueError("LaWAM training processors require dataset metadata with a valid `fps` value.")
    action_hz = float(dataset_fps)
    features = {**config.input_features, **config.output_features}
    processor_stats = dataset_stats
    if not config.flow_use_state:
        features.pop(OBS_STATE, None)
        if dataset_stats is not None and OBS_STATE in dataset_stats:
            processor_stats = {key: value for key, value in dataset_stats.items() if key != OBS_STATE}

    default_steps = make_default_policy_processor_steps(config, processor_stats)
    default_steps.rename_observations.rename_map = dict(rename_map or {})
    input_steps: list[ProcessorStep] = [
        default_steps.rename_observations,
        default_steps.add_batch_dim,
        default_steps.to_device,
        NormalizerProcessorStep(
            features=features,
            norm_map=config.normalization_mapping,
            stats=processor_stats,
        ),
    ]
    if config.pre_snap_gripper_action:
        input_steps.append(
            LaWAMPreSnapGripperProcessorStep(
                gripper_dim=config.gripper_dim,
                threshold=config.gripper_threshold,
            )
        )
    input_steps.extend(_make_lawam_model_input_steps(config, action_hz=action_hz))

    output_steps: list[ProcessorStep] = []
    if config.clip_normalized_actions:
        output_steps.append(LaWAMClipActionsProcessorStep())
    if config.pre_snap_gripper_action:
        output_steps.append(
            LaWAMPreSnapGripperProcessorStep(
                gripper_dim=config.gripper_dim,
                threshold=config.gripper_threshold,
            )
        )
    output_steps.append(
        UnnormalizerProcessorStep(
            features=features,
            norm_map=config.normalization_mapping,
            stats=processor_stats,
        )
    )
    if config.binarize_gripper_action:
        output_steps.append(
            LaWAMBinarizeGripperProcessorStep(
                gripper_dim=config.gripper_dim,
                threshold=config.gripper_threshold,
            )
        )
    output_steps.append(default_steps.to_cpu)

    return make_policy_processor_pipelines(
        input_steps=input_steps,
        output_steps=output_steps,
    )


def make_lawam_pre_post_processors_from_pretrained(
    config: LaWAMConfig,
    pretrained_path: str,
    *,
    revision: str | None = None,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    dataset_meta=None,
    preprocessor_overrides: dict[str, Any] | None = None,
    postprocessor_overrides: dict[str, Any] | None = None,
    preprocessor_config_filename: str | None = None,
    postprocessor_config_filename: str | None = None,
) -> (
    tuple[
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
        PolicyProcessorPipeline[PolicyAction, PolicyAction],
    ]
    | None
):
    """Rebuild LaWAM processors for fresh SFT and load serialized pipelines otherwise."""
    del pretrained_path, revision, postprocessor_overrides
    del preprocessor_config_filename, postprocessor_config_filename
    if dataset_meta is None:
        dataset_meta = getattr(config, "_runtime_dataset_meta", None)
    if dataset_meta is None or dataset_stats is None:
        return None

    rename_map = (preprocessor_overrides or {}).get("rename_observations_processor", {}).get("rename_map", {})
    return make_lawam_pre_post_processors(
        config,
        dataset_stats=dataset_stats,
        dataset_meta=dataset_meta,
        rename_map=rename_map,
    )
