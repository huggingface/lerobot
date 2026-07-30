# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea
# Modified for LeRobot in 2026.

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from torch import Tensor
from torch.nn import functional

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    ObservationProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    batch_to_transition,
    hotswap_stats,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    ACTION_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from lerobot.utils.import_utils import require_package

from .configuration_g05 import G05Config


@dataclass
class G05LiberoObservationStep(ObservationProcessorStep):
    """Match the released G0.5 LIBERO observation boundary."""

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        observation = observation.copy()
        for key, image in observation.items():
            if key.startswith("observation.images."):
                # LIBERO returns both cameras upside down relative to the training data.
                observation[key] = torch.flip(image, dims=(-2, -1))

        robot_state = observation.pop("observation.robot_state", None)
        if robot_state is None:
            return observation
        position = robot_state["eef"]["pos"]
        axis_angle = self._quat2axisangle(robot_state["eef"]["quat"])
        gripper = robot_state["gripper"]["qpos"][..., :1]
        observation[OBS_STATE] = torch.cat((position, axis_angle, gripper), dim=-1).float()
        return observation

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features = {feature_type: values.copy() for feature_type, values in features.items()}
        features[PipelineFeatureType.STATE] = {
            OBS_STATE: PolicyFeature(type=PipelineFeatureType.STATE, shape=(7,))
        }
        return features

    @staticmethod
    def _quat2axisangle(quaternion: Tensor) -> Tensor:
        """Convert LIBERO xyzw quaternions to the axis-angle representation used in training."""
        scalar = quaternion[..., 3:].clamp(-1, 1)
        denominator = torch.sqrt((1 - scalar.square()).clamp_min(0))
        scale = 2 * torch.acos(scalar) / denominator.clamp_min(1e-8)
        return quaternion[..., :3] * torch.where(denominator > 1e-8, scale, scale.new_zeros(()))


@dataclass
class G05LiberoActionStep(ProcessorStep):
    """Convert the trained [0, 1] gripper convention to LIBERO's {-1, +1} commands."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        action = transition.get(TransitionKey.ACTION)
        if action is not None:
            action = action.clone()
            # G0.5 encodes 0 as closed and 1 as open. LIBERO uses the opposite
            # actuator signs: +1 closes and -1 opens.
            action[..., -1] = torch.where(action[..., -1] > 0.5, -1.0, 1.0)
            transition[TransitionKey.ACTION] = action
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_g05_libero_pre_post_processors() -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Build the environment-side adapters required by the released LIBERO checkpoint."""
    return (
        PolicyProcessorPipeline(steps=[G05LiberoObservationStep()]),
        PolicyProcessorPipeline(steps=[G05LiberoActionStep()]),
    )


def _apply_normalization(
    values: Tensor, specs: list[dict], *, inverse: bool, step_index: int | None = None
) -> Tensor:
    """Apply the released G0.5 per-part normalizer at the LeRobot boundary."""
    if not specs:
        return values
    output: list[Tensor] = []
    offset = 0
    for spec in specs:
        mode = spec["mode"]
        stats = {name: values.new_tensor(value) for name, value in spec["stats"].items()}
        first_stat = next(iter(stats.values()))
        width = spec.get("width", first_stat.shape[-1])
        width = int(width)
        current = values[..., offset : offset + width]
        offset += width

        if first_stat.shape[-1] != width:
            raise ValueError("G0.5 normalization statistics do not match the configured part width")
        if first_stat.ndim >= 2:
            if current.ndim >= 3:
                horizon = current.shape[-2]
                if horizon > first_stat.shape[-2]:
                    raise ValueError("G0.5 action horizon exceeds the published normalization statistics")
                stats = {name: value[..., :horizon, :] for name, value in stats.items()}
            elif step_index is not None:
                index = min(step_index, first_stat.shape[-2] - 1)
                stats = {name: value[..., index, :] for name, value in stats.items()}

        if mode == "z-score":
            mean, std = stats["mean"], stats["std"]
            scale = 1 / (std + 1e-8)
            shift = -mean / (std + 1e-8)
            constant = std < 1e-4
            scale = torch.where(constant, torch.ones_like(scale), scale)
            shift = torch.where(constant, -mean, shift)
        elif mode == "q01/q99":
            low, high = stats["q01"], stats["q99"]
            value_range = high - low
            constant = value_range < 1e-4
            value_range = torch.where(constant, torch.full_like(value_range, 2), value_range)
            scale = 2.0 / value_range
            shift = -1 - scale * low
            shift = torch.where(constant, -low, shift)
        else:
            raise ValueError(f"unsupported G0.5 normalization mode: {mode}")

        if inverse:
            current = (current - shift) / scale
        if not inverse:
            current = (current * scale + shift).clamp(-5, 5)
            current = current.nan_to_num()
        output.append(current)
    if offset != values.shape[-1]:
        raise ValueError("G0.5 normalization specs do not cover the physical tensor")
    return torch.cat(output, dim=-1)


def _pad_last_dim(values: Tensor, indices: list[int], target_dim: int) -> tuple[Tensor, Tensor]:
    physical_dim = values.shape[-1]
    indices = indices or list(range(physical_dim))
    if len(indices) != physical_dim or len(set(indices)) != len(indices):
        raise ValueError("G0.5 layout indices must uniquely map every physical dimension")
    if min(indices, default=0) < 0 or max(indices, default=-1) >= target_dim:
        raise ValueError("G0.5 layout index exceeds the checkpoint dimension")
    output = values.new_zeros(*values.shape[:-1], target_dim)
    output[..., indices] = values
    dimension_is_pad = torch.ones(values.shape[0], target_dim, dtype=torch.bool, device=values.device)
    dimension_is_pad[:, indices] = False
    return output, dimension_is_pad


@ProcessorStepRegistry.register(name="g05_stepwise_normalizer")
@dataclass
class G05StepwiseNormalizerStep(ProcessorStep):
    """Apply checkpoint-bound state and per-horizon action statistics."""

    state_normalization: list[dict]
    action_normalization: list[dict]

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = dict(transition.get(TransitionKey.OBSERVATION) or {})
        state = observation.get(OBS_STATE)
        if state is not None:
            observation[OBS_STATE] = _apply_normalization(state, self.state_normalization, inverse=False)
            transition[TransitionKey.OBSERVATION] = observation

        action = transition.get(TransitionKey.ACTION)
        if action is not None:
            transition[TransitionKey.ACTION] = _apply_normalization(
                action, self.action_normalization, inverse=False
            )
        return transition

    def get_config(self) -> dict[str, Any]:
        return {
            "state_normalization": self.state_normalization,
            "action_normalization": self.action_normalization,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="g05_prepare_inputs")
@dataclass
class G05PrepareInputsStep(ProcessorStep):
    tokenizer_path: str
    action_tokenizer_path: str
    camera_keys: list[str]
    dummy_camera_keys: list[str]
    image_size: tuple[int, int]
    patch_size: int
    spatial_merge_size: int
    n_obs_steps: int
    internal_state_dim: int
    internal_action_dim: int
    state_indices: list[int]
    action_indices: list[int]
    embodiment: str
    max_task_tokens: int
    max_prompt_length: int
    image_token_id: int
    vision_start_token_id: int
    vision_end_token_id: int
    state_token_id: int
    eov_token_id: int
    pad_token_id: int
    eos_token_id: int
    camera_order: list[str] | None = None
    optional_camera_keys: list[str] | None = None
    append_eov: bool = True

    def __post_init__(self) -> None:
        require_package("transformers", extra="g05")
        if not self.tokenizer_path:
            raise ValueError("tokenizer_path must resolve inside the G0.5 artifact")
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, local_files_only=True)
        self._action_tokenizer = None
        if self.action_tokenizer_path:
            from .action_tokenizer import G05ActionCodecModel, G05ActionTokenizer

            codec = G05ActionCodecModel.from_pretrained(
                self.action_tokenizer_path,
                local_files_only=True,
            )
            self._action_tokenizer = G05ActionTokenizer(codec, self._tokenizer)

    def get_config(self) -> dict[str, Any]:
        return {
            item.name: (
                "" if item.name in {"tokenizer_path", "action_tokenizer_path"} else getattr(self, item.name)
            )
            for item in fields(self)
        }

    def _prepare_images(self, observation: dict[str, Any], state: Tensor) -> Tensor:
        images: list[Tensor] = []
        camera_order = self.camera_order or [*self.camera_keys, *self.dummy_camera_keys]
        for key in camera_order:
            value = observation.get(key)
            if value is None:
                optional_camera_keys = self.optional_camera_keys or []
                if key not in self.dummy_camera_keys and key not in optional_camera_keys:
                    raise ValueError(f"missing required G0.5 camera {key!r}")
                value = state.new_zeros(
                    state.shape[0], self.n_obs_steps, 3, self.image_size[0], self.image_size[1]
                )
            if value.ndim == 4:
                value = value.unsqueeze(1)
            if value.ndim != 5:
                raise ValueError(f"camera {key!r} must have shape [B,T,C,H,W] or [B,C,H,W]")
            if value.shape[1] != self.n_obs_steps:
                raise ValueError(f"camera {key!r} has {value.shape[1]} frames; expected {self.n_obs_steps}")
            batch_size, steps, channels, height, width = value.shape
            # LeRobot images are either uint8 [0, 255] or floating point [0, 1].
            # Checking dtype avoids a GPU synchronization for every camera slot.
            value = value.float() / 255 if value.dtype == torch.uint8 else value.float()
            if (height, width) != tuple(self.image_size):
                value = functional.interpolate(
                    value.reshape(batch_size * steps, channels, height, width),
                    size=self.image_size,
                    mode="bilinear",
                    align_corners=False,
                ).reshape(batch_size, steps, channels, *self.image_size)
            images.append(value * 2 - 1)
        return torch.stack(images, dim=1)

    def _prompt_ids(
        self, tasks: list[str], num_images: int, *, append_eov: bool | None = None
    ) -> tuple[Tensor, Tensor]:
        append_eov = self.append_eov if append_eov is None else append_eov
        image_tokens = (self.image_size[0] // self.patch_size // self.spatial_merge_size) * (
            self.image_size[1] // self.patch_size // self.spatial_merge_size
        )
        rows: list[list[int]] = []
        for task in tasks:
            # Match the source Qwen35Backend contract: only tokenizers whose
            # bos_token is non-empty use the instruct chat envelope. Released
            # base-model checkpoints have bos_token=None and were trained on the
            # bare multimodal prefix.
            uses_chat_template = bool(getattr(self._tokenizer, "bos_token", None))
            row = (
                self._tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
                if uses_chat_template
                else []
            )
            for _ in range(num_images * self.n_obs_steps):
                row += [self.vision_start_token_id]
                row += [self.image_token_id] * image_tokens
                row += [self.vision_end_token_id]
            # Static and dynamic template segments are tokenized separately in G0.5.
            row += self._tokenizer.encode("Embodiment: ", add_special_tokens=False)
            row += self._tokenizer.encode(self.embodiment, add_special_tokens=False)
            row += self._tokenizer.encode("; Task: ", add_special_tokens=False)
            task_ids = self._tokenizer.encode(task.strip(), add_special_tokens=False)
            row += task_ids[: self.max_task_tokens]
            row += self._tokenizer.encode(" State: ", add_special_tokens=False)
            row += [self.state_token_id] * self.n_obs_steps
            assistant_prefix = (
                ";<|im_end|>\n<|im_start|>robot\nAction: " if uses_chat_template else ";Action: "
            )
            row += self._tokenizer.encode(assistant_prefix, add_special_tokens=False)
            if append_eov:
                row += [self.eov_token_id]
            if len(row) > self.max_prompt_length:
                raise ValueError("G0.5 prompt exceeds max_prompt_length")
            rows.append(row)
        width = max(map(len, rows))
        ids = torch.full((len(rows), width), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.bool)
        for index, row in enumerate(rows):
            # G0.5 right-aligns prefix batches, so shorter prompts are left-padded.
            ids[index, width - len(row) :] = torch.tensor(row)
            mask[index, width - len(row) :] = True
        return ids, mask

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = dict(transition.get(TransitionKey.OBSERVATION) or {})
        complementary = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        state = observation.get(OBS_STATE)
        if state is None:
            raise ValueError("G0.5 requires observation.state")
        state = state.float()
        state, _ = _pad_last_dim(state, self.state_indices, self.internal_state_dim)
        observation[OBS_STATE] = state
        complementary["pixel_values"] = self._prepare_images(observation, state)

        tasks = complementary.get("task")
        if isinstance(tasks, str):
            tasks = [tasks]
        if not isinstance(tasks, (list, tuple)) or not all(isinstance(task, str) for task in tasks):
            raise ValueError("G0.5 requires one task string per batch item")
        action = transition.get(TransitionKey.ACTION)
        ids, mask = self._prompt_ids(
            list(tasks),
            len(self.camera_keys) + len(self.dummy_camera_keys),
            # Teacher-forced training uses the published return_prefix contract,
            # where EOV is the final prefix token. CoT-conditioned inference
            # omits it so the VLM can generate context up to EOV first.
            append_eov=self.append_eov or action is not None,
        )
        complementary[OBS_LANGUAGE_TOKENS] = ids.to(state.device)
        complementary[OBS_LANGUAGE_ATTENTION_MASK] = mask.to(state.device)

        if action is not None:
            action = action.float()
            action, _ = _pad_last_dim(action, self.action_indices, self.internal_action_dim)
            transition[TransitionKey.ACTION] = action
            if self._action_tokenizer is not None:
                codec_device = next(self._action_tokenizer.model.parameters()).device
                if codec_device != action.device:
                    self._action_tokenizer.model.to(action.device)
                action_token_ids = self._action_tokenizer.encode(action)
                eos = torch.full((ids.shape[0], 1), self.eos_token_id, dtype=torch.long, device=action.device)
                complementary[ACTION_TOKENS] = torch.cat((action_token_ids, eos), dim=-1)

        transition[TransitionKey.OBSERVATION] = observation
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="g05_stepwise_action_unnormalizer")
@dataclass
class G05StepwiseActionUnnormalizerStep(ProcessorStep):
    """Select the matching horizon statistics for each queued action."""

    action_normalization: list[dict]
    action_horizon: int

    def __post_init__(self) -> None:
        self._step_index = 0

    def reset(self) -> None:
        self._step_index = 0

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            raise ValueError("G0.5 postprocessor requires an action tensor")
        step_index = self._step_index % self.action_horizon
        transition[TransitionKey.ACTION] = _apply_normalization(
            action,
            self.action_normalization,
            inverse=True,
            step_index=step_index,
        )
        if action.ndim < 3:
            self._step_index += 1
        return transition

    def get_config(self) -> dict[str, Any]:
        return {
            "action_normalization": self.action_normalization,
            "action_horizon": self.action_horizon,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_g05_pre_post_processors(
    config: G05Config,
    dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    *,
    tokenizer_path: str | Path | None = None,
    action_tokenizer_path: str | Path | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    if config.state_token_id is None or config.eov_token_id is None:
        raise ValueError("converted G0.5 config must define state_token_id and eov_token_id")
    steps = make_default_policy_processor_steps(config, dataset_stats)
    relative_step = RelativeActionsProcessorStep(
        enabled=bool(config.relative_action_mask),
        action_names=["relative" if value else "absolute" for value in config.relative_action_mask] or None,
        exclude_joints=["absolute"],
    )
    prepare = G05PrepareInputsStep(
        tokenizer_path=str(tokenizer_path or ""),
        action_tokenizer_path=str(action_tokenizer_path or ""),
        camera_keys=config.camera_keys,
        dummy_camera_keys=config.dummy_camera_keys,
        image_size=config.image_size,
        patch_size=config.vision_patch_size,
        spatial_merge_size=config.vision_spatial_merge_size,
        n_obs_steps=config.n_obs_steps,
        internal_state_dim=config.internal_state_dim,
        internal_action_dim=config.internal_action_dim,
        state_indices=config.state_indices,
        action_indices=config.action_indices,
        embodiment=config.embodiment,
        max_task_tokens=config.max_task_tokens,
        max_prompt_length=config.max_prompt_length,
        image_token_id=config.image_token_id,
        vision_start_token_id=config.vision_start_token_id,
        vision_end_token_id=config.vision_end_token_id,
        state_token_id=config.state_token_id,
        eov_token_id=config.eov_token_id,
        pad_token_id=config.pad_token_id,
        eos_token_id=config.eos_token_id,
        camera_order=config.camera_order,
        optional_camera_keys=config.optional_camera_keys,
        append_eov=not config.action_attend_cot,
    )
    if config.normalization_strategy == "lerobot":
        normalize_step = steps.normalize
        unnormalize_step = steps.unnormalize
    else:
        normalize_step = G05StepwiseNormalizerStep(
            state_normalization=config.state_normalization,
            action_normalization=config.action_normalization,
        )
        unnormalize_step = G05StepwiseActionUnnormalizerStep(
            action_normalization=config.action_normalization,
            action_horizon=config.n_action_steps,
        )
    return make_policy_processor_pipelines(
        input_steps=[
            steps.rename_observations,
            steps.add_batch_dim,
            relative_step,
            steps.to_device,
            normalize_step,
            prepare,
        ],
        output_steps=[
            unnormalize_step,
            AbsoluteActionsProcessorStep(
                enabled=bool(config.relative_action_mask), relative_step=relative_step
            ),
            steps.to_cpu,
        ],
    )


def make_g05_pre_post_processors_from_pretrained(
    config: G05Config,
    pretrained_path: str,
    *,
    revision: str | None = None,
    dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    preprocessor_config_filename: str = f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
    postprocessor_config_filename: str = f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Load processors while resolving the tokenizer inside the same artifact."""
    artifact_root = Path(pretrained_path)
    if not artifact_root.is_dir():
        artifact_root = Path(
            snapshot_download(
                pretrained_path,
                revision=revision,
                allow_patterns=[
                    f"{config.tokenizer_subdir}/*",
                    f"{config.action_tokenizer_subdir}/*",
                    preprocessor_config_filename,
                    postprocessor_config_filename,
                    "*.safetensors",
                ],
            )
        )
    tokenizer_path = artifact_root / config.tokenizer_subdir
    action_tokenizer_path = artifact_root / config.action_tokenizer_subdir
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(
            f"self-contained G0.5 artifact is missing tokenizer directory: {tokenizer_path}"
        )
    if not action_tokenizer_path.is_dir():
        raise FileNotFoundError(
            f"self-contained G0.5 artifact is missing action tokenizer directory: {action_tokenizer_path}"
        )
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=artifact_root,
        config_filename=preprocessor_config_filename,
        overrides={
            "g05_prepare_inputs": {
                "tokenizer_path": str(tokenizer_path),
                "action_tokenizer_path": str(action_tokenizer_path) if config.discrete_action else "",
            }
        },
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=artifact_root,
        config_filename=postprocessor_config_filename,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    if config.normalization_strategy == "lerobot" and dataset_stats is not None:
        preprocessor = hotswap_stats(preprocessor, dataset_stats)
        postprocessor = hotswap_stats(postprocessor, dataset_stats)
    relative_step = next(
        (step for step in preprocessor.steps if isinstance(step, RelativeActionsProcessorStep)), None
    )
    if relative_step is not None:
        for step in postprocessor.steps:
            if isinstance(step, AbsoluteActionsProcessorStep):
                step.relative_step = relative_step
    return preprocessor, postprocessor
