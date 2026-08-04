"""Dedicated X-VLA-RMoE processors.

The pretrained X-VLA policy uses absolute 20-D EE6D actions, while
lerobot/libero_plus stores normalized relative OSC commands. This module
converts the demonstrations to the pretrained absolute action contract.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.policies.xvla.processor_xvla import (
    LiberoProcessorStep,
    XVLAAddDomainIdProcessorStep,
    XVLALiberoActionToEE6DProcessorStep as XVLABaseLiberoActionToEE6DProcessorStep,
    XVLARotation6DToAxisAngleProcessorStep,
)
from lerobot.policies.xvla.utils import rotate6d_to_axis_angle
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    ObservationProcessorStep,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    IMAGENET_STATS,
    OBS_IMAGES,
    OBS_PREFIX,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_xvla_rmoe import XVLARMoEConfig

DEFAULT_LIBERO_RENAME_MAP = {
    "observation.images.front": "observation.images.image",
    "observation.images.wrist": "observation.images.image2",
}


def _axis_angle_to_rotation_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle vectors to rotation matrices with Rodrigues' formula."""
    dtype = axis_angle.dtype
    vector = axis_angle.float()
    angle = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    axis = vector / angle.clamp_min(1e-8)
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*axis.shape[:-1], 3, 3)
    identity = torch.eye(3, device=vector.device).expand(*axis.shape[:-1], 3, 3)
    rotation = identity + torch.sin(angle)[..., None] * skew
    rotation = rotation + (1 - torch.cos(angle))[..., None] * (skew @ skew)
    return rotation.to(dtype)


def _axis_angle_to_rotation_6d(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle to X-VLA's two-column 6-D rotation convention."""
    rotation = _axis_angle_to_rotation_matrix(axis_angle)
    return torch.cat((rotation[..., :, 0], rotation[..., :, 1]), -1)


@dataclass
@ProcessorStepRegistry.register(name="xvlarmoe_image_preprocessor")
class XVLARMoEImageProcessorStep(ProcessorStep):
    """Canonicalize uint8 training and float inference images exactly once."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observations = transition.get(TransitionKey.OBSERVATION)
        if not observations:
            return transition
        output_observations = observations.copy()
        for key, image in observations.items():
            if not key.startswith(OBS_IMAGES) or not isinstance(image, torch.Tensor):
                continue
            image = image.float()
            min_value, max_value = image.min().item(), image.max().item()
            if min_value >= 0.0 and max_value > 1.0:
                if max_value > 255.0:
                    raise ValueError(f"Image {key!r} exceeds uint8 range: max={max_value}")
                image = image / 255.0
                min_value, max_value = image.min().item(), image.max().item()
            if min_value >= 0.0 and max_value <= 1.0:
                mean = image.new_tensor(IMAGENET_STATS["mean"])
                std = image.new_tensor(IMAGENET_STATS["std"])
                while mean.ndim < image.ndim:
                    mean, std = mean.unsqueeze(0), std.unsqueeze(0)
                image = (image - mean) / std
            elif min_value < -2.2 or max_value > 2.7:
                raise ValueError(
                    f"Image {key!r} is neither raw [0, 255]/[0, 1] nor ImageNet-normalized: "
                    f"min={min_value}, max={max_value}"
                )
            output_observations[key] = image
        output = transition.copy()
        output[TransitionKey.OBSERVATION] = output_observations
        return output

    def transform_features(self, features):
        return features

    def get_config(self) -> dict[str, Any]:
        return {}


@dataclass
@ProcessorStepRegistry.register(name="xvlarmoe_legacy_libero_observation")
class XVLARMoELegacyLiberoObservationProcessorStep(ObservationProcessorStep):
    """Build the 8-D proprio state seen by the existing RMoE fine-tuning checkpoints."""

    def observation(self, observation):
        output = observation.copy()
        agentview_key = f"{OBS_IMAGES}.image"
        if agentview_key in output:
            output[agentview_key] = torch.flip(output[agentview_key], dims=[2, 3])

        state_key = f"{OBS_PREFIX}robot_state"
        if state_key in output:
            state = output.pop(state_key)
            quat = state["eef"]["quat"]
            w = quat[..., 3:4].clamp(-1.0, 1.0)
            denominator = torch.sqrt((1.0 - w.square()).clamp_min(0.0))
            axis_angle = quat[..., :3] * (2.0 * torch.acos(w) / denominator.clamp_min(1e-8))
            axis_angle = torch.where(denominator < 1e-8, torch.zeros_like(axis_angle), axis_angle)
            output[OBS_STATE] = torch.cat(
                (state["eef"]["pos"], axis_angle, state["gripper"]["qpos"]), dim=-1
            ).float()
        return output

    def transform_features(self, features):
        return features


@dataclass
@ProcessorStepRegistry.register(name="xvlarmoe_libero_action_to_ee6d")
class XVLARMoELiberoActionToEE6DProcessorStep(XVLABaseLiberoActionToEE6DProcessorStep):
    """Use the original X-VLA cumulative absolute-pose target conversion."""


@dataclass
@ProcessorStepRegistry.register(name="xvlarmoe_dataset_state_to_pretrained")
class XVLARMoEDatasetStateToPretrainedProcessorStep(ProcessorStep):
    """Convert LIBERO dataset proprio from 8-D axis-angle to pretrained 20-D rot6d."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observations = transition.get(TransitionKey.OBSERVATION)
        if not observations:
            return transition
        state = observations.get(OBS_STATE)
        if not isinstance(state, torch.Tensor) or state.shape[-1] == 20:
            return transition
        if state.shape[-1] != 8:
            raise ValueError(f"Expected 8-D or 20-D LIBERO state, got {tuple(state.shape)}")
        proprio = state.new_zeros(*state.shape[:-1], 20)
        proprio[..., :3] = state[..., :3]
        proprio[..., 3:9] = _axis_angle_to_rotation_6d(state[..., 3:6])
        output_observations = observations.copy()
        output_observations[OBS_STATE] = proprio
        output = transition.copy()
        output[TransitionKey.OBSERVATION] = output_observations
        return output

    def transform_features(self, features):
        return features

    def get_config(self) -> dict[str, Any]:
        return {}


@dataclass
@ProcessorStepRegistry.register(name="xvlarmoe_ee6d_to_libero_action")
class XVLARMoEEE6DToLiberoActionProcessorStep(ProcessorStep):
    """Decode X-VLA's first arm slot to an absolute 7-D LIBERO command."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None or not isinstance(action, torch.Tensor):
            return transition
        if action.shape[-1] < 10:
            raise ValueError(f"Expected at least 10 EE6D values, got {tuple(action.shape)}")
        leading_shape = action.shape[:-1]
        rotation = action[..., 3:9].detach().float().cpu().reshape(-1, 6).numpy()
        axis_angle = rotate6d_to_axis_angle(rotation).reshape(*leading_shape, 3)
        axis_angle = torch.from_numpy(np.asarray(axis_angle)).to(action.device, action.dtype)
        gripper = torch.where(action[..., 9:10] > 0.5, 1.0, -1.0).to(action.dtype)
        output = transition.copy()
        output[TransitionKey.ACTION] = torch.cat((action[..., :3], axis_angle, gripper), -1)
        return output

    def transform_features(self, features):
        return features

    def get_config(self) -> dict[str, Any]:
        return {}


def reconcile_xvlarmoe_processors(config, preprocessor, postprocessor):
    """Inject LIBERO adapters into a processor loaded from plain XVLA."""
    rename_step = next(
        (step for step in preprocessor.steps if isinstance(step, RenameObservationsProcessorStep)), None
    )
    if rename_step is not None and not rename_step.rename_map:
        rename_step.rename_map = DEFAULT_LIBERO_RENAME_MAP.copy()

    domain_step = next(
        (step for step in preprocessor.steps if isinstance(step, XVLAAddDomainIdProcessorStep)), None
    )
    if domain_step is not None:
        domain_step.domain_id = 0

    insert_at = next(
        (
            index
            for index, step in enumerate(preprocessor.steps)
            if isinstance(step, (DeviceProcessorStep, NormalizerProcessorStep))
        ),
        len(preprocessor.steps),
    )
    if not any(isinstance(step, XVLARMoEImageProcessorStep) for step in preprocessor.steps):
        preprocessor.steps.insert(insert_at, XVLARMoEImageProcessorStep())
        insert_at += 1
    if config.action_mode.lower() == "ee6d" and not any(
        isinstance(step, XVLARMoELiberoActionToEE6DProcessorStep) for step in preprocessor.steps
    ):
        preprocessor.steps.insert(insert_at, XVLARMoELiberoActionToEE6DProcessorStep())
        insert_at += 1
    if not any(
        isinstance(step, XVLARMoEDatasetStateToPretrainedProcessorStep) for step in preprocessor.steps
    ):
        preprocessor.steps.insert(insert_at, XVLARMoEDatasetStateToPretrainedProcessorStep())
    return preprocessor, postprocessor


def make_xvlarmoe_pre_post_processors(
    config: XVLARMoEConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
):
    """Preserve original XVLA policy processing and convert 7-D training actions."""
    features = {**config.input_features, **config.output_features}
    input_steps = [
        RenameObservationsProcessorStep(rename_map=DEFAULT_LIBERO_RENAME_MAP.copy()),
        AddBatchDimensionProcessorStep(),
        TokenizerProcessorStep(
            tokenizer_name=config.tokenizer_name,
            max_length=config.tokenizer_max_length,
            padding=config.pad_language_to,
            padding_side=config.tokenizer_padding_side,
        ),
        XVLARMoEImageProcessorStep(),
        XVLAAddDomainIdProcessorStep(domain_id=0),
    ]
    if config.action_mode.lower() == "ee6d":
        input_steps.append(XVLARMoELiberoActionToEE6DProcessorStep())
    input_steps.append(XVLARMoEDatasetStateToPretrainedProcessorStep())
    input_steps.extend(
        [
            DeviceProcessorStep(device=config.device),
            NormalizerProcessorStep(
                features=features, norm_map=config.normalization_mapping, stats=dataset_stats
            ),
        ]
    )
    return (
        PolicyProcessorPipeline(steps=input_steps, name=POLICY_PREPROCESSOR_DEFAULT_NAME),
        PolicyProcessorPipeline(
            steps=[
                UnnormalizerProcessorStep(
                    features=config.output_features,
                    norm_map=config.normalization_mapping,
                    stats=dataset_stats,
                ),
                DeviceProcessorStep(device="cpu"),
            ],
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


def make_xvlarmoe_libero_pre_post_processors():
    """Use pretrained X-VLA geometry while policy preprocessing owns image normalization."""
    return (
        PolicyProcessorPipeline(steps=[LiberoProcessorStep()]),
        PolicyProcessorPipeline(steps=[XVLARotation6DToAxisAngleProcessorStep()]),
    )
