# ------------------------------------------------------------------------------
# Copyright 2025 The HuggingFace Inc. team and 2toINF (https://github.com/2toINF)
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
# ------------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.xvla.configuration_xvla import XVLAConfig
from lerobot.policies.xvla.utils import rotate6d_to_axis_angle
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    ObservationProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_IMAGES,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


def make_xvla_pre_post_processors(
    config: XVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Build the LeRobot processor pipelines for XVLA.
    """

    features = {**config.input_features, **config.output_features}
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        TokenizerProcessorStep(
            tokenizer_name=config.tokenizer_name,
            max_length=config.tokenizer_max_length,
            padding=config.pad_language_to,
            padding_side=config.tokenizer_padding_side,
        ),
        DeviceProcessorStep(device=config.device),
        XVLAAddDomainIdProcessorStep(),
        NormalizerProcessorStep(
            features=features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


# Custom XVLA processor steps
@dataclass
class LiberoProcessorStep(ObservationProcessorStep):
    """
    Processes LIBERO observations into the LeRobot format.

    This step handles the specific observation structure from LIBERO environments,
    which includes nested robot_state dictionaries and image observations.

    **State Processing:**
    -   Processes the `robot_state` dictionary which contains nested end-effector,
        gripper, and joint information.
    -   Extracts and concatenates:
        - End-effector position (3D)
        - End-effector quaternion converted to axis-angle (3D)
        - Gripper joint positions (2D)
    -   Maps the concatenated state to `"observation.state"`.

    **Image Processing:**
    -   Rotates images by 180 degrees by flipping both height and width dimensions.
    -   This accounts for the HuggingFaceVLA/libero camera orientation convention.
    """

    def _process_observation(self, observation):
        """
        Processes both image and robot_state observations from LIBERO.
        """
        processed_obs = observation.copy()
        for key in list(processed_obs.keys()):
            if key.startswith(f"{OBS_IMAGES}."):
                img = processed_obs[key]

                if key == f"{OBS_IMAGES}.image":
                    # Flip both H and W
                    img = torch.flip(img, dims=[2, 3])

                processed_obs[key] = img
        # Process robot_state into a flat state vector
        if "observation.robot_state" in processed_obs:
            robot_state = processed_obs.pop("observation.robot_state")

            # Extract components
            eef_pos = robot_state["eef"]["pos"]  # (B, 3,)
            eef_mat = robot_state["eef"]["mat"]  # (B, 3, 3)
            eef_rot6d = self._mat_to_rotate6d(eef_mat)  # (B, 6)

            extra = torch.zeros((eef_pos.shape[0], 1), dtype=torch.float32, device=eef_pos.device)

            proprio_state = torch.cat((eef_pos, eef_rot6d, extra), dim=-1)  # (B, 10)
            state = torch.cat((proprio_state, torch.zeros_like(proprio_state)), dim=-1)  # (B, 20)
            # ensure float32
            state = state.float()
            if state.dim() == 1:
                state = state.unsqueeze(0)

            processed_obs[OBS_STATE] = state
        return processed_obs

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Transforms feature keys from the LIBERO format to the LeRobot standard.
        """
        new_features: dict[PipelineFeatureType, dict[str, PolicyFeature]] = {}

        # copy over non-STATE features
        for ft, feats in features.items():
            if ft != PipelineFeatureType.STATE:
                new_features[ft] = feats.copy()

        # rebuild STATE features
        state_feats = {}

        # add our new flattened state
        state_feats["observation.state"] = PolicyFeature(
            key="observation.state",
            shape=(20,),
            dtype="float32",
        )

        new_features[PipelineFeatureType.STATE] = state_feats

        return new_features

    def _mat_to_rotate6d(self, rot_mats: torch.Tensor) -> torch.Tensor:
        """
        Convert batched rotation matrices (B, 3, 3) into 6D rotation representation (B, 6).

        Args:
            rot_mats (Tensor): Rotation matrices of shape (B, 3, 3)

        Returns:
            Tensor: 6D rotation representation, shape (B, 6)

        Raises:
            TypeError: if input is not a torch tensor
            ValueError: if shape is not (B, 3, 3)
        """

        if not isinstance(rot_mats, torch.Tensor):
            raise TypeError(f"mat_to_rot6d expects a torch.Tensor, got {type(rot_mats)}")

        if rot_mats.ndim != 3 or rot_mats.shape[1:] != (3, 3):
            raise ValueError(f"mat_to_rot6d expects shape (B, 3, 3), got {tuple(rot_mats.shape)}")

        rot_mats = rot_mats.to(torch.float32)

        col1 = rot_mats[:, :3, 0]  # (B, 3)
        col2 = rot_mats[:, :3, 1]  # (B, 3)

        rot6d = torch.cat([col1, col2], dim=-1)  # (B, 6)

        return rot6d

    def observation(self, observation):
        return self._process_observation(observation)


@dataclass
@ProcessorStepRegistry.register(name="xvla_image_scale")
class XVLAImageScaleProcessorStep(ProcessorStep):
    """Scale image observations by 255 to convert from [0, 1] to [0, 255] range.

    This processor step multiplies all image observations by 255, which is required
    for XVLA models that expect images in uint8-like range.

    Args:
        image_keys: List of observation keys that contain images to scale.
                   If None, will automatically detect keys starting with "observation.images."
    """

    image_keys: list[str] | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Scale image observations by 255."""
        new_transition = transition.copy()
        obs = new_transition.get(TransitionKey.OBSERVATION, {})
        if obs is None:
            return new_transition

        # Make a copy of observations to avoid modifying the original
        obs = obs.copy()

        # Determine which keys to scale
        keys_to_scale = self.image_keys
        if keys_to_scale is None:
            # Auto-detect image keys
            keys_to_scale = [k for k in obs if k.startswith("observation.images.")]

        # Scale each image
        for key in keys_to_scale:
            if key in obs and isinstance(obs[key], torch.Tensor):
                obs[key] = obs[key] * 255

        new_transition[TransitionKey.OBSERVATION] = obs
        return new_transition

    def transform_features(self, features):
        """Image scaling doesn't change feature structure."""
        return features

    def get_config(self) -> dict[str, Any]:
        """Return serializable configuration."""
        return {
            "image_keys": self.image_keys,
        }


@dataclass
@ProcessorStepRegistry.register(name="xvla_add_domain_id")
class XVLAAddDomainIdProcessorStep(ProcessorStep):
    """Add domain_id to complementary data.

    This processor step adds a domain_id tensor to the complementary data,
    which is used by XVLA to identify different robot embodiments or task domains.

    Args:
        domain_id: The domain ID to add (default: 3)
        device: Device to place the domain_id tensor on (default: "cuda")
    """

    domain_id: int = 0
    device: str = "cuda"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Add domain_id to complementary data."""
        new_transition = transition.copy()
        comp = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        comp = {} if comp is None else comp.copy()

        # Infer batch size from observation tensors
        obs = new_transition.get(TransitionKey.OBSERVATION, {})
        batch_size = 1
        if obs:
            for v in obs.values():
                if isinstance(v, torch.Tensor):
                    batch_size = v.shape[0]
                    break

        # Add domain_id tensor
        comp["domain_id"] = torch.tensor([int(self.domain_id)] * batch_size, dtype=torch.long).to(self.device)

        new_transition[TransitionKey.COMPLEMENTARY_DATA] = comp
        return new_transition

    def transform_features(self, features):
        """Domain ID addition doesn't change feature structure."""
        return features

    def get_config(self) -> dict[str, Any]:
        """Return serializable configuration."""
        return {
            "domain_id": self.domain_id,
            "device": self.device,
        }


@dataclass
@ProcessorStepRegistry.register(name="xvla_rotation_6d_to_axis_angle")
class XVLARotation6DToAxisAngleProcessorStep(ProcessorStep):
    """Convert 6D rotation representation to axis-angle and reorganize action dimensions.

    This processor step takes actions with 6D rotation representation and converts them to
    axis-angle representation, reorganizing the action dimensions as:
    - action[:, :3] -> target_eef (end-effector position)
    - action[:, 3:9] -> 6D rotation (converted to axis-angle, 3D)
    - action[:, 9:10] -> gripper action

    Final output: [target_eef (3), axis_angle (3), gripper (1)] = 7D action

    Args:
        expected_action_dim: Expected input action dimension (default: 10, supports 6D rotation + extras)
    """

    expected_action_dim: int = 10

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Convert 6D rotation to axis-angle in action."""
        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)

        if action is None or not isinstance(action, torch.Tensor):
            return new_transition

        # Convert to numpy for processing
        device = action.device
        dtype = action.dtype
        action_np = action.cpu().numpy()

        # Extract components
        # action shape: (B, D) where D >= 10
        target_eef = action_np[:, :3]  # (B, 3)
        rotation_6d = action_np[:, 3:9]  # (B, 6)
        target_act = action_np[:, 9:10]  # (B, 1)

        # Convert 6D rotation to axis-angle
        target_axis = rotate6d_to_axis_angle(rotation_6d)  # (B, 3)

        # Concatenate: [eef (3), axis_angle (3), gripper (1)] = 7D
        action_np = np.concatenate([target_eef, target_axis, target_act], axis=-1)

        # Convert gripper action to -1 or 1
        action_np[:, -1] = np.where(action_np[:, -1] > 0.5, 1.0, -1.0)

        # Convert back to tensor
        action = torch.from_numpy(action_np).to(device=device, dtype=dtype)

        new_transition[TransitionKey.ACTION] = action
        return new_transition

    def transform_features(self, features):
        """Rotation conversion changes action dimension from 10 to 7."""
        # Note: This is a simplified version. In practice, you might want to
        # update the action feature shape in the features dict.
        return features

    def get_config(self) -> dict[str, Any]:
        """Return serializable configuration."""
        return {
            "expected_action_dim": self.expected_action_dim,
        }


def make_xvla_libero_pre_post_processors() -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Build the LeRobot processor pipelines for XVLA with LIBERO environment.
    """
    pre_processor_steps: list[ProcessorStep] = []
    post_processor_steps: list[ProcessorStep] = []
    pre_processor_steps.extend([LiberoProcessorStep(), XVLAAddDomainIdProcessorStep()])
    post_processor_steps.extend([XVLARotation6DToAxisAngleProcessorStep()])
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=pre_processor_steps,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=post_processor_steps,
        ),
    )

__all__ = [
    "XVLAAddDomainIdProcessorStep",
]