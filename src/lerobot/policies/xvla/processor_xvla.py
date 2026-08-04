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

from lerobot.configs import PipelineFeatureType, PolicyFeature
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

from .configuration_xvla import XVLAConfig
from .utils import axis_angle_to_rotation_matrix, matrix_to_rotate6d, rotate6d_to_axis_angle


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
        XVLAImageToFloatProcessorStep(),
        XVLAImageNetNormalizeProcessorStep(),
        XVLAAddDomainIdProcessorStep(),
    ]
    if config.action_mode.lower() == "ee6d":
        input_steps.append(XVLALiberoActionToEE6DProcessorStep())
    input_steps.extend(
        [
            DeviceProcessorStep(device=config.device),
            NormalizerProcessorStep(
                features=features, norm_map=config.normalization_mapping, stats=dataset_stats
            ),
        ]
    )
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


def reconcile_xvla_processors(config, preprocessor, postprocessor):
    """Inject the LIBERO action adapter into a preprocessor loaded from a pretrained
    XVLA checkpoint whose saved config predates `XVLALiberoActionToEE6DProcessorStep`.
    """
    if config.action_mode.lower() != "ee6d":
        return preprocessor, postprocessor
    if any(isinstance(step, XVLALiberoActionToEE6DProcessorStep) for step in preprocessor.steps):
        return preprocessor, postprocessor
    insert_at = next(
        (
            index
            for index, step in enumerate(preprocessor.steps)
            if isinstance(step, (DeviceProcessorStep, NormalizerProcessorStep))
        ),
        len(preprocessor.steps),
    )
    preprocessor.steps.insert(insert_at, XVLALiberoActionToEE6DProcessorStep())
    return preprocessor, postprocessor


# Custom XVLA processor steps
@dataclass
@ProcessorStepRegistry.register(name="xvla_libero_action_to_ee6d")
class XVLALiberoActionToEE6DProcessorStep(ProcessorStep):
    """Convert a chunk of 7-D LIBERO per-step delta actions into the pretrained
    XVLA policy's 20-D EE6D *absolute*-pose action layout.

    LIBERO/robosuite records demonstrations as `OSC_POSE` deltas in `Box(-1, 1)^7`
    (see docs/source/libero_plus.mdx): `[dx, dy, dz, dRx, dRy, dRz, gripper]`, each
    relative to the pose *at that step*. The controller turns a raw delta into an
    absolute goal pose via (robosuite `osc.py`/`control_utils.py`, using the stock
    `OSC_POSE` config `output_max=[0.05]*3+[0.5]*3`, `input_max=1`):
        goal_pos = current_pos + delta_pos * 0.05
        goal_ori = R(delta_ori * 0.5) @ current_ori
    `lerobot/xvla-libero` is evaluated with `--env.control_mode=absolute`
    (docs/source/xvla.mdx), i.e. its EE6D position/rotation channels are *absolute*
    target poses for each step of the action chunk, not deltas. So the training
    target must compose every raw per-step delta onto the measured pose at the
    same step (`observation.state`: `[eef_pos(3), eef_axis_angle(3),
    gripper_qpos(2)]`). OSC does not reach each goal in one dataset frame, so
    accumulating deltas from the first pose produces increasingly wrong targets.

    Gripper is commanded directly per step (not a delta), so it only needs sign
    binarization: LIBERO's no-op action gripper=-1 is "open" (see `get_libero_dummy_action`).
    """

    model_action_dim: int = 20
    position_scale: float = 0.05
    rotation_scale: float = 0.5

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None or not isinstance(action, torch.Tensor):
            return transition
        if action.shape[-1] == self.model_action_dim:
            return transition
        if action.shape[-1] != 7:
            raise ValueError(f"Expected 7-D LIBERO action, got {tuple(action.shape)}")

        observation = transition.get(TransitionKey.OBSERVATION) or {}
        state = observation.get(OBS_STATE)
        if state is None or not isinstance(state, torch.Tensor) or state.shape[-1] < 6:
            raise ValueError(
                "XVLALiberoActionToEE6DProcessorStep needs `observation.state` "
                "(current eef position + axis-angle orientation) to compose delta "
                f"actions into absolute targets, got: {state if state is None else state.shape}"
            )
        squeeze_chunk_dim = action.ndim == 2
        if squeeze_chunk_dim:
            action = action.unsqueeze(1)

        if state.ndim == 2:
            if action.shape[-2] != 1:
                raise ValueError(
                    "Absolute LIBERO action chunks require action-aligned observation.state values; "
                    f"got state {tuple(state.shape)} for action {tuple(action.shape)}"
                )
            state = state.unsqueeze(-2)
        elif state.ndim != 3 or state.shape[-2] != action.shape[-2]:
            raise ValueError(
                "Expected one observation.state per LIBERO action step, got "
                f"state {tuple(state.shape)} and action {tuple(action.shape)}"
            )

        current_pos = state[..., :3].float()
        current_rot = axis_angle_to_rotation_matrix(state[..., 3:6].float())

        delta_pos = action[..., :3].float() * self.position_scale
        delta_rot = axis_angle_to_rotation_matrix(action[..., 3:6].float() * self.rotation_scale)

        target_pos = current_pos + delta_pos
        target_rot = delta_rot @ current_rot

        converted = action.new_zeros(*action.shape[:-1], self.model_action_dim)
        converted[..., :3] = target_pos.to(action.dtype)
        converted[..., 3:9] = matrix_to_rotate6d(target_rot).to(action.dtype)
        converted[..., 9] = (action[..., 6] > 0).to(action.dtype)

        if squeeze_chunk_dim:
            converted = converted.squeeze(1)

        output = transition.copy()
        output[TransitionKey.ACTION] = converted
        # Future states are training-only target context. Keep policy inputs
        # causal and identical to inference by retaining only the current state.
        if state.shape[-2] > 1:
            output_observations = observation.copy()
            output_observations[OBS_STATE] = state[:, 0, :]
            output[TransitionKey.OBSERVATION] = output_observations
        return output

    def transform_features(self, features):
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "model_action_dim": self.model_action_dim,
            "position_scale": self.position_scale,
            "rotation_scale": self.rotation_scale,
        }


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
        robot_state_str = OBS_PREFIX + "robot_state"
        if robot_state_str in processed_obs:
            robot_state = processed_obs.pop(robot_state_str)

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
        state_feats[OBS_STATE] = PolicyFeature(
            key=OBS_STATE,
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
            keys_to_scale = [k for k in obs if k.startswith(OBS_IMAGES)]

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
@ProcessorStepRegistry.register(name="xvla_image_to_float")
class XVLAImageToFloatProcessorStep(ProcessorStep):
    """Convert image observations from [0, 255] to [0, 1] range.

    This processor step divides image observations by 255 to convert from uint8-like
    range [0, 255] to float range [0, 1]. This is typically used when loading images
    that are stored as uint8 values.

    Args:
        image_keys: List of observation keys that contain images to convert.
                   If None, will automatically detect keys starting with "observation.images."
        validate_range: If True, validates that input values are in [0, 255] range (default: True)

    Raises:
        ValueError: If validate_range is True and image values are not in [0, 255] range.
    """

    image_keys: list[str] | None = None
    validate_range: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Convert image observations from [0, 255] to [0, 1]."""
        new_transition = transition.copy()
        obs = new_transition.get(TransitionKey.OBSERVATION, {})
        if obs is None:
            return new_transition

        # Make a copy of observations to avoid modifying the original
        obs = obs.copy()

        # Determine which keys to convert
        keys_to_convert = self.image_keys
        if keys_to_convert is None:
            # Auto-detect image keys
            keys_to_convert = [k for k in obs if k.startswith(OBS_IMAGES)]

        # Convert each image
        for key in keys_to_convert:
            if key in obs and isinstance(obs[key], torch.Tensor):
                tensor = obs[key]

                min_val = tensor.min().item()
                max_val = tensor.max().item()

                if max_val <= 1.0:
                    obs[key] = tensor.float()  # ensure float dtype, but no division
                    continue
                # Validate that values are in [0, 255] range if requested
                if self.validate_range and (min_val < 0.0 or max_val > 255.0):
                    raise ValueError(
                        f"Image '{key}' has values outside [0, 255] range: "
                        f"min={min_val:.4f}, max={max_val:.4f}. "
                        f"Cannot convert to [0, 1] range."
                    )

                # Convert to float and divide by 255
                obs[key] = tensor.float() / 255.0

        new_transition[TransitionKey.OBSERVATION] = obs
        return new_transition

    def transform_features(self, features):
        """Image conversion doesn't change feature structure."""
        return features

    def get_config(self) -> dict[str, Any]:
        """Return serializable configuration."""
        return {
            "image_keys": self.image_keys,
            "validate_range": self.validate_range,
        }


@dataclass
@ProcessorStepRegistry.register(name="xvla_imagenet_normalize")
class XVLAImageNetNormalizeProcessorStep(ProcessorStep):
    """Normalize image observations using ImageNet statistics.

    This processor step applies ImageNet normalization (mean and std) to image observations.
    It validates that input values are in the [0, 1] range before normalizing.

    The normalization formula is: (image - mean) / std

    Args:
        image_keys: List of observation keys that contain images to normalize.
                   If None, will automatically detect keys starting with "observation.images."

    Raises:
        ValueError: If image values are not in the [0, 1] range.
    """

    image_keys: list[str] | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Normalize image observations using ImageNet statistics."""
        new_transition = transition.copy()
        obs = new_transition.get(TransitionKey.OBSERVATION, {})
        if obs is None:
            return new_transition

        # Make a copy of observations to avoid modifying the original
        obs = obs.copy()

        # Determine which keys to normalize
        keys_to_normalize = self.image_keys
        if keys_to_normalize is None:
            # Auto-detect image keys
            keys_to_normalize = [k for k in obs if k.startswith(OBS_IMAGES)]

        # Normalize each image
        for key in keys_to_normalize:
            if key in obs and isinstance(obs[key], torch.Tensor):
                tensor = obs[key]

                # Validate that values are in [0, 1] range
                min_val = tensor.min().item()
                max_val = tensor.max().item()
                if min_val < 0.0 or max_val > 1.0:
                    raise ValueError(
                        f"Image '{key}' has values outside [0, 1] range: "
                        f"min={min_val:.4f}, max={max_val:.4f}. "
                        f"ImageNet normalization requires input values in [0, 1]."
                    )

                # Apply ImageNet normalization
                mean = torch.tensor(IMAGENET_STATS["mean"], device=tensor.device, dtype=tensor.dtype)
                std = torch.tensor(IMAGENET_STATS["std"], device=tensor.device, dtype=tensor.dtype)

                # Expand mean/std to match tensor dims (e.g., BCHW or BNCHW)
                while mean.dim() < tensor.dim():
                    mean = mean.unsqueeze(0)
                    std = std.unsqueeze(0)

                # Normalize: (image - mean) / std
                obs[key] = (tensor - mean) / std

        new_transition[TransitionKey.OBSERVATION] = obs
        return new_transition

    def transform_features(self, features):
        """ImageNet normalization doesn't change feature structure."""
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
    """

    domain_id: int = 0

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
        comp["domain_id"] = torch.tensor([int(self.domain_id)] * batch_size, dtype=torch.long)

        new_transition[TransitionKey.COMPLEMENTARY_DATA] = comp
        return new_transition

    def transform_features(self, features):
        """Domain ID addition doesn't change feature structure."""
        return features

    def get_config(self) -> dict[str, Any]:
        """Return serializable configuration."""
        return {
            "domain_id": self.domain_id,
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
        # NumPy does not support torch.bfloat16. LIBERO ultimately consumes a NumPy
        # action, so promote bfloat16 policy outputs to float32 for both conversion
        # steps while preserving every already-NumPy-compatible floating dtype.
        output_dtype = torch.float32 if dtype == torch.bfloat16 else dtype
        action_np = action.detach().to(device="cpu", dtype=output_dtype).numpy()

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
        action = torch.from_numpy(action_np).to(device=device, dtype=output_dtype)

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
    pre_processor_steps.extend(
        [LiberoProcessorStep(), XVLAImageNetNormalizeProcessorStep(), XVLAAddDomainIdProcessorStep()]
    )
    post_processor_steps.extend([XVLARotation6DToAxisAngleProcessorStep()])
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=pre_processor_steps,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=post_processor_steps,
        ),
    )
