from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode, functional as tvf

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_being_h05 import BeingH05Config
from .semantic import ACTION_SLOTS, STATE_SLOTS, atomic4_to_named, pack_named, unpack_action

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
@ProcessorStepRegistry.register(name="being_h05_semantic_pack")
class BeingH05SemanticPackStep(ProcessorStep):
    image_keys: list[str]
    prompt_template: str
    chunk_size: int
    atomic_4_adapter: bool = False
    state_slots: dict[str, tuple[int, int]] = field(default_factory=lambda: dict(STATE_SLOTS))
    action_slots: dict[str, tuple[int, int]] = field(default_factory=lambda: dict(ACTION_SLOTS))

    def get_config(self) -> dict[str, Any]:
        return {
            "image_keys": self.image_keys,
            "prompt_template": self.prompt_template,
            "chunk_size": self.chunk_size,
            "atomic_4_adapter": self.atomic_4_adapter,
            "state_slots": self.state_slots,
            "action_slots": self.action_slots,
        }

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = dict(transition.get(TransitionKey.OBSERVATION) or {})
        complementary = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        task = complementary.get("task")
        if task is None:
            raise ValueError("Being-H0.5 requires the already-selected LeRobot task string.")
        tasks = [task] if isinstance(task, str) else list(task)
        if not all(isinstance(item, str) for item in tasks):
            raise TypeError("Being-H0.5 task values must be strings.")
        # Audit hook: this value is captured before model-specific formatting.
        complementary["being_h05_raw_task"] = list(tasks)
        complementary["being_h05_prompt"] = [
            self.prompt_template.format(task_description=item, k=self.chunk_size) for item in tasks
        ]

        action = transition.get(TransitionKey.ACTION)
        if self.atomic_4_adapter:
            if OBS_STATE not in observation:
                raise ValueError("atomic_4 mapping requires observation.state.")
            named = atomic4_to_named(observation[OBS_STATE], action)
        else:
            named = {}
            for semantic in self.state_slots:
                key = f"observation.state.{semantic}"
                if key in observation:
                    named[semantic] = observation[key]
            if action is not None and action.shape[-1] == 12:
                named.update(
                    {
                        "action.eef_position": action[..., 0:3],
                        "action.eef_rotation": action[..., 3:6],
                        "action.gripper_position": action[..., 6:7],
                        "action.base_motion": action[..., 7:11],
                        "action.control_mode": action[..., 11:12],
                    }
                )

        state_values = {key: value for key, value in named.items() if not key.startswith("action.")}
        if not state_values:
            raise ValueError("No named Being-H0.5 state modalities were present.")
        state, state_mask = pack_named(state_values, self.state_slots)
        observation["being_h05.state"] = state
        observation["being_h05.state_valid"] = state_mask

        images = []
        image_present = []
        for key in self.image_keys:
            value = observation.get(key)
            if value is None:
                value = torch.zeros(state.shape[0], 3, 224, 224, dtype=state.dtype, device=state.device)
                image_present.append(torch.zeros(state.shape[0], dtype=torch.bool, device=state.device))
            else:
                if value.ndim != 4:
                    raise ValueError(f"{key} must have shape (B,C,H,W), got {tuple(value.shape)}")
                processed = []
                for frame in value:
                    if frame.is_floating_point():
                        frame = (frame.clamp(0, 1) * 255).round().to(torch.uint8)
                    else:
                        frame = frame.to(torch.uint8)
                    pil = Image.fromarray(frame.permute(1, 2, 0).cpu().numpy())
                    pil = tvf.resize(
                        pil,
                        224,
                        interpolation=InterpolationMode.BICUBIC,
                        antialias=True,
                    )
                    pil = tvf.center_crop(pil, [224, 224])
                    processed.append(tvf.to_tensor(pil))
                value = torch.stack(processed).to(device=state.device)
                mean = value.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
                std = value.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
                value = (value - mean) / std
                image_present.append(torch.ones(state.shape[0], dtype=torch.bool, device=state.device))
            images.append(value)
        observation["being_h05.pixel_values"] = torch.stack(images, dim=1)
        observation["being_h05.image_valid"] = torch.stack(image_present, dim=1)

        if action is not None:
            action_values = {
                key.removeprefix("action."): value
                for key, value in named.items()
                if key.startswith("action.")
            }
            for binary_key in ("gripper_position", "control_mode"):
                if binary_key in action_values:
                    action_values[binary_key] = (action_values[binary_key] > 0.5).to(
                        action_values[binary_key].dtype
                    )
            semantic_action, action_mask = pack_named(action_values, self.action_slots)
            complementary["being_h05.action_valid"] = action_mask
            transition[TransitionKey.ACTION] = semantic_action
        transition[TransitionKey.OBSERVATION] = observation
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register(name="being_h05_semantic_unpack")
class BeingH05SemanticUnpackStep(ProcessorStep):
    atomic_4_adapter: bool = False

    def get_config(self) -> dict[str, Any]:
        return {"atomic_4_adapter": self.atomic_4_adapter}

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        named = unpack_action(action)
        gripper = (named["gripper_position"] > 0.5).to(action.dtype)
        control_mode = (named["control_mode"] > 0.5).to(action.dtype)
        if self.atomic_4_adapter:
            # The published RoboCasa client inverts gripper convention but not
            # control mode before passing commands to PandaOmron.
            gripper = 1 - 2 * gripper
            control_mode = 2 * control_mode - 1
        transition[TransitionKey.ACTION] = torch.cat(
            [
                named["eef_position"],
                named["eef_rotation"],
                gripper,
                named["base_motion"],
                control_mode,
            ],
            dim=-1,
        )
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_being_h05_pre_post_processors(
    config: BeingH05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,  # noqa: ARG001
):
    semantic_step = BeingH05SemanticPackStep(
        image_keys=config.image_keys,
        prompt_template=config.prompt_template,
        chunk_size=config.chunk_size,
        atomic_4_adapter=config.atomic_4_adapter,
    )
    pre = PolicyProcessorPipeline(
        steps=[AddBatchDimensionProcessorStep(), semantic_step, DeviceProcessorStep(config.device)],
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
    )
    post = PolicyProcessorPipeline[PolicyAction, PolicyAction](
        steps=[
            BeingH05SemanticUnpackStep(atomic_4_adapter=config.atomic_4_adapter),
            DeviceProcessorStep("cpu"),
        ],
        name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return pre, post
