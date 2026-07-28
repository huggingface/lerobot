from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deployment_safety import (
    MAX_RELATIVE_TARGET,
    apply_action_safety,
    project_sim_state_to_calibration,
    sha256_file,
)

FIXED_CHECKPOINT = Path(
    "/home/ubuntu24/Teleop/artifacts/training/task1_picklift_real24_act_v1/"
    "full_100k/checkpoints/100000/pretrained_model"
)
FIXED_MODEL_SHA256 = "ba233bd25aad5bb63a8a863a8fe1f8ebfdb804547a5add33520c8e1e93f890fb"
CONTROL_HZ = 20
STATE_SHAPE = (6,)
FRONT_RGB_SHAPE = (480, 640, 3)
ACTION_SHAPE = (6,)


@dataclass(frozen=True)
class PolicyStep:
    raw_action: np.ndarray
    calibration_clipped_action: np.ndarray
    sent_action: np.ndarray
    calibration_clip_mask: np.ndarray
    relative_clip_mask: np.ndarray
    safety_reference_state: np.ndarray
    sim_state_projection_mask: np.ndarray
    sim_state_projection_delta: np.ndarray

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "raw_action": self.raw_action.tolist(),
            "calibration_clipped_action": self.calibration_clipped_action.tolist(),
            "sent_action": self.sent_action.tolist(),
            "calibration_clip_mask": self.calibration_clip_mask.tolist(),
            "relative_clip_mask": self.relative_clip_mask.tolist(),
            "safety_reference_state": self.safety_reference_state.tolist(),
            "sim_state_projection_mask": self.sim_state_projection_mask.tolist(),
            "sim_state_projection_delta": self.sim_state_projection_delta.tolist(),
            "sim_state_projected": bool(self.sim_state_projection_mask.any()),
            "raw_action_finite": bool(np.isfinite(self.raw_action).all()),
            "sent_action_finite": bool(np.isfinite(self.sent_action).all()),
            "calibration_clip_count": int(self.calibration_clip_mask.sum()),
            "relative_clip_count": int(self.relative_clip_mask.sum()),
        }


def validate_observation(state: np.ndarray, front_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    state_array = np.asarray(state)
    image_array = np.asarray(front_rgb)
    if state_array.shape != STATE_SHAPE:
        raise RuntimeError(f"observation.state must have shape {STATE_SHAPE}, got {state_array.shape}.")
    if not np.issubdtype(state_array.dtype, np.floating):
        raise RuntimeError(f"observation.state must be floating point, got {state_array.dtype}.")
    if not np.isfinite(state_array).all():
        raise RuntimeError("observation.state contains NaN or infinity.")
    if image_array.shape != FRONT_RGB_SHAPE:
        raise RuntimeError(
            f"active Remote canonical front must have RGB shape {FRONT_RGB_SHAPE}, "
            f"got {image_array.shape}."
        )
    if image_array.dtype != np.uint8:
        raise RuntimeError(f"active Remote canonical front must be uint8 RGB, got {image_array.dtype}.")
    return np.ascontiguousarray(state_array, dtype=np.float32), np.ascontiguousarray(image_array)


class Task1ActSimInference:
    """Hardware-free ACT inference endpoint for the Remote adapter.

    This class imports no robot, camera, serial, torque, or environment module.
    The caller owns reset/step/success/termination semantics and supplies only
    the active canonical observation at 20 Hz.
    """

    def __init__(
        self,
        checkpoint: Path = FIXED_CHECKPOINT,
        device: str = "cuda",
        max_relative_target: float = MAX_RELATIVE_TARGET,
    ) -> None:
        checkpoint = Path(checkpoint)
        if checkpoint.resolve() != FIXED_CHECKPOINT.resolve():
            raise RuntimeError("Only the frozen Task1 100k checkpoint is permitted.")
        model_hash = sha256_file(checkpoint / "model.safetensors")
        if model_hash != FIXED_MODEL_SHA256:
            raise RuntimeError("Frozen Task1 100k checkpoint hash mismatch.")

        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.act import ACTPolicy

        self.checkpoint = checkpoint
        self.model_hash = model_hash
        self.device = torch.device(device)
        self.max_relative_target = float(max_relative_target)
        self.model = ACTPolicy.from_pretrained(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.model.config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )
        self.reset_episode()

    def reset_episode(self) -> None:
        self.model.reset()
        self.step_index = 0

    def infer(self, state: np.ndarray, front_rgb: np.ndarray) -> PolicyStep:
        from lerobot.policies.utils import prepare_observation_for_inference

        state_array, image_array = validate_observation(state, front_rgb)
        observation = prepare_observation_for_inference(
            {
                "observation.state": state_array,
                "observation.images.front": image_array,
            },
            device=self.device,
            task="Task 1 PickLift v1",
            robot_type="so101_follower",
        )
        with torch.inference_mode():
            processed = self.preprocessor(observation)
            raw_tensor = self.postprocessor(self.model.select_action(processed))
        raw_action = raw_tensor.detach().cpu().numpy().reshape(-1)
        if raw_action.shape != ACTION_SHAPE:
            raise RuntimeError(f"ACT output must have shape {ACTION_SHAPE}, got {raw_action.shape}.")
        sim_state = project_sim_state_to_calibration(state_array)
        stages = apply_action_safety(
            raw_action=raw_action,
            current_state=sim_state["state"],
            max_relative_target=self.max_relative_target,
        )
        stages.update(
            {
                "safety_reference_state": sim_state["state"],
                "sim_state_projection_mask": sim_state["projection_mask"],
                "sim_state_projection_delta": sim_state["projection_delta"],
            }
        )
        self.step_index += 1
        return PolicyStep(**stages)
