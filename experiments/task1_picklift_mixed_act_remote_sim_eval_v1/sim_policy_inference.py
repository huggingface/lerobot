from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch


FIXED_CHECKPOINT = Path(
    "/home/ubuntu24/Teleop/artifacts/training/"
    "task1_picklift_real24_questsim24_act_v1/full_100k/checkpoints/"
    "100000/pretrained_model"
)
FIXED_MODEL_SHA256 = (
    "e054e682057f09a4653af00a4580da173d3d1658ef5c34244bdbf3ca1a125de5"
)
JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
DATASET_UNITS = (
    "degrees",
    "degrees",
    "degrees",
    "degrees",
    "degrees",
    "range_0_100_percent",
)
STATE_SHAPE = (6,)
FRONT_RGB_SHAPE = (480, 640, 3)
ACTION_SHAPE = (6,)
CONTROL_HZ = 20
MAX_RELATIVE_TARGET = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_observation(
    state: np.ndarray,
    front_rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    state_array = np.asarray(state)
    image_array = np.asarray(front_rgb)
    if state_array.shape != STATE_SHAPE:
        raise RuntimeError(
            f"observation.state must have shape {STATE_SHAPE}, "
            f"got {state_array.shape}."
        )
    if state_array.dtype != np.float32:
        raise RuntimeError(
            "observation.state must be float32 dataset units, "
            f"got {state_array.dtype}."
        )
    if not bool(np.isfinite(state_array).all()):
        raise RuntimeError("observation.state contains NaN or infinity.")
    if image_array.shape != FRONT_RGB_SHAPE:
        raise RuntimeError(
            f"active Remote canonical front must have RGB shape "
            f"{FRONT_RGB_SHAPE}, got {image_array.shape}."
        )
    if image_array.dtype != np.uint8:
        raise RuntimeError(
            "active Remote canonical front must be uint8 RGB, "
            f"got {image_array.dtype}."
        )
    return np.ascontiguousarray(state_array), np.ascontiguousarray(image_array)


def requested_action_from_raw(raw_action: Any) -> np.ndarray:
    raw = np.asarray(raw_action, dtype=np.float32)
    if raw.shape != ACTION_SHAPE:
        raise RuntimeError(
            f"ACT output must have shape {ACTION_SHAPE}, got {raw.shape}."
        )
    if not bool(np.isfinite(raw).all()):
        raise RuntimeError("ACT output contains NaN or infinity.")
    return np.ascontiguousarray(raw.copy())


@dataclass(frozen=True)
class PolicyStep:
    raw_action: np.ndarray
    requested_action: np.ndarray

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "raw_action": self.raw_action.tolist(),
            "requested_action": self.requested_action.tolist(),
            "raw_action_finite": bool(np.isfinite(self.raw_action).all()),
            "requested_action_finite": bool(
                np.isfinite(self.requested_action).all()
            ),
            "raw_equals_requested": bool(
                np.array_equal(self.raw_action, self.requested_action)
            ),
            "runner_state_projection": "none",
            "runner_absolute_calibration_clamp": "none",
            "runner_relative_clamp": "none",
            "max_relative_target": None,
            "action_units": list(DATASET_UNITS),
            "joint_order": list(JOINT_ORDER),
        }


class Task1MixedActSimInference:
    """ACT inference for the formal Nexus adapter with no Follower safety path."""

    def __init__(
        self,
        checkpoint: Path = FIXED_CHECKPOINT,
        device: str = "cuda",
    ) -> None:
        checkpoint = Path(checkpoint)
        if checkpoint.resolve() != FIXED_CHECKPOINT.resolve():
            raise RuntimeError("Only the frozen mixed Task1 100k checkpoint is allowed.")
        model_hash = sha256_file(checkpoint / "model.safetensors")
        if model_hash != FIXED_MODEL_SHA256:
            raise RuntimeError("Frozen mixed Task1 checkpoint hash mismatch.")

        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.act import ACTPolicy

        self.checkpoint = checkpoint
        self.model_hash = model_hash
        self.device = torch.device(device)
        self.model = ACTPolicy.from_pretrained(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        if self.model.config.chunk_size != 67:
            raise RuntimeError("Frozen mixed ACT chunk_size drifted.")
        if self.model.config.n_action_steps != 67:
            raise RuntimeError("Frozen mixed ACT n_action_steps drifted.")
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.model.config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={
                "device_processor": {"device": str(self.device)}
            },
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
            robot_type="so101",
        )
        with torch.inference_mode():
            processed = self.preprocessor(observation)
            raw_tensor = self.postprocessor(self.model.select_action(processed))
        raw_action = requested_action_from_raw(
            raw_tensor.detach().cpu().numpy().reshape(-1)
        )
        requested_action = requested_action_from_raw(raw_action)
        self.step_index += 1
        return PolicyStep(
            raw_action=raw_action,
            requested_action=requested_action,
        )
