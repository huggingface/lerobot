from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

FIXED_CHECKPOINT = Path(
    "/home/ubuntu24/Teleop/artifacts/training/"
    "task1_picklift_real24_questsim24_act_v2/full_100k/checkpoints/"
    "100000/pretrained_model"
)
FIXED_MODEL_SHA256 = (
    "b7faae880393bdbf5e44ebeaab1f399f732d6ee325be698f999c90eb865cee68"
)
FIXED_PREPROCESSOR_SHA256 = (
    "adc8a12dd079a93b4e6fd4e7f15e93126c9927463593a4de3174643e59fca28a"
)
FIXED_NORMALIZER_STATS_SHA256 = (
    "422dc7786245bb41c7799e0efc12570970db1b5802f58bb1170af6bdcae78893"
)
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
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


def validate_saved_imagenet_processor(checkpoint: Path) -> dict[str, Any]:
    train_config_path = checkpoint / "train_config.json"
    preprocessor_path = checkpoint / "policy_preprocessor.json"
    stats_path = (
        checkpoint
        / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    )
    train_config = json.loads(train_config_path.read_text(encoding="utf-8"))
    if train_config["dataset"].get("use_imagenet_stats") is not True:
        raise RuntimeError("Frozen mixed v2 checkpoint does not declare ImageNet stats.")
    if sha256_file(preprocessor_path) != FIXED_PREPROCESSOR_SHA256:
        raise RuntimeError("Frozen mixed v2 policy preprocessor hash mismatch.")
    if sha256_file(stats_path) != FIXED_NORMALIZER_STATS_SHA256:
        raise RuntimeError("Frozen mixed v2 normalizer stats hash mismatch.")
    stats = load_file(stats_path)
    mean = stats["observation.images.front.mean"].cpu().numpy().reshape(-1)
    std = stats["observation.images.front.std"].cpu().numpy().reshape(-1)
    if not np.allclose(mean, IMAGENET_MEAN, rtol=0.0, atol=1.0e-7):
        raise RuntimeError(f"Saved visual mean is not ImageNet mean: {mean}.")
    if not np.allclose(std, IMAGENET_STD, rtol=0.0, atol=1.0e-7):
        raise RuntimeError(f"Saved visual std is not ImageNet std: {std}.")
    return {
        "status": "pass_checkpoint_owned_imagenet_stats",
        "use_imagenet_stats": True,
        "policy_preprocessor_path": str(preprocessor_path),
        "policy_preprocessor_sha256": FIXED_PREPROCESSOR_SHA256,
        "normalizer_stats_path": str(stats_path),
        "normalizer_stats_sha256": FIXED_NORMALIZER_STATS_SHA256,
        "visual_mean": mean.tolist(),
        "visual_std": std.tolist(),
    }


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


class Task1MixedV2ActSimInference:
    """ACT inference for the formal Nexus adapter with no Follower safety path."""

    def __init__(
        self,
        checkpoint: Path = FIXED_CHECKPOINT,
        device: str = "cuda",
    ) -> None:
        checkpoint = Path(checkpoint)
        if checkpoint.resolve() != FIXED_CHECKPOINT.resolve():
            raise RuntimeError(
                "Only the frozen mixed v2 Task1 100k checkpoint is allowed."
            )
        model_hash = sha256_file(checkpoint / "model.safetensors")
        if model_hash != FIXED_MODEL_SHA256:
            raise RuntimeError("Frozen mixed v2 Task1 checkpoint hash mismatch.")
        self.processor_contract = validate_saved_imagenet_processor(checkpoint)

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
