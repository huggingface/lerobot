#!/usr/bin/env python
"""Privileged grasp-factor ablation: separates what's left of the grasp failure (after xyz is
oracle-corrected) into gripper-close-*timing* vs *orientation*, via sequential/cumulative
privileged (diagnostic-only, not deployable) corrections. No model/execution/env code is edited.

Four conditions, same 3 seeds, strictly cumulative:
  A. policy_baseline           -- the checkpoint, completely unmodified.
  B. oracle_xyz                -- xyz forced to the target card's live position during the
                                   grasp-approach window (same mechanism/window as
                                   `oracle_xyz_grasp_diagnostic.py`); rotation/gripper/timing left
                                   to the policy.
  C. oracle_xyz_close_timing   -- B, PLUS: the gripper is privilege-forced closed starting the
                                   instant EE first enters the target's grasp-tolerance-or-contact
                                   condition (`oracle_close_step` below), held closed for a short,
                                   fixed window, then released back to the policy's own gripper
                                   prediction. Rotation is still the policy's own.
  D. oracle_xyz_orientation_close_timing -- C, PLUS: rotation forced to a heuristic "oracle"
                                   orientation (the target card's own current orientation, as a
                                   best-effort proxy for a canonical grasp approach angle --
                                   VLABench defines no ground-truth grasp orientation at all, see
                                   module docstring's own caveat below).

Interpretation (as specified by the requester):
  - success appears at C but not B -> gripper-close *timing* was the added bottleneck beyond xyz.
  - success appears at D but not C -> *orientation* was the added bottleneck.
  - failure persists through D -> look past grasp itself, at lift / phase / execution.

Definitions:
  - `first_near_contact_step`: first step EE-to-target-card distance <= GRASP_TOLERANCE (0.05m,
    same constant used throughout this session's other diagnostics).
  - `first_physical_contact_step`: first step `target.is_grasped(physics, robot)` is True -- a
    pure MuJoCo contact query (see `place_phase_forensics.py`'s own docstring), independent of
    distance or of whether the *commanded* gripper is closed.
  - `oracle_close_step` = min(the two above) -- "first time EE enters the target's grasp
    tolerance/contact condition," evaluated live against *this condition's own* trajectory (so it
    can differ across conditions, since B/C/D's xyz differs from A's).
  - `CLOSE_HOLD_STEPS = 30`: the "short duration" the privileged close is held once triggered (C,
    D only) before control reverts to the policy's own gripper channel -- a deliberate finite
    window per the request, not an indefinite override.
  - Orientation-oracle caveat: VLABench's own success condition
    (`IsGraspedCondition` AND `LiftCondition`, see `place_phase_forensics.py`) never checks
    orientation at all, and no per-object "correct grasp pose" is defined anywhere in VLABench's
    source (checked: `Entity.get_grasped_keypoints` only gives a *position* fallback, no
    orientation). The card's own current orientation is used here as the least-arbitrary available
    proxy, not a verified ground truth -- condition D's results should be read as "does *an*
    orientation correction help," not "the correct orientation was supplied."

Usage:
    uv run python examples/safediff_vla/privileged_grasp_factor_ablation.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from lerobot.envs import make_env, make_env_pre_post_processors
from lerobot.envs.configs import VLABenchEnv
from lerobot.envs.utils import NEW_ROLLOUT_OPTION, preprocess_observation
from lerobot.envs.vlabench import VLABenchEnv as VLABenchEnvImpl
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

CHECKPOINT = "outputs/train/safediff_vla_temporal_decoder_padfix_20k/checkpoints/020000/pretrained_model"
TASK = "select_poker"
RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.second_image": "observation.images.camera2",
    "observation.images.wrist_image": "observation.images.camera3",
}
GRIPPER_INDEX = 6
GRIPPER_THRESHOLD = 0.5
GRASP_TOLERANCE = 0.05  # meters -- shared with target_grounding_diagnostic.py
LIFT_THRESHOLD = 0.9  # meters -- VLABench SelectPokerTask condition_config target_height
WINDOW_LEAD = 20  # steps of oracle-corrected approach before the unmodified policy's own close
CLOSE_HOLD_STEPS = 30
CLOSE_STEP_HINT = {1000: 48, 1001: 69, 1002: 65}  # unmodified-policy close steps, already measured
CONDITIONS = (
    "policy_baseline",
    "oracle_xyz",
    "oracle_xyz_close_timing",
    "oracle_xyz_orientation_close_timing",
)


def is_success_from_info(info: dict) -> bool:
    if "final_info" in info:
        final_info = info["final_info"]
        if isinstance(final_info, dict):
            v = final_info.get("is_success", [False])
            return bool(v.tolist()[0] if hasattr(v, "tolist") else v)
        if len(final_info) > 0 and isinstance(final_info[0], dict):
            return bool(final_info[0].get("is_success", False))
        return False
    if "is_success" in info:
        v = info["is_success"]
        return bool(v.tolist()[0] if hasattr(v, "tolist") else v)
    return False


def card_world_xyz_euler(env_impl: VLABenchEnvImpl, target_name: str) -> tuple[np.ndarray, np.ndarray]:
    physics = env_impl._env.physics
    ent = env_impl._env.task.entities[target_name]
    pos = np.asarray(ent.get_xpos(physics), dtype=float)
    quat_wxyz = np.asarray(ent.get_xqaut(physics), dtype=float)
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    euler = Rotation.from_quat(quat_xyzw).as_euler("xyz", degrees=False)
    return pos, euler


def geodesic_angle(euler_a: np.ndarray, euler_b: np.ndarray) -> float:
    r_a = Rotation.from_euler("xyz", euler_a)
    r_b = Rotation.from_euler("xyz", euler_b)
    return float((r_a.inv() * r_b).magnitude())


def run_condition_episode(
    policy: SafeDiffVLAPolicy, device: str, seed: int, condition: str, videos_dir: Path
) -> dict:
    use_xyz = condition != "policy_baseline"
    use_close_timing = condition in ("oracle_xyz_close_timing", "oracle_xyz_orientation_close_timing")
    use_orientation = condition == "oracle_xyz_orientation_close_timing"

    env_cfg = VLABenchEnv(task=TASK)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = envs[TASK][0]
    env_impl: VLABenchEnvImpl = env.envs[0]

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=CHECKPOINT,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy.config
    )

    close_hint = CLOSE_STEP_HINT[seed]
    xyz_window_start = max(1, close_hint - WINDOW_LEAD) if use_xyz else None

    policy.reset()
    observation, _ = env.reset(seed=[seed], options={NEW_ROLLOUT_OPTION: True})
    target_name = env_impl._env.task.target_entity
    base = env_impl._robot_base_xyz if env_impl._robot_base_xyz is not None else np.zeros(3)

    frames: list[np.ndarray] = []
    step_records: list[dict] = []
    first_near_contact_step = None
    first_physical_contact_step = None
    oracle_close_step = None
    oracle_close_active_until = None
    original_policy_close_step = None
    dist_at_original_policy_close = None
    dist_at_oracle_close = None
    geodesic_at_original_policy_close = None
    geodesic_at_oracle_close = None
    max_card_height = -np.inf
    ever_grasped = False
    success = False
    step_ix = 0
    max_steps = 500
    try:
        while step_ix < max_steps:
            frames.append(env_impl.render())

            obs_t = preprocess_observation(observation)
            obs_t["task"] = [env.call("task_description")[0]] if hasattr(env, "call") else [""]
            obs_t = env_preprocessor(obs_t)
            obs_norm = preprocessor(obs_t)

            with torch.no_grad():
                action_norm = policy.select_action(obs_norm)
            action_raw = postprocessor(action_norm)
            action_raw_np = action_raw.clone().cpu().numpy().reshape(-1)
            policy_gripper = float(action_raw_np[GRIPPER_INDEX])

            card_pos_world, card_euler = card_world_xyz_euler(env_impl, target_name)
            card_xyz_robot_frame = card_pos_world - base

            if use_xyz and step_ix >= xyz_window_start:
                action_raw_np[:3] = card_xyz_robot_frame
            if use_orientation and step_ix >= xyz_window_start:
                action_raw_np[3:6] = card_euler

            # Live contact/tolerance detection, against *this condition's own* trajectory.
            physics = env_impl._env.physics
            ee_pos_world = np.asarray(env_impl._env.task.robot.get_end_effector_pos(physics), dtype=float)
            ee_to_card_dist = float(np.linalg.norm(ee_pos_world - card_pos_world))
            is_grasped_now = bool(
                env_impl._env.task.entities[target_name].is_grasped(physics, env_impl._env.task.robot)
            )
            if is_grasped_now:
                ever_grasped = True
            if first_near_contact_step is None and ee_to_card_dist <= GRASP_TOLERANCE:
                first_near_contact_step = step_ix + 1
            if first_physical_contact_step is None and is_grasped_now:
                first_physical_contact_step = step_ix + 1
            if oracle_close_step is None and (
                first_near_contact_step is not None or first_physical_contact_step is not None
            ):
                oracle_close_step = min(
                    v for v in (first_near_contact_step, first_physical_contact_step) if v is not None
                )
                oracle_close_active_until = oracle_close_step + CLOSE_HOLD_STEPS
                dist_at_oracle_close = ee_to_card_dist
                geodesic_at_oracle_close = geodesic_angle(action_raw_np[3:6], card_euler)

            if original_policy_close_step is None and policy_gripper <= GRIPPER_THRESHOLD:
                original_policy_close_step = step_ix + 1
                dist_at_original_policy_close = ee_to_card_dist
                geodesic_at_original_policy_close = geodesic_angle(action_raw_np[3:6], card_euler)

            gripper_forced = False
            if (
                use_close_timing
                and oracle_close_step is not None
                and step_ix + 1 <= oracle_close_active_until
            ):
                action_raw_np[GRIPPER_INDEX] = 0.0  # dataset convention: 0 = fully closed
                gripper_forced = True

            action_transition = env_postprocessor(
                {"action": torch.from_numpy(action_raw_np[None, :]).to(action_raw.dtype)}
            )
            action_env_np = action_transition["action"].clone().cpu().numpy().reshape(-1)

            step_records.append(
                {
                    "step_ix": step_ix,
                    "policy_gripper": policy_gripper,
                    "gripper_forced_closed": gripper_forced,
                    "ee_to_card_dist": ee_to_card_dist,
                    "is_grasped_now": is_grasped_now,
                    "card_height": float(card_pos_world[2]),
                }
            )
            max_card_height = max(max_card_height, float(card_pos_world[2]))

            observation, reward, terminated, truncated, info = env.step(action_env_np[None, :])
            step_ix += 1
            if bool(np.asarray(terminated).any()) or bool(np.asarray(truncated).any()):
                success = is_success_from_info(info)
                break
    finally:
        env.close()

    fps = env_impl.metadata.get("render_fps", 20) if hasattr(env_impl, "metadata") else 20
    videos_dir.mkdir(parents=True, exist_ok=True)
    video_path = videos_dir / f"{condition}_seed{seed}.mp4"
    if frames:
        write_video(video_path, frames, fps=fps)

    return {
        "seed": seed,
        "condition": condition,
        "target_card_identity": target_name,
        "episode_len": step_ix,
        "first_near_contact_step": first_near_contact_step,
        "first_physical_contact_step": first_physical_contact_step,
        "original_policy_close_step": original_policy_close_step,
        "oracle_close_step": oracle_close_step,
        "ee_to_card_dist_at_original_policy_close": dist_at_original_policy_close,
        "ee_to_card_dist_at_oracle_close": dist_at_oracle_close,
        "orientation_geodesic_error_at_original_policy_close_rad": geodesic_at_original_policy_close,
        "orientation_geodesic_error_at_oracle_close_rad": geodesic_at_oracle_close,
        "grasp_contact_success": ever_grasped,
        "max_card_height_m": None if max_card_height == -np.inf else max_card_height,
        "lift_success": (max_card_height >= LIFT_THRESHOLD) if max_card_height != -np.inf else False,
        "task_success": success,
        "video_path": str(video_path),
        "step_records": step_records,
    }


def main() -> None:
    global CHECKPOINT
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="outputs/eval/safediff_vla_privileged_grasp_factor_ablation")
    args = parser.parse_args()
    CHECKPOINT = args.checkpoint

    set_seed(args.seeds[0])
    logger.info("=== loading policy from %s (unmodified) ===", CHECKPOINT)
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT)
    policy = policy.to(args.device)
    policy.eval()
    assert policy.config.architecture == "temporal_decoder"
    assert policy.config.execute_horizon == 50
    assert not policy.config.use_temporal_ensembling

    out_dir = Path(args.output_dir)
    all_results = []
    for condition in CONDITIONS:
        for seed in args.seeds:
            logger.info("=== condition=%s seed=%d ===", condition, seed)
            result = run_condition_episode(policy, args.device, seed, condition, out_dir / "videos")
            all_results.append(result)
            logger.info(
                "condition=%s seed=%d target=%s near_contact=%s phys_contact=%s policy_close=%s "
                "oracle_close=%s dist@policy_close=%s dist@oracle_close=%s geo@policy_close=%s "
                "geo@oracle_close=%s grasp=%s lift=%s task_success=%s",
                condition,
                seed,
                result["target_card_identity"],
                result["first_near_contact_step"],
                result["first_physical_contact_step"],
                result["original_policy_close_step"],
                result["oracle_close_step"],
                result["ee_to_card_dist_at_original_policy_close"],
                result["ee_to_card_dist_at_oracle_close"],
                result["orientation_geodesic_error_at_original_policy_close_rad"],
                result["orientation_geodesic_error_at_oracle_close_rad"],
                result["grasp_contact_success"],
                result["lift_success"],
                result["task_success"],
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(all_results, indent=2))
    logger.info("wrote %s", out_dir / "summary.json")

    logger.info("=== success by condition ===")
    for condition in CONDITIONS:
        n_success = sum(1 for r in all_results if r["condition"] == condition and r["task_success"])
        n_total = sum(1 for r in all_results if r["condition"] == condition)
        logger.info("%s: %d/%d task_success", condition, n_success, n_total)


if __name__ == "__main__":
    main()
