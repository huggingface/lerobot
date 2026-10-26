#!/usr/bin/env python
"""Pure diagnostic re-rollout (NO training) of `phase_aware_100k_50k` (the checkpoint selected
after the 100k long run) on the SAME 18 dev episodes (seeds 1000-1002, 6 tasks) already evaluated
in `eval_grounded_grasp_v2_5_phase_aware_loss_100k.py` -- re-run only because that script's own
instrumentation never captured ROTATION (`predicted_delta_rot_rad`/`actual_ee_euler_rad`), needed
for `plot_failure_timeseries.py`'s time-series figures. Everything else (position, gripper, phase)
is identical to that script's own `install_env_instrumentation`/`install_v2_5_policy_
instrumentation` -- copied, extended with 2 new fields, not refactored.

New fields per step:
    predicted_delta_rot_rad -- `LocalGraspRefiner`'s own predicted rotation residual
        (`metrics["delta_rot_rad"]`, first horizon step only -- same one actually executed),
        only set on LOCAL_GRASP steps (same as `local_delta_xyz_m`).
    actual_ee_euler_rad -- the physical end-effector's own orientation this step, converted
        quat(w,x,y,z) -> Euler 'xyz' radians via scipy (VLABench's own `get_end_effector_quat`
        returns (w,x,y,z); scipy expects (x,y,z,w), reordered before conversion).

Usage:
    MUJOCO_GL=egl uv run python examples/safediff_vla/rollout_with_rotation_for_failure_plots.py --tasks select_fruit
    MUJOCO_GL=egl uv run python examples/safediff_vla/rollout_with_rotation_for_failure_plots.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import RENAME_MAP, STAGE2_TASK_NAMES  # noqa: E402

from lerobot.envs import make_env, make_env_pre_post_processors  # noqa: E402
from lerobot.envs.configs import VLABenchEnv  # noqa: E402
from lerobot.envs.vlabench import VLABenchEnv as VLABenchEnvImpl  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.local_grasp_refiner import LOCAL_GRASP, PRE_GRASP  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402
from lerobot.policies.safediff_vla.rotation_encoding import GRIPPER_INDEX_RAW  # noqa: E402
from lerobot.scripts.lerobot_eval import eval_policy  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

CHECKPOINT = "outputs/train/safediff_vla_grounded_grasp_v2_5_phase_aware_loss_100k/checkpoints/050000/pretrained_model"
SEEDS = [1000, 1001, 1002]
OUT_DIR = Path("outputs/eval/safediff_vla_grounded_grasp_v2_5_phase_aware_loss_50k_with_rotation")


def robot_base(env: VLABenchEnvImpl) -> np.ndarray:
    base = getattr(env, "_robot_base_xyz", None)
    return np.asarray(base, dtype=float) if base is not None else np.array([0.0, -0.4, 0.78])


def quat_wxyz_to_euler_xyz(quat_wxyz) -> list[float]:
    w, x, y, z = quat_wxyz
    return Rotation.from_quat([x, y, z, w]).as_euler("xyz").tolist()


def extract_physics_record(env: VLABenchEnvImpl, step_ix: int) -> dict[str, Any]:
    physics = env._env.physics
    task = env._env.task
    robot = task.robot
    target_name = task.target_entity
    ee_pos = np.asarray(robot.get_end_effector_pos(physics), dtype=float)
    ee_euler = quat_wxyz_to_euler_xyz(robot.get_end_effector_quat(physics))
    try:
        target_pos = np.asarray(task.entities[target_name].get_xpos(physics), dtype=float)
    except Exception:  # noqa: BLE001
        target_pos = np.array([np.nan, np.nan, np.nan])
    try:
        is_grasped = bool(task.entities[target_name].is_grasped(physics, robot))
    except Exception:  # noqa: BLE001
        is_grasped = False
    return {
        "step_ix": step_ix,
        "ee_pos": ee_pos.tolist(),
        "actual_ee_euler_rad": ee_euler,
        "target_entity": target_name,
        "target_pos": target_pos.tolist(),
        "is_grasped": is_grasped,
        "ee_to_target_dist": float(np.linalg.norm(ee_pos - target_pos)),
    }


def install_env_instrumentation(policy_log: list[dict]) -> dict:
    cls = VLABenchEnvImpl
    orig_step = cls.step
    orig_reset = cls.reset
    state: dict[str, Any] = {"current_episode_steps": []}

    def merge_policy_log(record: dict, base: np.ndarray) -> None:
        keys = (
            "stage", "predicted_target_world", "predicted_target_dist_to_gt_target",
            "local_delta_xyz_m", "predicted_delta_rot_rad", "just_entered_local",
            "just_latched_gripper", "just_transitioned_post_grasp", "applied_gripper",
            "distance_to_predicted_target_at_entry_m",
        )
        for k in keys:
            record[k] = None
        if not policy_log:
            return
        p = policy_log[-1]
        record["stage"] = p["stage"]
        record["just_entered_local"] = p.get("just_entered_local", False)
        record["just_latched_gripper"] = p.get("just_latched_gripper", False)
        record["just_transitioned_post_grasp"] = p.get("just_transitioned_post_grasp", False)
        record["applied_gripper"] = p.get("applied_gripper")
        record["local_delta_xyz_m"] = p.get("local_delta_xyz_m")
        record["predicted_delta_rot_rad"] = p.get("predicted_delta_rot_rad")
        record["distance_to_predicted_target_at_entry_m"] = p.get("distance_to_predicted_target_at_entry_m")
        if p.get("predicted_target_robot_frame") is not None:
            predicted_target_world = np.asarray(p["predicted_target_robot_frame"], dtype=float) + base
            target_pos = np.asarray(record["target_pos"], dtype=float)
            record["predicted_target_world"] = predicted_target_world.tolist()
            record["predicted_target_dist_to_gt_target"] = float(np.linalg.norm(predicted_target_world - target_pos))

    def patched_reset(self, seed=None, **kwargs):
        result = orig_reset(self, seed=seed, **kwargs)
        if seed is not None:
            record = extract_physics_record(self, step_ix=0)
            merge_policy_log(record, robot_base(self))
            state["current_episode_steps"] = [record]
        return result

    def patched_step(self, action):
        result = orig_step(self, action)
        terminated = result[2]
        if not terminated:
            step_ix = len(state["current_episode_steps"])
            record = extract_physics_record(self, step_ix=step_ix)
            merge_policy_log(record, robot_base(self))
            state["current_episode_steps"].append(record)
        return result

    cls.step = patched_step
    cls.reset = patched_reset
    state["_originals"] = (cls, orig_step, orig_reset)
    return state


def uninstall_env_instrumentation(state: dict) -> None:
    cls, orig_step, orig_reset = state["_originals"]
    cls.step = orig_step
    cls.reset = orig_reset


def install_v2_5_policy_instrumentation(policy: SafeDiffVLAPolicy) -> dict:
    orig_plan = SafeDiffVLAPolicy._plan_temporal_decoder
    orig_enter_local = SafeDiffVLAPolicy._maybe_enter_local_grasp
    orig_local_step = SafeDiffVLAPolicy._plan_local_grasp_step
    orig_select_local = SafeDiffVLAPolicy._select_action_local_grasp
    log: list[dict] = []

    def wrapped_plan(self, batch):
        actions, metrics = orig_plan(self, batch)
        if "predicted_target_xyz" in metrics:
            target_pos_m = metrics["predicted_target_xyz"] * self.action_pos_std + self.action_pos_mean
            log.append({"stage": int(self._v25_stage), "predicted_target_robot_frame": target_pos_m[0].detach().cpu().numpy().tolist()})
        return actions, metrics

    def wrapped_enter_local(self, action, current_state):
        stage_before = self._v25_stage
        distance_to_predicted_target_m = None
        if self._v25_stage == PRE_GRASP and self._v25_last_target_xyz is not None:
            ee_pos_m = current_state[..., :3] * self.state_pos_std + self.state_pos_mean
            target_pos_m = self._v25_last_target_xyz * self.action_pos_std + self.action_pos_mean
            distance_to_predicted_target_m = float((ee_pos_m - target_pos_m).norm(dim=-1)[0].item())
        result = orig_enter_local(self, action, current_state)
        if stage_before == PRE_GRASP and self._v25_stage == LOCAL_GRASP:
            entry = {
                "stage": int(self._v25_stage),
                "predicted_target_robot_frame": None,
                "just_entered_local": True,
                "distance_to_predicted_target_at_entry_m": distance_to_predicted_target_m,
            }
            if log:
                log[-1]["just_entered_local"] = True
                log[-1]["distance_to_predicted_target_at_entry_m"] = distance_to_predicted_target_m
            else:
                log.append(entry)
        return result

    def wrapped_local_step(self, batch):
        action, metrics = orig_local_step(self, batch)
        target_pos_m = metrics["predicted_target_xyz"] * self.action_pos_std + self.action_pos_mean
        log.append(
            {
                "stage": int(self._v25_stage),
                "predicted_target_robot_frame": target_pos_m[0].detach().cpu().numpy().tolist(),
                "local_delta_xyz_m": metrics["delta_xyz_m"][0].detach().cpu().numpy().tolist(),
                "predicted_delta_rot_rad": metrics["delta_rot_rad"][0].detach().cpu().numpy().tolist(),
            }
        )
        return action, metrics

    def wrapped_select_local(self, batch, current_state):
        was_latched = self._v25_gripper_closed_latch
        action = orig_select_local(self, batch, current_state)
        just_latched = (not was_latched) and self._v25_gripper_closed_latch
        if log:
            log[-1]["stage"] = int(self._v25_stage)
            log[-1]["just_latched_gripper"] = just_latched
            log[-1]["just_transitioned_post_grasp"] = just_latched
            log[-1]["applied_gripper"] = float(action[0, GRIPPER_INDEX_RAW].item())
        return action

    policy._plan_temporal_decoder = wrapped_plan.__get__(policy, SafeDiffVLAPolicy)
    policy._maybe_enter_local_grasp = wrapped_enter_local.__get__(policy, SafeDiffVLAPolicy)
    policy._plan_local_grasp_step = wrapped_local_step.__get__(policy, SafeDiffVLAPolicy)
    policy._select_action_local_grasp = wrapped_select_local.__get__(policy, SafeDiffVLAPolicy)
    return {"log": log}


def uninstall_v2_5_policy_instrumentation(policy: SafeDiffVLAPolicy) -> None:
    for attr in ("_plan_temporal_decoder", "_maybe_enter_local_grasp", "_plan_local_grasp_step", "_select_action_local_grasp"):
        if attr in policy.__dict__:
            del policy.__dict__[attr]


def run_task(policy: SafeDiffVLAPolicy, device: str, task: str, out_dir: Path) -> None:
    assert policy.config.architecture == "temporal_decoder_grounded_grasp_v2_5"
    set_seed(SEEDS[0])

    env_cfg = VLABenchEnv(task=task)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = envs[task][0]
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=CHECKPOINT,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)

    policy_inst = install_v2_5_policy_instrumentation(policy)
    env_state = install_env_instrumentation(policy_inst["log"])
    episode_records: list[dict] = []
    videos_dir = out_dir / "videos"

    def episode_callback(episode_ix: int, rollout_data: dict, env_idx: int, done_index: int) -> None:
        ep_len = done_index + 1
        steps = list(env_state["current_episode_steps"])
        episode_records.append({"episode_ix": episode_ix, "episode_len": ep_len, "steps": steps})
        policy_inst["log"].clear()
        logger.info("  [%s] episode %d len=%d n_steps=%d", task, episode_ix, ep_len, len(steps))

    try:
        info = eval_policy(
            env=env, policy=policy, env_preprocessor=env_preprocessor, env_postprocessor=env_postprocessor,
            preprocessor=preprocessor, postprocessor=postprocessor, n_episodes=len(SEEDS),
            max_episodes_rendered=len(SEEDS), videos_dir=videos_dir, start_seed=SEEDS[0], episode_callback=episode_callback,
        )
    finally:
        env.close()
        uninstall_env_instrumentation(env_state)
        uninstall_v2_5_policy_instrumentation(policy)

    for ep, rec in zip(info["per_episode"], episode_records, strict=True):
        rec["seed"] = ep["seed"]
        rec["success"] = ep["success"]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "episodes_raw.json").write_text(json.dumps(episode_records, indent=2))
    logger.info("[%s] wrote %s (successes: %s)", task, out_dir / "episodes_raw.json", [ep["success"] for ep in info["per_episode"]])


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", default=STAGE2_TASK_NAMES)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    out_dir = Path(args.output_dir)

    logger.info("=== loading checkpoint: %s ===", CHECKPOINT)
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT)
    policy = policy.to(args.device)
    policy.eval()
    for task in args.tasks:
        logger.info("--- task=%s ---", task)
        run_task(policy, args.device, task, out_dir / task)


if __name__ == "__main__":
    main()
