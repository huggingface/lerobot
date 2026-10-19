#!/usr/bin/env python
"""PHASE 1A (new "temporal grounding stability" direction, NOT LocalGraspRefiner/v2.5-based):
pure diagnostic re-rollout (NO training, NO behavior change) of the frozen `GroundedGrasp-v2`
checkpoint (`architecture="temporal_decoder_grounded_grasp_v2"`), instrumented with a MONITOR-ONLY
per-step grounding forward.

Why this script exists (see PHASE 0 audit): v2's own `predicted_target_xyz` is only recomputed
inside `_plan_temporal_decoder`, which `ActionExecutor.select_action` only calls when its action
queue is empty -- with the canonical checkpoint's `action_horizon=execute_horizon=50`, that means
v2's own target is a STEP FUNCTION updated at most once per 50 env steps (often exactly once per
episode), not a per-frame quantity. There is therefore no way, with any *existing* logger on this
branch, to measure genuine frame-to-frame grounding jitter for v2 -- every prior "jitter" number on
this branch comes from v2.5's `LocalGraspRefiner`/LOCAL_GRASP stage, which this new direction is
explicitly NOT building on.

This script adds exactly one new capability: a MONITOR-ONLY forward pass, run every single env
step regardless of whether the action-chunk planner replans this step, that re-executes v2's own
(frozen, unmodified) grounding submodules --

    `TargetQueryExtractor` -> `VisualGroundingCrossAttention` -> `TargetPointHead`

-- on the current observation, producing `monitor_predicted_target_xyz`. This NEVER feeds back into
`select_action`'s returned action or the executor's queue -- it is a pure side-channel read, wired
in as a wrapper around `SafeDiffVLAPolicy.select_action` that calls the policy's own already-frozen
submodules AFTER the original method has already produced its action (see `wrapped_select_action`
below). `monitor_predicted_target_xyz` is logged ALONGSIDE (never in place of) the planner's own
cached `planner_predicted_target_xyz` (`self._v2_last_target_xyz`, updated only on an actual
replan -- `planner_replanned_this_step` flags exactly which steps that is).

Fields logged per step (see `install_policy_instrumentation`/`extract_physics_record`):
    monitor_target_world, planner_target_world (None until the first replan), planner_replanned_
    this_step, target_pos (GT, diagnostic only -- never fed to the policy), ee_pos (actual, world
    frame), actual_ee_euler_rad, commanded_xyz_world / commanded_rot_euler_raw /
    commanded_gripper_raw (the ACTUAL action sent to the env this step, post reactive-close),
    phase (v2's own PRE_GRASP=0/POST_GRASP=1), reactive_close_event (this step), is_grasped
    (privileged contact, diagnostic only), native success (from `eval_policy`'s own `ep["success"]`).

Dev set: 6 tasks x seeds 1000-1002 (`STAGE2_TASK_NAMES`, `select_ingredient` excluded -- same 18
episodes every prior diagnostic on this branch used), same process-per-task isolation convention
(see `run_grounded_grasp_v2_monitor_baseline_isolated.py`).

Usage:
    MUJOCO_GL=egl uv run python examples/safediff_vla/eval_grounded_grasp_v2_monitor_baseline.py --tasks select_book
    MUJOCO_GL=egl uv run python examples/safediff_vla/eval_grounded_grasp_v2_monitor_baseline.py --tasks select_book --seeds 1000 --smoke-test
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
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402
from lerobot.policies.safediff_vla.rotation_encoding import GRIPPER_INDEX_RAW, ROTATION_SLICE_RAW  # noqa: E402
from lerobot.scripts.lerobot_eval import eval_policy  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

CHECKPOINT = "outputs/train/safediff_vla_grounded_grasp_v2_20k/checkpoints/020000/pretrained_model"
SEEDS = [1000, 1001, 1002]
OUT_DIR = Path("outputs/eval/safediff_vla_grounded_grasp_v2_monitor_baseline")


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
            "phase", "reactive_close_event", "planner_replanned_this_step",
            "monitor_target_world", "monitor_target_dist_to_gt",
            "planner_target_world", "planner_target_dist_to_gt",
            "commanded_xyz_world", "commanded_rot_euler_raw", "commanded_gripper_raw",
        )
        for k in keys:
            record[k] = None
        if not policy_log:
            return
        p = policy_log[-1]
        record["phase"] = p.get("phase")
        record["reactive_close_event"] = p.get("reactive_close_event")
        record["planner_replanned_this_step"] = p.get("planner_replanned_this_step")
        record["commanded_rot_euler_raw"] = p.get("commanded_rot_euler_raw")
        record["commanded_gripper_raw"] = p.get("commanded_gripper_raw")
        target_pos = np.asarray(record["target_pos"], dtype=float)
        if p.get("monitor_target_robot_frame") is not None:
            monitor_world = np.asarray(p["monitor_target_robot_frame"], dtype=float) + base
            record["monitor_target_world"] = monitor_world.tolist()
            record["monitor_target_dist_to_gt"] = float(np.linalg.norm(monitor_world - target_pos))
        if p.get("planner_target_robot_frame") is not None:
            planner_world = np.asarray(p["planner_target_robot_frame"], dtype=float) + base
            record["planner_target_world"] = planner_world.tolist()
            record["planner_target_dist_to_gt"] = float(np.linalg.norm(planner_world - target_pos))
        if p.get("commanded_xyz_robot_frame") is not None:
            record["commanded_xyz_world"] = (np.asarray(p["commanded_xyz_robot_frame"], dtype=float) + base).tolist()

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


def install_policy_instrumentation(policy: SafeDiffVLAPolicy) -> dict:
    """Wraps `select_action` (fires every env step, unconditionally) and `_plan_temporal_decoder`
    (fires only on an actual chunk replan) to log exactly one entry per env step. The monitor-only
    forward re-runs v2's own frozen `target_query_extractor` / `visual_grounding_cross_attn` /
    `target_point_head` on THIS step's `batch` -- the exact same computation
    `_plan_temporal_decoder`'s own v2 branch does internally -- but never writes to
    `self._v2_last_target_xyz`/`self._v2_close_triggered`/the executor's queue, and is called AFTER
    `orig_select_action` has already produced and returned its action, so it cannot perturb
    `select_action`'s own control flow or output in any way.
    """
    orig_select_action = SafeDiffVLAPolicy.select_action
    orig_plan = SafeDiffVLAPolicy._plan_temporal_decoder
    log: list[dict] = []
    replanned_flag = {"flag": False}

    def wrapped_plan(self, batch):
        replanned_flag["flag"] = True
        return orig_plan(self, batch)

    def wrapped_select_action(self, batch):
        phase_before = self._v2_grasp_phase
        close_triggered_before = self._v2_close_triggered
        replanned_flag["flag"] = False

        action = orig_select_action(self, batch)

        planner_replanned_this_step = replanned_flag["flag"]
        phase_after = self._v2_grasp_phase
        reactive_close_event = (not close_triggered_before) and self._v2_close_triggered

        # Monitor-only forward: pure re-derivation of v2's own grounding pipeline on THIS batch,
        # no side effects on any policy state (`self._v2_*`), no effect on `action` (already
        # computed and will be returned unmodified below).
        with torch.no_grad():
            latent_tokens, latent_pad_mask, latent_modality_ids = self._encode_multimodal_latent(batch)
            target_language_query = self.target_query_extractor(latent_tokens, latent_pad_mask, latent_modality_ids)
            grounded_target_feature = self.visual_grounding_cross_attn(
                target_language_query, latent_tokens, latent_pad_mask, latent_modality_ids
            )
            monitor_target_xyz = self.target_point_head(grounded_target_feature)

        monitor_target_m = (monitor_target_xyz * self.action_pos_std + self.action_pos_mean)[0].detach().cpu().numpy()
        planner_target_m = None
        if self._v2_last_target_xyz is not None:
            planner_target_m = (
                (self._v2_last_target_xyz * self.action_pos_std + self.action_pos_mean)[0].detach().cpu().numpy()
            )
        commanded_xyz_m = (action[..., :3] * self.action_pos_std + self.action_pos_mean)[0].detach().cpu().numpy()
        commanded_rot_raw = (
            action[..., ROTATION_SLICE_RAW] * self.action_rot_std + self.action_rot_mean
        )[0].detach().cpu().numpy()
        commanded_gripper_raw = float(
            (action[0, GRIPPER_INDEX_RAW] * self.action_grip_std + self.action_grip_mean).item()
        )

        log.append(
            {
                "phase": int(phase_after),
                "reactive_close_event": bool(reactive_close_event),
                "planner_replanned_this_step": bool(planner_replanned_this_step),
                "monitor_target_robot_frame": monitor_target_m.tolist(),
                "planner_target_robot_frame": planner_target_m.tolist() if planner_target_m is not None else None,
                "commanded_xyz_robot_frame": commanded_xyz_m.tolist(),
                "commanded_rot_euler_raw": commanded_rot_raw.tolist(),
                "commanded_gripper_raw": commanded_gripper_raw,
            }
        )
        del phase_before  # logged implicitly via reactive_close_event / phase_after
        return action

    policy._plan_temporal_decoder = wrapped_plan.__get__(policy, SafeDiffVLAPolicy)
    policy.select_action = wrapped_select_action.__get__(policy, SafeDiffVLAPolicy)
    return {"log": log}


def uninstall_policy_instrumentation(policy: SafeDiffVLAPolicy) -> None:
    for attr in ("_plan_temporal_decoder", "select_action"):
        if attr in policy.__dict__:
            del policy.__dict__[attr]


def run_task(policy: SafeDiffVLAPolicy, device: str, task: str, seeds: list[int], out_dir: Path) -> None:
    assert policy.config.architecture == "temporal_decoder_grounded_grasp_v2"
    set_seed(seeds[0])

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

    policy_inst = install_policy_instrumentation(policy)
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
            preprocessor=preprocessor, postprocessor=postprocessor, n_episodes=len(seeds),
            max_episodes_rendered=len(seeds), videos_dir=videos_dir, start_seed=seeds[0], episode_callback=episode_callback,
        )
    finally:
        env.close()
        uninstall_env_instrumentation(env_state)
        uninstall_policy_instrumentation(policy)

    for ep, rec in zip(info["per_episode"], episode_records, strict=True):
        rec["seed"] = ep["seed"]
        rec["success"] = ep["success"]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "episodes_raw.json").write_text(json.dumps(episode_records, indent=2))
    logger.info(
        "[%s] wrote %s (successes: %s)", task, out_dir / "episodes_raw.json", [ep["success"] for ep in info["per_episode"]]
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", default=STAGE2_TASK_NAMES)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--smoke-test", action="store_true", help="Use only the first seed, for a quick sanity check.")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    seeds = args.seeds[:1] if args.smoke_test else args.seeds

    logger.info("=== loading checkpoint: %s ===", CHECKPOINT)
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT)
    policy = policy.to(args.device)
    policy.eval()
    for task in args.tasks:
        logger.info("--- task=%s seeds=%s ---", task, seeds)
        run_task(policy, args.device, task, seeds, out_dir / task)


if __name__ == "__main__":
    main()
