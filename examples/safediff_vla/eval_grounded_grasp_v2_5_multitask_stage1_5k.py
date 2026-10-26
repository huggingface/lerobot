#!/usr/bin/env python
"""B-track Stage 1 evaluation: closed-loop rollout comparison of the existing SINGLE-TASK
`temporal_decoder_grounded_grasp_v2_5` checkpoint (10k, `select_poker`-only closed-loop validation
so far) against the new MULTI-TASK-ADAPTED checkpoint from
`train_grounded_grasp_v2_5_multitask_stage1_5k.py` (5k more steps, `LocalGraspRefiner`-only,
task-balanced across select_fruit/toy/drink/book/ingredient/mahjong/poker), across all 7 tasks, 3
seeds each (seeds 1000-1002 -- already-seen seeds, development use only, same convention as
`eval_grounded_grasp_v2_5_5k_dev.py`).

Generalizes that script's instrumentation (same non-invasive monkeypatch of `VLABenchEnv.step`/
`.reset`, same instance-level wraps of `SafeDiffVLAPolicy`'s v2.5 stage-transition methods) from
`select_poker`-only to any of the 7 tasks: `extract_physics_record` reads `task.target_entity`'s
own position/grasped-state generically (`Entity.is_grasped` is defined once at the base class in
VLABench, not per-object-type) instead of poker's own "which Poker-typed entity is nearest"
machinery -- that machinery doesn't generalize (a `select_book` episode has no `Poker` entities),
and identity-of-nearest-distractor isn't one of the fields this evaluation was asked to record.

Per episode, records exactly what the B-track instructions ask for: success, grasp/contact success
(ground-truth `is_grasped`), whether/when LOCAL_GRASP was entered, EE<->predicted-target distance
at that entry, the minimum EE<->ground-truth-target distance achieved from local entry onward, a
boolean did-refinement-help comparison (entry distance vs. that minimum), the gripper-close
timestep, and the rollout video path -- plus a few already-free diagnostics (n_local_steps,
mean local delta magnitude, distance before entry) kept for context.

Usage:
    MUJOCO_GL=egl uv run python examples/safediff_vla/eval_grounded_grasp_v2_5_multitask_stage1_5k.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import RENAME_MAP, TASK_NAMES  # noqa: E402

from lerobot.envs import make_env, make_env_pre_post_processors  # noqa: E402
from lerobot.envs.configs import VLABenchEnv  # noqa: E402
from lerobot.envs.vlabench import VLABenchEnv as VLABenchEnvImpl  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.local_grasp_refiner import LOCAL_GRASP, POST_GRASP, PRE_GRASP  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402
from lerobot.policies.safediff_vla.rotation_encoding import GRIPPER_INDEX_RAW  # noqa: E402
from lerobot.scripts.lerobot_eval import eval_policy  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

CHECKPOINT_SINGLE_TASK = "outputs/train/safediff_vla_grounded_grasp_v2_5_5k/checkpoints/010000/pretrained_model"
CHECKPOINT_MULTITASK_STAGE1 = (
    "outputs/train/safediff_vla_grounded_grasp_v2_5_multitask_stage1_5k/checkpoints/005000/pretrained_model"
)
SEEDS = [1000, 1001, 1002]
OUT_DIR = Path("outputs/eval/safediff_vla_grounded_grasp_v2_5_multitask_stage1_5k")


def robot_base(env: VLABenchEnvImpl) -> np.ndarray:
    base = getattr(env, "_robot_base_xyz", None)
    return np.asarray(base, dtype=float) if base is not None else np.array([0.0, -0.4, 0.78])


def extract_physics_record(env: VLABenchEnvImpl, step_ix: int) -> dict[str, Any]:
    """Generic version of `eval_grounded_grasp_v2_5_5k_dev.py`'s poker-only `extract_physics_record`
    -- reads the task's OWN `target_entity` position/grasped-state, works for any of the 7 tasks."""
    physics = env._env.physics
    task = env._env.task
    robot = task.robot
    target_name = task.target_entity
    ee_pos = np.asarray(robot.get_end_effector_pos(physics), dtype=float)
    try:
        target_pos = np.asarray(task.entities[target_name].get_xpos(physics), dtype=float)
    except Exception:  # noqa: BLE001 - best-effort diagnostic only
        target_pos = np.array([np.nan, np.nan, np.nan])
    try:
        is_grasped = bool(task.entities[target_name].is_grasped(physics, robot))
    except Exception:  # noqa: BLE001
        is_grasped = False
    return {
        "step_ix": step_ix,
        "ee_pos": ee_pos.tolist(),
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
            "local_delta_xyz_m", "just_entered_local", "just_latched_gripper",
            "just_transitioned_post_grasp", "applied_gripper", "distance_to_predicted_target_at_entry_m",
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
    """Verbatim copy of `eval_grounded_grasp_v2_5_5k_dev.py`'s instrumentation -- these wraps are
    already fully generic (pure v2.5 stage-transition internals, no poker-specific code)."""
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


def summarize_episode(steps: list[dict], seed: int, success: bool, episode_len: int, video: Path, local_grasp_radius_m: float) -> dict:
    grasp_success = any(s["is_grasped"] for s in steps)
    local_steps = [s for s in steps if s.get("stage") == LOCAL_GRASP]
    local_entry_step = next((s["step_ix"] for s in steps if s.get("just_entered_local")), None)
    gripper_close_timestep = next((s["step_ix"] for s in steps if s.get("just_latched_gripper")), None)
    post_grasp_transition_step = next((s["step_ix"] for s in steps if s.get("just_transitioned_post_grasp")), None)
    grasp_timestep = next((s["step_ix"] for s in steps if s["is_grasped"]), None)

    entry_step_record = next((s for s in steps if s["step_ix"] == local_entry_step), None) if local_entry_step is not None else None
    distance_to_predicted_target_at_entry_m = (
        entry_step_record.get("distance_to_predicted_target_at_entry_m") if entry_step_record is not None else None
    )
    entry_within_local_grasp_radius = (
        distance_to_predicted_target_at_entry_m <= local_grasp_radius_m
        if distance_to_predicted_target_at_entry_m is not None
        else None
    )
    ee_to_gt_target_dist_at_local_entry = entry_step_record["ee_to_target_dist"] if entry_step_record is not None else None

    before_local = [s for s in steps if local_entry_step is None or s["step_ix"] < local_entry_step]
    after_local = [s for s in steps if local_entry_step is not None and s["step_ix"] >= local_entry_step]
    min_ee_to_gt_target_dist_before_local_entry = min((s["ee_to_target_dist"] for s in before_local), default=None)
    min_ee_to_gt_target_dist_after_local_entry = min((s["ee_to_target_dist"] for s in after_local), default=None)
    local_refinement_improved_distance = (
        min_ee_to_gt_target_dist_after_local_entry < ee_to_gt_target_dist_at_local_entry
        if ee_to_gt_target_dist_at_local_entry is not None and min_ee_to_gt_target_dist_after_local_entry is not None
        else None
    )
    delta_mags = [float(np.linalg.norm(s["local_delta_xyz_m"])) for s in local_steps if s.get("local_delta_xyz_m") is not None]

    return {
        "seed": seed,
        "success": success,
        "grasp_success": grasp_success,
        "target_entity_identity": steps[0]["target_entity"],
        "episode_len": episode_len,
        "entered_local_grasp": local_entry_step is not None,
        "local_entry_step": local_entry_step,
        "n_local_steps": len(local_steps),
        "distance_to_predicted_target_at_entry_m": distance_to_predicted_target_at_entry_m,
        "entry_within_local_grasp_radius": entry_within_local_grasp_radius,
        "ee_to_gt_target_dist_at_local_entry": ee_to_gt_target_dist_at_local_entry,
        "gripper_close_timestep": gripper_close_timestep,
        "post_grasp_transition_step": post_grasp_transition_step,
        "grasp_timestep": grasp_timestep,
        "min_ee_to_gt_target_dist_before_local_entry": min_ee_to_gt_target_dist_before_local_entry,
        "min_ee_to_gt_target_dist_after_local_entry": min_ee_to_gt_target_dist_after_local_entry,
        "local_refinement_improved_distance": local_refinement_improved_distance,
        "episode_min_ee_to_gt_target_dist": min(s["ee_to_target_dist"] for s in steps),
        "mean_local_delta_xyz_magnitude_m": (sum(delta_mags) / len(delta_mags)) if delta_mags else None,
        "video": str(video),
    }


def run(policy: SafeDiffVLAPolicy, checkpoint_path: str, device: str, task: str, seeds: list[int], out_dir: Path) -> dict:
    assert policy.config.architecture == "temporal_decoder_grounded_grasp_v2_5"
    set_seed(seeds[0])

    env_cfg = VLABenchEnv(task=task)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = envs[task][0]
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint_path,
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
            preprocessor=preprocessor, postprocessor=postprocessor, n_episodes=len(seeds),
            max_episodes_rendered=len(seeds), videos_dir=videos_dir, start_seed=seeds[0], episode_callback=episode_callback,
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

    per_episode_summary = [
        summarize_episode(
            rec["steps"], rec["seed"], rec["success"], rec["episode_len"],
            videos_dir / f"eval_episode_{rec['episode_ix']}.mp4", policy.config.local_grasp_radius_m,
        )
        for rec in episode_records
    ]

    def agg_mean(key: str) -> float | None:
        vals = [e[key] for e in per_episode_summary if e.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "checkpoint": checkpoint_path,
        "task": task,
        "n_episodes": len(seeds),
        "seeds": seeds,
        "per_episode": per_episode_summary,
        "aggregate": {
            "n_success": sum(1 for e in per_episode_summary if e["success"]),
            "n_grasp_success": sum(1 for e in per_episode_summary if e["grasp_success"]),
            "n_entered_local_grasp": sum(1 for e in per_episode_summary if e["entered_local_grasp"]),
            "n_entry_within_local_grasp_radius": sum(1 for e in per_episode_summary if e["entry_within_local_grasp_radius"]),
            "n_local_refinement_improved_distance": sum(1 for e in per_episode_summary if e["local_refinement_improved_distance"]),
            "mean_distance_to_predicted_target_at_entry_m": agg_mean("distance_to_predicted_target_at_entry_m"),
            "mean_ee_to_gt_target_dist_at_local_entry": agg_mean("ee_to_gt_target_dist_at_local_entry"),
            "mean_min_ee_to_gt_target_dist_before_local_entry": agg_mean("min_ee_to_gt_target_dist_before_local_entry"),
            "mean_min_ee_to_gt_target_dist_after_local_entry": agg_mean("min_ee_to_gt_target_dist_after_local_entry"),
            "mean_episode_min_ee_to_gt_target_dist": agg_mean("episode_min_ee_to_gt_target_dist"),
            "mean_local_delta_xyz_magnitude_m": agg_mean("mean_local_delta_xyz_magnitude_m"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("[%s] aggregate: %s", task, json.dumps(summary["aggregate"]))
    return summary


def build_comparison(all_results: dict[str, dict[str, dict]], out_dir: Path) -> dict:
    """Per-task, per-checkpoint-label aggregate table plus the 3 questions the B-track
    instructions ask this evaluation to answer."""
    def agg_of(label: str, task: str) -> dict | None:
        entry = all_results.get(label, {}).get(task, {})
        return entry.get("aggregate")

    rows = []
    for task in TASK_NAMES:
        a = agg_of("single_task_v2_5_10k", task)
        b = agg_of("multitask_stage1_v2_5_5k", task)
        rows.append({
            "task": task,
            "single_task_10k_n_success": a["n_success"] if a else None,
            "multitask_stage1_5k_n_success": b["n_success"] if b else None,
            "single_task_10k_n_local_refinement_improved": a["n_local_refinement_improved_distance"] if a else None,
            "multitask_stage1_5k_n_local_refinement_improved": b["n_local_refinement_improved_distance"] if b else None,
            "single_task_10k_mean_min_dist_after_local_entry": a["mean_min_ee_to_gt_target_dist_after_local_entry"] if a else None,
            "multitask_stage1_5k_mean_min_dist_after_local_entry": b["mean_min_ee_to_gt_target_dist_after_local_entry"] if b else None,
            "unavailable": a is None or b is None,
        })
    poker_a = agg_of("single_task_v2_5_10k", "select_poker")
    poker_b = agg_of("multitask_stage1_v2_5_5k", "select_poker")
    table = {
        "rows": rows,
        "poker_catastrophic_forgetting_check": (
            {
                "single_task_10k_n_success": poker_a["n_success"],
                "multitask_stage1_5k_n_success": poker_b["n_success"],
                "single_task_10k_n_local_refinement_improved": poker_a["n_local_refinement_improved_distance"],
                "multitask_stage1_5k_n_local_refinement_improved": poker_b["n_local_refinement_improved_distance"],
                "regressed": poker_b["n_success"] < poker_a["n_success"],
            }
            if poker_a and poker_b else {"unavailable": True}
        ),
        "new_tasks_local_refinement_improved_rate": {
            task: (agg_of("multitask_stage1_v2_5_5k", task)["n_local_refinement_improved_distance"] / 3)
            for task in TASK_NAMES if task != "select_poker" and agg_of("multitask_stage1_v2_5_5k", task) is not None
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps(table, indent=2))
    logger.info("wrote %s", out_dir / "comparison.json")
    return table


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--single-task-checkpoint", default=CHECKPOINT_SINGLE_TASK)
    parser.add_argument("--multitask-checkpoint", default=CHECKPOINT_MULTITASK_STAGE1)
    parser.add_argument("--tasks", nargs="+", default=TASK_NAMES)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip (task, checkpoint) pairs that already have a summary.json under --output-dir "
        "(e.g. after a partial run interrupted by an upstream env-reset failure).",
    )
    parser.add_argument(
        "--only-checkpoint", choices=["single_task", "multitask"], default=None,
        help="Run only one checkpoint label (VLABench/MuJoCo state has been observed to leak "
        "across many sequential env constructions in one long-lived process -- see "
        "run_multitask_eval_isolated.py, which uses this flag to give every (checkpoint, task) "
        "pair its own fresh subprocess).",
    )
    parser.add_argument(
        "--merge-only", action="store_true",
        help="Skip all rollouts; just read existing summary.json files under --output-dir for "
        "every (checkpoint, task) pair and rebuild all_results.json/comparison.json from them.",
    )
    args = parser.parse_args()
    out_dir = Path(args.output_dir)

    checkpoints = {
        "single_task_v2_5_10k": args.single_task_checkpoint,
        "multitask_stage1_v2_5_5k": args.multitask_checkpoint,
    }
    if args.only_checkpoint == "single_task":
        checkpoints = {"single_task_v2_5_10k": args.single_task_checkpoint}
    elif args.only_checkpoint == "multitask":
        checkpoints = {"multitask_stage1_v2_5_5k": args.multitask_checkpoint}

    if args.merge_only:
        all_results = {}
        for label in checkpoints:
            all_results[label] = {}
            for task in args.tasks:
                summary_path = out_dir / label / task / "summary.json"
                all_results[label][task] = (
                    json.loads(summary_path.read_text()) if summary_path.exists()
                    else {"error": "no summary.json found", "task": task}
                )
        (out_dir / "all_results.json").write_text(json.dumps(all_results, indent=2))
        if set(args.tasks) == set(TASK_NAMES) and set(checkpoints) == {"single_task_v2_5_10k", "multitask_stage1_v2_5_5k"}:
            build_comparison(all_results, out_dir)
        logger.info("=== merge-only done. wrote results under %s ===", out_dir)
        return

    all_results: dict[str, dict[str, dict]] = {}
    for label, ckpt in checkpoints.items():
        logger.info("=== loading checkpoint %s: %s ===", label, ckpt)
        policy = SafeDiffVLAPolicy.from_pretrained(ckpt)
        policy = policy.to(args.device)
        policy.eval()
        all_results[label] = {}
        for task in args.tasks:
            summary_path = out_dir / label / task / "summary.json"
            if args.resume and summary_path.exists():
                logger.info("--- [%s] task=%s: resuming from existing %s ---", label, task, summary_path)
                all_results[label][task] = json.loads(summary_path.read_text())
                continue
            logger.info("--- [%s] task=%s ---", label, task)
            # `select_mahjong`'s own layout sampler is documented (see `envs/vlabench.py`'s
            # `_ENSURE_ENV_MAX_ATTEMPTS` comment) to diverge often enough that even 20 in-process
            # retries occasionally aren't enough -- retry the whole task a few times (each attempt
            # builds a fresh env/reset, drawing fresh OS-entropy-reseeded layouts) before giving up,
            # and record a task-level failure rather than aborting the entire multi-task eval.
            last_error: Exception | None = None
            for attempt in range(3):
                try:
                    all_results[label][task] = run(policy, ckpt, args.device, task, args.seeds, out_dir / label / task)
                    last_error = None
                    break
                except RuntimeError as e:
                    last_error = e
                    logger.warning("[%s/%s] attempt %d/3 failed: %s", label, task, attempt + 1, e)
            if last_error is not None:
                logger.error("[%s/%s] all attempts failed -- recording as unavailable", label, task)
                all_results[label][task] = {"error": str(last_error), "task": task, "checkpoint": ckpt}
        del policy
        torch.cuda.empty_cache()

    (out_dir / "all_results.json").write_text(json.dumps(all_results, indent=2))
    if set(args.tasks) == set(TASK_NAMES):
        build_comparison(all_results, out_dir)
    logger.info("=== done. wrote results under %s ===", out_dir)


if __name__ == "__main__":
    main()
