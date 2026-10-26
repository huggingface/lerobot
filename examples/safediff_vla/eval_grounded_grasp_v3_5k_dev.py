#!/usr/bin/env python
"""DEVELOPMENT-ONLY closed-loop rollout evaluation of the fresh `train_grounded_grasp_v3_5k.py`
checkpoint (`temporal_decoder_grounded_grasp_v3`) on `select_poker` seeds 1000-1002 -- the SAME
seeds used for every prior 5k pilot's development eval on this branch (`eval_grounded_grasp_v2_5k.py`
included). These are already-seen seeds, used here only to sanity-check the coarse/fine split
before ever looking at held-out seeds 3000-3009. Do NOT extend this script to held-out seeds as
part of this pilot.

Self-contained: reads live MuJoCo physics via a non-invasive `VLABenchEnv.step`/`.reset`
monkeypatch (same pattern as `eval_grounded_grasp_v2_5k.py`), plus an instance-level wrap of this
one policy object's own `plan_action_chunk` (to capture the coarse/residual/final target and
visual-selection stats each time a chunk is actually (re)planned) and
`_grounded_grasp_v3_reactive_close` (observes, does not change, its proximity-trigger/phase-flip/
one-shot-guard behavior). No model/loss/execution-policy code is touched by this script.

Per episode, records (extending `eval_grounded_grasp_v2_5k.py`'s field set with the coarse/fine
split's own diagnostics):
  - task success, grasp/contact success, wrong_object_grounding (both at min-distance and
    majority-of-steps definitions), proximity trigger / phase transition timestep -- same as v2.
  - AT EACH reference step (first chunk, proximity trigger, min EE-distance): coarse_xyz<->GT
    error, final_xyz<->GT error, and whether refinement reduced error at that step.
  - per-episode and aggregate: mean/median coarse-xyz error, mean/median final-xyz error, and
    `fraction_steps_refinement_reduced_error` (of steps with both a coarse and final prediction,
    the fraction where |final-GT| < |coarse-GT|).
  - per-episode and aggregate visual-selection entropy / top-1 mass / top-k mass (averaged over
    every chunk-(re)plan in the episode).
  - min EE<->target-card distance over the whole episode, video path.

Usage:
    MUJOCO_GL=egl uv run python examples/safediff_vla/eval_grounded_grasp_v3_5k_dev.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.envs import make_env, make_env_pre_post_processors
from lerobot.envs.configs import VLABenchEnv
from lerobot.envs.vlabench import VLABenchEnv as VLABenchEnvImpl
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy
from lerobot.policies.safediff_vla.rotation_encoding import GRIPPER_INDEX_RAW
from lerobot.scripts.lerobot_eval import eval_policy
from lerobot.utils.random_utils import set_seed

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

CHECKPOINT = "outputs/train/safediff_vla_grounded_grasp_v3_5k/checkpoints/005000/pretrained_model"
TASK = "select_poker"
RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.second_image": "observation.images.camera2",
    "observation.images.wrist_image": "observation.images.camera3",
}
GRIPPER_THRESHOLD = 0.5
# Development seeds ONLY -- already seen on this branch. Held-out seeds 3000-3009 are
# deliberately NOT evaluated by this pilot.
SEEDS = [1000, 1001, 1002]


def robot_base(env: VLABenchEnvImpl) -> np.ndarray:
    base = getattr(env, "_robot_base_xyz", None)
    return np.asarray(base, dtype=float) if base is not None else np.array([0.0, -0.4, 0.78])


def all_card_positions(env: VLABenchEnvImpl) -> dict[str, list[float]]:
    physics = env._env.physics
    task = env._env.task
    out = {}
    for name, ent in task.entities.items():
        if type(ent).__name__ != "Poker":
            continue
        try:
            out[name] = np.asarray(ent.get_xpos(physics), dtype=float).tolist()
        except Exception:  # noqa: BLE001 - best-effort diagnostic only
            pass
    return out


def extract_physics_record(env: VLABenchEnvImpl, step_ix: int) -> dict[str, Any]:
    physics = env._env.physics
    task = env._env.task
    robot = task.robot
    target_name = task.target_entity
    ee_pos = np.asarray(robot.get_end_effector_pos(physics), dtype=float)
    cards = all_card_positions(env)
    card_pos = np.array(cards.get(target_name, [np.nan, np.nan, np.nan]))
    try:
        is_grasped = bool(task.entities[target_name].is_grasped(physics, robot))
    except Exception:  # noqa: BLE001 - best-effort diagnostic only
        is_grasped = False
    other_card_grasped = False
    for name, ent in task.entities.items():
        if name == target_name or type(ent).__name__ != "Poker":
            continue
        try:
            if ent.is_grasped(physics, robot):
                other_card_grasped = True
        except Exception:  # noqa: BLE001
            pass
    return {
        "step_ix": step_ix,
        "ee_pos": ee_pos.tolist(),
        "target_entity": target_name,
        "card_pos": card_pos.tolist(),
        "all_card_positions": cards,
        "is_grasped": is_grasped,
        "wrong_object_grasped_this_step": other_card_grasped,
        "ee_to_card_dist": float(np.linalg.norm(ee_pos - card_pos)),
    }


def install_policy_instrumentation(policy: SafeDiffVLAPolicy) -> dict:
    """Instance-level wraps (not class patches) of:
      - `plan_action_chunk`: captures coarse/residual/final target-xyz (un-normalized to physical
        meters, robot-base frame) and visual-selection stats each time a chunk is actually
        (re)planned (much rarer than every env step, since action_horizon=execute_horizon=50 here).
      - `_grounded_grasp_v3_reactive_close`: same observe-only wrap as
        `eval_grounded_grasp_v2_5k.py`'s, reading the most recently cached diagnostics above so
        every env step's record carries the CURRENT chunk's coarse/final target, not just steps
        where a fresh plan happened.
    """
    orig_plan = SafeDiffVLAPolicy.plan_action_chunk
    orig_close = SafeDiffVLAPolicy._grounded_grasp_v3_reactive_close
    plan_log: list[dict] = []
    close_log: list[dict] = []

    def wrapped_plan(self, batch):
        actions, metrics = orig_plan(self, batch)
        if self.architecture == "temporal_decoder_grounded_grasp_v3" and "predicted_target_xyz" in metrics:
            # `predicted_*_target_xyz` are in MEAN_STD-normalized action-position space (the same
            # space `TargetPointHead`/`CoarseTargetHead`/`FineResidualHead` are trained against) --
            # un-normalize to physical meters (robot-base frame) via the policy's own registered
            # stats BEFORE converting to world frame below, exactly like
            # `_grounded_grasp_v3_reactive_close`'s own un-normalization. Un-normalizing
            # `residual_xyz` directly (adding `action_pos_mean`) would be wrong -- it's a DELTA, not
            # an absolute position -- so its physical-scale value is derived as the difference of
            # the two already-un-normalized absolute positions instead.
            coarse_m = (
                metrics["predicted_coarse_target_xyz"][0] * self.action_pos_std + self.action_pos_mean
            ).detach().cpu().numpy()
            final_m = (
                metrics["predicted_target_xyz"][0] * self.action_pos_std + self.action_pos_mean
            ).detach().cpu().numpy()
            plan_log.append(
                {
                    "coarse_xyz": coarse_m,
                    "residual_xyz": final_m - coarse_m,
                    "final_xyz": final_m,
                    "entropy": float(metrics["visual_selection_entropy"][0].item()),
                    "top1_mass": float(metrics["visual_selection_top1_mass"][0].item()),
                    "topk_mass": float(metrics["visual_selection_topk_mass"][0].item()),
                }
            )
        return actions, metrics

    def wrapped_close(self, action, current_state):
        original_gripper = float(action[0, GRIPPER_INDEX_RAW].item())
        phase_before = self._v3_grasp_phase
        was_triggered_before = self._v3_close_triggered
        result = orig_close(self, action, current_state)
        close_log.append(
            {
                "original_gripper": original_gripper,
                "final_gripper": float(result[0, GRIPPER_INDEX_RAW].item()),
                "phase_before": phase_before,
                "phase_after": self._v3_grasp_phase,
                "close_triggered_after": bool(self._v3_close_triggered),
                "just_triggered_this_step": bool((not was_triggered_before) and self._v3_close_triggered),
                "diagnostic": dict(plan_log[-1]) if plan_log else None,
            }
        )
        return result

    policy.plan_action_chunk = wrapped_plan.__get__(policy, SafeDiffVLAPolicy)
    policy._grounded_grasp_v3_reactive_close = wrapped_close.__get__(policy, SafeDiffVLAPolicy)
    return {"plan_log": plan_log, "close_log": close_log}


def uninstall_policy_instrumentation(policy: SafeDiffVLAPolicy) -> None:
    for attr in ("plan_action_chunk", "_grounded_grasp_v3_reactive_close"):
        if attr in policy.__dict__:
            del policy.__dict__[attr]


def install_env_instrumentation(close_log: list[dict]) -> dict:
    cls = VLABenchEnvImpl
    orig_step = cls.step
    orig_reset = cls.reset
    state: dict[str, Any] = {"current_episode_steps": []}

    def merge_policy_log(record: dict, base: np.ndarray) -> None:
        keys = (
            "original_decoder_gripper", "applied_gripper", "phase_after_this_step",
            "close_triggered_after_this_step", "proximity_trigger_fired_this_step",
            "coarse_target_world", "final_target_world",
            "coarse_target_dist_to_gt_card", "final_target_dist_to_gt_card",
            "refinement_reduced_error", "visual_selection_entropy", "visual_selection_top1_mass",
            "visual_selection_topk_mass", "nearest_card_to_final_target", "target_is_nearest",
        )
        for k in keys:
            record[k] = None
        record["phase_after_this_step"] = 0
        record["close_triggered_after_this_step"] = False
        record["proximity_trigger_fired_this_step"] = False
        if not close_log:
            return
        p = close_log[-1]
        record["original_decoder_gripper"] = p["original_gripper"]
        record["applied_gripper"] = p["final_gripper"]
        record["phase_after_this_step"] = p["phase_after"]
        record["close_triggered_after_this_step"] = p["close_triggered_after"]
        record["proximity_trigger_fired_this_step"] = p["just_triggered_this_step"]
        diag = p.get("diagnostic")
        if diag is None:
            return
        card_pos = np.asarray(record["card_pos"], dtype=float)
        coarse_world = diag["coarse_xyz"] + base
        final_world = diag["final_xyz"] + base
        record["coarse_target_world"] = coarse_world.tolist()
        record["final_target_world"] = final_world.tolist()
        coarse_err = float(np.linalg.norm(coarse_world - card_pos))
        final_err = float(np.linalg.norm(final_world - card_pos))
        record["coarse_target_dist_to_gt_card"] = coarse_err
        record["final_target_dist_to_gt_card"] = final_err
        record["refinement_reduced_error"] = final_err < coarse_err
        record["visual_selection_entropy"] = diag["entropy"]
        record["visual_selection_top1_mass"] = diag["top1_mass"]
        record["visual_selection_topk_mass"] = diag["topk_mass"]
        dists = {
            name: float(np.linalg.norm(final_world - np.asarray(pos, dtype=float)))
            for name, pos in record["all_card_positions"].items()
        }
        nearest = min(dists, key=dists.get) if dists else None
        record["nearest_card_to_final_target"] = nearest
        record["target_is_nearest"] = nearest == record["target_entity"]

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


def _mean_median(values: list[float]) -> tuple[float | None, float | None]:
    values = sorted(values)
    if not values:
        return None, None
    mean = sum(values) / len(values)
    n = len(values)
    median = values[n // 2] if n % 2 == 1 else (values[n // 2 - 1] + values[n // 2]) / 2
    return mean, median


def summarize_episode(steps: list[dict], seed: int, success: bool, episode_len: int, video: Path) -> dict:
    grasp_success = any(s["is_grasped"] for s in steps)
    wrong_object_grasped_ever = any(s.get("wrong_object_grasped_this_step") for s in steps)

    proximity_trigger_step = next((s["step_ix"] for s in steps if s.get("proximity_trigger_fired_this_step")), None)
    phase_transition_step = next(
        (s["step_ix"] for s in steps if s.get("phase_after_this_step") == 1 and steps[max(s["step_ix"] - 1, 0)].get("phase_after_this_step") == 0),
        None,
    )

    min_idx = min(range(len(steps)), key=lambda i: steps[i]["ee_to_card_dist"])
    min_step = steps[min_idx]

    def ref(step_ix: int | None) -> dict | None:
        if step_ix is None:
            return None
        s = next((x for x in steps if x["step_ix"] == step_ix), None)
        if s is None:
            return None
        return {
            "step_ix": step_ix,
            "coarse_target_world": s.get("coarse_target_world"),
            "final_target_world": s.get("final_target_world"),
            "actual_target_card_world": s["card_pos"],
            "coarse_target_dist_to_gt_card": s.get("coarse_target_dist_to_gt_card"),
            "final_target_dist_to_gt_card": s.get("final_target_dist_to_gt_card"),
            "refinement_reduced_error": s.get("refinement_reduced_error"),
            "nearest_card_to_final_target": s.get("nearest_card_to_final_target"),
            "target_is_nearest": s.get("target_is_nearest"),
            "ee_to_card_dist": s["ee_to_card_dist"],
        }

    coarse_errs = [s["coarse_target_dist_to_gt_card"] for s in steps if s.get("coarse_target_dist_to_gt_card") is not None]
    final_errs = [s["final_target_dist_to_gt_card"] for s in steps if s.get("final_target_dist_to_gt_card") is not None]
    refine_flags = [s["refinement_reduced_error"] for s in steps if s.get("refinement_reduced_error") is not None]
    entropies = [s["visual_selection_entropy"] for s in steps if s.get("visual_selection_entropy") is not None]
    top1_masses = [s["visual_selection_top1_mass"] for s in steps if s.get("visual_selection_top1_mass") is not None]
    topk_masses = [s["visual_selection_topk_mass"] for s in steps if s.get("visual_selection_topk_mass") is not None]
    target_is_nearest_values = [s["target_is_nearest"] for s in steps if s.get("target_is_nearest") is not None]

    mean_coarse, median_coarse = _mean_median(coarse_errs)
    mean_final, median_final = _mean_median(final_errs)
    frac_target_is_nearest = (
        sum(target_is_nearest_values) / len(target_is_nearest_values) if target_is_nearest_values else None
    )

    return {
        "seed": seed,
        "success": success,
        "grasp_success": grasp_success,
        "wrong_object_grasped_ever": wrong_object_grasped_ever,
        "target_card_identity": steps[0]["target_entity"],
        "episode_len": episode_len,
        "proximity_trigger_step": proximity_trigger_step,
        "post_grasp_phase_transition_step": phase_transition_step,
        "phase_transition_matches_proximity_trigger": phase_transition_step == proximity_trigger_step,
        "episode_min_ee_to_card_dist": min_step["ee_to_card_dist"],
        "episode_min_dist_step_ix": min_step["step_ix"],
        "at_first_chunk": ref(1),
        "at_proximity_trigger": ref(proximity_trigger_step),
        "at_min_distance": ref(min_step["step_ix"]),
        "wrong_object_grounding_at_min_distance": (
            (not min_step["target_is_nearest"]) if min_step.get("target_is_nearest") is not None else None
        ),
        "wrong_object_grounding_majority_of_steps": (
            frac_target_is_nearest < 0.5 if frac_target_is_nearest is not None else None
        ),
        "fraction_steps_target_is_nearest": frac_target_is_nearest,
        "mean_coarse_target_dist_to_gt_card": mean_coarse,
        "median_coarse_target_dist_to_gt_card": median_coarse,
        "mean_final_target_dist_to_gt_card": mean_final,
        "median_final_target_dist_to_gt_card": median_final,
        "fraction_steps_refinement_reduced_error": (sum(refine_flags) / len(refine_flags)) if refine_flags else None,
        "mean_visual_selection_entropy": (sum(entropies) / len(entropies)) if entropies else None,
        "mean_visual_selection_top1_mass": (sum(top1_masses) / len(top1_masses)) if top1_masses else None,
        "mean_visual_selection_topk_mass": (sum(topk_masses) / len(topk_masses)) if topk_masses else None,
        "video": str(video),
    }


def run(policy: SafeDiffVLAPolicy, device: str, seeds: list[int], out_dir: Path) -> None:
    assert policy.config.architecture == "temporal_decoder_grounded_grasp_v3"
    assert policy.config.use_temporal_ensembling is False
    set_seed(seeds[0])

    env_cfg = VLABenchEnv(task=TASK)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = envs[TASK][0]
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
    env_state = install_env_instrumentation(policy_inst["close_log"])
    episode_records: list[dict] = []
    videos_dir = out_dir / "videos"

    def episode_callback(episode_ix: int, rollout_data: dict, env_idx: int, done_index: int) -> None:
        ep_len = done_index + 1
        steps = list(env_state["current_episode_steps"])
        episode_records.append({"episode_ix": episode_ix, "episode_len": ep_len, "steps": steps})
        policy_inst["plan_log"].clear()
        policy_inst["close_log"].clear()  # per-episode: episode boundary == policy.reset() call inside rollout()
        logger.info("  episode %d len=%d n_steps=%d", episode_ix, ep_len, len(steps))

    try:
        info = eval_policy(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=len(seeds),
            max_episodes_rendered=len(seeds),
            videos_dir=videos_dir,
            start_seed=seeds[0],
            episode_callback=episode_callback,
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

    per_episode_summary = [
        summarize_episode(rec["steps"], rec["seed"], rec["success"], rec["episode_len"], videos_dir / f"eval_episode_{rec['episode_ix']}.mp4")
        for rec in episode_records
    ]

    def agg_mean(key: str) -> float | None:
        vals = [e[key] for e in per_episode_summary if e.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "checkpoint": CHECKPOINT,
        "task": TASK,
        "n_episodes": len(seeds),
        "seeds": seeds,
        "note": "DEVELOPMENT eval on already-seen seeds only -- held-out seeds 3000-3009 not evaluated in this pilot.",
        "per_episode": per_episode_summary,
        "aggregate": {
            "n_success": sum(1 for e in per_episode_summary if e["success"]),
            "n_grasp_success": sum(1 for e in per_episode_summary if e["grasp_success"]),
            "n_wrong_object_grasped_ever": sum(1 for e in per_episode_summary if e["wrong_object_grasped_ever"]),
            "n_wrong_object_grounding_at_min_distance": sum(
                1 for e in per_episode_summary if e["wrong_object_grounding_at_min_distance"]
            ),
            "n_wrong_object_grounding_majority_of_steps": sum(
                1 for e in per_episode_summary if e["wrong_object_grounding_majority_of_steps"]
            ),
            "mean_min_ee_to_card_dist": agg_mean("episode_min_ee_to_card_dist"),
            "mean_coarse_target_dist_to_gt_card": agg_mean("mean_coarse_target_dist_to_gt_card"),
            "mean_final_target_dist_to_gt_card": agg_mean("mean_final_target_dist_to_gt_card"),
            "mean_fraction_steps_refinement_reduced_error": agg_mean("fraction_steps_refinement_reduced_error"),
            "mean_visual_selection_entropy": agg_mean("mean_visual_selection_entropy"),
            "mean_visual_selection_top1_mass": agg_mean("mean_visual_selection_top1_mass"),
            "mean_visual_selection_topk_mass": agg_mean("mean_visual_selection_topk_mass"),
            "n_phase_transition_matches_proximity_trigger": sum(
                1 for e in per_episode_summary if e["phase_transition_matches_proximity_trigger"]
            ),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("wrote %s and %s", out_dir / "episodes_raw.json", out_dir / "summary.json")
    logger.info("aggregate: %s", json.dumps(summary["aggregate"], indent=2))


def main() -> None:
    global CHECKPOINT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="outputs/eval/safediff_vla_grounded_grasp_v3_5k_dev")
    args = parser.parse_args()
    CHECKPOINT = args.checkpoint

    logger.info("=== loading temporal_decoder_grounded_grasp_v3 checkpoint from %s ===", CHECKPOINT)
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT)
    policy = policy.to(args.device)
    policy.eval()

    run(policy, args.device, args.seeds, Path(args.output_dir))


if __name__ == "__main__":
    main()
