#!/usr/bin/env python
"""FINAL unseen-seed evaluation for the SafeDiff-VLA paper submission (machine "B" share: seeds
3005-3009; "A" covers the other half of the seed range under the same protocol/config). NO
architecture/loss/controller/checkpoint changes are made by this script -- it only evaluates
already-existing, already-frozen checkpoints. `select_ingredient` excluded throughout (confirmed
train/eval scene mismatch, see `_multitask_common.py`).

Model variants (5 -- `TemporalDecoder-RawEuler` excluded: its checkpoint,
`outputs/train/safediff_vla_temporal_decoder_v2_baseline_20k`, does not exist on this machine and
was NOT retrained for this eval, per explicit user decision):
    SmolVLA                        -- architecture="smolvla_nominal", fresh backbone-only policy
                                       (`lerobot/smolvla_vlabench`), no SafeDiff-VLA head weights
                                       exist for this row by construction.
    TemporalDecoder-SinCos-PadFix   -- outputs/train/safediff_vla_temporal_decoder_padfix_20k/checkpoints/020000
    GroundedGrasp-v2                -- outputs/train/safediff_vla_grounded_grasp_v2_20k/checkpoints/020000
    GroundedGrasp-v2.5              -- outputs/train/safediff_vla_grounded_grasp_v2_5_5k/checkpoints/010000
                                       (NOT that directory's "last"=015000 continuation -- 010000 is
                                       the checkpoint every prior comparison script on this branch
                                       called "single_task_v2_5_10k"; kept consistent with that
                                       established identity, recorded explicitly in the saved
                                       metadata for a human to verify).
    SafeDiff-VLA-Final              -- outputs/train/safediff_vla_grounded_grasp_v2_5_phase_aware_loss_100k/checkpoints/050000
                                       (the FROZEN final candidate: v2.5 + phase-aware loss, 100k
                                       long run, checkpoint selected from dev sweep -- not
                                       reconsidered here under any circumstance).

Universal (architecture-independent) instrumentation via `VLABenchEnvImpl.step`/`reset`: ee_pos,
target_pos, is_grasped (privileged sim contact -- explicitly NOT treated as official grasp success,
see `contact_based_grasp_diagnostic` below), ee_to_target_dist. This alone is enough for every
variant, including SmolVLA (no grounding head at all).

Conditional (only when the policy exposes it) instrumentation: `_plan_temporal_decoder`'s own
`predicted_target_xyz` (present for every grounded-grasp architecture: v2/v2.5, absent/no-op for
plain `temporal_decoder` and `smolvla_nominal`) for target-localization error; `_plan_local_grasp_
step`/`_maybe_enter_local_grasp` (only `architecture=="temporal_decoder_grounded_grasp_v2_5"` has
these methods at all) for "LOCAL_GRASP entered".

Scene validity sanity check at episode reset: target entity resolves + has a finite xpos, robot
base position is finite and within a broad plausible bound, at least one camera frame is non-
degenerate (non-constant). On failure the episode is tagged `INVALID_ENVIRONMENT` and excluded from
the success-rate denominator (NOT scored as a policy failure) -- see `is_invalid_environment`.

Process isolation: this script itself handles exactly ONE (model, task) pair per invocation (one
fresh OS process, per the required "process-per-task isolation" -- same VLABench/MuJoCo state-leak
workaround as every other multitask eval on this branch). `run_final_eval_paper_isolated.py` is the
driver that launches one subprocess per (model, task).

Usage:
    MUJOCO_GL=egl uv run python examples/safediff_vla/final_eval_paper_multivariant.py --model SafeDiff-VLA-Final --task select_book
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import RENAME_MAP, STAGE2_TASK_NAMES  # noqa: E402

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.envs import make_env, make_env_pre_post_processors  # noqa: E402
from lerobot.envs.configs import VLABenchEnv  # noqa: E402
from lerobot.envs.vlabench import VLABenchEnv as VLABenchEnvImpl  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.configuration_safediff_vla import SafeDiffVLAConfig  # noqa: E402
from lerobot.policies.safediff_vla.local_grasp_refiner import LOCAL_GRASP  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402
from lerobot.scripts.lerobot_eval import eval_policy  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

MACHINE = "B"
SEEDS = [3005, 3006, 3007, 3008, 3009]
TASKS = STAGE2_TASK_NAMES
OUT_DIR = Path(f"outputs/final_eval/{MACHINE}_final_seeds_{SEEDS[0]}_{SEEDS[-1]}")

_BASE_KWARGS = {
    "device": "cuda",
    "push_to_hub": False,
    "backbone_name": "lerobot/smolvla_vlabench",
    "vlm_model_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    "freeze_backbone": True,
    "freeze_vision_encoder": True,
    "use_lora": False,
    "decoder_hidden_dim": 512,
    "decoder_num_layers": 4,
    "decoder_num_heads": 8,
    "decoder_ffn_dim": 1024,
    "decoder_dropout": 0.1,
    "action_horizon": 50,
    "execute_horizon": 50,
    "use_temporal_ensembling": False,
}
_STANDARD_FEATURES = {
    "input_features": {
        "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        "observation.images.camera2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        "observation.images.camera3": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
    },
    "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
}

MODEL_VARIANTS: dict[str, dict[str, Any]] = {
    "SmolVLA": {"checkpoint": None, "architecture": "smolvla_nominal"},
    "TemporalDecoder-SinCos-PadFix": {
        "checkpoint": "outputs/train/safediff_vla_temporal_decoder_padfix_20k/checkpoints/020000/pretrained_model",
        "architecture": "temporal_decoder",
    },
    "GroundedGrasp-v2": {
        "checkpoint": "outputs/train/safediff_vla_grounded_grasp_v2_20k/checkpoints/020000/pretrained_model",
        "architecture": "temporal_decoder_grounded_grasp_v2",
    },
    "GroundedGrasp-v2.5": {
        "checkpoint": "outputs/train/safediff_vla_grounded_grasp_v2_5_5k/checkpoints/010000/pretrained_model",
        "architecture": "temporal_decoder_grounded_grasp_v2_5",
    },
    "SafeDiff-VLA-Final": {
        "checkpoint": "outputs/train/safediff_vla_grounded_grasp_v2_5_phase_aware_loss_100k/checkpoints/050000/pretrained_model",
        "architecture": "temporal_decoder_grounded_grasp_v2_5",
    },
}


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2]).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def load_policy(model_name: str, device: str) -> tuple[SafeDiffVLAPolicy, str, dict]:
    variant = MODEL_VARIANTS[model_name]
    if variant["checkpoint"] is None:  # SmolVLA nominal -- fresh construction, no saved head
        cfg = SafeDiffVLAConfig(architecture=variant["architecture"], **_BASE_KWARGS, **_STANDARD_FEATURES)
        policy = SafeDiffVLAPolicy(cfg)
        ckpt_id = f"smolvla_nominal(backbone={_BASE_KWARGS['backbone_name']}, no_trained_head)"
        ckpt_hash = None
    else:
        ckpt_path = variant["checkpoint"]
        policy = SafeDiffVLAPolicy.from_pretrained(ckpt_path)
        assert policy.config.architecture == variant["architecture"], (
            f"{model_name}: expected architecture={variant['architecture']!r}, checkpoint says {policy.config.architecture!r}"
        )
        ckpt_id = ckpt_path
        ckpt_hash = file_sha256(Path(ckpt_path) / "model.safetensors")
    policy = policy.to(device)
    policy.eval()
    recipe = {
        "model": model_name,
        "architecture": policy.config.architecture,
        "checkpoint_identifier": ckpt_id,
        "checkpoint_model_safetensors_sha256": ckpt_hash,
        "action_horizon": policy.config.action_horizon,
        "execute_horizon": policy.config.execute_horizon,
        "local_grasp_radius_m": getattr(policy.config, "local_grasp_radius_m", None),
        "local_grasp_use_phase_aware_loss": getattr(policy.config, "local_grasp_use_phase_aware_loss", None),
        "lambda_local_terminal_xyz": getattr(policy.config, "lambda_local_terminal_xyz", None),
        "lambda_local_approach_xyz": getattr(policy.config, "lambda_local_approach_xyz", None),
        "lambda_local_grip_weighted": getattr(policy.config, "lambda_local_grip_weighted", None),
        "local_grasp_grip_close_weight": getattr(policy.config, "local_grasp_grip_close_weight", None),
        "grounded_grasp_gripper_open_threshold": getattr(policy.config, "grounded_grasp_gripper_open_threshold", None),
    }
    return policy, ckpt_id, recipe


def robot_base(env: VLABenchEnvImpl) -> np.ndarray:
    base = getattr(env, "_robot_base_xyz", None)
    return np.asarray(base, dtype=float) if base is not None else np.array([0.0, -0.4, 0.78])


def scene_validity_check(env: VLABenchEnvImpl) -> tuple[bool, str]:
    """Target exists / robot base plausible / target xyz finite&plausible / camera non-degenerate."""
    physics = env._env.physics
    task = env._env.task
    try:
        target_name = task.target_entity
        target_pos = np.asarray(task.entities[target_name].get_xpos(physics), dtype=float)
    except Exception as e:  # noqa: BLE001
        return False, f"target_entity unresolved: {e}"
    if not np.all(np.isfinite(target_pos)):
        return False, f"target_pos non-finite: {target_pos.tolist()}"
    if not np.all(np.abs(target_pos) < 10.0):  # broad plausibility bound, meters
        return False, f"target_pos implausible: {target_pos.tolist()}"
    base = robot_base(env)
    if not np.all(np.isfinite(base)) or not np.all(np.abs(base) < 10.0):
        return False, f"robot_base implausible: {base.tolist()}"
    try:
        obs = env._env.task.get_observation(physics) if hasattr(env._env.task, "get_observation") else None
    except Exception:  # noqa: BLE001
        obs = None
    # Camera non-degenerate: best-effort via the physics renderer directly (works regardless of
    # whether `get_observation` exposes images under a name this script can guess).
    try:
        frame = physics.render(height=64, width=64, camera_id=0)
        if frame is None or np.asarray(frame).std() < 1e-6:
            return False, "camera frame appears degenerate (near-zero variance)"
    except Exception as e:  # noqa: BLE001
        return False, f"camera render failed: {e}"
    del obs
    return True, "ok"


def extract_physics_record(env: VLABenchEnvImpl, step_ix: int) -> dict[str, Any]:
    physics = env._env.physics
    task = env._env.task
    robot = task.robot
    target_name = task.target_entity
    ee_pos = np.asarray(robot.get_end_effector_pos(physics), dtype=float)
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
        "target_entity": target_name,
        "target_pos": target_pos.tolist(),
        "is_grasped": is_grasped,
        "ee_to_target_dist": float(np.linalg.norm(ee_pos - target_pos)),
    }


def install_env_instrumentation(policy_log: list[dict]) -> dict:
    cls = VLABenchEnvImpl
    orig_step = cls.step
    orig_reset = cls.reset
    state: dict[str, Any] = {"current_episode_steps": [], "invalid_reason": None}

    def merge_policy_log(record: dict, base: np.ndarray) -> None:
        for k in ("stage", "predicted_target_world", "predicted_target_dist_to_gt_target"):
            record[k] = None
        if not policy_log:
            return
        p = policy_log[-1]
        record["stage"] = p.get("stage")
        if p.get("predicted_target_robot_frame") is not None:
            predicted_target_world = np.asarray(p["predicted_target_robot_frame"], dtype=float) + base
            target_pos = np.asarray(record["target_pos"], dtype=float)
            record["predicted_target_world"] = predicted_target_world.tolist()
            record["predicted_target_dist_to_gt_target"] = float(np.linalg.norm(predicted_target_world - target_pos))

    def patched_reset(self, seed=None, **kwargs):
        result = orig_reset(self, seed=seed, **kwargs)
        if seed is not None:
            valid, reason = scene_validity_check(self)
            state["invalid_reason"] = None if valid else reason
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
    """Conditional: only wraps methods that actually exist for this architecture."""
    log: list[dict] = []
    installed: dict[str, Any] = {"log": log, "wrapped": []}

    if hasattr(policy, "_plan_temporal_decoder"):
        orig_plan = type(policy)._plan_temporal_decoder

        def wrapped_plan(self, batch):
            actions, metrics = orig_plan(self, batch)
            entry: dict[str, Any] = {"stage": getattr(self, "_v25_stage", None)}
            if "predicted_target_xyz" in metrics and hasattr(self, "action_pos_std"):
                target_pos_m = metrics["predicted_target_xyz"] * self.action_pos_std + self.action_pos_mean
                entry["predicted_target_robot_frame"] = target_pos_m[0].detach().cpu().numpy().tolist()
            log.append(entry)
            return actions, metrics

        policy._plan_temporal_decoder = wrapped_plan.__get__(policy, SafeDiffVLAPolicy)
        installed["wrapped"].append("_plan_temporal_decoder")

    if hasattr(policy, "_plan_local_grasp_step"):
        orig_local_step = type(policy)._plan_local_grasp_step

        def wrapped_local_step(self, batch):
            action, metrics = orig_local_step(self, batch)
            target_pos_m = metrics["predicted_target_xyz"] * self.action_pos_std + self.action_pos_mean
            log.append({"stage": int(self._v25_stage), "predicted_target_robot_frame": target_pos_m[0].detach().cpu().numpy().tolist()})
            return action, metrics

        policy._plan_local_grasp_step = wrapped_local_step.__get__(policy, SafeDiffVLAPolicy)
        installed["wrapped"].append("_plan_local_grasp_step")

    return installed


def uninstall_policy_instrumentation(policy: SafeDiffVLAPolicy, installed: dict) -> None:
    for attr in installed["wrapped"]:
        if attr in policy.__dict__:
            del policy.__dict__[attr]


def summarize_episode(steps: list[dict], seed: int, success: bool, episode_len: int, invalid_reason: str | None) -> dict:
    if invalid_reason is not None:
        return {
            "seed": seed, "status": "INVALID_ENVIRONMENT", "invalid_reason": invalid_reason,
            "native_success": None, "episode_len": episode_len,
        }
    contact_based_grasp_diagnostic = any(s["is_grasped"] for s in steps)  # NOT official success -- contact loophole, see module docstring
    entered_local_grasp = any(s.get("stage") == LOCAL_GRASP for s in steps)
    terminal_ee_to_target_dist = steps[-1]["ee_to_target_dist"]
    loc_errors = [s["predicted_target_dist_to_gt_target"] for s in steps if s.get("predicted_target_dist_to_gt_target") is not None]
    target_localization_error_m = float(np.mean(loc_errors)) if loc_errors else None
    insufficient_approach = not contact_based_grasp_diagnostic  # never achieved contact at all -- same convention as trajectory_shape_diagnostic.py
    return {
        "seed": seed,
        "status": "OK",
        "native_success": bool(success),
        "contact_based_grasp_diagnostic": contact_based_grasp_diagnostic,
        "insufficient_approach": insufficient_approach,
        "terminal_ee_to_target_dist_m": terminal_ee_to_target_dist,
        "target_localization_error_m": target_localization_error_m,
        "entered_local_grasp": entered_local_grasp,
        "episode_len": episode_len,
    }


def run_model_task(model_name: str, task: str, root_dir: Path) -> None:
    out_dir = root_dir / "runs" / model_name / task  # recipe.json + raw per-step traces live here
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, ckpt_id, recipe = load_policy(model_name, device)
    recipe["task"] = task
    recipe["machine"] = MACHINE
    recipe["git_commit"] = git_commit()
    recipe["seeds"] = SEEDS

    set_seed(SEEDS[0])
    env_cfg = VLABenchEnv(task=task)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = envs[task][0]
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=MODEL_VARIANTS[model_name]["checkpoint"],
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )
    if MODEL_VARIANTS[model_name]["checkpoint"] is None:
        # `make_pre_post_processors` silently DROPS `preprocessor_overrides` on the from-scratch
        # (no `pretrained_path`) construction path (see `lerobot/policies/factory.py`'s SafeDiffVLA/
        # SmolVLA branch -- only `dataset_stats` is forwarded there), so the rename step above never
        # actually applies for SmolVLA-nominal. Apply it manually at the boundary instead -- same
        # `rename_batch_keys` helper `lerobot_train.py`'s own `_preprocess_dataset_batch` uses.
        from lerobot.processor.rename_processor import rename_batch_keys

        _orig_preprocessor = preprocessor
        preprocessor = lambda batch, _p=_orig_preprocessor: _p(rename_batch_keys(batch, RENAME_MAP))  # noqa: E731
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)

    policy_inst = install_policy_instrumentation(policy)
    env_state = install_env_instrumentation(policy_inst["log"])
    videos_dir = root_dir / "videos" / model_name / task
    episode_results: list[dict] = []

    def episode_callback(episode_ix: int, rollout_data: dict, env_idx: int, done_index: int) -> None:
        ep_len = done_index + 1
        steps = list(env_state["current_episode_steps"])
        invalid_reason = env_state["invalid_reason"]
        seed = SEEDS[episode_ix]
        summary = summarize_episode(steps, seed, None, ep_len, invalid_reason)  # success filled in after eval_policy returns
        summary["_steps"] = steps
        summary["_episode_ix"] = episode_ix
        episode_results.append(summary)
        policy_inst["log"].clear()
        logger.info("  [%s/%s] seed=%d PID=%d len=%d status=%s", model_name, task, seed, os.getpid(), ep_len, summary["status"])

    try:
        info = eval_policy(
            env=env, policy=policy, env_preprocessor=env_preprocessor, env_postprocessor=env_postprocessor,
            preprocessor=preprocessor, postprocessor=postprocessor, n_episodes=len(SEEDS),
            max_episodes_rendered=len(SEEDS), videos_dir=videos_dir, start_seed=SEEDS[0], episode_callback=episode_callback,
        )
    finally:
        env.close()
        uninstall_env_instrumentation(env_state)
        uninstall_policy_instrumentation(policy, policy_inst)

    config_hash = hashlib.sha256(json.dumps(recipe, sort_keys=True, default=str).encode()).hexdigest()[:16]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "recipe.json").write_text(json.dumps(recipe, indent=2))

    jsonl_path = root_dir / "episode_results.jsonl"
    with open(jsonl_path, "a") as f:
        for ep, summary in zip(info["per_episode"], episode_results, strict=True):
            if summary["status"] == "OK":
                summary["native_success"] = bool(ep["success"])
            steps = summary.pop("_steps")
            episode_ix = summary.pop("_episode_ix")
            record = {
                "machine": MACHINE,
                "git_commit": recipe["git_commit"],
                "checkpoint_identifier": ckpt_id,
                "checkpoint_model_safetensors_sha256": recipe["checkpoint_model_safetensors_sha256"],
                "config_hash": config_hash,
                "model_variant": model_name,
                "task": task,
                "seed": summary["seed"],
                "pid": os.getpid(),
                **{k: v for k, v in summary.items() if k != "seed"},
                "video_path": str(videos_dir / f"eval_episode_{episode_ix}.mp4"),
            }
            f.write(json.dumps(record) + "\n")
            # Also stash the raw per-step trace next to the video for later visual/diagnostic work.
            (videos_dir / f"eval_episode_{episode_ix}_steps.json").write_text(json.dumps(steps))
    logger.info("[%s/%s] appended %d episode records to %s", model_name, task, len(episode_results), jsonl_path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=list(MODEL_VARIANTS))
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    run_model_task(args.model, args.task, Path(args.output_dir))


if __name__ == "__main__":
    main()
