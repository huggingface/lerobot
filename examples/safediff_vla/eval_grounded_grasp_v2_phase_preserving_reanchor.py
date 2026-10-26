#!/usr/bin/env python
"""PHASE 2C: phase-preserving chunk reanchoring. INFERENCE-ONLY wrapper around the UNCHANGED,
frozen `GroundedGrasp-v2` checkpoint -- no decoder retraining, no queue clear, no full replan, no
EMA/median smoothing, no z/rotation/gripper modification, no LocalGraspRefiner. The ONLY thing this
adds: when monitor-only grounding persistently disagrees with the chunk's own committed target, the
remaining QUEUED xy positions (never popped yet) are shifted by a small bounded amount -- the
chunk's own temporal progression (offset, z trajectory, rotation, gripper schedule) is completely
untouched.

Mechanism (see `SafeDiffVLAPolicy.select_action`'s v2 branch, reimplemented here with ONE insertion
point rather than monkeypatched around it, since the insertion must happen BETWEEN the executor's
internal queue management and `_grounded_grasp_v2_reactive_close`):
    1. Every env step, re-run v2's OWN frozen grounding submodules (`target_query_extractor` ->
       `visual_grounding_cross_attn` -> `target_point_head`) on the current observation -- pure
       read-only "monitor" forward, identical to Phase 1A/2A/2B's own monitor logger.
    2. `planner_anchor_xyz` = `self._v2_last_target_xyz`, the SAME value `_plan_temporal_decoder`
       already caches at the moment the current chunk was committed -- no new storage.
    3. `effective_anchor_xy = planner_anchor_xy + cumulative_correction_xy` (cumulative_correction
       resets to 0 every time a NEW chunk is actually committed, since a fresh `planner_anchor_xyz`
       makes any prior correction moot).
    4. Trigger D (Phase 2A's own, reused verbatim, called "grounding disagreement" not
       "confidence" per instruction): `monitor_jump_m > 5cm` (vs. the PREVIOUS step's monitor
       reading, 3D) AND `gap_m > 10cm` (||monitor_xyz - effective_anchor_xyz||, 3D) -- generalized
       from Phase 2A's original (pre-correction) definition to use the EFFECTIVE anchor so an
       already-corrected chunk does not keep re-triggering against its own stale original anchor.
    5. Minimal persistence: trigger must hold >=2 CONSECUTIVE steps before a correction fires (not
       on a single-frame blip).
    6. Gated to PRE_GRASP (`self._v2_grasp_phase == 0`) AND chunk_offset < 30 (Phase 2B found the
       close-related gripper/z sequence starts ~offset 35 -- correction is withheld once the chunk
       approaches that region, so it never perturbs the trajectory right before/during contact) AND
       a non-empty queue already exists (a step that itself triggers a fresh 50-step replan has
       nothing stale to correct).
    7. On fire: `delta_xy_m = monitor_xy_m - effective_anchor_xy_m`, clipped to
       `||delta_xy_m|| <= MAX_CORRECTION_M (5cm)`, added to EVERY remaining item currently in
       `self._executor._action_queue` (xy only, z/rotation/gripper slices untouched) and to
       `cumulative_correction_xy`. A COOLDOWN of 10 steps (see `COOLDOWN_STEPS` below for the
       Phase-2B-profile-based justification) then suppresses further corrections, and the
       persistence counter resets to 0 (a fresh 2-frame persistence is required after cooldown).
    `_grounded_grasp_v2_reactive_close` itself is called completely UNCHANGED afterward -- it still
    measures EE distance against the RAW (uncorrected) `self._v2_last_target_xyz`, a known,
    deliberately-not-fixed side effect this experiment is designed to surface, not paper over (see
    the module docstring's note on this in the Phase 2C report).

Cooldown justification (`COOLDOWN_STEPS = 10`): Phase 2B's own `chunk_temporal_profile.png` showed
the XY-correction-magnitude curve rising to its peak around offset 5-15 and decaying over a
similar-width window -- i.e. one "alignment correction" naturally plays out over roughly 10-15
steps. 10 is picked as the round number at the lower end of that observed timescale (reused, not
re-derived, from the same K=10 window Phase 2A already used for `recent_var_m`) -- NOT swept.

Usage:
    MUJOCO_GL=egl uv run python examples/safediff_vla/eval_grounded_grasp_v2_phase_preserving_reanchor.py \
        --tasks select_poker --seeds 1001
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
from eval_grounded_grasp_v2_monitor_baseline import CHECKPOINT, extract_physics_record, robot_base  # noqa: E402

from lerobot.envs import make_env, make_env_pre_post_processors  # noqa: E402
from lerobot.envs.configs import VLABenchEnv  # noqa: E402
from lerobot.envs.vlabench import VLABenchEnv as VLABenchEnvImpl  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402
from lerobot.scripts.lerobot_eval import eval_policy  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

SEEDS = [1000, 1001, 1002]
OUT_DIR_ROOT = Path("outputs/eval/safediff_vla_grounded_grasp_v2_phase_preserving_reanchor")
REPRESENTATIVE_EPISODES = [
    ("select_poker", 1001), ("select_fruit", 1001), ("select_book", 1002), ("select_mahjong", 1000),
]

TAU_JUMP_M = 0.05
TAU_GAP_M = 0.10
PERSISTENCE_FRAMES = 2
MAX_CORRECTION_M = 0.05
COOLDOWN_STEPS = 10
OFFSET_LIMIT = 30


def quat_wxyz_to_euler_xyz(quat_wxyz) -> list[float]:
    w, x, y, z = quat_wxyz
    return Rotation.from_quat([x, y, z, w]).as_euler("xyz").tolist()


def install_env_instrumentation(policy_log: list[dict]) -> dict:
    cls = VLABenchEnvImpl
    orig_step = cls.step
    orig_reset = cls.reset
    state: dict[str, Any] = {"current_episode_steps": []}

    def merge_policy_log(record: dict, base: np.ndarray) -> None:
        keys = (
            "phase", "reactive_close_event", "planner_replanned_this_step", "chunk_offset",
            "monitor_target_world", "monitor_target_dist_to_gt",
            "planner_target_world", "planner_target_dist_to_gt",
            "effective_anchor_world", "effective_anchor_dist_to_gt",
            "commanded_xyz_world", "commanded_rot_euler_raw", "commanded_gripper_raw",
            "reanchor_event", "applied_correction_m", "gap_before_m", "gap_after_m",
            "cumulative_correction_xy_m",
        )
        for k in keys:
            record[k] = None
        if not policy_log:
            return
        p = policy_log[-1]
        for k in (
            "phase", "reactive_close_event", "planner_replanned_this_step", "chunk_offset",
            "commanded_rot_euler_raw", "commanded_gripper_raw",
            "reanchor_event", "applied_correction_m", "gap_before_m", "gap_after_m",
            "cumulative_correction_xy_m",
        ):
            record[k] = p.get(k)
        target_pos = np.asarray(record["target_pos"], dtype=float)
        if p.get("monitor_target_robot_frame") is not None:
            w = np.asarray(p["monitor_target_robot_frame"], dtype=float) + base
            record["monitor_target_world"] = w.tolist()
            record["monitor_target_dist_to_gt"] = float(np.linalg.norm(w - target_pos))
        if p.get("planner_target_robot_frame") is not None:
            w = np.asarray(p["planner_target_robot_frame"], dtype=float) + base
            record["planner_target_world"] = w.tolist()
            record["planner_target_dist_to_gt"] = float(np.linalg.norm(w - target_pos))
        if p.get("effective_anchor_robot_frame") is not None:
            w = np.asarray(p["effective_anchor_robot_frame"], dtype=float) + base
            record["effective_anchor_world"] = w.tolist()
            record["effective_anchor_dist_to_gt"] = float(np.linalg.norm(w - target_pos))
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


def install_reanchor_policy(policy: SafeDiffVLAPolicy, enabled: bool) -> dict:
    """`enabled=False` reproduces condition A (original, untouched E50 behavior) with the SAME
    monitor-only logging as Phase 1A, for apples-to-apples plotting against condition B
    (`enabled=True`) without needing two different instrumentation code paths."""
    from lerobot.policies.safediff_vla.rotation_encoding import GRIPPER_INDEX_RAW, ROTATION_SLICE_RAW

    orig_plan = SafeDiffVLAPolicy._plan_temporal_decoder
    orig_reactive_close = SafeDiffVLAPolicy._grounded_grasp_v2_reactive_close
    orig_reset = SafeDiffVLAPolicy.reset
    log: list[dict] = []
    replanned_flag = {"flag": False}

    def wrapped_plan(self, batch):
        replanned_flag["flag"] = True
        return orig_plan(self, batch)

    def wrapped_reset(self):
        orig_reset(self)
        self._pc_cumulative_correction_xy_m = None  # lazy zeros_like on first use
        self._pc_prev_monitor_xyz_m = None
        self._pc_consecutive_trigger = 0
        self._pc_cooldown_remaining = 0
        self._pc_chunk_offset = -1
        self._pc_prev_close_triggered = False

    def wrapped_select_action(self, batch):
        self.eval()
        current_state = self._current_state(batch)
        assert self.architecture == "temporal_decoder_grounded_grasp_v2"

        with torch.no_grad():
            latent_tokens, latent_pad_mask, latent_modality_ids = self._encode_multimodal_latent(batch)
            target_language_query = self.target_query_extractor(latent_tokens, latent_pad_mask, latent_modality_ids)
            grounded_target_feature = self.visual_grounding_cross_attn(
                target_language_query, latent_tokens, latent_pad_mask, latent_modality_ids
            )
            monitor_target_xyz = self.target_point_head(grounded_target_feature)
        monitor_xyz_m = (monitor_target_xyz * self.action_pos_std + self.action_pos_mean)[0].detach()
        monitor_xy_m = monitor_xyz_m[:2]

        had_queue = len(self._executor._action_queue) > 0
        replanned_flag["flag"] = False

        entry: dict[str, Any] = {}
        reanchor_event = False
        applied_correction_m = None
        gap_before_m = None
        gap_after_m = None

        if enabled:
            if self._pc_cumulative_correction_xy_m is None:
                self._pc_cumulative_correction_xy_m = torch.zeros(2, device=monitor_xy_m.device, dtype=monitor_xy_m.dtype)
            self._pc_chunk_offset = self._pc_chunk_offset + 1 if self._pc_chunk_offset >= 0 else 0
            planner_anchor_m = None
            if self._v2_last_target_xyz is not None:
                planner_anchor_m = (self._v2_last_target_xyz * self.action_pos_std + self.action_pos_mean)[0].detach()
                effective_anchor_xy_m = planner_anchor_m[:2] + self._pc_cumulative_correction_xy_m
                gap_m = float((monitor_xyz_m - torch.cat([effective_anchor_xy_m, planner_anchor_m[2:3]])).norm().item())
            else:
                gap_m = None
            # trigger D uses the FULL 3D monitor jump (Phase 2A's own definition) but gap is the
            # xy-effective-anchor-relative one computed above (already folds in prior corrections).
            jump3d_m = (
                float((monitor_xyz_m - self._pc_prev_monitor_xyz_m).norm().item())
                if getattr(self, "_pc_prev_monitor_xyz_m", None) is not None
                else None
            )
            trig_now = (jump3d_m is not None) and (gap_m is not None) and (jump3d_m > TAU_JUMP_M) and (gap_m > TAU_GAP_M)
            self._pc_consecutive_trigger = self._pc_consecutive_trigger + 1 if trig_now else 0
            self._pc_cooldown_remaining = max(0, self._pc_cooldown_remaining - 1)

            can_fire = (
                had_queue and self._v2_grasp_phase == 0 and self._pc_chunk_offset < OFFSET_LIMIT
                and planner_anchor_m is not None and self._pc_consecutive_trigger >= PERSISTENCE_FRAMES
                and self._pc_cooldown_remaining == 0
            )
            if can_fire:
                effective_anchor_xy_m = planner_anchor_m[:2] + self._pc_cumulative_correction_xy_m
                delta_xy_m = monitor_xy_m - effective_anchor_xy_m
                norm = delta_xy_m.norm().clamp_min(1e-8)
                applied = delta_xy_m * min(1.0, MAX_CORRECTION_M / norm.item())
                applied_delta_xy_norm = applied / self.action_pos_std[:2]
                for i in range(len(self._executor._action_queue)):
                    item = self._executor._action_queue[i].clone()
                    item[..., :2] = item[..., :2] + applied_delta_xy_norm
                    self._executor._action_queue[i] = item
                gap_before_m = gap_m
                self._pc_cumulative_correction_xy_m = self._pc_cumulative_correction_xy_m + applied
                new_effective_xy = planner_anchor_m[:2] + self._pc_cumulative_correction_xy_m
                gap_after_m = float((monitor_xyz_m - torch.cat([new_effective_xy, planner_anchor_m[2:3]])).norm().item())
                applied_correction_m = float(applied.norm().item())
                reanchor_event = True
                self._pc_cooldown_remaining = COOLDOWN_STEPS
                self._pc_consecutive_trigger = 0

            self._pc_prev_monitor_xyz_m = monitor_xyz_m
            entry["effective_anchor_robot_frame"] = (
                (planner_anchor_m[:2] + self._pc_cumulative_correction_xy_m).tolist() + [planner_anchor_m[2].item()]
                if planner_anchor_m is not None else None
            )
            entry["chunk_offset"] = int(self._pc_chunk_offset)
            entry["cumulative_correction_xy_m"] = self._pc_cumulative_correction_xy_m.tolist()

        action = self._executor.select_action(current_state, lambda: self.plan_action_chunk(batch))
        if replanned_flag["flag"] and enabled:
            self._pc_chunk_offset = 0
            self._pc_cumulative_correction_xy_m = torch.zeros(2, device=monitor_xy_m.device, dtype=monitor_xy_m.dtype)
            self._pc_consecutive_trigger = 0
            self._pc_cooldown_remaining = 0
            entry["chunk_offset"] = 0
            entry["cumulative_correction_xy_m"] = [0.0, 0.0]
            entry["effective_anchor_robot_frame"] = None  # anchor just reset; next step recomputes it

        action = orig_reactive_close(self, action, current_state)

        xyz_m = (action[..., :3] * self.action_pos_std + self.action_pos_mean)[0].detach().cpu().numpy()
        rot_raw = (action[..., ROTATION_SLICE_RAW] * self.action_rot_std + self.action_rot_mean)[0].detach().cpu().numpy()
        grip_raw = float((action[0, GRIPPER_INDEX_RAW] * self.action_grip_std + self.action_grip_mean).item())
        planner_m = None
        if self._v2_last_target_xyz is not None:
            planner_m = (self._v2_last_target_xyz * self.action_pos_std + self.action_pos_mean)[0].detach().cpu().numpy()

        reactive_close_event = bool(self._v2_close_triggered) and not self._pc_prev_close_triggered
        self._pc_prev_close_triggered = bool(self._v2_close_triggered)

        entry.update(
            {
                "phase": int(self._v2_grasp_phase),
                "reactive_close_event": reactive_close_event,
                "planner_replanned_this_step": bool(replanned_flag["flag"]),
                "monitor_target_robot_frame": monitor_xyz_m.cpu().numpy().tolist(),
                "planner_target_robot_frame": planner_m.tolist() if planner_m is not None else None,
                "commanded_xyz_robot_frame": xyz_m.tolist(),
                "commanded_rot_euler_raw": rot_raw.tolist(),
                "commanded_gripper_raw": grip_raw,
                "reanchor_event": reanchor_event,
                "applied_correction_m": applied_correction_m,
                "gap_before_m": gap_before_m,
                "gap_after_m": gap_after_m,
            }
        )
        log.append(entry)
        return action

    policy._plan_temporal_decoder = wrapped_plan.__get__(policy, SafeDiffVLAPolicy)
    policy.select_action = wrapped_select_action.__get__(policy, SafeDiffVLAPolicy)
    policy.reset = wrapped_reset.__get__(policy, SafeDiffVLAPolicy)
    return {"log": log}


def uninstall_reanchor_policy(policy: SafeDiffVLAPolicy) -> None:
    for attr in ("_plan_temporal_decoder", "select_action", "reset"):
        if attr in policy.__dict__:
            del policy.__dict__[attr]


def run_task(policy: SafeDiffVLAPolicy, device: str, task: str, seeds: list[int], out_dir: Path, enabled: bool) -> None:
    set_seed(seeds[0])
    env_cfg = VLABenchEnv(task=task)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = envs[task][0]
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=CHECKPOINT,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)

    policy_inst = install_reanchor_policy(policy, enabled=enabled)
    env_state = install_env_instrumentation(policy_inst["log"])
    episode_records: list[dict] = []
    videos_dir = out_dir / "videos"

    def episode_callback(episode_ix, rollout_data, env_idx, done_index):
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
        uninstall_reanchor_policy(policy)

    for ep, rec in zip(info["per_episode"], episode_records, strict=True):
        rec["seed"] = ep["seed"]
        rec["success"] = ep["success"]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "episodes_raw.json").write_text(json.dumps(episode_records, indent=2))
    logger.info("[%s] wrote %s (successes: %s)", task, out_dir / "episodes_raw.json", [ep["success"] for ep in info["per_episode"]])


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--condition", choices=["A", "B"], default="B")
    parser.add_argument("--tasks", nargs="+", default=STAGE2_TASK_NAMES)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else OUT_DIR_ROOT / f"condition_{args.condition}"

    logger.info("=== loading checkpoint: %s (condition=%s) ===", CHECKPOINT, args.condition)
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT)
    policy = policy.to(args.device)
    policy.eval()
    for task in args.tasks:
        logger.info("--- task=%s seeds=%s condition=%s ---", task, args.seeds, args.condition)
        run_task(policy, args.device, task, args.seeds, out_dir / task, enabled=(args.condition == "B"))


if __name__ == "__main__":
    main()
