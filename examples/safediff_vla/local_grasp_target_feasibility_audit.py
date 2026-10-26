#!/usr/bin/env python
"""Pre-design-doc feasibility audit for the proposed `LocalGraspTargetHead`/`LocalGraspCorrectionHead`
(coarse-object vs. fine-grasp-point split for `architecture="temporal_decoder_grounded_grasp_v2_5"`'s
LOCAL_GRASP phase -- see the design note this script's numbers feed into). NOT a model/architecture
change -- pure `lerobot/vlabench_unified` parquet-column analysis (no video decode, no GPU, no
policy forward pass), same status as `audit_grasp_target_supervision.py` but broken out PER TASK
(that script pools all tasks) and extended with an action-vs-state consistency check the existing
audit does not do.

Answers three questions, per one of the 6 `STAGE2_TASK_NAMES` tasks:

  1. Coverage -- of episodes with a real (non-padded) open->close gripper transition at all, what
     fraction, and how many steps into the episode does it typically land (`utils.find_grasp_target`
     only ever sees a fixed `action_horizon`-step future window at train time, so a transition that
     lands very late in a long episode may fall outside every window a real training sample ever
     sees -- this is a DIFFERENT coverage number from `audit_grasp_target_supervision.py`'s per-frame
     `in_horizon_coverage`, this one is per-EPISODE "does a transition exist at all").
  2. Consistency -- `find_grasp_target`'s label is the COMMANDED `action[..., :3]` xyz at the
     transition step (see that function's docstring), not the physically-realized
     `observation.state[..., :3]` xyz. This script measures how far apart those two candidate
     "demo grasp TCP xyz" definitions actually are, per task, by also locating the first frame
     where `observation.state`'s OWN gripper channel confirms closed (one frame after the action
     transition, per `utils._state_gripper_is_open`'s docstring) and diffing positions.
  3. Scale -- a "typical reach distance" reference (episode-start xyz -> transition xyz) so the
     correction magnitudes this feeds into a design doc can be read relative to something, not as
     bare centimeters.

Outputs:
    outputs/eval/safediff_vla_local_grasp_target_feasibility_audit/summary.json

Usage:
    uv run python examples/safediff_vla/local_grasp_target_feasibility_audit.py
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import REPO_ID, STAGE2_TASK_NAMES, build_episode_task_map  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.safediff_vla.rotation_encoding import GRIPPER_INDEX_RAW  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

GRIPPER_OPEN_THRESHOLD = 0.5
N_EPISODES_PER_TASK = 60
SEED = 0


def action_open(v: np.ndarray) -> np.ndarray:
    return v > GRIPPER_OPEN_THRESHOLD


def state_open(v: np.ndarray) -> np.ndarray:
    return v < GRIPPER_OPEN_THRESHOLD


def analyze_episode(action_xyz: np.ndarray, action_grip: np.ndarray, state_xyz: np.ndarray, state_grip: np.ndarray) -> dict:
    """All arrays `[T, ...]` for one episode, full length (no horizon windowing -- this is an
    episode-level "does a real event exist, and where" check, distinct from the per-frame
    windowed check `audit_grasp_target_supervision.py` already covers)."""
    length = len(action_grip)
    a_open = action_open(action_grip)
    # First action-only open->close transition anywhere in the episode.
    transition_ix = None
    for t in range(length - 1):
        if a_open[t] and not a_open[t + 1]:
            transition_ix = t + 1  # action[transition_ix] is the first CLOSED command
            break
    if transition_ix is None:
        return {"has_transition": False}

    commanded_xyz = action_xyz[transition_ix]

    # First frame where observation.state's OWN gripper channel confirms closed, at/after the
    # action transition (expected ~transition_ix or +1, per `_state_gripper_is_open`'s docstring).
    s_open = state_open(state_grip)
    confirm_ix = None
    for t in range(transition_ix, length):
        if not s_open[t]:
            confirm_ix = t
            break

    result = {
        "has_transition": True,
        "episode_length": length,
        "transition_ix": transition_ix,
        "transition_frac_of_episode": transition_ix / length,
        "commanded_xyz": commanded_xyz.tolist(),
        "reach_dist_m": float(np.linalg.norm(commanded_xyz - state_xyz[0])),
    }
    if confirm_ix is not None:
        actual_xyz = state_xyz[confirm_ix]
        result.update(
            {
                "confirm_ix": confirm_ix,
                "confirm_lag_steps": confirm_ix - transition_ix,
                "actual_xyz": actual_xyz.tolist(),
                "commanded_vs_actual_dist_m": float(np.linalg.norm(commanded_xyz - actual_xyz)),
            }
        )
    else:
        result.update({"confirm_ix": None, "confirm_lag_steps": None, "actual_xyz": None, "commanded_vs_actual_dist_m": None})
    return result


def summarize(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=float)
    return {
        "n": len(arr),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def run_task(task: str, episodes: list[int]) -> dict:
    ds = LeRobotDataset(REPO_ID, episodes=episodes)
    raw = ds.select_columns(["action", "observation.state", "episode_index", "frame_index"]).to_pandas()
    per_episode = []
    for ep, group in raw.groupby("episode_index", sort=True):
        group = group.sort_values("frame_index")
        action = np.stack(group["action"].to_numpy())
        state = np.stack(group["observation.state"].to_numpy())
        analysis = analyze_episode(
            action[:, :3], action[:, GRIPPER_INDEX_RAW], state[:, :3], state[:, GRIPPER_INDEX_RAW]
        )
        analysis["episode_index"] = int(ep)
        per_episode.append(analysis)

    n_total = len(per_episode)
    with_transition = [e for e in per_episode if e["has_transition"]]
    with_confirm = [e for e in with_transition if e["confirm_ix"] is not None]

    return {
        "task": task,
        "n_episodes_sampled": n_total,
        "n_with_open_close_transition": len(with_transition),
        "coverage_frac": len(with_transition) / n_total if n_total else None,
        "n_with_state_confirm": len(with_confirm),
        "confirm_rate_given_transition": len(with_confirm) / len(with_transition) if with_transition else None,
        "transition_frac_of_episode": summarize([e["transition_frac_of_episode"] for e in with_transition]),
        "confirm_lag_steps": summarize([e["confirm_lag_steps"] for e in with_confirm]),
        "reach_dist_m": summarize([e["reach_dist_m"] for e in with_transition]),
        "commanded_vs_actual_dist_m": summarize([e["commanded_vs_actual_dist_m"] for e in with_confirm]),
        "per_episode": per_episode,
    }


def main() -> None:
    rng = random.Random(SEED)
    episode_task_map = build_episode_task_map(REPO_ID)
    task_to_episodes: dict[str, list[int]] = {t: [] for t in STAGE2_TASK_NAMES}
    for ep, t in episode_task_map.items():
        if t in task_to_episodes:
            task_to_episodes[t].append(ep)

    results = {}
    for task in STAGE2_TASK_NAMES:
        all_eps = sorted(task_to_episodes[task])
        sample = sorted(rng.sample(all_eps, min(N_EPISODES_PER_TASK, len(all_eps))))
        logger.info("=== %s: %d episodes available, sampling %d ===", task, len(all_eps), len(sample))
        task_result = run_task(task, sample)
        results[task] = task_result
        cva = task_result["commanded_vs_actual_dist_m"]
        cov = task_result["coverage_frac"]
        reach = task_result["reach_dist_m"]
        logger.info(
            "[%s] coverage=%.2f  commanded_vs_actual_dist(cm): mean=%.2f median=%.2f p90=%.2f (n=%d)  "
            "reach_dist(cm): mean=%.1f median=%.1f",
            task,
            cov if cov is not None else float("nan"),
            cva.get("mean", float("nan")) * 100,
            cva.get("median", float("nan")) * 100,
            cva.get("p90", float("nan")) * 100,
            cva.get("n", 0),
            reach.get("mean", float("nan")) * 100,
            reach.get("median", float("nan")) * 100,
        )

    out_path = Path("outputs/eval/safediff_vla_local_grasp_target_feasibility_audit/summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("=== wrote %s ===", out_path)


if __name__ == "__main__":
    main()
