#!/usr/bin/env python
"""B-track Stage 1 prerequisite: per-task dataset audit for the 7 tasks
(select_fruit/toy/drink/book/ingredient/mahjong/poker) `train_grounded_grasp_v2_5_multitask_stage1_5k.py`
trains on. Pure parquet/tensor audit -- no simulator, no policy, no video decoding (same style as
`dataset_position_bias_audit.py`/`grounded_grasp_supervision_audit.py` on this branch). Answers,
per task, exactly what the B-track instructions ask to have on record BEFORE training:
  - episode count / frame count
  - open->close gripper transition count (how many real grasp events the demonstrations contain)
  - valid K=10 local-window sample count: for every FRAME (not just transition frames) in the
    task, whether `find_grasp_target`'s own `valid_mask` is True AND the transition lands within
    `local_grasp_window_k=10` steps -- i.e. `LocalGraspRefiner`'s own training eligibility
    (`local_eligible` in `modeling_safediff_vla.py`'s v2_5 branch), computed here exactly the same
    way (same imported functions, same raw-physical-unit convention, same action_horizon=50) but
    over the WHOLE task-restricted dataset rather than one random training minibatch.
  - local-loss mask coverage: valid K=10 count / frame count (dataset-wide equivalent of the
    `local_eligible_frac` metric already logged per-batch during training).

`lerobot/vlabench_unified`'s data lives in exactly 3 parquet files covering all 10977 episodes
(verified locally cached -- no download needed for this audit).

Usage:
    uv run python examples/safediff_vla/multitask_grasp_supervision_audit.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import REPO_ID, TASK_NAMES, build_episode_task_map  # noqa: E402

from lerobot.policies.safediff_vla.modeling_safediff_vla import _first_grasp_transition_idx  # noqa: E402
from lerobot.policies.safediff_vla.rotation_encoding import GRIPPER_INDEX_RAW  # noqa: E402
from lerobot.policies.safediff_vla.utils import _state_gripper_is_open, find_grasp_target  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

ACTION_HORIZON = 50  # matches every v2/v2.5 training script on this branch
GRIPPER_OPEN_THRESHOLD = 0.5  # matches `grounded_grasp_gripper_open_threshold`'s config default
LOCAL_GRASP_WINDOW_K = 10  # matches `local_grasp_window_k`'s config default
OUT_DIR = Path("outputs/eval/safediff_vla_grounded_grasp_multitask_supervision_audit")


def load_all_data_files() -> pd.DataFrame:
    """All 3 data parquet files for `lerobot/vlabench_unified`, concatenated. Only
    `episode_index`/`observation.state`/`action` are needed."""
    n_files = 3  # verified against meta/episodes: exactly 3 (data/chunk_index, data/file_index) pairs
    frames = []
    for file_index in range(n_files):
        path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=f"data/chunk-000/file-{file_index:03d}.parquet",
        )
        frames.append(pd.read_parquet(path, columns=["episode_index", "observation.state", "action"]))
    return pd.concat(frames, ignore_index=True)


def audit_episode(state: np.ndarray, action: np.ndarray) -> dict:
    """`state`/`action`: `[T, 7]` raw physical values for one episode, frame-ordered.

    Returns per-episode counts: frame count, open->close transition count (over the REAL episode
    trajectory, not a fixed horizon window), and valid-K=10 frame count (sliding a 50-step action
    window from every frame, `find_grasp_target` + `_first_grasp_transition_idx` exactly as
    `modeling_safediff_vla.py`'s v2_5 branch computes `local_eligible`).
    """
    t = state.shape[0]
    state_t = torch.from_numpy(state).float()
    action_t = torch.from_numpy(action).float()

    # ---- real open->close transition count (whole-episode gripper trace, state convention) ----
    state_open = _state_gripper_is_open(state_t[:, GRIPPER_INDEX_RAW], GRIPPER_OPEN_THRESHOLD)
    n_transitions = int((state_open[:-1] & ~state_open[1:]).sum().item())

    # ---- per-frame local_eligible, sliding a 50-step action window from every frame ----
    # Build [T, ACTION_HORIZON, 7] by padding the trajectory's tail with repeated last frame, and
    # an accompanying [T, ACTION_HORIZON] action_is_pad mask -- same semantics as
    # `DatasetReader._get_query_indices`'s end-of-episode padding.
    pad_len = ACTION_HORIZON
    action_padded = np.concatenate([action, np.repeat(action[-1:], pad_len, axis=0)], axis=0)  # [T+H, 7]
    action_padded_t = torch.from_numpy(action_padded).float()
    idx = np.arange(t)[:, None] + np.arange(ACTION_HORIZON)[None, :]  # [T, H]
    windows = action_padded_t[torch.from_numpy(idx)]  # [T, H, 7]
    is_pad = torch.from_numpy(idx >= t)  # [T, H] True past the real episode end

    gt_xyz, valid_mask = find_grasp_target(
        state_t, windows, gripper_index=GRIPPER_INDEX_RAW, gripper_open_threshold=GRIPPER_OPEN_THRESHOLD,
        action_is_pad=is_pad,
    )
    first_transition_idx = _first_grasp_transition_idx(
        state_t, windows, GRIPPER_INDEX_RAW, GRIPPER_OPEN_THRESHOLD, is_pad
    )
    local_eligible = valid_mask & (first_transition_idx <= LOCAL_GRASP_WINDOW_K)

    return {
        "n_frames": t,
        "n_open_close_transitions": n_transitions,
        "n_target_valid_frames": int(valid_mask.sum().item()),
        "n_local_eligible_k10_frames": int(local_eligible.sum().item()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("=== building episode -> task map ===")
    episode_task = build_episode_task_map()
    logger.info("matched %d episodes across %d tasks", len(episode_task), len(TASK_NAMES))

    logger.info("=== loading all 3 data parquet files (state/action only) ===")
    data = load_all_data_files()
    data = data[data["episode_index"].isin(episode_task.keys())]
    logger.info("restricted to %d frames across %d episodes", len(data), data["episode_index"].nunique())

    per_task = {t: {"n_episodes": 0, "n_frames": 0, "n_open_close_transitions": 0,
                     "n_target_valid_frames": 0, "n_local_eligible_k10_frames": 0} for t in TASK_NAMES}
    per_episode_records = []

    grouped = data.groupby("episode_index", sort=False)
    for ep_idx, group in grouped:
        task = episode_task[int(ep_idx)]
        # `groupby` preserves each episode's original row order, which is already frame-ordered
        # (verified: `frame_index` is monotonic per episode in this dataset).
        state = np.stack(group["observation.state"].to_numpy())
        action = np.stack(group["action"].to_numpy())
        stats = audit_episode(state, action)
        per_episode_records.append({"episode_index": int(ep_idx), "task": task, **stats})
        agg = per_task[task]
        agg["n_episodes"] += 1
        agg["n_frames"] += stats["n_frames"]
        agg["n_open_close_transitions"] += stats["n_open_close_transitions"]
        agg["n_target_valid_frames"] += stats["n_target_valid_frames"]
        agg["n_local_eligible_k10_frames"] += stats["n_local_eligible_k10_frames"]

    for task, agg in per_task.items():
        agg["local_loss_mask_coverage"] = (
            agg["n_local_eligible_k10_frames"] / agg["n_frames"] if agg["n_frames"] else None
        )
        agg["target_valid_coverage"] = (
            agg["n_target_valid_frames"] / agg["n_frames"] if agg["n_frames"] else None
        )
        agg["mean_open_close_transitions_per_episode"] = (
            agg["n_open_close_transitions"] / agg["n_episodes"] if agg["n_episodes"] else None
        )

    summary = {
        "repo_id": REPO_ID,
        "action_horizon": ACTION_HORIZON,
        "gripper_open_threshold": GRIPPER_OPEN_THRESHOLD,
        "local_grasp_window_k": LOCAL_GRASP_WINDOW_K,
        "per_task": per_task,
    }
    (OUT_DIR / "per_task_summary.json").write_text(json.dumps(summary, indent=2))
    (OUT_DIR / "per_episode.json").write_text(json.dumps(per_episode_records, indent=2))
    logger.info("wrote %s and %s", OUT_DIR / "per_task_summary.json", OUT_DIR / "per_episode.json")
    logger.info("=== per-task summary ===")
    for task in TASK_NAMES:
        agg = per_task[task]
        logger.info(
            "%-20s episodes=%4d frames=%6d open->close=%4d (%.2f/ep) target_valid=%6d (%.1f%%) "
            "local_eligible_k10=%6d (%.1f%% mask coverage)",
            task, agg["n_episodes"], agg["n_frames"], agg["n_open_close_transitions"],
            agg["mean_open_close_transitions_per_episode"],
            agg["n_target_valid_frames"], 100 * agg["target_valid_coverage"],
            agg["n_local_eligible_k10_frames"], 100 * agg["local_loss_mask_coverage"],
        )


if __name__ == "__main__":
    main()
