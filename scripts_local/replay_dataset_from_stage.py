"""Replay a recorded sim_assembling episode starting from an arbitrary stage / frame.

Requires the dataset to contain `observation.sim_state.{qpos,qvel,act}` columns
(landed in lerobot-214; see env.py + sim_assembling.py).

Modes:
  - replay-actions (default): set mujoco state at `start_frame`, then push the
    recorded actions from that frame onward through the env.
  - policy: set state, then roll forward with a loaded pretrained policy
    (applies make_pre_post_processors per feedback_eval_preprocessor_critical).

Example:
  uv run --no-sync python scripts_local/replay_dataset_from_stage.py \\
    --dataset_repo_id=domrachev03/sim_3stage_v5_smoke \\
    --episode=0 --start_stage=4 --video_out=/tmp/replay_ep0_stage4.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

import lerobot.envs.sim_assembling  # noqa: F401 — registers gym id
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _resolve_start_frame(ep_meta: dict, start_stage, start_frame: int | None) -> int:
    if start_frame is not None:
        return int(start_frame)
    if start_stage is None:
        return 0
    names = ep_meta.get("sparse_subtask_names") or ep_meta.get("dense_subtask_names")
    starts = ep_meta.get("sparse_subtask_start_frames") or ep_meta.get("dense_subtask_start_frames")
    if names is None or starts is None:
        raise ValueError("Episode meta missing subtask frame ranges; pass --start_frame instead.")
    if isinstance(start_stage, str):
        # Try int first (CLI passes "4" as str), then fall back to stage name.
        try:
            idx = int(start_stage)
        except ValueError:
            try:
                idx = names.index(start_stage)
            except ValueError as e:
                raise ValueError(f"Stage '{start_stage}' not in {names}") from e
    else:
        idx = int(start_stage)
    if idx < 0 or idx >= len(starts):
        raise IndexError(f"Stage idx {idx} out of range [0, {len(starts) - 1}]")
    return int(starts[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_repo_id", required=True)
    ap.add_argument("--root", default=None, help="local dataset root (skip HF hub lookup)")
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--start_stage", default=None, help="int stage idx or stage name")
    ap.add_argument("--start_frame", type=int, default=None, help="override stage → use this frame")
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--video_out", required=True)
    ap.add_argument("--cam", default="cam_front", choices=["cam_front", "cam_side", "cam_gripper"])
    ap.add_argument("--policy", default=None, help="optional pretrained_model dir for policy rollout")
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    ds = LeRobotDataset(args.dataset_repo_id, root=args.root) if args.root else LeRobotDataset(args.dataset_repo_id)
    if args.episode < 0 or args.episode >= ds.num_episodes:
        raise IndexError(f"episode {args.episode} out of range [0, {ds.num_episodes - 1}]")

    ep_meta = ds.meta.episodes[args.episode]
    ep_len = int(ep_meta["length"])
    start_frame = _resolve_start_frame(ep_meta, args.start_stage, args.start_frame)
    if start_frame >= ep_len:
        raise IndexError(f"start_frame {start_frame} >= ep_len {ep_len}")

    df = ds.hf_dataset.to_pandas()
    ep_df = df[df.episode_index == args.episode].reset_index(drop=True)
    needed = {"observation.sim_state.qpos", "observation.sim_state.qvel"}
    missing = needed - set(ep_df.columns)
    if missing:
        raise KeyError(f"Dataset {args.dataset_repo_id} missing required columns: {missing}. Re-record with lerobot-214 env mod.")

    qpos0 = np.asarray(ep_df["observation.sim_state.qpos"].iloc[start_frame], dtype=np.float64)
    qvel0 = np.asarray(ep_df["observation.sim_state.qvel"].iloc[start_frame], dtype=np.float64)
    # `act` is optional — only present when the mujoco model has stateful actuators (na > 0).
    if "observation.sim_state.act" in ep_df.columns:
        act0 = np.asarray(ep_df["observation.sim_state.act"].iloc[start_frame], dtype=np.float64)
    else:
        act0 = np.zeros(0, dtype=np.float64)

    env = gym.make("sim_assembling/AssembleBase-v0", render_mode="rgb_array")
    env.reset()
    inner = env.unwrapped
    while not hasattr(inner, "data") or not hasattr(inner, "model"):
        inner = inner.env  # walk wrapper chain
    if qpos0.shape[0] != inner.model.nq:
        raise ValueError(f"qpos dim mismatch: recorded {qpos0.shape[0]} vs env nq={inner.model.nq}")

    inner.data.qpos[:] = qpos0
    inner.data.qvel[:] = qvel0
    if act0.size > 0 and inner.model.na > 0:
        inner.data.act[:] = act0
    mujoco.mj_forward(inner.model, inner.data)

    # Resync AssemblingHILAdapter internal state to match restored sim state.
    # Without this, actions (which are EE deltas relative to `_ee_ref_pos`) and
    # gripper "stay" commands operate against the RESET-time reference instead
    # of the start_frame reference, causing the arm to fly back + gripper to
    # remain open. Walk wrapper chain to find the adapter.
    adapter = env
    while adapter is not None and not hasattr(adapter, "_ee_ref_pos"):
        adapter = getattr(adapter, "env", None)
    if adapter is None:
        raise RuntimeError("Could not find AssemblingHILAdapter in wrapper chain.")
    obs_now = inner._get_obs()
    adapter._ee_ref_pos = np.asarray(obs_now["state"]["ee_pos"], dtype=np.float64).copy()
    adapter._ee_ref_quat = np.asarray(obs_now["state"]["ee_quat"], dtype=np.float64).copy()
    # `_gripper_cmd` is 1=closed / 0=open (sim_assembling.py convention), while
    # obs `gripper_width` is 0=closed / 1=open (env.py convention) — invert.
    # This is the LAST commanded width; we approximate with current physical
    # width so subsequent "stay" actions preserve the restored grip state.
    adapter._gripper_cmd = 1.0 - float(obs_now["state"].get("gripper_width", 0.0))

    if args.policy is not None:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_policy, make_pre_post_processors
        cfg = PreTrainedConfig.from_pretrained(args.policy)
        policy = make_policy(cfg, pretrained_path=args.policy)
        policy.eval()
        pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=args.policy)
    else:
        policy = pre = post = None
        recorded_actions = np.stack([np.asarray(a, dtype=np.float32) for a in ep_df["action"].iloc[start_frame:].tolist()])

    frames = []
    img = env.render() if env.render_mode == "rgb_array" else inner.render()
    if isinstance(img, dict):
        frames.append(np.asarray(img[args.cam], dtype=np.uint8))
    else:
        frames.append(np.asarray(img, dtype=np.uint8))

    n_steps = min(args.max_steps, ep_len - start_frame)
    for t in range(n_steps):
        if policy is None:
            action = recorded_actions[t]
        else:
            # Rebuild obs dict via env's _get_obs path so processors see fresh state.
            obs_raw = inner._get_obs() if hasattr(inner, "_get_obs") else None
            # Walk back up to the adapter to get the flat obs.
            wrapper = env
            while wrapper is not env.unwrapped:
                if hasattr(wrapper, "_adapt_obs"):
                    obs_flat = wrapper._adapt_obs(obs_raw)
                    break
                wrapper = wrapper.env
            batch = pre({k: v for k, v in obs_flat.items() if isinstance(v, (np.ndarray, torch.Tensor))})
            with torch.no_grad():
                action = policy.select_action(batch)
            action = post(action).cpu().numpy().squeeze()

        _, _, terminated, truncated, _ = env.step(action)
        img = inner.render() if hasattr(inner, "render") else env.render()
        if isinstance(img, dict):
            frames.append(np.asarray(img[args.cam], dtype=np.uint8))
        else:
            frames.append(np.asarray(img, dtype=np.uint8))
        if terminated or truncated:
            break

    Path(args.video_out).parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(args.video_out, fps=args.fps) as w:
        for f in frames:
            w.append_data(f)
    print(f"Wrote {len(frames)} frames to {args.video_out} (start_frame={start_frame}, n_steps={len(frames) - 1})")


if __name__ == "__main__":
    main()
