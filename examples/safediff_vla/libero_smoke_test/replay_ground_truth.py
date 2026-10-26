#!/usr/bin/env python
"""Ground-truth action replay: feed a REAL recorded demonstration's own `action` sequence
straight into `env.step()` for the same task, and record video.

This is a stronger action-semantics check than rolling out an undertrained (1500-step) policy:
it removes the policy entirely, so any success/sane-motion result directly confirms (or refutes)
our understanding of the raw 7-D action layout and scale -- since we're just replaying the
dataset's own numbers, unmodified, through the same LiberoEnv the dataset was presumably
generated against. Does not touch find_grasp_target or any GroundedGrasp architecture.

Usage:
    uv run python examples/safediff_vla/libero_smoke_test/replay_ground_truth.py \
        --dataset libero --episode-in-task 0 --output-dir outputs/libero_smoke_test/replay_libero
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from adapter import EVAL_CAMERA_NAME_MAPPING, SMOKE_TEST_SUITE, SMOKE_TEST_SUITE_TASK_ID, SMOKE_TEST_TASK_TEXT

from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.configs import LiberoPlusEnv as LiberoPlusEnvConfig
from lerobot.envs.factory import make_env

REPO_IDS = {"libero": "lerobot/libero", "libero_plus": "lerobot/libero_plus"}


def annotate_frame(image: np.ndarray, lines: list[str]) -> Image.Image:
    img = Image.fromarray(image).convert("RGB")
    canvas = Image.new("RGB", (img.width, img.height + 14 * len(lines) + 6), "black")
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    for i, line in enumerate(lines):
        draw.text((2, img.height + 2 + 14 * i), line, fill="lime")
    return canvas


def load_episode_actions(dataset: str, episode_in_task: int) -> tuple[np.ndarray, int]:
    """Fetch the `episode_in_task`-th recorded episode's own action sequence for our fixed task
    (by task text -> dataset-native task_index, then by chronological order within that task,
    matching LiberoEnv's own `init_state_id` convention)."""
    from huggingface_hub import HfFileSystem
    import pyarrow.dataset as ds
    import pyarrow.compute as pc
    import pandas as pd

    fs = HfFileSystem()
    repo_id = REPO_IDS[dataset]

    tasks_df = pd.read_parquet(fs.open(f"datasets/{repo_id}/meta/tasks.parquet"))
    task_index = int(tasks_df.loc[SMOKE_TEST_TASK_TEXT, "task_index"])

    paths = sorted(fs.glob(f"datasets/{repo_id}/data/**/*.parquet"))
    dset = ds.dataset(paths, filesystem=fs, format="parquet")
    tbl = dset.to_table(columns=["episode_index", "timestamp", "action"], filter=pc.field("task_index") == task_index)
    df = tbl.to_pandas().sort_values(["episode_index", "timestamp"])

    ep_ids = sorted(df["episode_index"].unique().tolist())
    if episode_in_task >= len(ep_ids):
        raise ValueError(f"episode_in_task={episode_in_task} out of range (task has {len(ep_ids)} episodes)")
    chosen_ep = ep_ids[episode_in_task]
    ep_df = df[df["episode_index"] == chosen_ep].sort_values("timestamp")
    actions = np.stack(ep_df["action"].values).astype(np.float32)
    print(f"[replay] task_index={task_index} chosen dataset episode_index={chosen_ep} "
          f"({episode_in_task}-th episode of this task) n_steps={len(actions)}")
    return actions, chosen_ep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("libero", "libero_plus"), required=True)
    parser.add_argument("--episode-in-task", type=int, default=0)
    parser.add_argument("--n-frames", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    actions, chosen_ep = load_episode_actions(args.dataset, args.episode_in_task)

    camera_name_mapping = EVAL_CAMERA_NAME_MAPPING[args.dataset]
    common = dict(
        task=SMOKE_TEST_SUITE,
        task_ids=[SMOKE_TEST_SUITE_TASK_ID],
        obs_type="pixels_agent_pos",
        observation_height=256,
        observation_width=256,
        init_states=True,
        hard_reset=True,
        control_mode="relative",
        camera_name_mapping=camera_name_mapping,
    )
    env_cfg = LiberoEnvConfig(fps=10, **common) if args.dataset == "libero" else LiberoPlusEnvConfig(fps=20, **common)
    envs = make_env(env_cfg, n_envs=1)
    vec_env = envs[SMOKE_TEST_SUITE][SMOKE_TEST_SUITE_TASK_ID]

    # `seed=` on reset() does NOT select which LIBERO init_state is used -- `LiberoEnv.init_state_id`
    # defaults to `episode_index` (0 for a freshly constructed single env) and is otherwise only
    # advanced by repeated reset() calls. Set it directly so this replay actually starts from
    # `episode_in_task`'s own init state, matching the demo whose actions we're about to replay
    # (first found and fixed here: three "different" episode_in_task replays were silently all
    # starting from init_state_id=0 before this fix, which explains their identical divergence).
    if hasattr(vec_env, "envs"):
        vec_env.envs[0].init_state_id = args.episode_in_task
    else:
        vec_env.set_attr("init_state_id", args.episode_in_task)
    obs, info = vec_env.reset()

    frame_idxs = sorted(set(np.linspace(0, len(actions) - 1, args.n_frames).astype(int).tolist()))
    success_flags = []
    for t, a in enumerate(actions):
        obs, reward, terminated, truncated, info = vec_env.step(a[None, :])
        is_success = bool(info.get("is_success", [False])[0]) if "is_success" in info else False
        if "final_info" in info:
            fi = info["final_info"]
            if isinstance(fi, dict) and "is_success" in fi:
                is_success = bool(fi["is_success"][0])
        success_flags.append(is_success)
        if t in frame_idxs:
            img = vec_env.envs[0].render()
            grip = a[6]
            annotated = annotate_frame(
                img,
                [
                    f"REPLAY dataset_ep={chosen_ep} t={t}/{len(actions)}",
                    f"raw_action_xyz={a[0:3].round(3).tolist()} rot={a[3:6].round(3).tolist()}",
                    f"raw_action_grip={grip:+.3f} success_so_far={any(success_flags)}",
                ],
            )
            annotated.save(out_dir / f"t{t:03d}_replay.png")
        if terminated[0] or truncated[0]:
            break

    print(f"[replay] finished at t={t}, success at any point: {any(success_flags)}")
    print(f"[replay] wrote frames to {out_dir}")


if __name__ == "__main__":
    main()
