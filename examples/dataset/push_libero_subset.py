#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Build a task-filtered subset of a LIBERO-format LeRobotDataset and push it to the Hub.

`lerobot/libero` bundles all 4 LIBERO suites (40 tasks) into one dataset with no
per-episode task column in its metadata -- only each frame's `task_index` says which
task it belongs to. This script resolves a task selection to episode indices, builds a
physically reduced copy with `split_dataset` (re-indexed episodes/frames/videos, so it's
a real standalone dataset, not just a filtered view), and optionally pushes it to the Hub
so it renders in the Hub's dataset viewer.

Task selection (pick exactly one):
    --task-text "<exact task language string>"
    --suite-task <suite> <task_id>   (resolved via the `libero` benchmark, requires lerobot[libero])
    --episodes 0,4,12                (explicit episode indices, bypasses task resolution)

Usage:
    # Preview which episodes libero_object task 0 resolves to, without building anything.
    python examples/dataset/push_libero_subset.py \
        --new-repo-id ${HF_USER}/libero_object_task0 \
        --suite-task libero_object 0 \
        --dry-run

    # Build the subset locally (no upload) so you can inspect it first.
    python examples/dataset/push_libero_subset.py \
        --new-repo-id ${HF_USER}/libero_object_task0 \
        --suite-task libero_object 0

    # Build and push as a public dataset.
    python examples/dataset/push_libero_subset.py \
        --new-repo-id ${HF_USER}/libero_object_task0 \
        --suite-task libero_object 0 \
        --push

    # Select by exact task text instead, and push privately.
    python examples/dataset/push_libero_subset.py \
        --new-repo-id ${HF_USER}/libero_stack_cups \
        --task-text "stack the two cups and put them in the basket" \
        --push --private

    # Select explicit episode indices, e.g. from your own bookkeeping.
    python examples/dataset/push_libero_subset.py \
        --new-repo-id ${HF_USER}/libero_custom_subset \
        --episodes 813,818,825,850 \
        --push
"""

import argparse

from lerobot.datasets.dataset_tools import recompute_stats, split_dataset
from lerobot.datasets.io_utils import write_info
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import require_package

# `lerobot/libero`'s published info.json gives these two vector features a single
# generic `names` entry each (["actions"] / ["state"]) instead of one name per
# dimension. split_dataset() copies that metadata through verbatim, so any viewer
# that pairs values with `names` positionally (rather than falling back on a
# length mismatch) only picks up 1 of the 7/8 series. Overwrite with real
# per-dimension names -- matching LIBERO's OSC-pose delta action (dx,dy,dz,
# droll,dpitch,dyaw,gripper) and eef-pose + gripper state -- before saving.
LIBERO_ACTION_NAMES = ["delta_x", "delta_y", "delta_z", "delta_roll", "delta_pitch", "delta_yaw", "gripper"]
LIBERO_STATE_NAMES = [
    "eef_x", "eef_y", "eef_z", "eef_ori_x", "eef_ori_y", "eef_ori_z", "gripper_0", "gripper_1",
]


def fix_action_state_names(dataset: LeRobotDataset) -> None:
    """Patch mismatched per-dimension `names` on the action/state features and persist to disk."""
    for key, names in ((ACTION, LIBERO_ACTION_NAMES), (OBS_STATE, LIBERO_STATE_NAMES)):
        feature = dataset.meta.info.features.get(key)
        if feature is None:
            continue
        dim = feature["shape"][-1]
        if feature.get("names") != names and len(names) == dim:
            feature["names"] = names
    write_info(dataset.meta.info, dataset.meta.root)


def resolve_suite_task_text(suite: str, task_id: int) -> str:
    """Resolve a LIBERO (suite, task_id) pair -- the same numbering used by
    `lerobot-eval --env.task_ids` -- to its language instruction string."""
    require_package("hf-libero", extra="libero", import_name="libero")
    from libero.libero import benchmark

    bench = benchmark.get_benchmark_dict()
    if suite not in bench:
        raise ValueError(f"Unknown LIBERO suite '{suite}'. Available: {', '.join(sorted(bench.keys()))}")
    suite_obj = bench[suite]()
    if not (0 <= task_id < len(suite_obj.tasks)):
        raise ValueError(
            f"task_id {task_id} out of range for suite '{suite}' (has {len(suite_obj.tasks)} tasks)."
        )
    return suite_obj.tasks[task_id].language


def episodes_for_task_text(dataset: LeRobotDataset, task_text: str) -> list[int]:
    """Find every episode whose frames are labeled with `task_text`.

    The dataset's episode metadata has no task column (only frame rows carry
    `task_index`), so this scans the frame-level `episode_index`/`task_index` columns.
    """
    task_index = dataset.meta.get_task_index(task_text)
    if task_index is None:
        raise ValueError(
            f"Task {task_text!r} not found in dataset '{dataset.repo_id}'. "
            f"Known tasks: {list(dataset.meta.tasks.index)}"
        )
    table = dataset.select_columns(["episode_index", "task_index"])
    episodes = sorted({int(ep) for ep, idx in zip(table["episode_index"], table["task_index"], strict=True) if idx == task_index})
    if not episodes:
        raise ValueError(f"No episodes found for task_index {task_index} ({task_text!r}).")
    return episodes


def parse_episodes_arg(raw: str) -> list[int]:
    return sorted({int(item) for item in raw.split(",") if item.strip()})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo-id", default="lerobot/libero", help="Source dataset repo id.")
    parser.add_argument(
        "--new-repo-id",
        required=True,
        help="Target repo id for the subset, e.g. <hf_user>/libero_object_task0.",
    )
    parser.add_argument("--root", default=None, help="Local root of the source dataset, if not the HF cache.")
    parser.add_argument(
        "--new-root", default=None, help="Local root to write the subset to, if not the HF cache."
    )

    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--task-text", help="Exact task language string to select episodes for.")
    selection.add_argument(
        "--suite-task",
        nargs=2,
        metavar=("SUITE", "TASK_ID"),
        help="LIBERO suite name and task_id, e.g. --suite-task libero_object 0.",
    )
    selection.add_argument("--episodes", help="Explicit comma-separated episode indices, e.g. 0,4,12.")

    parser.add_argument(
        "--split-name",
        default="subset",
        help="Internal name for the local copy while it's being built. Does not affect --new-repo-id.",
    )
    parser.add_argument("--private", action="store_true", help="Push as a private dataset (default: public).")
    parser.add_argument(
        "--push",
        action="store_true",
        help="Actually upload to the Hub. Without this flag the subset is only built locally.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print the selected episode list, then exit without building or pushing anything.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()

    print(f"Loading source dataset '{args.repo_id}'...")
    dataset = LeRobotDataset(args.repo_id, root=args.root)
    print(f"  {dataset.meta.total_episodes} episodes, {dataset.meta.total_frames} frames")

    if args.episodes:
        episodes = parse_episodes_arg(args.episodes)
        invalid = [e for e in episodes if not (0 <= e < dataset.meta.total_episodes)]
        if invalid:
            raise ValueError(f"Episode indices out of range for this dataset: {invalid}")
    else:
        if args.task_text:
            task_text = args.task_text
        else:
            suite, task_id = args.suite_task
            task_text = resolve_suite_task_text(suite, int(task_id))
            print(f"Resolved {suite} task_id={task_id} -> {task_text!r}")
        episodes = episodes_for_task_text(dataset, task_text)

    n_frames = sum(int(dataset.meta.episodes[e]["length"]) for e in episodes)
    print(f"Selected {len(episodes)} episodes / {n_frames} frames for the subset.")

    if args.dry_run:
        print(episodes)
        return

    print(f"Building local subset (split name '{args.split_name}')...")
    result = split_dataset(dataset, splits={args.split_name: episodes}, output_dir=args.new_root)
    subset = result[args.split_name]
    fix_action_state_names(subset)
    # `lerobot/libero`'s per-episode metadata carries no `stats/*` columns at all, so
    # split_dataset() has nothing to aggregate from and writes an empty meta/stats.json.
    # A feature missing from stats.json gets silently skipped by MEAN_STD normalization
    # at train time (no error) -- action/observation.state would train unnormalized.
    # Recompute real stats directly from the copied parquet frame data instead.
    recompute_stats(subset)
    print(
        f"Built subset at {subset.root} "
        f"({subset.meta.total_episodes} episodes, {subset.meta.total_frames} frames)."
    )

    if not args.push:
        print("Not pushing (pass --push to upload). To push later:")
        print(
            f"  LeRobotDataset({args.new_repo_id!r}, root={str(subset.root)!r})"
            f".push_to_hub(private={args.private or None})"
        )
        return

    # split_dataset() names the local copy after the source repo_id; re-load it under
    # the requested target repo_id (mirrors the pattern lerobot-edit-dataset uses) so
    # push_to_hub() creates/updates the right repo instead of `{repo_id}_{split_name}`.
    to_push = LeRobotDataset(args.new_repo_id, root=subset.root)
    print(f"Pushing to https://huggingface.co/datasets/{args.new_repo_id} (private={bool(args.private)})...")
    to_push.push_to_hub(private=args.private or None)
    print(f"Done: https://huggingface.co/datasets/{args.new_repo_id}")


if __name__ == "__main__":
    main()
