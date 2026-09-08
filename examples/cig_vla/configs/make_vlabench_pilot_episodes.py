#!/usr/bin/env python
"""Build a stratified episode subset of lerobot/vlabench_unified for a cheap pilot CIG-VLA
training run, covering every one of its 295 tasks (not the 4 primitive categories we've
eval-ed VLABench on so far -- see this repo's other VLABench discussion/commits).

Per task: episodes = max(1, round(fraction * n_episodes_for_that_task)), sampled with a
fixed seed so the subset is reproducible. The per-task episode count is heavily skewed (295
tasks, 1 to 2498 episodes each, median 8) -- 100 of the 295 tasks have fewer than 5 episodes
total, so a flat `fraction` cut could drop a task to zero. The `max(1, ...)` floor guarantees
every task keeps at least one episode in the pilot subset.

Usage:
    uv run python examples/cig_vla/configs/make_vlabench_pilot_episodes.py
    uv run python examples/cig_vla/configs/make_vlabench_pilot_episodes.py --fraction 0.5 --out examples/cig_vla/configs/vlabench_pilot50_episodes.json

Then feed the "episodes" list to lerobot-train, e.g.:
    EPISODES=$(python3 -c "import json; print(json.load(open('examples/cig_vla/configs/vlabench_pilot30_episodes.json'))['episodes'])")
    uv run lerobot-train ... --dataset.episodes="$EPISODES"

Full-dataset training (no filter) is just the same command with --dataset.episodes omitted.
"""

import argparse
import json
import random

import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "lerobot/vlabench_unified"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fraction", type=float, default=0.3, help="Fraction of each task's episodes to keep.")
    parser.add_argument("--seed", type=int, default=1000, help="RNG seed for the per-task sample (reproducible).")
    parser.add_argument(
        "--out",
        type=str,
        default="examples/cig_vla/configs/vlabench_pilot30_episodes.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    ep_path = hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset", filename="meta/episodes/chunk-000/file-000.parquet"
    )
    ep = pd.read_parquet(ep_path, columns=["episode_index", "tasks", "length"])
    ep["task_str"] = ep["tasks"].apply(lambda x: x[0] if len(x) else "")

    rng = random.Random(args.seed)
    selected: list[int] = []
    for _task, group in ep.groupby("task_str", sort=True):
        indices = group["episode_index"].tolist()
        n = len(indices)
        k = max(1, round(args.fraction * n))
        chosen = sorted(rng.sample(indices, k)) if k < n else sorted(indices)
        selected.extend(chosen)
    selected.sort()

    total_frames = int(ep.set_index("episode_index").loc[selected, "length"].sum())
    full_frames = int(ep["length"].sum())
    num_tasks = ep["task_str"].nunique()

    print(f"dataset: {REPO_ID}")
    print(f"tasks:    {num_tasks} (all represented, min 1 episode each)")
    print(f"episodes: {len(selected)} / {len(ep)} ({100 * len(selected) / len(ep):.1f}%)")
    print(f"frames:   {total_frames} / {full_frames} ({100 * total_frames / full_frames:.1f}%)")

    out = {
        "repo_id": REPO_ID,
        "seed": args.seed,
        "fraction": args.fraction,
        "num_tasks": int(num_tasks),
        "num_episodes": len(selected),
        "num_frames": total_frames,
        "episodes": selected,
    }
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
