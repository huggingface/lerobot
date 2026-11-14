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

"""Select the DROID episodes of the praxis fine-tuning recipe for ``--dataset.episodes``.

The praxis recipe trains on the episodes that have a language instruction and hold a
training window plus a margin (93 frames + 0.5 s at 15 Hz: 102 frames or more). The episodes
without an instruction (about half of ``lerobot/droid_1.0.1``) are left out; the text dropout
of the objective provides the unconditional training.

    python examples/praxis/select_droid_episodes.py --repo-id lerobot/droid_1.0.1 \\
        --revision refs/pr/11 --out droid_episodes.json

The output is a JSON list for ``lerobot-train --dataset.episodes="$(cat droid_episodes.json)"``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lerobot.policies.praxis.praxis_core.train.data import MARGIN_S, window_fits

__all__ = ["select_episodes"]


def select_episodes(
    episodes: pd.DataFrame,
    window_frames: int = 93,
    fps: float = 15.0,
    max_episodes: int | None = None,
) -> list[int]:
    """Return the episode indices with an instruction that hold a window plus the margin.

    Args:
        episodes: The ``episodes`` table of ``LeRobotDatasetMetadata`` (columns ``episode_index``,
            ``tasks``, ``length``).
        window_frames: Frames of one training window.
        fps: Frame rate of the dataset.
        max_episodes: Keep only the first ``max_episodes`` selected episodes.

    Returns:
        The selected ``episode_index`` values, in order.
    """
    out: list[int] = []
    for index, tasks, length in zip(
        episodes["episode_index"], episodes["tasks"], episodes["length"], strict=True
    ):
        prompts = [str(t).strip() for t in (tasks if tasks is not None else [])]
        if not any(prompts):
            continue
        if not window_fits(int(length), window_frames, fps):
            continue
        out.append(int(index))
        if max_episodes is not None and len(out) >= max_episodes:
            break
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo-id", default="lerobot/droid_1.0.1")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--root", default=None, help="a local dataset directory instead of the Hub")
    parser.add_argument("--window-frames", type=int, default=93)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--out", default=None, help="write the JSON list here (default: stdout)")
    args = parser.parse_args()
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root, revision=args.revision)
    episodes = meta.episodes.to_pandas() if hasattr(meta.episodes, "to_pandas") else meta.episodes
    selected = select_episodes(episodes, args.window_frames, float(meta.fps), args.max_episodes)
    text = json.dumps(selected)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(
            f"{len(selected)} of {len(episodes)} episodes selected (instruction, >= {args.window_frames} frames + {MARGIN_S} s) -> {args.out}"
        )
    else:
        print(text)


if __name__ == "__main__":
    main()
