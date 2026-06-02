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
"""Emit the ``--dataset.episodes`` include-list for a LeRobot dataset, minus a
set of excluded episode indices.

``LeRobotDatasetConfig.episodes`` is an *include* list (train only on the listed
episodes), so "exclude episode X" means "pass every episode except X". This
helper builds that complement.

For ``pepijn223/robocasa_pretrain_human300_v4`` the default exclusion set is the
63 episodes that carry NO ``subtask`` annotation (in fact no persistent language
rows at all) — see the scan in this PR's discussion. Training the steerable
SmolVLA/pi052 policy on those episodes would feed it frames with empty subtask
targets, so we drop them.

Usage (prints a compact ``[0,1,2,...]`` list to stdout, logs to stderr):

    python scripts/build_episode_filter.py \
        --repo-id pepijn223/robocasa_pretrain_human300_v4

    # capture in a shell script
    EPISODES=$(python scripts/build_episode_filter.py --repo-id <id>)
    lerobot-train ... --dataset.episodes="$EPISODES"

The helper reads ``meta/info.json`` from the Hub to learn ``total_episodes`` and
validates that every excluded index is in ``[0, total_episodes)`` before emitting
the complement. Pass ``--no-validate-hub`` to skip the network round-trip and use
``--total-episodes`` directly (e.g. for an offline / local dataset).
"""

from __future__ import annotations

import argparse
import json
import sys

# Episodes in pepijn223/robocasa_pretrain_human300_v4 with no `subtask`
# annotation (no persistent language rows at all). 63 episodes / 179,009 frames.
DEFAULT_EXCLUDE: tuple[int, ...] = (
    1065, 2972, 6971, 8129, 9167, 9170, 9171, 9177, 9190, 9196, 9199, 9204,
    9207, 9208, 9210, 9217, 9232, 9234, 9240, 9243, 9254, 9256, 9258, 9259,
    9261, 9263, 9264, 15928, 16350, 18729, 20026, 21703, 25314, 25319, 25321,
    25324, 25333, 25340, 25356, 25366, 25374, 25388, 25392, 25825, 25893,
    26347, 26357, 26374, 26375, 26388, 26394, 26398, 26400, 26409, 26422,
    26423, 26426, 26895, 26905, 26915, 26954, 27064, 30812,
)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _total_episodes_from_hub(repo_id: str, revision: str | None) -> int:
    """Return ``total_episodes`` from the dataset's ``meta/info.json`` on the Hub."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo_id,
        filename="meta/info.json",
        repo_type="dataset",
        revision=revision,
    )
    with open(path) as f:
        info = json.load(f)
    total = int(info["total_episodes"])
    if total <= 0:
        raise ValueError(f"info.json reports non-positive total_episodes={total!r}")
    return total


def build_include_list(total_episodes: int, exclude: set[int]) -> list[int]:
    """Return ``[0, total_episodes)`` with ``exclude`` removed, ascending."""
    out_of_range = sorted(e for e in exclude if e < 0 or e >= total_episodes)
    if out_of_range:
        raise ValueError(
            f"{len(out_of_range)} excluded index(es) outside [0, {total_episodes}): "
            f"{out_of_range[:10]}{'...' if len(out_of_range) > 10 else ''}. "
            "The dataset may have changed — re-run the subtask scan before training."
        )
    return [e for e in range(total_episodes) if e not in exclude]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default="pepijn223/robocasa_pretrain_human300_v4")
    p.add_argument("--revision", default=None, help="Dataset revision/branch (default: main).")
    p.add_argument(
        "--exclude-file",
        default=None,
        help="Optional JSON file with a list of episode indices to exclude. "
        "Overrides the built-in default set.",
    )
    p.add_argument(
        "--total-episodes",
        type=int,
        default=None,
        help="Total episode count. If omitted, read from meta/info.json on the Hub.",
    )
    p.add_argument(
        "--no-validate-hub",
        action="store_true",
        help="Do not fetch info.json from the Hub; requires --total-episodes.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Write the list to this file instead of stdout.",
    )
    args = p.parse_args()

    if args.exclude_file:
        with open(args.exclude_file) as f:
            data = json.load(f)
        # Accept either a bare list or the {"missing_episode_indices": [...]} report shape.
        exclude = set(data["missing_episode_indices"] if isinstance(data, dict) else data)
    else:
        exclude = set(DEFAULT_EXCLUDE)

    if args.total_episodes is not None:
        total = args.total_episodes
        if not args.no_validate_hub:
            hub_total = _total_episodes_from_hub(args.repo_id, args.revision)
            if hub_total != total:
                raise ValueError(
                    f"--total-episodes={total} disagrees with Hub info.json total_episodes={hub_total}."
                )
    else:
        if args.no_validate_hub:
            raise SystemExit("--no-validate-hub requires --total-episodes.")
        total = _total_episodes_from_hub(args.repo_id, args.revision)

    include = build_include_list(total, exclude)
    _log(
        f"[build_episode_filter] repo={args.repo_id} total={total} "
        f"excluded={len(exclude)} kept={len(include)}"
    )

    # Compact JSON (no spaces) so the resulting CLI arg stays as short as possible.
    payload = "[" + ",".join(map(str, include)) + "]"
    if args.out:
        with open(args.out, "w") as f:
            f.write(payload)
        _log(f"[build_episode_filter] wrote {len(payload)} bytes to {args.out}")
    else:
        sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
