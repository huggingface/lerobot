#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Resolve LeRobot v3 episode/segment prompts without copying concatenated video files.

Use --pairs for same-dataset target/prompt rows, --pools for compact episode groups, or
--episode-index to prepare one recorded physical demonstration for rollout. Encode the output
with encode_human_videos.py to write pairs.jsonl or pools.jsonl and a prompts.jsonl index.
"""

import argparse
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

from lerobot.datasets.io_utils import load_episodes
from lerobot.streaming.location import LocationKind, StorageLocation


def episode_video_spec(
    root: Path, info: dict, episode: dict, camera: str, start: int, stop: int, data_root: str | None = None
) -> dict:
    """Map a half-open episode frame range to presentation times in its v3 video container."""
    fps = float(info["fps"])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Dataset fps must be positive and finite")
    if not 0 <= start < stop <= int(episode["length"]):
        raise ValueError(f"Invalid prompt frame interval [{start}, {stop})")
    prefix = f"videos/{camera}"
    offset = float(episode[f"{prefix}/from_timestamp"])
    if not math.isfinite(offset) or offset < 0:
        raise ValueError("Invalid episode video offset")
    relative = info["video_path"].format(
        video_key=camera,
        chunk_index=int(episode[f"{prefix}/chunk_index"]),
        file_index=int(episode[f"{prefix}/file_index"]),
    )
    if data_root is None:
        video = str((root / relative).resolve())
    else:
        location = StorageLocation.parse(data_root).pinned()
        if location.kind is not LocationKind.HF_DATASET:
            raise ValueError("streaming-data-root must be an hf://datasets URL")
        video = f"{location.uri}/{relative}"
    return {
        "prompt_video": video,
        "prompt_start_s": offset + start / fps,
        "prompt_end_s": offset + stop / fps,
        "prompt_episode_index": int(episode["episode_index"]),
        "prompt_start_frame": start,
        "prompt_end_frame": stop,
        "prompt_camera": camera,
        "source_fps": fps,
    }


def pool_member_rows(source: TextIO, dataset_id: str | None, split: str) -> Iterator[dict]:
    """Flatten compact pools for video encoding without creating episode-pair combinations."""
    for line in source:
        if not line.strip():
            continue
        pool = json.loads(line)
        if not dataset_id:
            raise ValueError("--dataset-id is required for prompt pools")
        if pool["dataset"] != dataset_id or pool["split"] != split:
            continue
        if len({int(m["episode_index"]) for m in pool["members"]}) < 2:
            raise ValueError("A prompt pool requires at least two distinct episodes")
        for member in pool["members"]:
            yield {
                "pool_id": pool["pool_id"],
                "dataset": pool["dataset"],
                "split": pool["split"],
                "prompt_episode": member["episode_index"],
                "prompt_from_frame": member["start_frame"],
                "prompt_to_frame": member["end_frame"],
                "detail_raw": pool["text"],
                "prompt_source_revision": pool.get("revision"),
            }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--streaming-data-root", help="Pinned hf://datasets payload root; metadata stays local"
    )
    parser.add_argument("--camera", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pairs", type=Path)
    source.add_argument("--pools", type=Path)
    source.add_argument("--episode-index", type=int)
    parser.add_argument("--dataset-id", help="Filter a multi-dataset candidate manifest by this dataset ID")
    parser.add_argument("--split", default="train")
    parser.add_argument("--prompt-text", help="Instruction for a standalone recorded demonstration")
    parser.add_argument("--max-video-frames", type=int, default=81)
    parser.add_argument(
        "--metadata-only", action="store_true", help="Resolve paths even if videos are not local"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.streaming_data_root:
        args.streaming_data_root = StorageLocation.parse(args.streaming_data_root).pinned().uri
    if args.max_video_frames < 1:
        parser.error("--max-video-frames must be positive")
    info = json.loads((args.dataset_root / "meta/info.json").read_text())
    if not str(info["codebase_version"]).startswith("v3.") or not info.get("video_path"):
        raise ValueError("This preparation tool requires a LeRobot v3 video dataset")
    episodes = {int(ep["episode_index"]): ep for ep in load_episodes(args.dataset_root)}
    missing = set()
    written = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # A temporary file prevents a rejected input from leaving a partial training manifest.
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    try:
        with temporary.open("w") as output:
            if args.episode_index is not None:
                ep = episodes[args.episode_index]
                rows = [
                    {
                        "prompt_episode": args.episode_index,
                        "prompt_from_frame": 0,
                        "prompt_to_frame": int(ep["length"]),
                        "detail_raw": args.prompt_text if args.prompt_text is not None else ep["tasks"][0],
                    }
                ]
                source_file = None
            else:
                source_file = (args.pools or args.pairs).open()
                rows = (
                    pool_member_rows(source_file, args.dataset_id, args.split)
                    if args.pools
                    else (json.loads(line) for line in source_file if line.strip())
                )
            try:
                for pair in rows:
                    if args.pairs is not None:
                        if "dataset" in pair:
                            if not args.dataset_id:
                                raise ValueError("--dataset-id is required for a multi-dataset manifest")
                            if pair["dataset"] != args.dataset_id:
                                continue
                        if pair.get("split") != args.split:
                            continue
                        if int(pair["target_episode"]) == int(pair["prompt_episode"]):
                            raise ValueError("Training prompts must come from another episode")
                        target = episodes[int(pair["target_episode"])]
                        start, stop = int(pair["target_from_frame"]), int(pair["target_to_frame"])
                        if not 0 <= start < stop <= int(target["length"]):
                            raise ValueError("Invalid target frame interval")
                    ep = episodes[int(pair["prompt_episode"])]
                    video = episode_video_spec(
                        args.dataset_root,
                        info,
                        ep,
                        args.camera,
                        int(pair["prompt_from_frame"]),
                        int(pair["prompt_to_frame"]),
                        args.streaming_data_root,
                    )
                    if args.streaming_data_root and pair.get("prompt_source_revision") not in (
                        None,
                        StorageLocation.parse(args.streaming_data_root).pinned_commit,
                    ):
                        raise ValueError("Prompt pool revision does not match streaming-data-root")
                    if not args.streaming_data_root and not Path(video["prompt_video"]).is_file():
                        missing.add(video["prompt_video"])
                        if not args.metadata_only:
                            raise FileNotFoundError(video["prompt_video"])
                    result = {
                        **pair,
                        **video,
                        "prompt_text": pair["detail_raw"],
                        "prompt_max_frames": args.max_video_frames,
                    }
                    if args.pairs is not None:
                        result.update(
                            episode_index=int(pair["target_episode"]),
                            start_frame=int(pair["target_from_frame"]),
                            end_frame=int(pair["target_to_frame"]),
                        )
                    output.write(json.dumps(result) + "\n")
                    written += 1
            finally:
                if source_file is not None:
                    source_file.close()
        if not written:
            raise ValueError("No matching prompts. Check --dataset-id and --split")
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Prepared {written} prompts; {len(missing)} video files are not local. Output: {args.output}")


if __name__ == "__main__":
    main()
