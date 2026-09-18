# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Materialize selected corrections as separate episodes alongside seed demonstrations.

Run in a subprocess: finalized inputs are read-only and output is always a new dataset.
Only executed intervention frames are copied as corrective targets. VLA proposals
remain in the rollout event journal and never become correction labels.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.language_render import active_at

DEFAULT_KEYS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def recording_frame(item, features):
    frame = {"task": item["task"]}
    for key, feature in features.items():
        if key in DEFAULT_KEYS:
            continue
        value = item[key]
        if hasattr(value, "numpy"):
            value = value.numpy()
        if feature["dtype"] in ("video", "image"):
            value = np.asarray(value)
            if value.ndim == 3 and value.shape[0] in (1, 3):
                value = value.transpose(1, 2, 0)
            if value.dtype != np.uint8:
                value = np.clip(value * 255, 0, 255).astype(np.uint8)
        if isinstance(value, np.ndarray) and value.ndim == 0 and tuple(feature["shape"]) == (1,):
            value = value.reshape(1)
        frame[key] = value
    return frame


def build_dataset(spec):
    if not spec.get("seed", {}).get("episodes"):
        raise ValueError("Select seed training episode IDs explicitly; leave held-out episodes out")
    output_root = Path(spec["output_root"]).resolve()
    if output_root.exists():
        raise ValueError("Output must be a new dataset directory")
    seed = LeRobotDataset(**spec["seed"])
    features = {key: value for key, value in seed.features.items() if key not in DEFAULT_KEYS}
    output = LeRobotDataset.create(
        spec["output_repo_id"],
        root=output_root,
        fps=seed.fps,
        robot_type=seed.meta.robot_type,
        features=features,
        image_writer_threads=6,
    )
    try:
        previous_episode = None
        for item in seed:
            episode = int(item["episode_index"])
            if previous_episode is not None and episode != previous_episode:
                output.save_episode()
            output.add_frame(recording_frame(item, features))
            previous_episode = episode
        if output.has_pending_frames():
            output.save_episode()
        for source in spec.get("corrections", []):
            if not source.get("episodes"):
                raise ValueError("Select reviewed correction episode IDs explicitly")
            dataset = LeRobotDataset(**source)
            if dataset.fps != seed.fps or dataset.meta.robot_type != seed.meta.robot_type:
                raise ValueError("Correction robot type and fps must match the seed dataset")
            for key, feature in features.items():
                other = dataset.features.get(key)
                if (
                    other is None
                    or feature["dtype"] != other["dtype"]
                    or list(feature["shape"]) != list(other["shape"])
                    or feature.get("names") != other.get("names")
                ):
                    raise ValueError(f"Incompatible correction feature: {key}")
            previous_episode = None
            segment_frames = 0
            for item in dataset:
                episode = int(item["episode_index"])
                intervention = bool(item["intervention"])
                if (episode != previous_episode or not intervention) and output.has_pending_frames():
                    output.save_episode()
                    segment_frames = 0
                previous_episode = episode
                if not intervention:
                    continue
                frame = recording_frame(item, features)
                # Each correction span becomes its own episode, with language time rebased to that span.
                if "language_persistent" in features:
                    active = active_at(
                        float(item["timestamp"]), persistent=item["language_persistent"], style="subtask"
                    )
                    if active is None:
                        raise ValueError("Correction has no active subtask annotation")
                    frame["language_persistent"] = [
                        {
                            "role": "assistant",
                            "content": active["content"],
                            "style": "subtask",
                            "timestamp": segment_frames / seed.fps,
                            "camera": None,
                            "tool_calls": None,
                        }
                    ]
                    frame["language_events"] = []
                output.add_frame(frame)
                segment_frames += 1
            if output.has_pending_frames():
                output.save_episode()
        (output_root / "aggregation.json").write_text(json.dumps(spec, indent=2))
    finally:
        output.finalize()
    return {"root": str(output_root), "episodes": output.num_episodes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()
    print(json.dumps(build_dataset(json.loads(Path(args.spec).read_text()))))


if __name__ == "__main__":
    main()
