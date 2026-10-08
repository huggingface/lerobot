# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Normalize an already-LeRobot input through the same episode conversion recipe."""

import json
from pathlib import Path

from lerobot.utils.constants import DEFAULT_FEATURES

from ..artifacts import file_checksum
from ..types import DatasetRef, InputItem, fingerprint


class LeRobotSource:
    def __init__(self, root, repo_id="processing/input"):
        self.root = Path(root).resolve()
        self.repo_id = repo_id

    def open(self):
        self.info = json.loads((self.root / "meta/info.json").read_text())
        if not self.info["codebase_version"].startswith("v3."):
            raise ValueError("Run the official format-upgrade recipe for pre-v3 datasets before this reader")
        self.fps = self.info["fps"]
        self.robot_type = self.info.get("robot_type")
        self.camera_keys = sorted(
            key for key, value in self.info["features"].items() if value["dtype"] in {"image", "video"}
        )
        for key in self.camera_keys:
            if (self.info["features"][key].get("info") or {}).get("is_depth_map"):
                raise ValueError("Depth needs a depth-preserving source recipe, not RGB resizing")
        self.features = {
            key: value
            for key, value in self.info["features"].items()
            if key not in {*DEFAULT_FEATURES, *self.camera_keys}
        }
        self.metadata_files = {}
        if (self.root / "meta/subtasks.parquet").exists():
            self.metadata_files["meta/subtasks.parquet"] = self.root / "meta/subtasks.parquet"

    def acquire(self):
        import pyarrow.parquet as pq

        self.open()
        native = []
        self.episodes = []
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and not any(part.startswith(".") for part in path.relative_to(self.root).parts):
                native.append({"path": str(path.relative_to(self.root)), "checksum": file_checksum(path)})
        self.dataset_ref = DatasetRef(str(self.root), fingerprint(native))
        for path in sorted((self.root / "meta/episodes").glob("chunk-*/file-*.parquet")):
            table = pq.read_table(path, columns=["episode_index", "dataset_from_index", "dataset_to_index"])
            self.episodes.extend(table.to_pylist())
        self.episodes.sort(key=lambda ep: ep["episode_index"])
        if not self.episodes or len({ep["episode_index"] for ep in self.episodes}) != len(self.episodes):
            raise ValueError("Missing or duplicated episode metadata")

    def discover(self, stage, store, upstream):
        for episode in self.episodes:
            yield InputItem(
                str(episode["episode_index"]),
                episode,
                episode["dataset_to_index"] - episode["dataset_from_index"],
                physical_seconds=(episode["dataset_to_index"] - episode["dataset_from_index"]) / self.fps,
                camera_seconds=(episode["dataset_to_index"] - episode["dataset_from_index"])
                / self.fps
                * len(self.camera_keys),
            )

    def frames(self, payload):
        import numpy as np

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        # One standard reader per persistent worker; it handles packed video
        # offsets and canonical language features instead of reinventing them.
        if not hasattr(self, "dataset"):
            self.dataset = LeRobotDataset(
                self.repo_id, root=self.root, video_backend="pyav", return_uint8=True
            )
        for index in range(payload["dataset_from_index"], payload["dataset_to_index"]):
            sample = self.dataset[index]
            if int(sample["episode_index"]) != payload["episode_index"]:
                raise ValueError("Episode metadata no longer matches frame data")
            frame = {}
            for key in [*self.features, *self.camera_keys]:
                value = sample[key]
                if hasattr(value, "numpy"):
                    value = value.numpy()
                if key in self.camera_keys:
                    value = np.transpose(value, (1, 2, 0))
                elif isinstance(value, np.ndarray):
                    value = value.reshape(self.features[key]["shape"])
                frame[key] = value
            yield {**frame, "task": sample["task"]}
