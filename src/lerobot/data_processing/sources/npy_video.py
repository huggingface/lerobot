# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""One complete recipe: acquire/unpack, discover, and stream NPY + RGB videos.

Other MCAP/HDF5/etc. recipes implement this small interface in one source file;
the runtime, encoding, assembly and publication code are shared.
"""

import json
import tempfile
from pathlib import Path

from ..artifacts import file_checksum
from ..types import DatasetRef, InputItem, fingerprint
from .acquisition import acquire_file, unpack_tar


class ArrayVideoSource:
    def __init__(self, manifest, archive_uri=None, archive_sha256=None, header=None):
        self.manifest_path = Path(manifest).resolve()
        self.archive_uri = archive_uri
        self.archive_sha256 = archive_sha256
        self.header = header

    def acquire(self):
        if self.archive_uri:
            from filelock import FileLock

            root = self.manifest_path.parent
            with FileLock(str(root) + ".acquire.lock"):
                if not root.exists():
                    archive = acquire_file(self.archive_uri, self.archive_sha256, root.with_suffix(".tar"))
                    with tempfile.TemporaryDirectory(
                        prefix=root.name + "-unpacking-", dir=root.parent
                    ) as directory:
                        temporary = Path(directory) / "payload"
                        unpack_tar(archive, temporary)
                        temporary.rename(root)
        self.open()
        inputs = []
        for episode in self.episodes:
            import numpy as np

            checksums = {}
            for relative in [*episode["arrays"].values(), *episode.get("videos", {}).values()]:
                path = self.root / relative
                if not path.resolve().is_relative_to(self.root):
                    raise ValueError("Source reference escapes its declared root")
                checksums[relative] = file_checksum(path)
            frames = (
                len(
                    np.load(
                        self.root / next(iter(episode["arrays"].values())), mmap_mode="r", allow_pickle=False
                    )
                )
                if episode["arrays"]
                else episode.get("frames")
            )
            if not isinstance(frames, int) or frames < 1:
                raise ValueError("Video-only sources require an explicit positive episode frames count")
            inputs.append({"episode": episode, "checksums": checksums, "frames": frames})
        self.inputs = inputs
        self.dataset_ref = DatasetRef(
            str(self.root), fingerprint({"manifest": self.manifest, "inputs": inputs})
        )

    def open(self):
        """Worker open: metadata only, never rescan the whole raw source per worker."""
        self.manifest = self.header or json.loads(self.manifest_path.read_text())
        self.root = self.manifest_path.parent
        self.fps = self.manifest["fps"]
        if not isinstance(self.fps, int) or self.fps < 1:
            raise ValueError("This recipe requires a positive integer source FPS")
        self.features = self.manifest["features"]
        self.robot_type = self.manifest.get("robot_type")
        if self.header:
            self.camera_keys = self.header["camera_keys"]
            return
        episodes = self.manifest["episodes"]
        if not episodes:
            raise ValueError("An empty source cannot be converted")
        self.episodes = sorted(episodes, key=lambda ep: ep["id"])
        self.camera_keys = sorted(self.episodes[0].get("videos", {}))
        if len({ep["id"] for ep in episodes}) != len(episodes):
            raise ValueError("Duplicate original episode IDs")
        for episode in self.episodes:
            if sorted(episode.get("videos", {})) != self.camera_keys:
                raise ValueError("Missing cameras need an explicit source-specific policy")
            if set(episode["arrays"]) != set(self.features):
                raise ValueError("Native array features differ from the declared feature schema")

    def worker_config(self):
        return {
            "manifest": str(self.manifest_path),
            "header": {
                "fps": self.fps,
                "features": self.features,
                "robot_type": self.robot_type,
                "camera_keys": self.camera_keys,
            },
        }

    def discover(self, stage, store, upstream):
        for value in self.inputs:
            seconds = value["frames"] / self.fps
            yield InputItem(
                value["episode"]["id"],
                value,
                value["frames"],
                physical_seconds=seconds,
                camera_seconds=seconds * len(self.camera_keys),
            )

    def frames(self, payload):
        import av
        import numpy as np

        episode = payload["episode"]
        for relative, expected in payload["checksums"].items():
            if file_checksum(self.root / relative) != tuple(expected):
                raise ValueError("Raw source changed after planning")
        arrays = {
            key: np.load(self.root / relative, mmap_mode="r", allow_pickle=False)
            for key, relative in episode["arrays"].items()
        }
        lengths = {len(array) for array in arrays.values()} if arrays else {payload["frames"]}
        if len(lengths) != 1 or next(iter(lengths)) < 1:
            raise ValueError("Native feature lengths are empty or inconsistent")
        count = next(iter(lengths))
        containers = {}
        try:
            for key, relative in episode.get("videos", {}).items():
                container = av.open(self.root / relative)
                containers[key] = container
                stream = container.streams.video[0]
                stream.thread_count = 1
                if stream.average_rate is None or abs(float(stream.average_rate) - self.fps) > 1e-6:
                    raise ValueError(
                        "Camera FPS differs from native data; implement explicit clock alignment in the recipe"
                    )
            decoders = {key: container.decode(video=0) for key, container in containers.items()}
            for index in range(count):
                frame = {key: np.asarray(array[index]).copy() for key, array in arrays.items()}
                for key, decoder in decoders.items():
                    video_frame = next(decoder, None)
                    if (
                        video_frame is None
                        or video_frame.time is None
                        or abs(video_frame.time - index / self.fps) > 0.5 / self.fps
                    ):
                        raise ValueError("Incomplete or asynchronous video clock")
                    frame[key] = video_frame.to_ndarray(format="rgb24")
                yield {**frame, "task": episode["task"]}
            if any(next(decoder, None) is not None for decoder in decoders.values()):
                raise ValueError("Video contains more frames than native data")
        finally:
            for container in containers.values():
                container.close()
