# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Streaming local video quality measurement with explicit sample coverage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2

from lerobot.utils.import_utils import _pyarrow_available, require_package

from ..errors import UndecodableInputError
from ..types import ItemResult, ModuleSpec, Outcome, Resources, WorkItem
from ..worker import WorkerContext

if TYPE_CHECKING or _pyarrow_available:
    import pyarrow as pa


class VideoQuality:
    """Decode sequentially in bounded memory; emit metrics, not implicit filtering."""

    def __init__(self, sample_every: int = 30, black_threshold: float = 5):
        require_package("pyarrow", "dataset")
        if sample_every < 1 or not 0 <= black_threshold <= 255:
            raise ValueError("Invalid sampling interval or black threshold")
        self.sample_every, self.black_threshold = sample_every, black_threshold
        self.schema = pa.schema(
            [
                ("item_id", pa.string()),
                ("decoded_frames", pa.int64()),
                ("sampled_frames", pa.int64()),
                ("black_frames", pa.int64()),
            ]
        )
        self.spec = ModuleSpec("video_quality", "1", "camera_stream", {"quality": self.schema}, Resources())

    def setup(self, context: WorkerContext) -> None:
        pass

    def teardown(self) -> None:
        pass

    def process_batch(self, items: list[WorkItem], context: WorkerContext) -> list[ItemResult]:
        results = []
        for item in items:
            capture = cv2.VideoCapture(item.payload["video_path"])
            count = sampled = black = 0
            try:
                if not capture.isOpened():
                    raise UndecodableInputError(f"Cannot decode video: {item.key}")
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if count % self.sample_every == 0:
                        sampled += 1
                        black += int(float(frame.mean()) <= self.black_threshold)
                    count += 1
                expected = item.payload.get("expected_frames")
                if not count or (expected is not None and count != expected):
                    raise UndecodableInputError(
                        f"Incomplete video {item.key}: decoded {count}, expected {expected}"
                    )
            finally:
                capture.release()
            table = pa.Table.from_pylist(
                [
                    {
                        "item_id": item.item_id,
                        "decoded_frames": count,
                        "sampled_frames": sampled,
                        "black_frames": black,
                    }
                ],
                schema=self.schema,
            )
            results.append(
                ItemResult(item.item_id, Outcome.COMPLETED, (context.write_parquet(item, "quality", table),))
            )
        return results
