# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Temporal reads must not materialize unrelated episode columns."""

import threading
from pathlib import Path
from typing import Any

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

import datasets

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from tests.fixtures.constants import DUMMY_REPO_ID


@pytest.mark.parametrize("image_window", [False, True])
@pytest.mark.parametrize("decode_threads", [1, 3])
@pytest.mark.parametrize("sampling_strategy", ["remaining", "round_robin"])
def test_temporal_reads_decode_only_requested_images(
    tmp_path: Path,
    lerobot_dataset_factory: Any,
    info_factory: Any,
    episodes_factory: Any,
    tasks_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
    image_window: bool,
    decode_threads: int,
    sampling_strategy: str,
) -> None:
    root = tmp_path / "dataset"
    info = info_factory(total_episodes=3, total_frames=18, total_tasks=1, use_videos=False)
    tasks = tasks_factory(total_tasks=1)
    episodes = episodes_factory(
        features=info.features, total_episodes=3, total_frames=18, tasks=tasks
    ).to_list()
    # Random multinomial lengths can contain empty episodes, making the local fixture incomplete.
    for episode, start, end in zip(episodes, (0, 3, 9), (3, 9, 18), strict=True):
        episode.update(length=end - start, dataset_from_index=start, dataset_to_index=end)
    reference = lerobot_dataset_factory(
        root=root,
        repo_id=DUMMY_REPO_ID,
        info=info,
        tasks=tasks,
        episodes_metadata=datasets.Dataset.from_list(episodes),
    )
    deltas = {
        "action": [offset / reference.fps for offset in range(16)],
        "state": [-1 / reference.fps, 0.0, 1 / reference.fps],
    }
    if image_window:
        deltas.update(
            {key: [-1 / reference.fps, 0.0, 1 / reference.fps] for key in reference.meta.camera_keys}
        )
    # Reload the independent map-style reader with identical temporal semantics.
    reference = LeRobotDataset(DUMMY_REPO_ID, root=root, delta_timestamps=deltas)
    expected = [reference[index] for index in range(len(reference))]
    stream = StreamingLeRobotDataset(
        DUMMY_REPO_ID,
        root=root,
        delta_timestamps=deltas,
        episode_pool_size=2,
        decode_threads=decode_threads,
        decoded_queue_size=4,
        sampling_strategy=sampling_strategy,
    )
    decode_image = datasets.Image.decode_example
    calls = 0
    lock = threading.Lock()

    def counted_decode(self: Any, value: Any, **kwargs: Any) -> Any:
        nonlocal calls
        with lock:
            calls += 1
        return decode_image(self, value, **kwargs)

    monkeypatch.setattr(datasets.Image, "decode_example", counted_decode)
    actual = list(stream)
    assert sorted(int(item["index"]) for item in actual) == list(range(len(reference)))
    for item in actual:
        wanted = expected[int(item["index"])]
        assert item.keys() == wanted.keys()
        for key, value in item.items():
            if isinstance(value, torch.Tensor):
                assert value.dtype == wanted[key].dtype
                assert value.shape == wanted[key].shape
                assert torch.equal(value, wanted[key]), key
            else:
                assert value == wanted[key], key
    images_per_anchor = len(reference.meta.camera_keys) * (4 if image_window else 1)
    assert calls == len(actual) * images_per_anchor
