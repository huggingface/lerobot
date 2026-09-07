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

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.configs.default import DatasetConfig  # noqa: E402
from lerobot.scripts.lerobot_train import _preprocess_dataset_batch, make_dataloaders  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_IMAGES  # noqa: E402
from lerobot.utils.logging_utils import MetricsTracker  # noqa: E402


def test_preprocess_dataset_batch_normalizes_and_renames_before_processor():
    seen_batch = None

    def preprocessor(batch):
        nonlocal seen_batch
        seen_batch = batch
        return batch

    batch = {
        "image": torch.full((1, 3, 2, 2), 255, dtype=torch.uint8),
        "actions": torch.ones(1, 2),
    }
    processed = _preprocess_dataset_batch(
        batch,
        camera_keys=["image"],
        rename_map={"image": f"{OBS_IMAGES}.camera1", "actions": ACTION},
        preprocessor=preprocessor,
    )

    assert seen_batch is processed
    assert set(processed) == {f"{OBS_IMAGES}.camera1", ACTION}
    assert processed[f"{OBS_IMAGES}.camera1"].dtype == torch.float32
    torch.testing.assert_close(processed[f"{OBS_IMAGES}.camera1"], torch.ones(1, 3, 2, 2))


@pytest.mark.parametrize("training", [False, True])
def test_preprocessing_removes_and_reports_worker_timings(training):
    tracker = MetricsTracker(3, 10, 1, {}) if training else None
    batch = {
        ACTION: torch.ones(3, 2),
        "_loader_timings": {
            "worker_loading_s": torch.tensor([8.0, 8.0, 8.0]),
            "video_loading_s": torch.tensor([3.0, 3.0, 3.0]),
        },
    }

    def processor(batch):
        assert set(batch) == {ACTION}
        return batch

    _preprocess_dataset_batch(batch, [], {}, processor, tracker)
    if training:
        assert tracker.to_dict()["worker_loading_s"] == 8.0
        assert tracker.to_dict()["video_loading_s"] == 3.0
        # Another worker delivers a shorter batch. Count batches, not duplicated per-sample times.
        batch["_loader_timings"] = {
            "worker_loading_s": torch.tensor([4.0]),
            "video_loading_s": torch.tensor([1.0]),
        }
        _preprocess_dataset_batch(batch, [], {}, processor, tracker)
        assert tracker.to_dict()["worker_loading_s"] == 6.0
        assert tracker.to_dict()["video_loading_s"] == 2.0
        assert all(meter.reduction == "mean" and meter.count == 2 for meter in tracker.metrics.values())


@pytest.mark.parametrize("enabled", [None, False, True])
def test_training_loader_enables_profiling(tmp_path, lerobot_dataset_factory, enabled):
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds", total_episodes=1, total_frames=10, use_videos=False
    )
    cfg = SimpleNamespace(
        dataset=DatasetConfig(
            repo_id="test/dataset", **({} if enabled is None else {"profile_loading": enabled})
        ),
        trainable_config=SimpleNamespace(),
        seed=1,
        resume=False,
        num_workers=0,
        batch_size=3,
        persistent_workers=False,
    )
    loader, _ = make_dataloaders(cfg, dataset, None, 0, SimpleNamespace(device_type="cpu"))
    batch = next(iter(loader))
    assert ("_loader_timings" in batch) is (enabled is not False)
    if enabled is not False:
        assert batch["_loader_timings"]["worker_loading_s"].dtype == torch.float32
        assert torch.all(batch["_loader_timings"]["video_loading_s"] == 0)


def test_default_profiling_does_not_break_streaming(caplog):
    class StreamingDataset(torch.utils.data.IterableDataset):
        meta = SimpleNamespace(has_language_columns=False)

        def __iter__(self):
            yield {ACTION: torch.ones(2)}

    cfg = SimpleNamespace(
        dataset=DatasetConfig(repo_id="test/dataset", streaming=True),
        trainable_config=SimpleNamespace(),
        num_workers=0,
        batch_size=3,
        persistent_workers=False,
    )
    loader, _ = make_dataloaders(cfg, StreamingDataset(), None, 0, SimpleNamespace(device_type="cpu"))
    assert set(next(iter(loader))) == {ACTION}
    assert "timings are unavailable" in caplog.text
