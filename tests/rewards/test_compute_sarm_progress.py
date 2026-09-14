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

"""Characterization tests for SARM's legacy progress artifact."""

from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from lerobot.rewards.sarm import SARMPrediction


def test_compute_sarm_progress_preserves_legacy_parquet_artifact(monkeypatch, tmp_path: Path) -> None:
    import importlib
    import sys

    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    # The legacy script keeps visualization helpers in the scoring module.
    # This artifact test does not exercise them, so avoid requiring the
    # optional matplotlib dependency just to import the scoring boundary.
    matplotlib = ModuleType("matplotlib")
    matplotlib.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.gridspec", ModuleType("matplotlib.gridspec"))
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", ModuleType("matplotlib.pyplot"))
    compute_rabc_weights = importlib.import_module("lerobot.rewards.sarm.compute_rabc_weights")

    image_key = "observation.images.top"

    class FakeDataset:
        root = tmp_path
        num_episodes = 1
        num_frames = 3
        meta = SimpleNamespace(episodes=[{"dataset_from_index": 0, "dataset_to_index": num_frames}])

        def __getitem__(self, index: int):
            return {
                image_key: torch.full((3, 2, 2), float(index)),
                "task": "pick up the cube",
            }

    class FakeModel:
        config = SimpleNamespace(
            image_key=image_key,
            state_key="observation.state",
            frame_gap=30,
            uses_dual_heads=False,
            n_obs_steps=2,
        )

        def predict_progress(self, batch, *, head_mode: str):  # noqa: ARG002
            progress = batch["video_features"][..., 0].float()
            batch_size, seq_len = progress.shape
            return SARMPrediction(
                progress=progress,
                stage_probabilities=torch.ones(batch_size, seq_len, 1),
                stage_confidence=torch.ones(batch_size, seq_len),
                valid_mask=torch.ones(batch_size, seq_len, dtype=torch.bool),
            )

    class FakePreprocessor:
        steps: list = []

        def eval(self):
            return self

        def __call__(self, batch):
            query_index = float(batch["index"])
            return {
                "text_features": torch.zeros(1, 4),
                "video_features": torch.full((1, 3, 4), query_index),
                "lengths": torch.tensor([3], dtype=torch.int32),
            }

    monkeypatch.setattr(
        compute_rabc_weights,
        "load_sarm_resources",
        lambda *_args, **_kwargs: (FakeDataset(), FakeModel(), FakePreprocessor()),
    )

    output_path = compute_rabc_weights.compute_sarm_progress(
        dataset_repo_id="local/test",
        reward_model_path="local/sarm",
        output_path=str(tmp_path / "sarm_progress.parquet"),
        device="cpu",
        num_visualizations=0,
    )

    table = pq.read_table(output_path)
    assert table.schema.names == ["index", "episode_index", "frame_index", "progress_sparse"]
    assert table.schema.field("index").type == pa.int64()
    assert table.schema.field("episode_index").type == pa.int64()
    assert table.schema.field("frame_index").type == pa.int64()
    assert table.schema.field("progress_sparse").type == pa.float32()
    assert table.schema.metadata == {b"reward_model_path": b"local/sarm"}
    assert table.column("progress_sparse").to_pylist() == [0.0, 1.0, 2.0]
