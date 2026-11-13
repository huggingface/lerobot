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

"""Tests for SARM's RA-BC progress script."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lerobot.rewards.sarm import SARMOutput


@pytest.mark.dataset
@pytest.mark.parametrize("dual_heads", [False, True], ids=["sparse", "sparse-and-dense"])
def test_compute_sarm_progress_writes_progress_parquet(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, dual_heads: bool
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("matplotlib")
    from lerobot.rewards.sarm import compute_rabc_weights

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
            uses_dual_heads=dual_heads,
            n_obs_steps=2,
        )

        def predict_progress(self, batch, *, head_mode: str):
            assert not torch.is_grad_enabled()
            progress = batch["video_features"][..., 0].float()
            if head_mode == "dense":
                progress = progress + 0.25
            batch_size, seq_len = progress.shape
            return SARMOutput(
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
        head_mode="both" if dual_heads else "sparse",
        device="cpu",
        num_visualizations=0,
    )

    table = pq.read_table(output_path)
    expected = {
        "index": [0, 1, 2],
        "episode_index": [0, 0, 0],
        "frame_index": [0, 1, 2],
        "progress_sparse": [0.0, 1.0, 2.0],
    }
    if dual_heads:
        expected["progress_dense"] = [0.25, 1.25, 2.25]
        assert table.schema.field("progress_dense").type == pa.float32()
    assert table.schema.names == list(expected)
    assert table.schema.field("index").type == pa.int64()
    assert table.schema.field("episode_index").type == pa.int64()
    assert table.schema.field("frame_index").type == pa.int64()
    assert table.schema.field("progress_sparse").type == pa.float32()
    assert table.schema.metadata == {b"reward_model_path": b"local/sarm"}
    assert table.to_pydict() == expected
