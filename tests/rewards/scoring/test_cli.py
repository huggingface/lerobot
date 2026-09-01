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

from pathlib import Path
from types import SimpleNamespace

from lerobot.rewards.robometer.configuration_robometer import RobometerConfig
from lerobot.rewards.scoring import ScoringSummary
from lerobot.scripts import lerobot_score_dataset
from lerobot.scripts.lerobot_score_dataset import ScoreDatasetConfig


def test_run_score_dataset_loads_robometer_and_calls_shared_workflow(monkeypatch, tmp_path):
    import lerobot.datasets

    reward_config = RobometerConfig(
        pretrained_path="old/model",
        device="cpu",
        vlm_config={"model_type": "fake", "text_config": {"vocab_size": 10}},
    )
    fake_model = object()
    fake_dataset = SimpleNamespace(
        root=tmp_path / "dataset",
        repo_id="user/dataset",
        revision="dataset-revision",
    )
    expected_summary = ScoringSummary(
        artifact_path=tmp_path / "signals.parquet",
        episode_count=1,
        new_episode_count=1,
        resumed_episode_count=0,
        frame_count=3,
        signal_nan_counts={},
        observed_ranges={},
    )
    captured: dict[str, object] = {}

    def fake_from_pretrained(path, *, revision):
        captured["config_load"] = (path, revision)
        return reward_config

    def fake_dataset_class(repo_id, **kwargs):
        captured["dataset_load"] = (repo_id, kwargs)
        return fake_dataset

    def fake_score(dataset, model, **kwargs):
        captured["score"] = (dataset, model, kwargs)
        return expected_summary

    monkeypatch.setattr(lerobot_score_dataset.RewardModelConfig, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(lerobot.datasets, "LeRobotDataset", fake_dataset_class)
    monkeypatch.setattr(lerobot_score_dataset, "make_reward_model", lambda config: fake_model)
    monkeypatch.setattr(lerobot_score_dataset, "score_dataset_with_reward_model", fake_score)

    cfg = ScoreDatasetConfig(
        dataset_repo_id="user/dataset",
        dataset_root=tmp_path / "dataset",
        dataset_revision="dataset-revision",
        reward_model_path="user/robometer",
        reward_model_revision="model-revision",
        output_path=tmp_path / "signals.parquet",
        episodes=[1],
        device="cpu",
        image_key="observation.images.wrist",
        batch_size=8,
        num_subsampled_frames=6,
    )

    summary = lerobot_score_dataset.run_score_dataset(cfg)

    assert summary == expected_summary
    assert captured["config_load"] == ("user/robometer", "model-revision")
    assert captured["dataset_load"] == (
        "user/dataset",
        {
            "root": tmp_path / "dataset",
            "revision": "dataset-revision",
            "download_videos": True,
        },
    )
    assert reward_config.pretrained_path == "user/robometer"
    assert reward_config.pretrained_revision == "model-revision"
    assert reward_config.image_key == "observation.images.wrist"
    assert "observation.images.wrist" in reward_config.input_features
    assert "observation.images.top" not in reward_config.input_features
    dataset, model, kwargs = captured["score"]
    assert dataset is fake_dataset
    assert model is fake_model
    assert kwargs == {
        "output_path": Path(tmp_path / "signals.parquet"),
        "model_id": "user/robometer",
        "model_revision": "model-revision",
        "episode_indices": [1],
        "resume": True,
        "batch_size": 8,
        "num_subsampled_frames": 6,
    }
