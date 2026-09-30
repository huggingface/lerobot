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

from lerobot.rewards.robometer.configuration_robometer import RobometerConfig
from lerobot.rewards.rynnvalue.configuration_rynnvalue import RynnValueConfig
from lerobot.scripts import lerobot_score
from lerobot.scripts.lerobot_score import ScoreConfig


def test_run_score_uses_dataset_score_api(monkeypatch, tmp_path):
    pytest.importorskip("datasets")
    pytest.importorskip("av")
    import lerobot.datasets

    reward_config = RobometerConfig(
        pretrained_path="old/model",
        device="cpu",
        vlm_config={"model_type": "fake", "text_config": {"vocab_size": 10}},
    )
    fake_model = object()
    fake_scorer = SimpleNamespace(name="robometer")
    captured: dict[str, object] = {}

    class FakeDataset:
        root = tmp_path / "dataset"

        def add_score(self, scorer, **kwargs) -> None:
            captured["add_score"] = (scorer, kwargs)

        def push_score_to_hub(self, name: str) -> None:
            captured["push_score_to_hub"] = name

    fake_dataset = FakeDataset()

    def fake_from_pretrained(path, *, revision):
        captured["config_load"] = (path, revision)
        return reward_config

    def fake_dataset_class(repo_id, **kwargs):
        captured["dataset_load"] = (repo_id, kwargs)
        return fake_dataset

    monkeypatch.setattr(lerobot_score.RewardModelConfig, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(lerobot.datasets, "LeRobotDataset", fake_dataset_class)
    monkeypatch.setattr(lerobot_score, "make_reward_model", lambda config: fake_model)
    monkeypatch.setattr(
        lerobot_score,
        "make_frame_scorer",
        lambda model, **kwargs: captured.update(make_scorer=(model, kwargs)) or fake_scorer,
    )

    cfg = ScoreConfig(
        dataset_repo_id="user/dataset",
        dataset_root=tmp_path / "dataset",
        dataset_revision="dataset-revision",
        reward_model_path="user/robometer",
        reward_model_revision="model-revision",
        name="robometer-4b",
        episodes=[1],
        device="cpu",
        image_key="observation.images.wrist",
        batch_size=8,
        num_subsampled_frames=6,
        push_to_hub=True,
    )

    assert lerobot_score.run_score(cfg) is None
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
    assert captured["make_scorer"] == (
        fake_model,
        {"batch_size": 8, "num_subsampled_frames": 6},
    )
    assert captured["add_score"] == (
        fake_scorer,
        {
            "name": "robometer-4b",
            "episodes": [1],
            "resume": True,
            "overwrite": False,
        },
    )
    assert captured["push_score_to_hub"] == "robometer-4b"


def test_run_score_passes_rynnvalue_options(monkeypatch):
    pytest.importorskip("datasets")
    pytest.importorskip("av")
    import lerobot.datasets

    reward_config = RynnValueConfig(pretrained_path="old/model", device="cpu", use_meta=False)
    fake_model = object()
    fake_scorer = SimpleNamespace(name="rynnvalue")
    captured: dict[str, object] = {}

    class FakeDataset:
        fps = 10

        def add_score(self, scorer, **kwargs) -> None:
            captured["add_score"] = (scorer, kwargs)

    monkeypatch.setattr(
        lerobot_score.RewardModelConfig,
        "from_pretrained",
        lambda path, *, revision: reward_config,
    )
    monkeypatch.setattr(lerobot.datasets, "LeRobotDataset", lambda *args, **kwargs: FakeDataset())
    monkeypatch.setattr(lerobot_score, "make_reward_model", lambda config: fake_model)
    monkeypatch.setattr(
        lerobot_score,
        "make_frame_scorer",
        lambda model, **kwargs: captured.update(make_scorer=(model, kwargs)) or fake_scorer,
    )

    cfg = ScoreConfig(
        dataset_repo_id="user/dataset",
        reward_model_path="user/rynnvalue",
        reward_model_revision="model-revision",
        image_key="observation.images.wrist",
        default_task="pick up the cube",
        inference_fps=2.0,
        max_frames=6,
        horizon_s=12.0,
        robot_description="a single-arm robot",
        camera_description="a third-person camera",
        use_meta=True,
    )

    lerobot_score.run_score(cfg)

    assert reward_config.image_key == "observation.images.wrist"
    assert reward_config.default_task == "pick up the cube"
    assert reward_config.robot_description == "a single-arm robot"
    assert reward_config.camera_description == "a third-person camera"
    assert reward_config.use_meta is True
    assert captured["make_scorer"] == (
        fake_model,
        {"dataset_fps": 10.0, "batch_size": 2, "inference_fps": 2.0, "max_frames": 6, "horizon_s": 12.0},
    )
    assert captured["add_score"] == (
        fake_scorer,
        {"name": "rynnvalue", "episodes": None, "resume": True, "overwrite": False},
    )
