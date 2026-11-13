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

from lerobot.scripts import lerobot_score
from lerobot.scripts.lerobot_score import ScoreConfig

pytestmark = pytest.mark.dataset


def test_score_config_loads_reward_model_from_path(monkeypatch):
    loaded_config = object()
    captured = {}

    def fake_from_pretrained(path, *, revision, cli_overrides):
        captured["load"] = (path, revision, cli_overrides)
        return loaded_config

    monkeypatch.setattr(lerobot_score.parser, "get_path_arg", lambda field: "user/robometer")
    monkeypatch.setattr(lerobot_score.parser, "get_pretrained_revision", lambda field: "model-revision")
    monkeypatch.setattr(lerobot_score.parser, "get_cli_overrides", lambda field: ["--device=cpu"])
    monkeypatch.setattr(lerobot_score.RewardModelConfig, "from_pretrained", fake_from_pretrained)

    cfg = ScoreConfig(dataset_repo_id="user/dataset", num_samples=4, head_mode="dense", inference_fps=0.5)

    assert cfg.reward_model is loaded_config
    assert captured["load"] == ("user/robometer", "model-revision", ["--device=cpu"])
    assert cfg.scorer_options == {"num_samples": 4, "head_mode": "dense", "inference_fps": 0.5}


def test_run_score_uses_reward_scoring_api(monkeypatch, tmp_path):
    pytest.importorskip("datasets")
    pytest.importorskip("av")

    reward_config = object()
    fake_model = object()
    fake_dataset = object()
    fake_scorer = SimpleNamespace(name="robometer")
    captured: dict[str, object] = {}

    def fake_dataset_class(repo_id, **kwargs):
        captured["dataset_load"] = (repo_id, kwargs)
        return fake_dataset

    monkeypatch.setattr(lerobot_score, "LeRobotDataset", fake_dataset_class)
    monkeypatch.setattr(
        lerobot_score, "make_reward_model", lambda config: captured.update(model_config=config) or fake_model
    )
    monkeypatch.setattr(
        lerobot_score,
        "make_frame_scorer",
        lambda model, **kwargs: captured.update(make_scorer=(model, kwargs)) or fake_scorer,
    )
    monkeypatch.setattr(
        lerobot_score,
        "score_dataset",
        lambda dataset, scorer, **kwargs: captured.update(score_dataset=(dataset, scorer, kwargs)),
    )
    monkeypatch.setattr(
        lerobot_score,
        "push_scores_to_hub",
        lambda dataset, name: captured.update(push_scores_to_hub=(dataset, name)),
    )

    cfg = ScoreConfig(
        dataset_repo_id="user/dataset",
        reward_model=reward_config,
        dataset_root=tmp_path / "dataset",
        dataset_revision="dataset-revision",
        name="robometer-4b",
        episodes=[1],
        batch_size=8,
        num_subsampled_frames=6,
        push_to_hub=True,
    )

    assert lerobot_score.run_score(cfg) is None
    assert captured["model_config"] is reward_config
    assert captured["make_scorer"] == (fake_model, {"batch_size": 8, "num_subsampled_frames": 6})
    assert captured["dataset_load"] == (
        "user/dataset",
        {
            "root": tmp_path / "dataset",
            "revision": "dataset-revision",
            "episodes": [1],
            "download_videos": True,
        },
    )
    assert captured["score_dataset"] == (
        fake_dataset,
        fake_scorer,
        {"name": "robometer-4b", "episodes": [1], "resume": True, "overwrite": False},
    )
    assert captured["push_scores_to_hub"] == (fake_dataset, "robometer-4b")


def test_run_score_requires_a_reward_model():
    pytest.importorskip("datasets")
    pytest.importorskip("av")

    with pytest.raises(ValueError, match="needs a reward model"):
        lerobot_score.run_score(ScoreConfig(dataset_repo_id="user/dataset"))
