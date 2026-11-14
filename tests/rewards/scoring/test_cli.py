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

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

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
    monkeypatch.setattr(
        lerobot_score.parser,
        "get_yaml_overrides",
        lambda field: ["--device=cuda", "--image_key=observation.images.front"],
    )
    monkeypatch.setattr(lerobot_score.parser, "get_cli_overrides", lambda field: ["--device=cpu"])
    monkeypatch.setattr(lerobot_score.RewardModelConfig, "from_pretrained", fake_from_pretrained)

    cfg = ScoreConfig(dataset_repo_id="user/dataset", num_samples=4, head_mode="dense")

    assert cfg.reward_model is loaded_config
    assert captured["load"] == (
        "user/robometer",
        "model-revision",
        ["--device=cuda", "--image_key=observation.images.front", "--device=cpu"],
    )
    assert cfg.scorer_options == {"num_samples": 4, "head_mode": "dense"}


def test_score_config_applies_yaml_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from lerobot.rewards.topreward import TOPRewardConfig

    checkpoint = tmp_path / "checkpoint"
    TOPRewardConfig(device="cpu", image_key="observation.images.top", fps=2.0).save_pretrained(checkpoint)
    config_path = tmp_path / "score.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset_repo_id": "user/dataset",
                "reward_model": {
                    "path": str(checkpoint),
                    "image_key": "observation.images.front",
                    "fps": 7.0,
                },
            }
        )
    )
    monkeypatch.setattr(lerobot_score.parser, "_config_path_args", {})
    monkeypatch.setattr(lerobot_score.parser, "_config_yaml_overrides", {})
    monkeypatch.setattr(sys, "argv", ["lerobot-score", f"--config_path={config_path}"])
    received: list[ScoreConfig] = []
    monkeypatch.setattr(lerobot_score, "run_score", received.append)
    monkeypatch.setattr(lerobot_score, "init_logging", lambda: None)

    lerobot_score.main()

    assert len(received) == 1
    reward_model = received[0].reward_model
    assert isinstance(reward_model, TOPRewardConfig)
    assert reward_model.pretrained_path == str(checkpoint)
    assert reward_model.image_key == "observation.images.front"
    assert reward_model.fps == 7.0


@pytest.mark.parametrize(
    "name,expected_name",
    [(None, "robometer"), ("robometer-4b", "robometer-4b"), ("", "")],
)
def test_run_score_uses_reward_scoring_api(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str | None, expected_name: str
) -> None:
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
        name=name,
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
        {"name": expected_name, "episodes": [1], "resume": True, "overwrite": False},
    )
    assert captured["push_scores_to_hub"] == (fake_dataset, expected_name)


def test_run_score_requires_a_reward_model():
    pytest.importorskip("datasets")
    pytest.importorskip("av")

    with pytest.raises(ValueError, match="needs a reward model"):
        lerobot_score.run_score(ScoreConfig(dataset_repo_id="user/dataset"))
