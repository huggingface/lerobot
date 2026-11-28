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

    cfg = ScoreConfig(repo_id="user/dataset", root=Path("dataset"))

    assert cfg.reward_model is loaded_config
    assert captured["load"] == (
        "user/robometer",
        "model-revision",
        ["--device=cuda", "--image_key=observation.images.front", "--device=cpu"],
    )


def test_score_config_applies_yaml_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from lerobot.rewards.topreward import TOPRewardConfig

    checkpoint = tmp_path / "checkpoint"
    TOPRewardConfig(device="cpu", image_key="observation.images.top", fps=2.0).save_pretrained(checkpoint)
    config_path = tmp_path / "score.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "repo_id": "user/dataset",
                "root": str(tmp_path / "dataset"),
                "reward_model": {
                    "path": str(checkpoint),
                    "image_key": "observation.images.front",
                    "fps": 7.0,
                    "num_samples": 5,
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
    assert reward_model.num_samples == 5


@pytest.mark.parametrize(
    "name,expected_name",
    [(None, "robometer"), ("robometer-4b", "robometer-4b"), ("", "")],
)
def test_run_score_uses_reward_scoring_api(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str | None, expected_name: str
) -> None:
    pytest.importorskip("datasets")
    pytest.importorskip("av")

    reward_config = SimpleNamespace(type="robometer")
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
        lambda model, *, batch_size: captured.update(make_scorer=(model, batch_size)) or fake_scorer,
    )
    monkeypatch.setattr(
        lerobot_score,
        "score_dataset",
        lambda dataset, scorer, **kwargs: captured.update(score_dataset=(dataset, scorer, kwargs)),
    )
    monkeypatch.setattr(
        lerobot_score,
        "push_scores_to_hub",
        lambda dataset, name, **kwargs: captured.update(push_scores_to_hub=(dataset, name, kwargs)),
    )

    def file_exists(repo_id, filename, *, repo_type, revision):
        captured["hub_check"] = (repo_id, filename, repo_type, revision)
        return False

    monkeypatch.setattr(lerobot_score, "HfApi", lambda: SimpleNamespace(file_exists=file_exists))

    cfg = ScoreConfig(
        repo_id="user/dataset",
        reward_model=reward_config,
        root=tmp_path / "dataset",
        revision="dataset-revision",
        score_name=name,
        episodes=[1],
        batch_size=8,
        push_to_hub=True,
        branch="scores",
        tag_version=False,
    )

    assert lerobot_score.run_score(cfg) is None
    assert captured["model_config"] is reward_config
    assert captured["make_scorer"] == (fake_model, 8)
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
    assert captured["hub_check"] == ("user/dataset", f"scores/{expected_name}.parquet", "dataset", "scores")
    assert captured["push_scores_to_hub"] == (
        fake_dataset,
        expected_name,
        {"branch": "scores", "tag_version": False, "overwrite": False},
    )


def test_run_score_requires_a_reward_model(tmp_path: Path):
    pytest.importorskip("datasets")
    pytest.importorskip("av")

    with pytest.raises(ValueError, match="needs a reward model"):
        lerobot_score.run_score(ScoreConfig(repo_id="user/dataset", root=tmp_path))


@pytest.mark.parametrize("name", [None, "robometer-4b"])
def test_run_score_rejects_existing_scores_before_loading_the_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str | None
) -> None:
    pytest.importorskip("datasets")
    pytest.importorskip("av")

    (tmp_path / "scores").mkdir()
    (tmp_path / "scores" / f"{name or 'robometer'}.parquet").touch()
    monkeypatch.setattr(lerobot_score, "make_reward_model", pytest.fail)
    cfg = ScoreConfig(
        repo_id="user/dataset",
        root=tmp_path,
        reward_model=SimpleNamespace(type="robometer"),
        score_name=name,
    )

    with pytest.raises(FileExistsError, match="--overwrite=true"):
        lerobot_score.run_score(cfg)


def test_run_score_rejects_scores_on_the_hub_before_loading_the_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("datasets")
    pytest.importorskip("av")

    checked = []

    def file_exists(repo_id, filename, *, repo_type, revision):
        checked.append((repo_id, filename, repo_type, revision))
        return True

    monkeypatch.setattr(lerobot_score, "HfApi", lambda: SimpleNamespace(file_exists=file_exists))
    monkeypatch.setattr(lerobot_score, "make_reward_model", pytest.fail)
    cfg = ScoreConfig(
        repo_id="user/dataset",
        root=tmp_path,
        reward_model=SimpleNamespace(type="robometer"),
        push_to_hub=True,
    )

    with pytest.raises(FileExistsError, match="hf://datasets/user/dataset"):
        lerobot_score.run_score(cfg)
    assert checked == [("user/dataset", "scores/robometer.parquet", "dataset", None)]


def test_run_score_defaults_the_dataset_root_to_the_lerobot_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("datasets")
    pytest.importorskip("av")

    # Like `lerobot-record` and `lerobot-edit-dataset`, write into $HF_LEROBOT_HOME/<repo_id>
    # instead of the read-only Hub snapshot cache `LeRobotDataset` uses without a root.
    monkeypatch.setattr(lerobot_score, "HF_LEROBOT_HOME", tmp_path)
    captured = {}
    monkeypatch.setattr(lerobot_score, "make_reward_model", lambda config: object())
    monkeypatch.setattr(lerobot_score, "make_frame_scorer", lambda model, *, batch_size: object())
    monkeypatch.setattr(
        lerobot_score, "LeRobotDataset", lambda repo_id, **kwargs: captured.update(root=kwargs["root"])
    )
    monkeypatch.setattr(lerobot_score, "score_dataset", lambda dataset, scorer, **kwargs: None)
    cfg = ScoreConfig(repo_id="user/dataset", reward_model=SimpleNamespace(type="robometer"))

    lerobot_score.run_score(cfg)

    assert captured["root"] == tmp_path / "user/dataset"

    # The existing-scores check uses the same default root.
    (tmp_path / "user/dataset/scores").mkdir(parents=True)
    (tmp_path / "user/dataset/scores/robometer.parquet").touch()
    with pytest.raises(FileExistsError, match="--overwrite=true"):
        lerobot_score.run_score(cfg)


def test_score_config_rejects_invalid_batch_size(tmp_path: Path):
    with pytest.raises(ValueError, match="batch_size"):
        ScoreConfig(repo_id="user/dataset", root=tmp_path, batch_size=0)
