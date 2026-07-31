# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import pytest

import lerobot.configs.train as tc
from lerobot.configs.train import TrainPipelineConfig


class _FakeHTTPError(tc.HfHubHTTPError):
    """HfHubHTTPError that can be raised without a real HTTP response object."""

    def __init__(self):
        pass


def test_from_pretrained_falls_back_to_latest_checkpoint_config(tmp_path, monkeypatch):
    """A Hub repo with no root train_config.json (an interrupted run that only pushed
    checkpoints/) resolves via the latest checkpoint's config."""
    # A real train_config.json written by save_pretrained, to be returned by the fallback.
    parsed = tc.draccus.parse(TrainPipelineConfig, args=["--dataset.repo_id", "u/d"])
    cfg_file = tmp_path / "train_config.json"
    parsed._save_pretrained(tmp_path)
    assert cfg_file.is_file()

    calls = []

    def fake_hf_hub_download(filename=None, **kwargs):
        calls.append(filename)
        if filename == "train_config.json":
            raise _FakeHTTPError()  # no root config
        if filename == "checkpoints/000010/pretrained_model/train_config.json":
            return str(cfg_file)
        raise AssertionError(f"unexpected filename {filename}")

    monkeypatch.setattr(tc, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(
        tc, "find_latest_hub_checkpoint", lambda repo_id, token=None, revision=None: "checkpoints/000010"
    )

    loaded = TrainPipelineConfig.from_pretrained("user/interrupted-run")
    assert loaded.dataset.repo_id == "u/d"
    # Tried the root config first, then fell back to the latest checkpoint's config.
    assert calls == ["train_config.json", "checkpoints/000010/pretrained_model/train_config.json"]


def test_from_pretrained_raises_when_no_root_config_and_no_checkpoints(monkeypatch):
    """No root config AND no checkpoints → a clear FileNotFoundError, not the raw HTTP error."""

    def fake_hf_hub_download(filename=None, **kwargs):
        raise _FakeHTTPError()

    monkeypatch.setattr(tc, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(tc, "find_latest_hub_checkpoint", lambda repo_id, token=None, revision=None: None)

    with pytest.raises(FileNotFoundError, match="train_config.json not found"):
        TrainPipelineConfig.from_pretrained("user/empty-repo")


@pytest.mark.parametrize("pass_dir", [False, True])
def test_resolve_resume_checkpoint_accepts_file_or_pretrained_model_dir(tmp_path, monkeypatch, pass_dir):
    """`--config_path` may point at the checkpoint's train_config.json or at its
    pretrained_model/ directory; both must resolve `policy.pretrained_path` to the
    pretrained_model/ directory (regression test for the directory case, which
    previously resolved one level too high and failed on model.safetensors)."""
    pretrained_dir = tmp_path / "checkpoints" / "000002" / "pretrained_model"
    pretrained_dir.mkdir(parents=True)
    (pretrained_dir / "train_config.json").touch()
    target = pretrained_dir if pass_dir else pretrained_dir / "train_config.json"

    from lerobot.policies.act.configuration_act import ACTConfig

    cfg = tc.draccus.parse(TrainPipelineConfig, args=["--dataset.repo_id", "u/d"])
    cfg.policy = ACTConfig()
    cfg.resume = True
    monkeypatch.setattr(tc.parser, "parse_arg", lambda name: str(target) if name == "config_path" else None)

    cfg._resolve_resume_checkpoint()

    assert cfg.policy.pretrained_path == pretrained_dir
    assert cfg.checkpoint_path == pretrained_dir.parent
