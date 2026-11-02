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

import datetime as dt
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import lerobot.configs.train as tc
from lerobot.common import train_utils
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
    pretrained_model/ directory; both must resolve `checkpoint_path` to the step directory above the
    pretrained_model/ directory (regression test for the directory case, which
    previously resolved one level too high and failed on model.safetensors)."""
    pretrained_dir = tmp_path / "checkpoints" / "000002" / "pretrained_model"
    pretrained_dir.mkdir(parents=True)
    (pretrained_dir / "train_config.json").touch()
    target = pretrained_dir if pass_dir else pretrained_dir / "train_config.json"

    from lerobot.policies.act.configuration_act import ACTConfig

    cfg = tc.draccus.parse(TrainPipelineConfig, args=["--dataset.repo_id", "u/d"])
    cfg.policy = ACTConfig(pretrained_path=Path("user/base_model"))
    cfg.resume = True
    monkeypatch.setattr(tc.parser, "parse_arg", lambda name: str(target) if name == "config_path" else None)

    cfg._resolve_resume_checkpoint()

    # The model the run was fine-tuned from survives the resume, for the published card.
    assert cfg.policy.pretrained_path == Path("user/base_model")
    assert cfg.checkpoint_path == pretrained_dir.parent


def _hub_resume_cfg(monkeypatch, cli_output_dir=None):
    """A `--resume` config on a Hub repo, with the Hub download stubbed out."""

    def fake_snapshot_download(repo_id, repo_type, allow_patterns, local_dir):
        (Path(local_dir) / "checkpoints" / "000200" / "pretrained_model").mkdir(parents=True, exist_ok=True)
        return local_dir

    monkeypatch.setattr(train_utils, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(train_utils, "find_latest_hub_checkpoint", lambda repo_id: "checkpoints/000200")
    args = {"config_path": "alice/run", "output_dir": cli_output_dir}
    monkeypatch.setattr(tc.parser, "parse_arg", lambda name: args.get(name))
    cfg = tc.draccus.parse(TrainPipelineConfig, args=["--dataset.repo_id", "u/d"])
    cfg.resume = True
    return cfg


def _ranks_clock(monkeypatch, clocks: dict[str, dt.datetime]) -> None:
    """Give every thread (a simulated rank) its own wall clock."""

    class _RankDateTime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return clocks[threading.current_thread().name]

    monkeypatch.setattr(tc, "dt", SimpleNamespace(datetime=_RankDateTime))


def test_multi_process_hub_resume_requires_an_explicit_output_dir(monkeypatch, tmp_path):
    """Two ranks whose clocks straddle a second boundary would each stamp their own resume dir, and
    a sharded save would then scatter its shards across both. Refuse instead of guessing."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WORLD_SIZE", "2")
    _ranks_clock(monkeypatch, {threading.current_thread().name: dt.datetime(2026, 11, 2, 11, 11, 45)})
    cfg = _hub_resume_cfg(monkeypatch)

    with pytest.raises(ValueError, match="needs an explicit --output_dir"):
        cfg._resolve_resume_checkpoint()
    assert not (tmp_path / "outputs").exists()  # nothing downloaded


def test_single_process_hub_resume_keeps_the_timestamped_default(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    _ranks_clock(monkeypatch, {threading.current_thread().name: dt.datetime(2026, 11, 2, 11, 11, 45)})
    cfg = _hub_resume_cfg(monkeypatch)

    cfg._resolve_resume_checkpoint()

    assert cfg.output_dir == Path("outputs/train/2026-11-02/11-11-45_resume")
    assert cfg.checkpoint_path == cfg.output_dir / "checkpoints" / "000200"


def test_concurrent_ranks_with_different_clocks_share_an_explicit_output_dir(monkeypatch, tmp_path):
    """Two ranks whose clocks are a second apart resolve the same explicit dir at once: they agree
    on it, and the checkpoint is downloaded into it once."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WORLD_SIZE", "2")
    _ranks_clock(
        monkeypatch,
        {"rank0": dt.datetime(2026, 11, 2, 11, 11, 45), "rank1": dt.datetime(2026, 11, 2, 11, 11, 46)},
    )
    resolved: dict[str, tuple[Path, Path]] = {}
    errors: list[BaseException] = []
    downloads: list[str] = []
    cfgs = {name: _hub_resume_cfg(monkeypatch, "outputs/train/run_resume") for name in ("rank0", "rank1")}

    def fake_snapshot_download(repo_id, repo_type, allow_patterns, local_dir):
        weights = Path(local_dir) / "checkpoints" / "000200" / "pretrained_model" / "model.safetensors"
        if not weights.is_file():
            time.sleep(0.2)  # a real download takes a while
            weights.parent.mkdir(parents=True, exist_ok=True)
            weights.write_text("weights")
            downloads.append(threading.current_thread().name)
        return local_dir

    monkeypatch.setattr(train_utils, "snapshot_download", fake_snapshot_download)

    def rank(name):
        try:
            cfgs[name]._resolve_resume_checkpoint()
            resolved[name] = (cfgs[name].output_dir, cfgs[name].checkpoint_path)
        except BaseException as e:  # noqa: BLE001 - collected and asserted below
            errors.append(e)

    threads = [threading.Thread(target=rank, args=(name,), name=name) for name in cfgs]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(downloads) == 1
    run_dir = Path("outputs/train/run_resume")
    assert resolved == dict.fromkeys(cfgs, (run_dir, run_dir / "checkpoints" / "000200"))
