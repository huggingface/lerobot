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
import json
import threading
from types import SimpleNamespace

import draccus
import httpx
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.configs.train import TrainPipelineConfig
from lerobot.jobs.hf import (
    _pod_forwarded_args,
    _poll_until_done,
    build_remote_config_file,
    build_repo_id,
    resolve_job_tags,
    resolve_wandb_api_key,
    submit_to_hf,
)


def test_resolve_job_tags_always_includes_lerobot_and_dedups():
    assert resolve_job_tags(None) == ["lerobot"]
    assert resolve_job_tags([]) == ["lerobot"]
    assert resolve_job_tags(["lelab"]) == ["lerobot", "lelab"]
    # lerobot isn't duplicated if passed explicitly; order is stable.
    assert resolve_job_tags(["lelab", "lerobot", "lelab"]) == ["lerobot", "lelab"]


def _fake_inspect(stage_value, *, as_enum=True):
    # huggingface_hub returns `stage` as an enum (with `.value`) in some versions and a plain str in others.
    stage = SimpleNamespace(value=stage_value) if as_enum else stage_value
    return lambda job_id: SimpleNamespace(status=SimpleNamespace(stage=stage))


@pytest.mark.parametrize("as_enum", [True, False], ids=["enum_stage", "str_stage"])
def test_poll_until_done_returns_terminal_stage(monkeypatch, as_enum):
    monkeypatch.setattr("lerobot.jobs.hf.inspect_job", _fake_inspect("COMPLETED", as_enum=as_enum))
    done = threading.Event()
    assert _poll_until_done("j", done, poll_interval=0.01) == "COMPLETED"
    assert done.is_set()


def test_poll_until_done_exits_when_done_already_set(monkeypatch):
    # Non-terminal forever; with done pre-set the loop must not block and returns None.
    monkeypatch.setattr("lerobot.jobs.hf.inspect_job", _fake_inspect("RUNNING"))
    done = threading.Event()
    done.set()
    assert _poll_until_done("j", done, poll_interval=0.01) is None


def test_poll_until_done_gives_up_after_repeated_network_failures(monkeypatch):
    monkeypatch.setattr(
        "lerobot.jobs.hf.inspect_job", lambda job_id: (_ for _ in ()).throw(httpx.ConnectError("boom"))
    )
    done = threading.Event()
    result = _poll_until_done("j", done, poll_interval=0.001, max_failures=3)
    assert result is None
    assert done.is_set()


def test_poll_until_done_propagates_programming_errors(monkeypatch):
    """A bug (e.g. TypeError) must surface, not be silently retried as a transient failure."""
    monkeypatch.setattr("lerobot.jobs.hf.inspect_job", lambda job_id: (_ for _ in ()).throw(TypeError("bug")))
    done = threading.Event()
    with pytest.raises(TypeError):
        _poll_until_done("j", done, poll_interval=0.001, max_failures=3)


def test_resolve_wandb_key_from_env(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "abc123")
    assert resolve_wandb_api_key() == "abc123"


def test_resolve_wandb_key_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.netrc here
    monkeypatch.setattr("netrc.netrc", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
    assert resolve_wandb_api_key() is None


def test_resolve_wandb_key_from_netrc(monkeypatch):
    # No env var → fall back to the wandb credentials in ~/.netrc.
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    class _FakeNetrc:
        def authenticators(self, host):
            assert host == "api.wandb.ai"
            return ("login", "account", "netrc-secret")

    monkeypatch.setattr("netrc.netrc", lambda *a, **k: _FakeNetrc())
    assert resolve_wandb_api_key() == "netrc-secret"


def test_resolve_wandb_key_netrc_without_wandb_entry(monkeypatch):
    # ~/.netrc exists but has no api.wandb.ai entry → None.
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    class _FakeNetrc:
        def authenticators(self, host):
            return None

    monkeypatch.setattr("netrc.netrc", lambda *a, **k: _FakeNetrc())
    assert resolve_wandb_api_key() is None


def test_build_repo_id_sanitizes_and_timestamps():
    now = dt.datetime(2026, 6, 19, 10, 22, 3)
    assert build_repo_id("alice", "act", now) == "alice/act_2026-06-19_10-22-03"
    # Runs of illegal characters collapse to a single dash; edges are trimmed.
    assert build_repo_id("alice", "my cool/run!!", now) == "alice/my-cool-run_2026-06-19_10-22-03"
    # A name with nothing usable falls back to "train".
    assert build_repo_id("alice", "///", now) == "alice/train_2026-06-19_10-22-03"


def test_pod_forwarded_args_drops_host_only_flags():
    """User overrides are replayed on the pod, minus flags that only make sense on the submitter.

    `--dataset.root` is a host-local path the pod can't read, so it must be dropped in both the
    `--name=value` and `--name value` forms; unrelated overrides are forwarded untouched.
    """
    argv = [
        "--config_path=u/d",
        "--dataset.root=/local/data",
        "--dataset.root",
        "/other/local/data",
        "--policy.repo_id=u/keep",
        "--steps=10",
        "--job.target=a10g-small",
    ]
    forwarded = _pod_forwarded_args(
        argv,
        drop_names=("--config_path", "--policy.repo_id", "--policy.push_to_hub", "--dataset.root"),
        drop_prefixes=("--job.",),
    )
    assert forwarded == ["--steps=10"]


def _minimal_cfg():
    return draccus.parse(
        TrainPipelineConfig,
        args=["--dataset.repo_id", "u/d", "--policy.type", "act", "--job.target", "a10g-small"],
    )


def test_validate_skips_repo_id_check_for_remote():
    """Remote runs auto-assign repo_id in submit_to_hf, so validate() must not demand it up front."""
    cfg = _minimal_cfg()  # remote target, push_to_hub default True, no explicit repo_id
    assert cfg.policy.repo_id is None
    cfg.validate()  # must not raise


def test_validate_requires_repo_id_for_local_push():
    """Local runs that push to the Hub still need an explicit repo_id."""
    cfg = draccus.parse(
        TrainPipelineConfig,
        args=["--dataset.repo_id", "u/d", "--policy.type", "act"],
    )
    with pytest.raises(ValueError, match="repo_id"):
        cfg.validate()


def test_build_remote_config_applies_overrides(tmp_path):
    cfg = _minimal_cfg()
    dest = tmp_path / "train_config.json"
    out = build_remote_config_file(cfg, "u/run", dest)
    assert out == dest
    data = json.loads(dest.read_text())
    # `job` is client-only orchestration and must be stripped for the pod.
    assert "job" not in data
    # save_checkpoint_to_hub defaults off → omitted so older images accept the config.
    assert "save_checkpoint_to_hub" not in data
    assert data["policy"]["push_to_hub"] is True
    assert data["policy"]["repo_id"] == "u/run"
    assert data["policy"]["device"] is None  # pod auto-detects its GPU
    assert data["dataset"]["root"] is None  # pod resolves the dataset by repo_id
    # the caller's cfg must be left untouched (function works on a deep copy)
    assert cfg.job.target == "a10g-small"
    assert cfg.save_checkpoint_to_hub is False


def test_build_remote_config_includes_checkpoint_flag_when_enabled(tmp_path):
    cfg = draccus.parse(
        TrainPipelineConfig,
        args=[
            "--dataset.repo_id",
            "u/d",
            "--policy.type",
            "act",
            "--job.target",
            "a10g-small",
            "--save_checkpoint_to_hub",
            "true",
        ],
    )
    dest = tmp_path / "train_config.json"
    build_remote_config_file(cfg, "u/run", dest)
    data = json.loads(dest.read_text())
    # explicitly enabled → kept in the config (requires a matching trainer image).
    assert data["save_checkpoint_to_hub"] is True
    assert "job" not in data


def test_build_remote_config_merges_tags_into_policy(tmp_path):
    cfg = _minimal_cfg()
    dest = tmp_path / "train_config.json"
    build_remote_config_file(cfg, "u/run", dest, tags=["lerobot", "lelab"])
    data = json.loads(dest.read_text())
    # tags propagate to the model the pod pushes.
    assert data["policy"]["tags"] == ["lerobot", "lelab"]


def test_build_remote_config_merges_tags_without_duplicating(tmp_path):
    cfg = _minimal_cfg()
    cfg.policy.tags = ["existing", "lerobot"]
    dest = tmp_path / "train_config.json"
    build_remote_config_file(cfg, "u/run", dest, tags=["lerobot", "lelab"])
    data = json.loads(dest.read_text())
    # pre-existing policy tags are kept; only genuinely-new tags are appended (no dup "lerobot").
    assert data["policy"]["tags"] == ["existing", "lerobot", "lelab"]


def test_submit_requires_login(monkeypatch):
    monkeypatch.setattr("lerobot.jobs.hf.get_token", lambda: None)
    cfg = draccus.parse(
        TrainPipelineConfig,
        args=["--dataset.repo_id", "u/d", "--policy.type", "act", "--job.target", "a10g-small"],
    )
    with pytest.raises(RuntimeError, match="hf auth login"):
        submit_to_hf(cfg)


def test_submit_passes_validation_and_submits(monkeypatch):
    """A type-based policy with no explicit repo_id is auto-assigned one and submitted."""
    from unittest.mock import MagicMock

    # Patch get_token
    monkeypatch.setattr("lerobot.jobs.hf.get_token", lambda: "tok")

    # Patch HfApi so whoami returns alice
    class FakeHfApi:
        def __init__(self, token=None):
            pass

        def whoami(self, token=None):
            return {"name": "alice"}

    monkeypatch.setattr("lerobot.jobs.hf.HfApi", FakeHfApi)

    # ensure_dataset_available returns None; patch it out so no Hub access happens
    # (hf.py imports it at module level, so patch it on lerobot.jobs.hf).
    monkeypatch.setattr("lerobot.jobs.hf.ensure_dataset_available", lambda *a, **kw: None)

    # Patch _stage_config_on_hub to skip network
    monkeypatch.setattr(
        "lerobot.jobs.hf._stage_config_on_hub",
        lambda cfg, repo_id, token, tags=None: repo_id,
    )

    # Patch run_job to return a fake job
    fake_job = MagicMock()
    fake_job.id = "job-123"
    run_job_calls = []

    def fake_run_job(**kwargs):
        run_job_calls.append(kwargs)
        return fake_job

    monkeypatch.setattr("lerobot.jobs.hf.run_job", fake_run_job)

    cfg = draccus.parse(
        TrainPipelineConfig,
        args=[
            "--dataset.repo_id",
            "u/d",
            "--policy.type",
            "act",
            "--job.target",
            "a10g-small",
            "--job.detach",
            "true",
        ],
    )

    # Must NOT raise (pre-fix this raised ValueError about missing repo_id)
    submit_to_hf(cfg)

    assert len(run_job_calls) == 1, "run_job should have been called exactly once"
    assert cfg.policy.repo_id is not None
    assert cfg.policy.repo_id.startswith("alice/")
    call = run_job_calls[0]
    # The pod runs `lerobot-train --config_path=<staged repo>` on the requested flavor/image.
    assert call["command"][0] == "lerobot-train"
    assert call["command"][1].startswith("--config_path=")
    assert call["flavor"] == "a10g-small"
    assert call["image"] == "huggingface/lerobot-gpu:latest"
    # The Hub token is forwarded so the pod can pull the (possibly private) dataset.
    assert call["secrets"]["HF_TOKEN"] == "tok"
    # Every job carries the lerobot tag as a queryable label.
    assert call["labels"].get("lerobot") == "true"


def test_submit_rejects_reward_model_training(monkeypatch):
    """Remote training only supports policies; reward-model runs fail fast with a clear error."""
    monkeypatch.setattr("lerobot.jobs.hf.get_token", lambda: "tok")

    class FakeHfApi:
        def __init__(self, token=None):
            pass

        def whoami(self, token=None):
            return {"name": "alice"}

    monkeypatch.setattr("lerobot.jobs.hf.HfApi", FakeHfApi)

    cfg = _minimal_cfg()
    cfg.reward_model = SimpleNamespace(type="reward")  # marks this as reward-model training
    monkeypatch.setattr(cfg, "validate", lambda: None)  # skip pretrained-path resolution

    with pytest.raises(ValueError, match="reward model"):
        submit_to_hf(cfg)


@pytest.mark.timeout(15)
def test_submit_returns_when_job_completes(monkeypatch):
    """Non-detach path must RETURN (not hang) once the job reaches a terminal stage."""
    from types import SimpleNamespace

    monkeypatch.setattr("lerobot.jobs.hf.get_token", lambda: "tok")

    class FakeHfApi:
        def __init__(self, token=None):
            pass

        def whoami(self, token=None):
            return {"name": "alice"}

    monkeypatch.setattr("lerobot.jobs.hf.HfApi", FakeHfApi)
    monkeypatch.setattr("lerobot.jobs.hf.ensure_dataset_available", lambda *a, **kw: None)
    monkeypatch.setattr(
        "lerobot.jobs.hf._stage_config_on_hub", lambda cfg, repo_id, token, tags=None: repo_id
    )
    monkeypatch.setattr("lerobot.jobs.hf.run_job", lambda **kw: SimpleNamespace(id="job-1", url="http://x"))
    # Job is already COMPLETED on the first poll.
    monkeypatch.setattr(
        "lerobot.jobs.hf.inspect_job",
        lambda job_id: SimpleNamespace(
            status=SimpleNamespace(stage=SimpleNamespace(value="COMPLETED"), message=None)
        ),
    )
    # Log stream ends immediately.
    monkeypatch.setattr("lerobot.jobs.hf.fetch_job_logs", lambda job_id, follow=True: iter(()))

    cfg = draccus.parse(
        TrainPipelineConfig,
        args=["--dataset.repo_id", "u/d", "--policy.type", "act", "--job.target", "a10g-small"],
    )
    # Runs in the pytest main thread (signal handler install requires it); the
    # @timeout marker fails the test instead of hanging if it regresses.
    submit_to_hf(cfg)


@pytest.mark.timeout(15)
def test_submit_returns_on_model_pushed_marker(monkeypatch):
    """Finish when the model-pushed log appears, even if the job stage never flips."""
    from types import SimpleNamespace

    monkeypatch.setattr("lerobot.jobs.hf.get_token", lambda: "tok")

    class FakeHfApi:
        def __init__(self, token=None):
            pass

        def whoami(self, token=None):
            return {"name": "alice"}

    monkeypatch.setattr("lerobot.jobs.hf.HfApi", FakeHfApi)
    monkeypatch.setattr("lerobot.jobs.hf.ensure_dataset_available", lambda *a, **kw: None)
    monkeypatch.setattr(
        "lerobot.jobs.hf._stage_config_on_hub", lambda cfg, repo_id, token, tags=None: repo_id
    )
    monkeypatch.setattr("lerobot.jobs.hf.run_job", lambda **kw: SimpleNamespace(id="job-1", url="http://x"))
    # Job stays RUNNING forever — only the log marker can end the command.
    monkeypatch.setattr(
        "lerobot.jobs.hf.inspect_job",
        lambda job_id: SimpleNamespace(
            status=SimpleNamespace(stage=SimpleNamespace(value="RUNNING"), message=None)
        ),
    )
    pushed_line = "INFO Model pushed to https://huggingface.co/alice/myrun"
    monkeypatch.setattr("lerobot.jobs.hf.fetch_job_logs", lambda job_id, follow=True: iter([pushed_line]))

    cfg = draccus.parse(
        TrainPipelineConfig,
        args=[
            "--dataset.repo_id",
            "u/d",
            "--policy.type",
            "act",
            "--policy.repo_id",
            "alice/myrun",
            "--job.target",
            "a10g-small",
        ],
    )
    # Must return via the model-pushed marker despite the perpetual RUNNING stage.
    submit_to_hf(cfg)


def test_submit_raises_when_wandb_enabled_without_key(monkeypatch):
    """wandb.enable with no key reachable anywhere fails fast, before submitting."""

    monkeypatch.setattr("lerobot.jobs.hf.get_token", lambda: "tok")

    class FakeHfApi:
        def __init__(self, token=None):
            pass

        def whoami(self, token=None):
            return {"name": "alice"}

    monkeypatch.setattr("lerobot.jobs.hf.HfApi", FakeHfApi)
    monkeypatch.setattr("lerobot.jobs.hf.resolve_wandb_api_key", lambda: None)

    cfg = draccus.parse(
        TrainPipelineConfig,
        args=[
            "--dataset.repo_id",
            "u/d",
            "--policy.type",
            "act",
            "--job.target",
            "a10g-small",
            "--wandb.enable",
            "true",
        ],
    )
    with pytest.raises(ValueError, match="WANDB_API_KEY"):
        submit_to_hf(cfg)


@pytest.mark.timeout(15)
def test_submit_raises_when_job_ends_in_error(monkeypatch):
    """A terminal non-COMPLETED stage with no model-pushed marker must raise with the status."""
    from types import SimpleNamespace

    monkeypatch.setattr("lerobot.jobs.hf.get_token", lambda: "tok")

    class FakeHfApi:
        def __init__(self, token=None):
            pass

        def whoami(self, token=None):
            return {"name": "alice"}

    monkeypatch.setattr("lerobot.jobs.hf.HfApi", FakeHfApi)
    monkeypatch.setattr("lerobot.jobs.hf.ensure_dataset_available", lambda *a, **kw: None)
    monkeypatch.setattr(
        "lerobot.jobs.hf._stage_config_on_hub", lambda cfg, repo_id, token, tags=None: repo_id
    )
    monkeypatch.setattr("lerobot.jobs.hf.run_job", lambda **kw: SimpleNamespace(id="job-1", url="http://x"))
    # Job fails: a terminal ERROR stage carrying the platform's status message.
    monkeypatch.setattr(
        "lerobot.jobs.hf.inspect_job",
        lambda job_id: SimpleNamespace(
            status=SimpleNamespace(stage=SimpleNamespace(value="ERROR"), message="Job timeout")
        ),
    )
    # Logs end without the model-pushed marker.
    monkeypatch.setattr("lerobot.jobs.hf.fetch_job_logs", lambda job_id, follow=True: iter(()))

    cfg = draccus.parse(
        TrainPipelineConfig,
        args=["--dataset.repo_id", "u/d", "--policy.type", "act", "--job.target", "a10g-small"],
    )
    with pytest.raises(RuntimeError, match=r"stage=ERROR \(Job timeout\)"):
        submit_to_hf(cfg)
