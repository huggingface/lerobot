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

import copy
import datetime as dt
import json
import threading
from importlib.metadata import entry_points
from types import SimpleNamespace

import draccus
import httpx
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.configs.default import JobConfig
from lerobot.configs.train import TrainJobConfig, TrainPipelineConfig
from lerobot.jobs.hf import (
    _build_resume_job,
    _pod_forwarded_args,
    _poll_until_done,
    build_launch,
    build_remote_config_file,
    build_repo_id,
    checkpoint_dp_world_size,
    describe_batch_size,
    flavor_gpu_count,
    follow_job,
    resolve_job_tags,
    resolve_nproc_per_node,
    submit_to_hf,
)
from lerobot.utils.constants import TRAINING_STATE_DIR, TRAINING_STEP

pytestmark = pytest.mark.dataset


def test_follow_job_detach_returns_without_watching(monkeypatch):
    """`detach` must short-circuit before any polling or log streaming starts."""

    def _boom(*a, **kw):
        raise AssertionError("detach must not touch the job")

    monkeypatch.setattr("lerobot.jobs.hf.inspect_job", _boom)
    monkeypatch.setattr("lerobot.jobs.hf.fetch_job_logs", _boom)
    # False = "stopped watching without a verdict", so callers stay quiet rather than
    # claiming success for a job that is still running.
    assert follow_job("job-1", detach=True) is False


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


def test_build_resume_job_requires_config_path(monkeypatch):
    """Resuming without --config_path fails with a clear ValueError, not a TypeError from Path(None)."""
    monkeypatch.setattr("lerobot.configs.parser.parse_arg", lambda *a, **k: None)
    cfg = SimpleNamespace(checkpoint_path=None, job_name="j")
    with pytest.raises(ValueError, match="config_path"):
        _build_resume_job(cfg, "alice")


def test_build_resume_job_local_checkpoint_requires_resolved_checkpoint_path(monkeypatch, tmp_path):
    """A local --config_path whose checkpoint validate() never resolved (--policy.path given) is rejected.

    `--policy.path` takes priority over `--resume` in TrainPipelineConfig, leaving `checkpoint_path`
    None; the helper must name that clash instead of crashing, and must not upload anything.
    """
    local_cfg = tmp_path / "train_config.json"
    local_cfg.write_text("{}")
    monkeypatch.setattr("lerobot.configs.parser.parse_arg", lambda *a, **k: str(local_cfg))
    monkeypatch.setattr("sys.argv", ["lerobot-train", "--resume=true", f"--config_path={local_cfg}"])

    def _boom(*a, **kw):
        raise AssertionError("must not upload a checkpoint that was never resolved")

    monkeypatch.setattr("lerobot.jobs.hf.push_checkpoint_to_hub", _boom)
    cfg = SimpleNamespace(checkpoint_path=None, job_name="j")
    with pytest.raises(ValueError, match="policy.path"):
        _build_resume_job(cfg, "alice")


def test_submit_resume_dispatches_to_build_resume_job(monkeypatch):
    """`--resume` takes the resume branch: no fresh repo or config staging, the checkpoint's repo is used."""
    monkeypatch.setattr("lerobot.jobs.hf.get_token", lambda: "tok")

    class FakeHfApi:
        def __init__(self, token=None):
            pass

        def whoami(self, token=None):
            return {"name": "alice"}

    monkeypatch.setattr("lerobot.jobs.hf.HfApi", FakeHfApi)
    monkeypatch.setattr("lerobot.jobs.hf.ensure_dataset_available", lambda *a, **kw: None)

    def _boom(*a, **kw):
        raise AssertionError("a resume must not stage a fresh config")

    monkeypatch.setattr("lerobot.jobs.hf._stage_config_on_hub", _boom)
    resume_args = ["--config_path=alice/ckpt", "--job.target=local"]
    resume_calls = []

    def fake_build_resume_job(cfg, username):
        resume_calls.append((cfg, username))
        return "alice/ckpt", resume_args

    monkeypatch.setattr("lerobot.jobs.hf._build_resume_job", fake_build_resume_job)
    monkeypatch.setattr("lerobot.jobs.hf.checkpoint_dp_world_size", lambda cfg: 1)
    run_job_calls = []

    def fake_run_job(**kwargs):
        run_job_calls.append(kwargs)
        return SimpleNamespace(id="job-1", url=None)

    monkeypatch.setattr("lerobot.jobs.hf.run_job", fake_run_job)

    cfg = draccus.parse(
        TrainPipelineConfig,
        args=[
            "--dataset.repo_id",
            "u/d",
            "--policy.type",
            "act",
            "--resume",
            "true",
            "--job.target",
            "a10g-small",
            "--job.detach",
            "true",
        ],
    )
    # validate() would look for --config_path on the real sys.argv; resolving the checkpoint is
    # _build_resume_job's job here, so skip it (as test_submit_rejects_reward_model_training does).
    monkeypatch.setattr(cfg, "validate", lambda: None)

    submit_to_hf(cfg)

    assert resume_calls == [(cfg, "alice")]
    # A single-GPU flavor keeps the plain command (no launcher, no env override).
    assert run_job_calls[0]["command"] == ["lerobot-train", *resume_args]
    assert run_job_calls[0]["env"] is None
    # The fresh-run branch must not have run: no auto-assigned model repo on the policy.
    assert cfg.policy.repo_id is None


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
    # Single-GPU flavor: no launcher and no env override, as before multi-GPU support.
    assert len(call["command"]) == 2
    assert call["env"] is None
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
@pytest.mark.parametrize("flavor", ["a10g-small", "a10g-largex2"], ids=["single_gpu", "torchrun"])
def test_submit_returns_on_model_pushed_marker(monkeypatch, flavor):
    """Finish when the model-pushed log appears, even if the job stage never flips.

    Under torchrun the marker still arrives verbatim: only rank 0 logs at INFO (`init_logging`
    gives other ranks a NullHandler), and torchrun only prefixes lines of tee'd streams, which the
    pod command doesn't request.
    """
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
            flavor,
        ],
    )
    # Must return via the model-pushed marker despite the perpetual RUNNING stage.
    submit_to_hf(cfg)


def test_submit_raises_when_trackio_enabled_without_space_id(monkeypatch):
    """trackio.enable without a Space fails fast, before submitting: the pod's local database is lost."""

    monkeypatch.setattr("lerobot.jobs.hf.get_token", lambda: "tok")

    class FakeHfApi:
        def __init__(self, token=None):
            pass

        def whoami(self, token=None):
            return {"name": "alice"}

    monkeypatch.setattr("lerobot.jobs.hf.HfApi", FakeHfApi)

    cfg = draccus.parse(
        TrainPipelineConfig,
        args=[
            "--dataset.repo_id",
            "u/d",
            "--policy.type",
            "act",
            "--job.target",
            "a10g-small",
            "--trackio.enable",
            "true",
        ],
    )
    with pytest.raises(ValueError, match="space_id"):
        submit_to_hf(cfg)


def test_submit_with_trackio_space_id_stages_config_and_forwards_only_hf_token(monkeypatch, tmp_path):
    """With a Space set, the staged config carries the trackio section and the pod gets only HF_TOKEN."""

    monkeypatch.setattr("lerobot.jobs.hf.get_token", lambda: "tok")

    class FakeHfApi:
        def __init__(self, token=None):
            pass

        def whoami(self, token=None):
            return {"name": "alice"}

    monkeypatch.setattr("lerobot.jobs.hf.HfApi", FakeHfApi)
    monkeypatch.setattr("lerobot.jobs.hf.ensure_dataset_available", lambda *a, **kw: None)
    staged = {}

    def fake_stage(cfg, repo_id, token, tags=None):
        # Build the real pod config (no upload) so the test sees what the pod would parse.
        path = build_remote_config_file(cfg, repo_id, tmp_path / "train_config.json", tags=tags)
        staged.update(json.loads(path.read_text()))
        return repo_id

    monkeypatch.setattr("lerobot.jobs.hf._stage_config_on_hub", fake_stage)
    run_job_calls = []

    def fake_run_job(**kwargs):
        run_job_calls.append(kwargs)
        return SimpleNamespace(id="job-1", url=None)

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
            "--trackio.enable",
            "true",
            "--trackio.space_id",
            "u/s",
        ],
    )
    submit_to_hf(cfg)

    assert len(run_job_calls) == 1
    # The Hub token alone authenticates both the dataset pull and the trackio Space sync.
    assert run_job_calls[0]["secrets"] == {"HF_TOKEN": "tok"}
    assert staged["trackio"]["enable"] is True
    assert staged["trackio"]["space_id"] == "u/s"


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


# --- multi-GPU launch ---------------------------------------------------------------------------


# The `hf jobs hardware` catalogue (2026-11): multi-GPU flavors end in `x<count>`.
@pytest.mark.parametrize(
    "flavor,gpus",
    [
        ("cpu-basic", 0),
        ("cpu-upgrade", 0),
        ("cpu-performance", 0),
        ("cpu-xl", 0),
        ("t4-small", 1),
        ("t4-medium", 1),
        ("a10g-small", 1),
        ("a10g-large", 1),
        ("a10g-largex2", 2),
        ("a10g-largex4", 4),
        ("a100-large", 1),
        ("a100x4", 4),
        ("a100x8", 8),
        ("h200", 1),
        ("h200x2", 2),
        ("h200x4", 4),
        ("h200x8", 8),
        ("rtx-pro-6000", 1),
        ("rtx-pro-6000x2", 2),
        ("rtx-pro-6000x4", 4),
        ("rtx-pro-6000x8", 8),
        ("l4x1", 1),
        ("l4x4", 4),
        ("l40sx1", 1),
        ("l40sx4", 4),
        ("l40sx8", 8),
    ],
)
def test_flavor_gpu_count_matches_hf_jobs_catalogue(flavor, gpus):
    assert flavor_gpu_count(flavor) == gpus


@pytest.mark.parametrize(
    "flavor,override,expected",
    [
        ("a10g-largex2", None, 2),  # auto: every GPU of the flavor
        ("h200x8", None, 8),
        ("a10g-large", None, 1),  # single-GPU flavor: one process
        ("cpu-basic", None, 1),  # CPU flavor: still one process
        ("a100x4", 1, 1),  # opt out of multi-GPU on a multi-GPU flavor
        ("h200x8", 4, 4),  # use a subset of the GPUs
    ],
)
def test_resolve_nproc_per_node(flavor, override, expected):
    assert resolve_nproc_per_node(flavor, flavor_gpu_count(flavor), override) == expected


@pytest.mark.parametrize(
    "flavor,override",
    [("a10g-largex2", 4), ("a10g-large", 2), ("some-new-gpux", 8)],
)
def test_resolve_nproc_per_node_lets_an_explicit_value_exceed_the_name_count(caplog, flavor, override):
    """The GPU count only comes from the flavor name, so an explicit value wins, with a warning."""
    with caplog.at_level("WARNING"):
        assert resolve_nproc_per_node(flavor, flavor_gpu_count(flavor), override) == override
    assert "exceeds" in caplog.text


@pytest.mark.parametrize(
    "flavor,recorded,expected",
    [
        ("a10g-largex2", 1, 1),  # a single-process checkpoint stays single-process
        ("h200x8", 2, 2),
        ("a10g-largex2", 2, 2),
        ("a10g-large", 1, 1),
    ],
)
def test_resolve_nproc_per_node_resume_defaults_to_the_checkpoint_world_size(flavor, recorded, expected):
    gpus = flavor_gpu_count(flavor)
    assert resolve_nproc_per_node(flavor, gpus, None, recorded=recorded, resume=True) == expected


def test_resolve_nproc_per_node_resume_with_unknown_world_size_runs_one_process(caplog):
    with caplog.at_level("WARNING"):
        assert resolve_nproc_per_node("h200x8", 8, None, recorded=None, resume=True) == 1
    assert "--job.nproc_per_node" in caplog.text


def test_resolve_nproc_per_node_resume_rejects_a_flavor_smaller_than_the_checkpoint():
    with pytest.raises(ValueError, match="trained on 4 processes"):
        resolve_nproc_per_node("a10g-largex2", 2, None, recorded=4, resume=True)


def test_resolve_nproc_per_node_resume_warns_when_an_explicit_value_changes_the_world(caplog):
    with caplog.at_level("WARNING"):
        assert resolve_nproc_per_node("h200x4", 4, 4, recorded=1, resume=True) == 4
    assert "effective batch size changes" in caplog.text


def test_resolve_nproc_per_node_resume_rejects_a_world_change_on_streaming():
    """The trainer hard-fails a streaming resume on another dp world size; fail before submitting."""
    with pytest.raises(ValueError, match="streaming"):
        resolve_nproc_per_node("h200x4", 4, 4, recorded=1, resume=True, streaming=True)
    # The matching size goes through.
    assert resolve_nproc_per_node("h200x4", 4, 2, recorded=2, resume=True, streaming=True) == 2


@pytest.mark.parametrize("value", [0, -1])
def test_job_config_rejects_non_positive_nproc_per_node(value):
    with pytest.raises(ValueError, match="nproc_per_node"):
        TrainJobConfig(target="a10g-largex2", nproc_per_node=value)


def test_job_config_parses_nproc_per_node_from_cli():
    cfg = draccus.parse(
        TrainPipelineConfig,
        args=["--dataset.repo_id", "u/d", "--policy.type", "act", "--job.target", "h200x8"]
        + ["--job.nproc_per_node", "4"],
    )
    assert cfg.job.nproc_per_node == 4
    assert _minimal_cfg().job.nproc_per_node is None  # auto by default


def test_nproc_per_node_is_training_only():
    """The launch size lives on the training job config; other entry points never see it."""
    assert not hasattr(JobConfig(), "nproc_per_node")


def test_build_launch_single_process_is_unchanged():
    assert build_launch(["--config_path=u/r"], 1) == (["lerobot-train", "--config_path=u/r"], None)


def test_build_launch_multi_process_uses_torchrun():
    command, env = build_launch(["--config_path=u/r"], 4)
    assert command == [
        "torchrun",
        "--standalone",
        "--nproc-per-node=4",
        "-m",
        "lerobot.scripts.lerobot_train",
        "--config_path=u/r",
    ]
    # The lerobot-gpu image bakes in CUDA_VISIBLE_DEVICES=0; every rank must see all GPUs.
    assert env == {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}


def test_torchrun_module_is_the_lerobot_train_entry_point():
    """`torchrun -m <module>` must run the same `main` as the `lerobot-train` console script."""
    (script,) = entry_points(group="console_scripts", name="lerobot-train")
    command, _ = build_launch([], 2)
    assert script.value == f"{command[4]}:main"


@pytest.mark.parametrize(
    "parallelism,world,dp_world",
    [([], 4, 4), (["--parallelism.dp_shard", "-1"], 4, 4), ([], 1, 1)],
    ids=["ddp", "fsdp", "single"],
)
def test_describe_batch_size_uses_the_resolved_dp_world(parallelism, world, dp_world):
    cfg = draccus.parse(
        TrainPipelineConfig,
        args=["--dataset.repo_id", "u/d", "--policy.type", "act", "--batch_size", "8"]
        + ["--accelerator.gradient_accumulation.steps", "2", *parallelism],
    )
    resolved = copy.deepcopy(cfg.parallelism)
    resolved.resolve(world)
    assert describe_batch_size(cfg, resolved) == (
        f"8 per worker x {dp_world} data-parallel worker(s) x 2 grad accum = {8 * dp_world * 2} "
        "samples per optimizer step"
    )


def test_build_remote_config_never_carries_nproc_per_node(tmp_path):
    """The launch size is client-side orchestration: the staged config must stay parseable by
    older trainer images, and its parallelism must stay unresolved so the pod binds it to the
    WORLD_SIZE torchrun gives it."""
    cfg = draccus.parse(
        TrainPipelineConfig,
        args=["--dataset.repo_id", "u/d", "--policy.type", "act", "--job.target", "h200x4"]
        + ["--job.nproc_per_node", "2"],
    )
    dest = tmp_path / "train_config.json"
    build_remote_config_file(cfg, "u/run", dest)
    text = dest.read_text()
    data = json.loads(text)
    assert "job" not in data
    assert "nproc_per_node" not in text
    assert data["parallelism"]["dp_replicate"] == 1
    assert data["parallelism"]["dp_shard"] == 1


def _write_training_step(training_state_dir, **state):
    training_state_dir.mkdir(parents=True)
    (training_state_dir / TRAINING_STEP).write_text(json.dumps({"step": 100, **state}))


def test_checkpoint_dp_world_size_reads_a_local_checkpoint(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / "checkpoints" / "000100"
    _write_training_step(checkpoint_dir / TRAINING_STATE_DIR, dp_world_size=2)
    monkeypatch.setattr("lerobot.configs.parser.parse_arg", lambda name, *a, **k: str(checkpoint_dir))
    cfg = _minimal_cfg()
    cfg.checkpoint_path = checkpoint_dir
    assert checkpoint_dp_world_size(cfg) == 2


def test_checkpoint_dp_world_size_fetches_only_the_step_file_of_a_hub_checkpoint(monkeypatch, tmp_path):
    training_state_dir = tmp_path / "snapshot" / TRAINING_STATE_DIR
    _write_training_step(training_state_dir, dp_world_size=4)
    downloads = []

    def fake_download(repo_id, filename, **kwargs):
        downloads.append((repo_id, filename))
        return str(training_state_dir / TRAINING_STEP)

    monkeypatch.setattr("lerobot.configs.parser.parse_arg", lambda name, *a, **k: "alice/run")
    monkeypatch.setattr("lerobot.jobs.hf.find_latest_hub_checkpoint", lambda repo_id: "checkpoints/000100")
    monkeypatch.setattr("lerobot.jobs.hf.hf_hub_download", fake_download)
    assert checkpoint_dp_world_size(_minimal_cfg()) == 4
    assert downloads == [("alice/run", f"checkpoints/000100/{TRAINING_STATE_DIR}/{TRAINING_STEP}")]


@pytest.mark.parametrize(
    "state",
    [{}, None],  # a checkpoint predating the record, and a missing training_step.json
    ids=["not_recorded", "unreadable"],
)
def test_checkpoint_dp_world_size_is_none_when_unknown(monkeypatch, tmp_path, state):
    checkpoint_dir = tmp_path / "checkpoints" / "000100"
    if state is None:
        checkpoint_dir.mkdir(parents=True)
    else:
        _write_training_step(checkpoint_dir / TRAINING_STATE_DIR, **state)
    monkeypatch.setattr("lerobot.configs.parser.parse_arg", lambda name, *a, **k: str(checkpoint_dir))
    cfg = _minimal_cfg()
    cfg.checkpoint_path = checkpoint_dir
    assert checkpoint_dp_world_size(cfg) is None


def _patch_submit_side_effects(monkeypatch, *, stage_config=None, recorded=None) -> list[dict]:
    """Stub the Hub/Jobs calls of `submit_to_hf`; return the list `run_job` kwargs land in.

    `recorded` is the data-parallel world size a resumed checkpoint reports.
    """
    monkeypatch.setattr("lerobot.jobs.hf.get_token", lambda: "tok")
    monkeypatch.setattr("lerobot.jobs.hf.checkpoint_dp_world_size", lambda cfg: recorded)

    class FakeHfApi:
        def __init__(self, token=None):
            pass

        def whoami(self, token=None):
            return {"name": "alice"}

    monkeypatch.setattr("lerobot.jobs.hf.HfApi", FakeHfApi)
    monkeypatch.setattr("lerobot.jobs.hf.ensure_dataset_available", lambda *a, **kw: None)
    monkeypatch.setattr(
        "lerobot.jobs.hf._stage_config_on_hub",
        stage_config or (lambda cfg, repo_id, token, tags=None: repo_id),
    )
    run_job_calls: list[dict] = []

    def fake_run_job(**kwargs):
        run_job_calls.append(kwargs)
        return SimpleNamespace(id="job-1", url=None)

    monkeypatch.setattr("lerobot.jobs.hf.run_job", fake_run_job)
    return run_job_calls


def _remote_cfg(flavor: str, *extra: str) -> TrainPipelineConfig:
    return draccus.parse(
        TrainPipelineConfig,
        args=["--dataset.repo_id", "u/d", "--policy.type", "act", "--batch_size", "16"]
        + ["--job.target", flavor, "--job.detach", "true", *extra],
    )


@pytest.mark.parametrize(
    "flavor,extra,nproc",
    [
        ("a10g-largex2", [], 2),
        ("h200x8", [], 8),
        ("h200x8", ["--job.nproc_per_node", "4"], 4),
    ],
)
def test_submit_fresh_run_on_multi_gpu_flavor_launches_torchrun(monkeypatch, capsys, flavor, extra, nproc):
    run_job_calls = _patch_submit_side_effects(monkeypatch)
    cfg = _remote_cfg(flavor, *extra)

    submit_to_hf(cfg)

    (call,) = run_job_calls
    assert call["flavor"] == flavor
    assert call["command"] == [
        "torchrun",
        "--standalone",
        f"--nproc-per-node={nproc}",
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--config_path={cfg.policy.repo_id}",
    ]
    assert call["env"] == {"CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(nproc))}
    # The staged config keeps the parallelism unresolved; the submit-time check works on a copy.
    assert cfg.parallelism.dp_replicate == 1
    out = capsys.readouterr().out
    assert f"torchrun, {nproc} processes" in out
    assert f"16 per worker x {nproc} data-parallel worker(s) x 1 grad accum = {16 * nproc} samples" in out


@pytest.mark.parametrize(
    "flavor,extra",
    [("a10g-large", []), ("l4x1", []), ("a100x4", ["--job.nproc_per_node", "1"])],
    ids=["single_gpu", "x1_suffix", "opt_out"],
)
def test_submit_single_process_keeps_plain_command(monkeypatch, capsys, flavor, extra):
    run_job_calls = _patch_submit_side_effects(monkeypatch)
    cfg = _remote_cfg(flavor, *extra)

    submit_to_hf(cfg)

    (call,) = run_job_calls
    assert call["command"] == ["lerobot-train", f"--config_path={cfg.policy.repo_id}"]
    assert call["env"] is None
    assert "= 16 samples per optimizer step" in capsys.readouterr().out


def _resume_cfg(monkeypatch, *job_args: str, streaming: bool = False) -> TrainPipelineConfig:
    """A `--resume` config on a Hub checkpoint repo, with the user's argv set to match."""
    argv = ["--config_path=alice/ckpt", "--resume=true", "--steps=500"]
    monkeypatch.setattr("sys.argv", ["lerobot-train", *argv, *job_args, "--job.detach=true"])
    monkeypatch.setattr(
        "lerobot.configs.parser.parse_arg",
        lambda name, *a, **k: "alice/ckpt" if name == "config_path" else None,
    )
    cfg = draccus.parse(
        TrainPipelineConfig,
        args=["--dataset.repo_id", "u/d", "--policy.type", "act", "--resume", "true"]
        + ["--dataset.streaming", str(streaming).lower(), *job_args, "--job.detach=true"],
    )
    # validate() would resolve the Hub checkpoint; _build_resume_job only needs config_path.
    monkeypatch.setattr(cfg, "validate", lambda: None)
    return cfg


@pytest.mark.parametrize(
    "job_args,recorded,nproc",
    [
        (["--job.target=h200x4", "--job.nproc_per_node=2"], 2, 2),  # explicit, matching
        (["--job.target=h200x4"], 2, 2),  # auto: the checkpoint's world, not the flavor's 4 GPUs
        (["--job.target=a10g-largex2"], 1, 1),  # a single-process checkpoint stays single-process
        (["--job.target=a10g-largex2"], None, 1),  # unknown world size: the pre-multi-GPU default
    ],
    ids=["explicit", "auto_multi", "auto_single", "unknown"],
)
def test_submit_resume_keeps_the_checkpoint_world_size(monkeypatch, job_args, recorded, nproc):
    """The resume path replays the user's flags under the launcher, minus every --job.* flag."""

    def _boom(*a, **kw):
        raise AssertionError("a resume must not stage a fresh config")

    run_job_calls = _patch_submit_side_effects(monkeypatch, stage_config=_boom, recorded=recorded)
    cfg = _resume_cfg(monkeypatch, *job_args)

    submit_to_hf(cfg)

    (call,) = run_job_calls
    train_args = ["--resume=true", "--steps=500", "--config_path=alice/ckpt", "--job.target=local"]
    command, env = build_launch(train_args, nproc)
    assert call["command"] == command
    assert call["env"] == env


def test_submit_resume_rejects_a_streaming_world_change_before_touching_the_hub(monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("must fail before touching the Hub")

    run_job_calls = _patch_submit_side_effects(monkeypatch, stage_config=_boom, recorded=1)
    monkeypatch.setattr("lerobot.jobs.hf.ensure_dataset_available", _boom)
    cfg = _resume_cfg(monkeypatch, "--job.target=h200x4", "--job.nproc_per_node=4", streaming=True)

    with pytest.raises(ValueError, match="streaming"):
        submit_to_hf(cfg)
    assert run_job_calls == []


@pytest.mark.parametrize(
    "flavor,parallelism",
    [
        # FSDP over 4 shards cannot fit the 2 processes of a 2-GPU flavor.
        ("a10g-largex2", ["--parallelism.dp_shard", "4"]),
        # e.g. resuming a 2-GPU DDP checkpoint (its config records dp_replicate=2) on 1 GPU.
        ("a10g-large", ["--parallelism.dp_replicate", "2"]),
    ],
)
def test_submit_rejects_parallelism_that_does_not_fit_the_launch(monkeypatch, flavor, parallelism):
    """The pod would fail on WORLD_SIZE; catch it before the dataset push and config staging."""

    def _boom(*a, **kw):
        raise AssertionError("must fail before touching the Hub")

    run_job_calls = _patch_submit_side_effects(monkeypatch, stage_config=_boom)
    monkeypatch.setattr("lerobot.jobs.hf.ensure_dataset_available", _boom)
    cfg = _remote_cfg(flavor, *parallelism)

    with pytest.raises(ValueError, match="does not fit"):
        submit_to_hf(cfg)
    assert run_job_calls == []


def test_submit_rejects_dcp_on_a_topology_that_resolves_unsharded(monkeypatch):
    """`dp_shard=-1` passes the parse-time DCP check but resolves to an unsharded run on 1 GPU;
    the trainer's factory would refuse it on the pod, so submit refuses it first."""

    def _boom(*a, **kw):
        raise AssertionError("must fail before touching the Hub")

    run_job_calls = _patch_submit_side_effects(monkeypatch, stage_config=_boom)
    monkeypatch.setattr("lerobot.jobs.hf.ensure_dataset_available", _boom)
    cfg = _remote_cfg("a10g-large", "--parallelism.dp_shard", "-1", "--checkpoint_format", "dcp")

    with pytest.raises(ValueError, match="requires a sharded run"):
        submit_to_hf(cfg)
    assert run_job_calls == []


def test_submit_launches_an_explicit_value_above_the_name_count(monkeypatch, caplog):
    run_job_calls = _patch_submit_side_effects(monkeypatch)
    cfg = _remote_cfg("a10g-largex2", "--job.nproc_per_node", "4")

    with caplog.at_level("WARNING"):
        submit_to_hf(cfg)

    (call,) = run_job_calls
    assert "--nproc-per-node=4" in call["command"]
    assert "exceeds" in caplog.text


def test_validate_warns_that_nproc_per_node_is_ignored_locally(caplog):
    cfg = draccus.parse(
        TrainPipelineConfig,
        args=["--dataset.repo_id", "u/d", "--policy.type", "act", "--policy.push_to_hub", "false"]
        + ["--job.nproc_per_node", "2"],
    )
    with caplog.at_level("WARNING"):
        cfg.validate()
    assert "only sizes HF Jobs launches" in caplog.text
