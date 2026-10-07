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

import shlex
import sys
from unittest.mock import MagicMock

import draccus
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.annotations.steerable_pipeline.config import (
    DEFAULT_ANNOTATE_JOB_IMAGE,
    AnnotationJobConfig,
    AnnotationPipelineConfig,
)
from lerobot.jobs.annotate import build_pod_command, build_pod_setup, submit_annotate_to_hf


def _parse(*args):
    return draccus.parse(AnnotationPipelineConfig, args=list(args))


def _set_argv(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["lerobot-annotate", *args])


# --- config ----------------------------------------------------------------


def test_annotation_job_defaults_are_local_with_vllm_image():
    cfg = AnnotationJobConfig()
    assert cfg.target is None
    assert cfg.is_remote is False
    assert cfg.image == DEFAULT_ANNOTATE_JOB_IMAGE
    assert cfg.timeout == "2h"
    assert cfg.lerobot_ref == "main"


def test_annotation_config_parses_job_target():
    cfg = _parse("--repo_id", "u/d", "--job.target", "h200")
    assert cfg.job.target == "h200"
    assert cfg.job.is_remote is True


def test_annotation_config_defaults_to_local():
    assert _parse("--repo_id", "u/d").job.is_remote is False


# --- pod command -----------------------------------------------------------


def test_pod_setup_installs_requested_ref():
    setup = build_pod_setup("my-branch")
    assert "git+https://github.com/huggingface/lerobot.git@my-branch" in setup
    # The vLLM image has neither ffmpeg (video decode) nor lerobot's pinned deps.
    assert "ffmpeg" in setup
    assert "'draccus==0.10.0'" in setup


def _annotate_argv(command):
    """Extract the `lerobot-annotate ...` argv from a `bash -c` pod command."""
    assert command[:2] == ["bash", "-c"]
    _setup, _, annotate = command[2].rpartition(" && ")
    return shlex.split(annotate)


def test_pod_command_forwards_user_flags_and_pins_local_target():
    command = build_pod_command(
        "u/d",
        "main",
        ["--repo_id=u/d", "--new_repo_id=u/d_annotated", "--push_to_hub=true", "--job.target=h200"],
    )
    argv = _annotate_argv(command)
    assert argv[0] == "lerobot-annotate"
    # --job.* is client-side orchestration; the pod must not re-dispatch itself.
    assert not any(a.startswith("--job.") for a in argv[1:-1])
    assert argv[-1] == "--job.target=local"
    assert "--new_repo_id=u/d_annotated" in argv
    assert "--push_to_hub=true" in argv


def test_pod_command_replaces_host_local_root_with_repo_id():
    """--root points at a directory only the client has; the pod resolves by repo_id."""
    command = build_pod_command("u/d", "main", ["--root", "/home/me/datasets/d", "--seed=7"])
    argv = _annotate_argv(command)
    assert "--root" not in argv
    assert "/home/me/datasets/d" not in argv
    assert argv.count("--repo_id=u/d") == 1
    assert "--seed=7" in argv


def test_pod_command_does_not_duplicate_repo_id():
    command = build_pod_command("u/d", "main", ["--repo_id", "u/d"])
    assert _annotate_argv(command).count("--repo_id=u/d") == 1


def test_pod_command_quotes_flags_containing_spaces_and_json():
    """serve_command and chat_template_kwargs must survive the trip through `bash -c`."""
    serve = "--vlm.serve_command=vllm serve Qwen/Qwen3.6-27B --max-model-len 32768 --port {port}"
    kwargs = '--vlm.chat_template_kwargs={"enable_thinking": false}'
    command = build_pod_command("u/d", "main", [serve, kwargs])
    argv = _annotate_argv(command)
    assert serve in argv
    assert kwargs in argv


# --- submission ------------------------------------------------------------


def test_submit_requires_login(monkeypatch):
    monkeypatch.setattr("lerobot.jobs.annotate.get_token", lambda: None)
    with pytest.raises(RuntimeError, match="hf auth login"):
        submit_annotate_to_hf(_parse("--repo_id", "u/d", "--job.target", "h200"))


def test_submit_requires_repo_id(monkeypatch):
    """A remote run over --root alone can't work: the pod can't see the client's disk."""
    monkeypatch.setattr("lerobot.jobs.annotate.get_token", lambda: "tok")
    cfg = _parse("--root", "/tmp/d", "--job.target", "h200")
    with pytest.raises(ValueError, match="--repo_id"):
        submit_annotate_to_hf(cfg)


@pytest.mark.parametrize("arg", ["--config_path=annotate.yaml", "--vlm=vlm.yaml", "--job=job.yaml"])
def test_submit_rejects_local_config_files(monkeypatch, arg):
    """draccus takes a config file for the whole config and for each nested one; the
    pod can read none of them, so a remote run must refuse rather than drop them."""
    monkeypatch.setattr("lerobot.jobs.annotate.get_token", lambda: "tok")
    _set_argv(monkeypatch, arg, "--job.target=h200")
    cfg = _parse("--repo_id", "u/d", "--job.target", "h200")
    with pytest.raises(ValueError, match="cannot read config files"):
        submit_annotate_to_hf(cfg)


def test_pod_command_drops_bare_job_config_file_arg():
    """`--job` isn't caught by the `--job.` prefix, and could carry a remote target
    that would make the pod submit a job of its own — recursively."""
    argv = _annotate_argv(build_pod_command("u/d", "main", ["--job", "job.yaml", "--seed=7"]))
    assert "--job" not in argv
    assert "job.yaml" not in argv
    assert argv[-1] == "--job.target=local"


def test_submit_dispatches_job(monkeypatch):
    monkeypatch.setattr("lerobot.jobs.annotate.get_token", lambda: "tok")
    monkeypatch.setattr("lerobot.jobs.annotate.HfApi", lambda token=None: MagicMock())
    monkeypatch.setattr("lerobot.jobs.annotate.ensure_dataset_available", lambda *a, **kw: None)

    run_job_calls = []

    def fake_run_job(**kwargs):
        run_job_calls.append(kwargs)
        return MagicMock(id="job-123")

    monkeypatch.setattr("lerobot.jobs.annotate.run_job", fake_run_job)
    _set_argv(monkeypatch, "--repo_id=u/d", "--push_to_hub=true", "--job.target=h200", "--job.detach=true")

    cfg = _parse("--repo_id", "u/d", "--push_to_hub", "true", "--job.target", "h200", "--job.detach", "true")
    submit_annotate_to_hf(cfg)

    assert len(run_job_calls) == 1
    call = run_job_calls[0]
    assert call["flavor"] == "h200"
    assert call["image"] == DEFAULT_ANNOTATE_JOB_IMAGE
    assert call["timeout"] == "2h"
    # The Hub token is forwarded so the pod can pull a private dataset and push the result.
    assert call["secrets"]["HF_TOKEN"] == "tok"
    assert call["labels"].get("lerobot") == "true"
    argv = _annotate_argv(call["command"])
    assert argv[0] == "lerobot-annotate"
    assert "--push_to_hub=true" in argv


@pytest.mark.timeout(15)
def test_submit_follows_job_to_completion(monkeypatch, capsys):
    """Non-detach path must stream logs and RETURN (not hang) once the job is terminal.

    Exercises the `follow_job` helper shared with the training submitter from the
    annotation side, which is why the job-state patches target `lerobot.jobs.hf`.
    Asserting on the completion message and not merely on "didn't hang" is what makes
    this fail if `follow_job` ever reports detached-without-a-verdict instead.
    """
    monkeypatch.setattr("lerobot.jobs.annotate.get_token", lambda: "tok")
    monkeypatch.setattr("lerobot.jobs.annotate.HfApi", lambda token=None: MagicMock())
    monkeypatch.setattr("lerobot.jobs.annotate.ensure_dataset_available", lambda *a, **kw: None)
    monkeypatch.setattr("lerobot.jobs.annotate.run_job", lambda **kw: MagicMock(id="job-1", url="http://x"))
    monkeypatch.setattr(
        "lerobot.jobs.hf.inspect_job",
        lambda job_id: MagicMock(status=MagicMock(stage=MagicMock(value="COMPLETED"), message=None)),
    )
    monkeypatch.setattr("lerobot.jobs.hf.fetch_job_logs", lambda job_id, follow=True: iter(()))
    _set_argv(monkeypatch, "--repo_id=u/d", "--job.target=h200")

    submit_annotate_to_hf(_parse("--repo_id", "u/d", "--push_to_hub", "true", "--job.target", "h200"))
    assert "Annotation complete" in capsys.readouterr().out


@pytest.mark.timeout(15)
def test_submit_raises_when_job_fails(monkeypatch):
    """A job that ends in a non-COMPLETED stage must surface as an error, not a silent return."""
    monkeypatch.setattr("lerobot.jobs.annotate.get_token", lambda: "tok")
    monkeypatch.setattr("lerobot.jobs.annotate.HfApi", lambda token=None: MagicMock())
    monkeypatch.setattr("lerobot.jobs.annotate.ensure_dataset_available", lambda *a, **kw: None)
    monkeypatch.setattr("lerobot.jobs.annotate.run_job", lambda **kw: MagicMock(id="job-1", url=None))
    monkeypatch.setattr(
        "lerobot.jobs.hf.inspect_job",
        lambda job_id: MagicMock(status=MagicMock(stage=MagicMock(value="ERROR"), message="Job timeout")),
    )
    monkeypatch.setattr("lerobot.jobs.hf.fetch_job_logs", lambda job_id, follow=True: iter(()))
    _set_argv(monkeypatch, "--repo_id=u/d", "--job.target=h200")

    with pytest.raises(RuntimeError, match="stage=ERROR .Job timeout."):
        submit_annotate_to_hf(_parse("--repo_id", "u/d", "--job.target", "h200"))


def test_submit_ensures_dataset_is_on_the_hub(monkeypatch):
    """A local-only dataset is pushed (privately) before the job can reach it by repo_id."""
    monkeypatch.setattr("lerobot.jobs.annotate.get_token", lambda: "tok")
    monkeypatch.setattr("lerobot.jobs.annotate.HfApi", lambda token=None: MagicMock())
    monkeypatch.setattr("lerobot.jobs.annotate.run_job", lambda **kw: MagicMock(id="job-1"))

    seen = []
    monkeypatch.setattr(
        "lerobot.jobs.annotate.ensure_dataset_available",
        lambda repo_id, *, api, tags=None: seen.append((repo_id, tags)),
    )
    _set_argv(monkeypatch, "--repo_id=u/d", "--job.target=h200", "--job.detach=true")

    submit_annotate_to_hf(
        _parse("--repo_id", "u/d", "--job.target", "h200", "--job.detach", "true", "--job.tags", '["lelab"]')
    )
    assert seen == [("u/d", ["lerobot", "lelab"])]
