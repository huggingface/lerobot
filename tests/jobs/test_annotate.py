# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Resolved config transport, pinned inputs, secrets and persistent outputs."""

import json
import uuid
from types import SimpleNamespace

import draccus
import pytest

pytest.importorskip("datasets")
from lerobot.annotations.steerable_pipeline.config import AnnotationPipelineConfig  # noqa: E402
from lerobot.data_processing.artifacts import ArtifactStore  # noqa: E402
from lerobot.jobs import annotate, processing  # noqa: E402

SHA = "a" * 40
IMAGE = "runtime@sha256:" + "b" * 64


@pytest.fixture
def cloud(monkeypatch):
    # Simulate remote persistence in one process; no external jobs/storage writes.
    monkeypatch.setattr(annotate, "require_persistent_remote", lambda store: None)
    monkeypatch.setattr(processing, "require_persistent_remote", lambda store: None)
    monkeypatch.setattr(annotate, "get_token", lambda: "fake-test-token")
    monkeypatch.setattr(processing, "get_token", lambda: "fake-test-token")
    monkeypatch.setattr(processing, "validate_hardware", lambda *args: None)
    monkeypatch.setattr(
        annotate,
        "HfApi",
        lambda **kwargs: SimpleNamespace(dataset_info=lambda *a, **k: SimpleNamespace(sha=SHA)),
    )
    monkeypatch.setattr(processing, "run_job", lambda **kwargs: SimpleNamespace(id="job-123"))
    cfg = AnnotationPipelineConfig(repo_id="owner/source", root="/host/local")
    cfg.runtime.run_uri = "memory://annotation-" + uuid.uuid4().hex
    cfg.runtime.hf_jobs.code_revision = SHA
    cfg.job.target, cfg.job.image, cfg.job.detach = "h200", IMAGE, True
    return cfg


def test_config_file_serialized_without_argv_replay(cloud, monkeypatch, tmp_path):
    cloud.plan.max_frames_per_prompt = 42
    cloud.vlm.serve_command = "vllm serve model --port {port}"
    file = tmp_path / "resolved.json"
    file.write_text(json.dumps(draccus.encode(cloud)))
    cfg = draccus.parse(AnnotationPipelineConfig, args=[f"--config_path={file}"])
    captured = []
    monkeypatch.setattr(
        processing, "run_job", lambda **kwargs: captured.append(kwargs) or SimpleNamespace(id="job-123")
    )
    annotate.submit_annotate_to_hf(cfg)
    store = ArtifactStore(cfg.runtime.run_uri)
    bundle = store.read_json(store.list("bundles/*.json")[0])
    decoded = draccus.decode(AnnotationPipelineConfig, bundle["config"])
    assert decoded.plan.max_frames_per_prompt == 42
    assert decoded.vlm.serve_command == cloud.vlm.serve_command
    assert decoded.root is None and decoded.job.target == "local"
    assert decoded.revision == SHA and decoded.runtime.backend == "local"
    assert "config_path" not in captured[0]["command"][2]
    assert SHA in captured[0]["command"][2]
    assert captured[0]["secrets"]["HF_TOKEN"] == "fake-test-token"
    assert "fake-test-token" not in json.dumps(bundle)
    assert store.list("submissions/hf/*/*.json")


def test_secrets_only_environment_references(cloud, monkeypatch):
    cloud.vlm.api_key = "do-not-persist-me"
    with pytest.raises(ValueError, match="environment reference"):
        annotate.submit_annotate_to_hf(cloud)
    assert not ArtifactStore(cloud.runtime.run_uri).list("bundles/*.json")
    cloud.vlm.api_key = "EMPTY"
    cloud.vlm.api_key_env = "TEST_VLM_KEY"
    monkeypatch.setenv("TEST_VLM_KEY", "runtime-only-secret")
    captured = []
    monkeypatch.setattr(
        processing, "run_job", lambda **kwargs: captured.append(kwargs) or SimpleNamespace(id="job-123")
    )
    annotate.submit_annotate_to_hf(cloud)
    assert captured[0]["secrets"]["TEST_VLM_KEY"] == "runtime-only-secret"
    bundle = ArtifactStore(cloud.runtime.run_uri).read_json(
        ArtifactStore(cloud.runtime.run_uri).list("bundles/*.json")[0]
    )
    assert "runtime-only-secret" not in json.dumps(bundle)


def test_plan_submits_no_jobs(cloud, monkeypatch):
    cloud.runtime.mode = "plan"
    monkeypatch.setattr(processing, "run_job", lambda **kwargs: pytest.fail("plan dispatched a paid job"))
    assert annotate.submit_annotate_to_hf(cloud).startswith("bundles/")


@pytest.mark.parametrize(
    "change,match",
    [
        ("root_only", "repo_id"),
        ("backend", "conflicts"),
        ("image", "immutable image"),
        ("code", "immutable.*Git"),
    ],
)
def test_unsafe_remote_settings_rejected(cloud, change, match):
    if change == "root_only":
        cloud.repo_id = None
    elif change == "backend":
        cloud.runtime.backend = "slurm"
    elif change == "image":
        cloud.job.image = "image:latest"
    else:
        cloud.runtime.hf_jobs.code_revision = "main"
    with pytest.raises(ValueError, match=match):
        annotate.submit_annotate_to_hf(cloud)


def test_remote_requires_login(cloud, monkeypatch):
    monkeypatch.setattr(annotate, "get_token", lambda: None)
    with pytest.raises(RuntimeError, match="hf auth login"):
        annotate.submit_annotate_to_hf(cloud)


def test_active_job_rejected_on_resume(cloud, monkeypatch):
    annotate.submit_annotate_to_hf(cloud)
    monkeypatch.setattr(
        processing, "inspect_job", lambda id: SimpleNamespace(id=id, status=SimpleNamespace(stage="RUNNING"))
    )
    with pytest.raises(RuntimeError, match="still active"):
        annotate.submit_annotate_to_hf(cloud)


def test_real_remote_requirement(tmp_path):
    for uri in (tmp_path, "memory://invalid"):
        with pytest.raises(ValueError, match="persistent remote"):
            processing.require_persistent_remote(ArtifactStore(uri))
