# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("pyarrow")
from lerobot.data_processing.artifacts import ArtifactStore  # noqa: E402
from lerobot.data_processing.configs import RuntimeConfig  # noqa: E402
from lerobot.jobs import processing  # noqa: E402
from tests.data_processing.test_runtime import make_plan  # noqa: E402

SHA = "a" * 40


def test_bounded_jobs_real_workers_and_completed_resume(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path / "remote-simulation")
    plan = make_plan(store, size=7, shard_size=1)
    cfg = RuntimeConfig(backend="hf_jobs", workers=5)
    cfg.hf_jobs.cpu_image = "cpu-runtime@sha256:" + "b" * 64
    cfg.hf_jobs.code_revision = SHA
    cfg.hf_jobs.max_parallel = 2
    monkeypatch.setattr(processing, "require_persistent_remote", lambda store: None)
    monkeypatch.setattr(processing, "get_token", lambda: "fake-test-token")
    monkeypatch.setattr(processing, "validate_hardware", lambda *args: None)
    monkeypatch.setenv("LEROBOT_PROCESSING_CODE_REVISION", SHA)
    active, history = {}, []

    def submit(**kwargs):
        id = str(len(history))
        history.append(kwargs)
        active[id] = kwargs
        assert len(active) <= 2
        assert kwargs["flavor"] == "cpu-upgrade"
        return SimpleNamespace(id=id)

    def finish(job):
        import shlex

        command = shlex.split(active.pop(job.id)["command"][2].split(" && exec ")[1])
        processing.main(command[3:])
        return True

    monkeypatch.setattr(processing, "run_job", submit)
    monkeypatch.setattr(processing, "follow_processing_job", finish)
    first = processing.run_hf_stage(store, plan, cfg)
    assert first.completed == 7 and len(history) == 5
    assert processing.run_hf_stage(store, plan, cfg) == first
    assert len(history) == 5


def test_bundle_tampering_and_code_mismatch(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path / "store")
    key, digest = processing.write_bundle(store, "worker", {}, SHA)
    monkeypatch.setenv("LEROBOT_PROCESSING_CODE_REVISION", "b" * 40)
    with pytest.raises(ValueError, match="pinned revision"):
        processing.execute_bundle(store, key, digest)
    with pytest.raises(ValueError, match="checksum"):
        processing.execute_bundle(store, key, "0" * 64)


def test_retained_release_survives_local_deletion_and_has_no_duplicate_video(tmp_path, monkeypatch):
    import shutil

    root = tmp_path / "dataset"
    for relative, data in [
        ("data/chunk-000/file-000.parquet", b"parquet-placeholder"),
        ("meta/info.json", b"{}"),
        ("videos/camera/file.mp4", b"video-placeholder"),
    ]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    source_video = tmp_path / "source-video.mp4"
    shutil.copyfile(root / "videos/camera/file.mp4", source_video)
    store = ArtifactStore(tmp_path / "remote-simulation")
    key = processing.retain_release(store, root, source={"repo_id": "owner/source", "revision": SHA})
    release = store.read_json(key)
    assert not any(entry["path"].startswith("videos/") for entry in release["files"])
    assert len(release["source_files"]) == 1
    shutil.rmtree(root)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **kw: str(source_video))
    restored = processing.restore_release(store, key, tmp_path / "restored")
    assert (restored / "videos/camera/file.mp4").read_bytes() == b"video-placeholder"
    assert (restored / "data/chunk-000/file-000.parquet").read_bytes() == b"parquet-placeholder"
    artifact = release["files"][0]["artifact"]
    Path(store.path(artifact["path"])).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        processing.restore_release(store, key, tmp_path / "corrupt-restore")


def test_persistent_credentials_rejected(tmp_path):
    store = ArtifactStore(tmp_path / "run")
    for value in (
        {"api_key": "literal"},
        {"url": "https://host/path?token=secret"},
        {"url": "https://user:password@host/path"},
    ):
        with pytest.raises(ValueError):
            processing.write_bundle(store, "worker", value, SHA)
    assert not store.list("bundles/*.json")


def test_hardware_admission_uses_current_inventory(monkeypatch):
    from lerobot.data_processing.types import Resources

    monkeypatch.setattr(
        processing,
        "list_jobs_hardware",
        lambda: [SimpleNamespace(name="cpu-upgrade", cpu="8 vCPU", ram="32 GB", accelerator=None)],
    )
    processing.validate_hardware("cpu-upgrade", Resources(cpus=8, memory_gb=32))
    for resource in (Resources(cpus=9), Resources(gpus=1), Resources(memory_gb=33)):
        with pytest.raises(ValueError, match="exceeds"):
            processing.validate_hardware("cpu-upgrade", resource)


def test_convert_bundle_rewrites_host_paths_and_selects_cpu(tmp_path, monkeypatch):
    from lerobot.data_processing.conversion import ConvertConfig

    cfg = ConvertConfig(
        source={
            "manifest": "/host/raw/manifest.json",
            "archive_uri": "https://host/archive.tar",
            "archive_sha256": "c" * 64,
        }
    )
    cfg.runtime.backend = "hf_jobs"
    cfg.runtime.run_uri = str(tmp_path / "simulation")
    cfg.runtime.hf_jobs.code_revision = SHA
    cfg.runtime.hf_jobs.cpu_image = "cpu@sha256:" + "b" * 64
    cfg.runtime.hf_jobs.detach = True
    captured = []
    monkeypatch.setattr(processing, "require_persistent_remote", lambda *args: None)
    monkeypatch.setattr(
        processing, "dispatch_bundle", lambda *a, **k: captured.append(k) or SimpleNamespace(id="job-1")
    )
    processing.submit_convert_to_hf(cfg)
    store = ArtifactStore(cfg.runtime.run_uri)
    bundle = store.read_json(store.list("bundles/*.json")[0])
    assert bundle["config"]["runtime"]["backend"] == "local"
    assert bundle["config"]["source"]["manifest"].startswith("inputs/")
    assert captured[0]["image"] == cfg.runtime.hf_jobs.cpu_image
    assert captured[0]["resources"].gpus == 0
