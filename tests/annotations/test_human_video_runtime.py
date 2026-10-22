# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Media survives materialization and cached resumes make no provider requests."""

import json

import pytest

pytest.importorskip("datasets")

from lerobot.annotations.processing_recipe import run_annotation_pipeline
from lerobot.annotations.steerable_pipeline.config import AnnotationPipelineConfig
from lerobot.annotations.steerable_pipeline.vlm_client import StubVlmClient

from ._helpers import StubFrameProvider, make_language_client
from .test_human_video import _Generator, responder

FACTORY = "tests.annotations.test_human_video_runtime:client"
GENERATOR = "tests.annotations.test_human_video_runtime:generator"


def client():
    return StubVlmClient(lambda messages: responder(messages) or make_language_client().responder(messages))


def generator():
    return _Generator()


def test_runtime_media_commit_and_resume(single_episode_root, monkeypatch):
    from lerobot.annotations import human_video_processing, processing

    monkeypatch.setattr(processing, "make_frame_provider", lambda *a, **k: StubFrameProvider())
    monkeypatch.setattr(human_video_processing, "make_frame_provider", lambda *a, **k: StubFrameProvider())
    cfg = AnnotationPipelineConfig()
    cfg.vlm.auto_serve = False
    cfg.plan.derive_task_from_video = "off"
    cfg.plan.n_task_rephrasings = 0
    cfg.interjections.enabled = cfg.vqa.enabled = False
    cfg.human_video.enabled = True
    cfg.human_video.max_segments_per_episode = 1
    result = run_annotation_pipeline(
        cfg, single_episode_root, client_factory=FACTORY, human_video_generator_factory=GENERATOR
    )
    manifest = single_episode_root / "meta/human_videos.jsonl"
    entries = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert [entry["status"] for entry in entries] == ["ok"]
    assert (single_episode_root / entries[0]["video_path"]).read_bytes().endswith(b"fake-video")
    assert single_episode_root / entries[0]["video_path"] in result.metadata_paths
    before = manifest.read_bytes()

    def no_requests(*args, **kwargs):
        pytest.fail("Cached episode repeated paid generation")

    monkeypatch.setattr(human_video_processing.HumanVideoStage, "setup", no_requests)
    cfg.runtime.mode = "resume"
    run_annotation_pipeline(
        cfg, single_episode_root, client_factory=FACTORY, human_video_generator_factory=GENERATOR
    )
    assert manifest.read_bytes() == before


def test_existing_subtask_changes_invalidate_media_without_rewriting_language(
    single_episode_root, monkeypatch
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from lerobot.annotations import human_video_processing, processing

    monkeypatch.setattr(processing, "make_frame_provider", lambda *a, **k: StubFrameProvider())
    monkeypatch.setattr(human_video_processing, "make_frame_provider", lambda *a, **k: StubFrameProvider())
    cfg = AnnotationPipelineConfig()
    cfg.vlm.auto_serve = False
    cfg.plan.derive_task_from_video = "off"
    cfg.plan.n_task_rephrasings = 0
    cfg.interjections.enabled = cfg.vqa.enabled = False
    run_annotation_pipeline(cfg, single_episode_root, client_factory=FACTORY)
    path = single_episode_root / "data/chunk-000/file-000.parquet"
    before = pq.read_table(path)
    cfg.plan.enabled = False
    cfg.human_video.enabled = True
    cfg.human_video.max_segments_per_episode = 1
    run_annotation_pipeline(
        cfg, single_episode_root, client_factory=FACTORY, human_video_generator_factory=GENERATOR
    )
    assert pq.read_table(path).equals(before)
    with monkeypatch.context() as context:
        context.setattr(
            human_video_processing.HumanVideoStage,
            "setup",
            lambda *a: pytest.fail("Unchanged existing subtasks repeated generation"),
        )
        run_annotation_pipeline(
            cfg, single_episode_root, client_factory=FACTORY, human_video_generator_factory=GENERATOR
        )
    rows = before["language_persistent"].to_pylist()
    for atoms in rows:
        for atom in atoms:
            if atom.get("style") == "subtask" and atom["timestamp"] == 0:
                atom["content"] = "pick up the cup"
    table = before.set_column(
        before.schema.get_field_index("language_persistent"),
        "language_persistent",
        pa.array(rows, type=before["language_persistent"].type),
    )
    pq.write_table(table, path)
    run_annotation_pipeline(
        cfg, single_episode_root, client_factory=FACTORY, human_video_generator_factory=GENERATOR
    )
    manifest = single_episode_root / "meta/human_videos.jsonl"
    assert json.loads(manifest.read_text().splitlines()[0])["subtask"] == "pick up the cup"
    assert pq.read_table(path).equals(table)
