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
