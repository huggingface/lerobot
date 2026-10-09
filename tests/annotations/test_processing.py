# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Real Parquet migration, checkpoint resume and publisher-label ownership."""

import pytest

pytest.importorskip("datasets")
import pyarrow.parquet as pq  # noqa: E402

from lerobot.annotations.processing import _language_array, run_annotation_pipeline  # noqa: E402
from lerobot.annotations.steerable_pipeline.config import AnnotationPipelineConfig  # noqa: E402
from lerobot.datasets.language import language_persistent_arrow_type  # noqa: E402
from tests.annotations.test_pipeline_recipe_render import _build_executor  # noqa: E402

FACTORY = "tests.annotations.test_processing:stub_client"
CLIENTS = []


def stub_client():
    client = _build_executor().plan.vlm
    responder = client.responder

    def joined_prompt(messages):
        flattened = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                content = "\n".join(block.get("text", "") for block in content if block.get("type") == "text")
            flattened.append({**message, "content": content})
        text = "\n".join(str(message.get("content", "")) for message in flattened)
        if '"subtasks"' in text:
            return {
                "subtasks": [
                    {"text": "grasp bottle", "start": 0.0, "end": 0.5},
                    {"text": "pour water", "start": 0.5, "end": 2.9},
                ]
            }
        return responder(flattened)

    client.responder = joined_prompt
    CLIENTS.append(client)
    return client


def config():
    cfg = AnnotationPipelineConfig()
    cfg.vlm.auto_serve = False
    cfg.plan.derive_task_from_video = "off"
    cfg.plan.n_task_rephrasings = 0
    cfg.interjections.enabled = False
    cfg.vqa.enabled = False
    return cfg


def test_runtime_preserves_source_labels_and_resumes(single_episode_root):
    root = single_episode_root
    path = root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    publisher = {
        "role": "assistant",
        "content": "Publisher subtask",
        "style": "subtask",
        "timestamp": 0.0,
        "camera": None,
        "tool_calls": None,
    }
    table = table.append_column(
        "language_persistent",
        _language_array([[publisher]] * table.num_rows, language_persistent_arrow_type()),
    )
    pq.write_table(table, path)
    cfg = config()
    CLIENTS.clear()
    summary = run_annotation_pipeline(cfg, root, client_factory=FACTORY)
    assert summary.validation_report.ok
    first = pq.read_table(path)
    assert first.select(table.column_names[:-1]).equals(table.select(table.column_names[:-1]))
    assert "subtask_index" in first.column_names
    atoms = first["language_persistent"].to_pylist()[0]
    assert atoms[0] == publisher
    assert len(atoms) > 1
    count = len(CLIENTS)
    run_annotation_pipeline(cfg, root, client_factory=FACTORY)
    assert len(CLIENTS) == count  # no model construction on a cached run
    assert pq.read_table(path).equals(first)


def test_disabled_families_and_unselected_episodes(fixture_dataset_root):
    import json

    from lerobot.datasets.feature_utils import get_hf_features_from_features
    from lerobot.datasets.io_utils import load_info, load_nested_dataset

    root = fixture_dataset_root
    untouched = root / "data/chunk-000/file-001.parquet"
    before = pq.read_table(untouched)
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"] = {
        field.name: {"dtype": str(field.type), "shape": [1], "names": None} for field in before.schema
    }
    info_path.write_text(json.dumps(info))
    cfg = config()
    cfg.only_episodes = (0,)
    run_annotation_pipeline(cfg, root, client_factory=FACTORY)
    after = pq.read_table(untouched)
    assert after.select(before.column_names).equals(before)
    assert after["language_persistent"].to_pylist() == [[]] * len(before)
    assert after["language_events"].to_pylist() == [[]] * len(before)
    # A partial annotation must remain readable as one dataset, including files
    # that have no generated labels. This failed with nested JSON Arrow types.
    loaded = load_nested_dataset(
        root / "data", features=get_hf_features_from_features(load_info(root).features)
    )
    assert len(loaded) == 24
    assert loaded[12]["language_persistent"] == []
    normalized = untouched.read_bytes()
    run_annotation_pipeline(cfg, root, client_factory=FACTORY)
    assert untouched.read_bytes() == normalized  # no rewrite once its schema is consistent
    cfg.plan.enabled = False
    result = run_annotation_pipeline(cfg, root, client_factory=FACTORY)
    assert not result.written_paths


def test_all_language_families_roundtrip(single_episode_root):
    cfg = config()
    cfg.interjections.enabled = True
    cfg.interjections.max_interjections_per_episode = 1
    cfg.interjections.interjection_min_t = 0.5
    cfg.vqa.enabled = True
    cfg.vqa.vqa_emission_hz = 1
    result = run_annotation_pipeline(cfg, single_episode_root, client_factory=FACTORY)
    assert result.validation_report.ok
    table = pq.read_table(result.written_paths[0])
    assert any(atoms for atoms in table["language_events"].to_pylist())


def test_plan_mode_does_not_construct_models(single_episode_root):
    cfg = config()
    cfg.runtime.mode = "plan"
    CLIENTS.clear()
    run_annotation_pipeline(cfg, single_episode_root, client_factory=FACTORY)
    assert not CLIENTS
    assert list(single_episode_root.rglob("plan.json"))


def test_quality_mask_skips_inference_and_preserves_native_rows(single_episode_root, monkeypatch):
    import torch

    from lerobot.annotations import processing

    class BlackFrames:
        def frames_at(self, record, timestamps):
            return [torch.zeros(3, 16, 16, dtype=torch.uint8) for _ in timestamps]

    monkeypatch.setattr(processing, "make_frame_provider", lambda *args, **kwargs: BlackFrames())
    cfg = config()
    cfg.quality.enabled = True
    cfg.interjections.enabled = True
    cfg.vqa.enabled = True
    CLIENTS.clear()
    path = single_episode_root / "data/chunk-000/file-000.parquet"
    before = pq.read_table(path)
    result = run_annotation_pipeline(cfg, single_episode_root, client_factory=FACTORY)
    assert not CLIENTS
    after = pq.read_table(path)
    assert after.select(before.column_names).equals(before)
    assert after["language_persistent"].to_pylist() == [[]] * len(before)
    quality = pq.read_table(single_episode_root / "meta/annotations/quality/episode-0.parquet").to_pylist()[0]
    assert not quality["usable"] and quality["reason"] == "black_frame_fraction"
    assert quality["timestamps"] == before["timestamp"].to_pylist()
    assert all(
        phase.episodes_skipped == 1
        for phase in result.phases
        if phase.name in {"plan", "vqa", "interjections", "plan_update"}
    )


def test_window_provenance_keeps_exact_source_references(single_episode_root):
    cfg = config()
    run_annotation_pipeline(cfg, single_episode_root, client_factory=FACTORY)
    rows = pq.read_table(single_episode_root / "meta/annotations/windows/episode-0.parquet").to_pylist()
    native = pq.read_table(single_episode_root / "data/chunk-000/file-000.parquet")
    owned = [row for row in rows if row["owned"]]
    assert [row["frame_index"] for row in owned] == native["frame_index"].to_pylist()
    assert [row["timestamp"] for row in owned] == native["timestamp"].to_pylist()


def test_masked_episodes_never_auto_start_model(single_episode_root, monkeypatch):
    import torch

    from lerobot.annotations import processing

    class BlackFrames:
        def frames_at(self, record, timestamps):
            return [torch.zeros(3, 8, 8, dtype=torch.uint8) for _ in timestamps]

    monkeypatch.setattr(processing, "make_frame_provider", lambda *args, **kwargs: BlackFrames())

    def forbidden(*args, **kwargs):
        raise AssertionError("Masked episode started inference service")

    monkeypatch.setattr(processing, "make_vlm_client", forbidden)
    cfg = config()
    cfg.vlm.auto_serve = True
    cfg.quality.enabled = True
    assert run_annotation_pipeline(cfg, single_episode_root).validation_report.ok


def test_split_episode_and_hf_reader(single_episode_root):
    import json

    from lerobot.datasets.feature_utils import get_hf_features_from_features
    from lerobot.datasets.io_utils import load_info, load_nested_dataset

    root = single_episode_root
    path = root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    pq.write_table(table.slice(0, 15), path)
    pq.write_table(table.slice(15), path.with_name("file-001.parquet"))
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"] = {
        field.name: {"dtype": str(field.type), "shape": [1], "names": None} for field in table.schema
    }
    info_path.write_text(json.dumps(info))
    cfg = config()
    cfg.runtime.workers = 2
    cfg.runtime.shard_size = 1
    result = run_annotation_pipeline(cfg, root, client_factory=FACTORY)
    assert len(result.written_paths) == 2
    loaded = load_nested_dataset(
        root / "data", features=get_hf_features_from_features(load_info(root).features)
    )
    assert len(loaded) == 30
    assert loaded[0]["language_persistent"][0]["style"] == "subtask"


def test_partial_update_preserves_disabled_families(single_episode_root):
    cfg = config()
    cfg.interjections.enabled = True
    cfg.interjections.max_interjections_per_episode = 1
    cfg.interjections.interjection_min_t = 0.5
    run_annotation_pipeline(cfg, single_episode_root, client_factory=FACTORY)
    path = single_episode_root / "data/chunk-000/file-000.parquet"
    before = pq.read_table(path)["language_events"].to_pylist()
    cfg.interjections.enabled = False
    cfg.plan.emit_memory = False
    run_annotation_pipeline(cfg, single_episode_root, client_factory=FACTORY)
    assert pq.read_table(path)["language_events"].to_pylist() == before


def test_serialized_job_keeps_real_enriched_parquet(single_episode_root, tmp_path, monkeypatch):
    import shutil

    import draccus

    from lerobot.data_processing.artifacts import ArtifactStore
    from lerobot.jobs import processing
    from lerobot.scripts import lerobot_annotate

    snapshot = tmp_path / ("a" * 40)
    shutil.copytree(single_episode_root, snapshot)
    monkeypatch.setattr(lerobot_annotate, "snapshot_download", lambda **kwargs: str(snapshot))
    monkeypatch.setattr(lerobot_annotate, "HF_LEROBOT_HOME", tmp_path / "cache")
    monkeypatch.setattr(
        lerobot_annotate,
        "run_annotation_pipeline",
        lambda cfg, root: run_annotation_pipeline(cfg, root, client_factory=FACTORY),
    )
    cfg = config()
    cfg.repo_id, cfg.revision = "owner/source", "a" * 40
    cfg.runtime.run_uri = str(tmp_path / "persistent-job-store")
    store = ArtifactStore(cfg.runtime.run_uri)
    key, digest = processing.write_bundle(store, "annotate", draccus.encode(cfg), "b" * 40)
    monkeypatch.setenv("LEROBOT_PROCESSING_CODE_REVISION", "b" * 40)
    release = processing.execute_bundle(store, key, digest)
    restored = processing.restore_release(store, release, tmp_path / "restored")
    table = pq.read_table(restored / "data/chunk-000/file-000.parquet")
    assert table.num_rows == 30 and table["language_persistent"].to_pylist()[0]
    assert (
        "language_persistent" not in pq.read_table(snapshot / "data/chunk-000/file-000.parquet").column_names
    )


def test_unrelated_vqa_settings_do_not_repeat_subtask_inference(single_episode_root):
    cfg = config()
    CLIENTS.clear()
    run_annotation_pipeline(cfg, single_episode_root, client_factory=FACTORY)
    count = len(CLIENTS)
    cfg.vqa.question_types = ("attribute",)
    cfg.vqa.K = 4
    run_annotation_pipeline(cfg, single_episode_root, client_factory=FACTORY)
    assert len(CLIENTS) == count


def test_model_revision_invalidates_subtask_inference(single_episode_root):
    cfg = config()
    CLIENTS.clear()
    run_annotation_pipeline(cfg, single_episode_root, client_factory=FACTORY)
    count = len(CLIENTS)
    cfg.vlm.model_revision = "a" * 40
    run_annotation_pipeline(cfg, single_episode_root, client_factory=FACTORY)
    assert len(CLIENTS) > count
