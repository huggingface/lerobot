# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Real Parquet migration, checkpoint resume and publisher-label ownership."""

import pytest

pytest.importorskip("datasets")
import pyarrow.parquet as pq  # noqa: E402

from lerobot.annotations.processing import _language_array, run_annotation_pipeline  # noqa: E402
from lerobot.annotations.steerable_pipeline.config import AnnotationPipelineConfig  # noqa: E402
from lerobot.datasets.language import language_persistent_arrow_type  # noqa: E402

from ._helpers import StubFrameProvider, make_language_client  # noqa: E402

FACTORY = "tests.annotations.test_processing:stub_client"
CLIENTS = []


def stub_client():
    client = make_language_client()
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


def test_semantic_cache_survives_relocation_and_endpoint_changes(single_episode_root, tmp_path, monkeypatch):
    import shutil

    root = single_episode_root
    moved = tmp_path / "moved"
    shutil.copytree(root, moved)
    cfg = config()
    cfg.runtime.run_uri = str(tmp_path / "shared-cache")
    CLIENTS.clear()
    run_annotation_pipeline(cfg, root, client_factory=FACTORY)
    count = len(CLIENTS)
    cfg.vlm.api_base = "http://another-host:9000/v1"
    cfg.vlm.serve_port = 9000
    cfg.vlm.api_key_env = "OTHER_TOKEN"
    monkeypatch.setenv("OTHER_TOKEN", "EMPTY")
    cfg.executor.episode_parallelism = 2
    run_annotation_pipeline(cfg, moved, client_factory=FACTORY)
    assert len(CLIENTS) == count
    original = pq.read_table(root / "data/chunk-000/file-000.parquet")
    assert pq.read_table(moved / "data/chunk-000/file-000.parquet").equals(original)


def test_prompt_override_invalidates_affected_inference(single_episode_root, monkeypatch):
    from lerobot.annotations.steerable_pipeline.prompts import load

    cfg = config()
    CLIENTS.clear()
    run_annotation_pipeline(cfg, single_episode_root, client_factory=FACTORY)
    count = len(CLIENTS)
    monkeypatch.setenv("LEROBOT_PROMPT_OVERRIDE_plan_subtasks", load("plan_subtasks") + "\nBe precise.")
    run_annotation_pipeline(cfg, single_episode_root, client_factory=FACTORY)
    assert len(CLIENTS) == count + 1


def test_unchanged_source_fingerprints_avoid_media_reads(single_episode_root, monkeypatch):
    from lerobot.annotations import processing_source

    root = single_episode_root
    video = root / "videos/camera/test.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"source-video")
    first = processing_source.LeRobotEpisodeSource(root)
    real_checksum = processing_source.file_checksum
    reads = []

    def checksum(path):
        if path == video:
            reads.append(path)
        return real_checksum(path)

    monkeypatch.setattr(processing_source, "file_checksum", checksum)
    assert processing_source.LeRobotEpisodeSource(root).dataset_ref == first.dataset_ref
    assert not reads
    assert processing_source.LeRobotEpisodeSource(root, verify_sources=True).dataset_ref == first.dataset_ref
    assert reads == [video]
    video.write_bytes(b"changed-vid!")
    assert processing_source.LeRobotEpisodeSource(root).dataset_ref != first.dataset_ref
    assert reads == [video, video]


def test_rejected_episode_does_not_discard_healthy_batch(fixture_dataset_root, monkeypatch):
    from lerobot.annotations import processing

    original = processing.PlanSubtasksMemoryModule.run_episode

    def run(self, record, staging):
        if record.episode_index != 0:
            original(self, record, staging)

    monkeypatch.setattr(processing.PlanSubtasksMemoryModule, "run_episode", run)
    cfg = config()
    cfg.interjections.enabled = True
    cfg.runtime.batch_size = 16
    root = fixture_dataset_root
    result = run_annotation_pipeline(cfg, root, client_factory=FACTORY)
    assert result.validation_report.ok
    assert any("no_subtasks" in warning for warning in result.validation_report.warnings)
    phases = {phase.name: phase for phase in result.phases}
    assert phases["plan"].episodes_processed == 1 and phases["plan"].episodes_skipped == 1
    assert phases["interjections"].episodes_skipped == 1
    assert (
        pq.read_table(root / "data/chunk-000/file-000.parquet")["language_persistent"].to_pylist()
        == [[]] * 12
    )
    assert pq.read_table(root / "data/chunk-000/file-001.parquet")["language_persistent"][0].as_py()


def test_materialized_cache_pruning_keeps_inference_resume(single_episode_root):
    cfg = config()
    cfg.runtime.retain_materialized_outputs = False
    CLIENTS.clear()
    run_annotation_pipeline(cfg, single_episode_root, client_factory=FACTORY)
    count = len(CLIENTS)
    assert not list((single_episode_root / ".annotate_staging/prepared").rglob("*-*"))
    run_annotation_pipeline(cfg, single_episode_root, client_factory=FACTORY)
    assert len(CLIENTS) == count


@pytest.mark.parametrize("version", ["v2.1", "v4.0", None, 3])
@pytest.mark.parametrize("mode", ["plan", "run"])
def test_runtime_rejects_non_v3_before_planning_or_inference(single_episode_root, version, mode):
    import json

    root = single_episode_root
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    if version is None:
        info.pop("codebase_version")
    else:
        info["codebase_version"] = version
    info_path.write_text(json.dumps(info))
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    cfg = config()
    cfg.runtime.mode = mode
    CLIENTS.clear()
    with pytest.raises(ValueError, match="LeRobot v3"):
        run_annotation_pipeline(cfg, root, client_factory=FACTORY)
    assert not CLIENTS
    assert not list(root.rglob("plan.json"))
    assert {path: path.read_bytes() for path in root.rglob("*") if path.is_file()} == before


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


@pytest.mark.parametrize("enabled", [("plan", "interjections", "vqa"), ("plan",), ("vqa",)])
def test_enabled_families_roundtrip_and_render(single_episode_root, monkeypatch, enabled):
    import json

    from lerobot.annotations import processing
    from lerobot.datasets.feature_utils import get_hf_features_from_features
    from lerobot.datasets.io_utils import load_info, load_nested_dataset
    from lerobot.datasets.language_render import render_sample
    from lerobot.datasets.recipe import MessageTurn, TrainingRecipe

    path = single_episode_root / "data/chunk-000/file-000.parquet"
    native = pq.read_table(path)
    info_path = single_episode_root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"] = {
        field.name: {"dtype": str(field.type), "shape": [1], "names": None} for field in native.schema
    }
    info_path.write_text(json.dumps(info))
    # The text-only fixture has no cameras; provide images so VQA cannot pass as a no-op.
    provider = StubFrameProvider()
    monkeypatch.setattr(processing, "make_frame_provider", lambda *a, **k: provider)
    cfg = config()
    for name in ("plan", "interjections", "vqa"):
        getattr(cfg, name).enabled = name in enabled
    cfg.plan.n_task_rephrasings = 1
    cfg.interjections.max_interjections_per_episode = 1
    cfg.interjections.interjection_min_t = 0.5
    cfg.vqa.vqa_emission_hz = 1
    cfg.vqa.K = 1
    cfg.vqa.question_types = ("count",)
    result = run_annotation_pipeline(cfg, single_episode_root, client_factory=FACTORY)
    assert result.validation_report.ok
    table = pq.read_table(path)
    assert table.select(native.column_names).equals(native)
    rows = list(
        load_nested_dataset(
            single_episode_root / "data",
            features=get_hf_features_from_features(load_info(single_episode_root).features),
        )
    )
    expected_persistent = {"subtask", "plan", "memory", "task_aug"} if "plan" in enabled else set()
    expected_events = ({"vqa"} if "vqa" in enabled else set()) | (
        {"interjection", None} if "interjections" in enabled else set()
    )
    assert {atom["style"] for row in rows for atom in row["language_persistent"]} == expected_persistent
    assert {atom["style"] for row in rows for atom in row["language_events"]} == expected_events
    assert all(row["language_persistent"] == rows[0]["language_persistent"] for row in rows)
    if "interjections" in enabled:
        interjected = [
            row for row in rows if any(a["style"] == "interjection" for a in row["language_events"])
        ]
        assert len(interjected) == 1
        assert any(
            a["style"] == "plan" and a["timestamp"] == pytest.approx(interjected[0]["timestamp"])
            for a in interjected[0]["language_persistent"]
        )
    for family, binding in (("plan", "subtask"), ("vqa", "vqa"), ("interjections", "speech")):
        if family not in enabled:
            continue
        recipe = TrainingRecipe(
            messages=[
                MessageTurn("user", "${task}", stream="high_level"),
                MessageTurn(
                    "assistant",
                    f"${{{binding}}}",
                    stream="high_level",
                    target=True,
                    if_present=binding,
                    tool_calls_from="speech" if binding == "speech" else None,
                ),
            ]
        )
        rendered = [
            render_sample(
                recipe=recipe,
                persistent=row["language_persistent"],
                events=row["language_events"],
                t=row["timestamp"],
                sample_idx=i,
                task="Pour water.",
            )
            for i, row in enumerate(rows)
        ]
        targets = [
            sample["messages_rendered"][-1]
            for sample in rendered
            if sample and sample["target_message_indices"]
        ]
        assert targets, f"No training targets rendered for {family}"
        if binding == "speech":
            assert all(t["tool_calls"][0]["function"]["name"] == "say" for t in targets)
        else:
            assert all(t["content"] for t in targets)


def test_plan_mode_does_not_construct_models(single_episode_root):
    cfg = config()
    cfg.runtime.mode = "plan"
    CLIENTS.clear()
    run_annotation_pipeline(cfg, single_episode_root, client_factory=FACTORY)
    assert not CLIENTS
    assert list(single_episode_root.rglob("plan.json"))


def test_window_provenance_keeps_exact_source_references(single_episode_root):
    cfg = config()
    run_annotation_pipeline(cfg, single_episode_root, client_factory=FACTORY)
    rows = pq.read_table(single_episode_root / "meta/annotations/windows/episode-0.parquet").to_pylist()
    native = pq.read_table(single_episode_root / "data/chunk-000/file-000.parquet")
    owned = [row for row in rows if row["owned"]]
    assert [row["frame_index"] for row in owned] == native["frame_index"].to_pylist()
    assert [row["timestamp"] for row in owned] == native["timestamp"].to_pylist()


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

    from lerobot.annotations.runtime.artifacts import ArtifactStore
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
