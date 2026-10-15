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


@pytest.mark.parametrize("failure", ["window", "clock", "frames"])
def test_bad_episode_is_rejected_without_losing_batch(fixture_dataset_root, monkeypatch, failure):
    from lerobot.annotations import processing

    root = fixture_dataset_root
    path = root / "data/chunk-000/file-000.parquet"
    if failure == "window":
        original = processing.PlanSubtasksMemoryModule._subtasks_for_window

        def spans(self, record, *args, **kwargs):
            return [] if record.episode_index == 0 else original(self, record, *args, **kwargs)

        monkeypatch.setattr(processing.PlanSubtasksMemoryModule, "_subtasks_for_window", spans)
    else:
        import pyarrow as pa

        table = pq.read_table(path)
        original_native = table
        name = "timestamp" if failure == "clock" else "frame_index"
        values = table[name].to_pylist()
        values[2] = values[1]
        table = table.set_column(
            table.schema.get_field_index(name), name, pa.array(values, type=table[name].type)
        )
        pq.write_table(table, path)
    monkeypatch.setattr(processing, "make_frame_provider", lambda *a, **k: StubFrameProvider())
    cfg = config()
    cfg.interjections.enabled = True
    cfg.plan.max_frames_per_prompt = 2
    cfg.plan.window_overlap_seconds = 0
    result = run_annotation_pipeline(cfg, root, client_factory=FACTORY)
    assert result.validation_report.ok
    assert any(
        {
            "window": "empty_subtask_window",
            "clock": "invalid_episode_clock",
            "frames": "invalid_episode_frames",
        }[failure]
        in warning
        for warning in result.validation_report.warnings
    )
    assert pq.read_table(root / "data/chunk-000/file-001.parquet")["language_persistent"][0].as_py()
    if failure != "window":
        calls = []
        original_batch = processing.LanguageModule.process_batch

        def observe(self, items, context):
            calls.extend(item.key for item in items)
            return original_batch(self, items, context)

        monkeypatch.setattr(processing.LanguageModule, "process_batch", observe)
        pq.write_table(original_native, path)
        repaired = run_annotation_pipeline(cfg, root, client_factory=FACTORY)
        assert repaired.validation_report.ok and set(calls) == {"0"}


def test_external_staging_plan_does_not_write_dataset(single_episode_root, tmp_path):
    root = single_episode_root
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    cfg = config()
    cfg.staging_dir = tmp_path / "external-stage"
    cfg.runtime.mode = "plan"
    run_annotation_pipeline(cfg, root, client_factory=FACTORY)
    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()} == before
    assert list(cfg.staging_dir.rglob("plan.json"))


@pytest.mark.parametrize(
    "change", ["append", "native", "metadata", "video", "shared_file_append", "selection"]
)
def test_episode_changes_only_recompute_affected_vlm_items(
    fixture_dataset_root, tmp_path, monkeypatch, change
):
    import json

    import pyarrow as pa

    from lerobot.annotations import processing
    from tests.fixtures.dataset_factories import build_annotation_dataset

    root = fixture_dataset_root
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    camera = "observation.images.top"
    if change == "video":
        info["features"][camera] = {"dtype": "video", "shape": [24, 32, 3], "names": None}
        info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        metadata = []
        for ep in (0, 1):
            video = root / f"videos/{camera}/chunk-000/file-{ep:03d}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(f"video-{ep}".encode())
            metadata.append(
                {
                    "episode_index": ep,
                    f"videos/{camera}/chunk_index": 0,
                    f"videos/{camera}/file_index": ep,
                    f"videos/{camera}/from_timestamp": 0.0,
                    f"videos/{camera}/to_timestamp": 1.2,
                }
            )
        path = root / "meta/episodes/chunk-000/file-000.parquet"
        path.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(metadata), path)
        info_path.write_text(json.dumps(info))
    first_path = root / "data/chunk-000/file-000.parquet"
    second_path = first_path.with_name("file-001.parquet")
    if change == "shared_file_append":
        pq.write_table(pa.concat_tables([pq.read_table(first_path), pq.read_table(second_path)]), first_path)
        second_path.unlink()
    calls = []
    original = processing.LanguageModule.process_batch

    def observe(self, items, context):
        calls.extend((self.phase, item.key) for item in items)
        return original(self, items, context)

    monkeypatch.setattr(processing.LanguageModule, "process_batch", observe)
    monkeypatch.setattr(processing, "make_frame_provider", lambda *a, **k: StubFrameProvider())
    cfg = config()
    cfg.interjections.enabled = cfg.vqa.enabled = True
    cfg.interjections.max_interjections_per_episode = 1
    cfg.interjections.interjection_min_t = 0.5
    cfg.vqa.K, cfg.vqa.vqa_emission_hz, cfg.vqa.question_types = 1, 1, ("count",)
    if change == "selection":
        cfg.only_episodes = (0,)
    run_annotation_pipeline(cfg, root, client_factory=FACTORY)
    calls.clear()
    if change in {"append", "shared_file_append"}:
        other = build_annotation_dataset(
            tmp_path / "new", [(2, 12, "Could you tidy the kitchen please?")], fps=10
        )
        new = pq.read_table(other / "data/chunk-000/file-002.parquet")
        target = first_path if change == "shared_file_append" else first_path.with_name("file-002.parquet")
        if change == "shared_file_append":
            old = pq.read_table(first_path)
            for name in set(old.column_names) - set(new.column_names):
                new = new.append_column(name, _language_array([[]] * len(new), old[name].type))
            new = pa.concat_tables([old, new.select(old.column_names)])
        pq.write_table(new, target)
        info = json.loads(info_path.read_text())
        info["total_episodes"] += 1
        info_path.write_text(json.dumps(info))
        expected = {"2"}
    elif change == "native":
        table = pq.read_table(second_path)
        times = table["timestamp"].to_pylist()
        times[2] += 0.01
        table = table.set_column(
            table.schema.get_field_index("timestamp"),
            "timestamp",
            pa.array(times, type=table["timestamp"].type),
        )
        pq.write_table(table, second_path)
        expected = {"1"}
    elif change == "metadata":
        path = root / "meta/episodes/notes.txt"
        path.parent.mkdir(parents=True)
        path.write_text("Administrative note, not a model input")
        expected = set()
    elif change == "selection":
        cfg.only_episodes = None
        expected = {"1"}
    else:
        (root / f"videos/{camera}/chunk-000/file-000.mp4").write_bytes(b"reencoded-video-0")
        expected = {"0"}
    run_annotation_pipeline(cfg, root, client_factory=FACTORY)
    assert {key for _, key in calls} == expected
    for phase in ("plan", "vqa", "interjections", "plan_update"):
        assert {key for name, key in calls if name == phase} == expected


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


def test_materialized_cache_pruning_keeps_inference_resume(single_episode_root, caplog):
    cfg = config()
    cfg.runtime.retain_materialized_outputs = False
    CLIENTS.clear()
    run_annotation_pipeline(cfg, single_episode_root, client_factory=FACTORY)
    count = len(CLIENTS)
    assert not list((single_episode_root / ".annotate_staging/prepared").rglob("*-*"))
    run_annotation_pipeline(cfg, single_episode_root, client_factory=FACTORY)
    assert len(CLIENTS) == count
    run_annotation_pipeline(cfg, single_episode_root, client_factory=FACTORY)
    assert len(CLIENTS) == count
    assert "Invalid artifact checksum" not in caplog.text
    assert not list((single_episode_root / ".annotate_staging/prepared").iterdir())


def test_consolidated_windows_preserve_unselected_episodes(fixture_dataset_root):
    import pyarrow as pa

    root = fixture_dataset_root
    first = root / "data/chunk-000/file-000.parquet"
    second = first.with_name("file-001.parquet")
    pq.write_table(pa.concat_tables([pq.read_table(first), pq.read_table(second)]), first)
    second.unlink()
    cfg = config()
    run_annotation_pipeline(cfg, root, client_factory=FACTORY)
    paths = list((root / "meta/annotations/windows").rglob("*.parquet"))
    assert paths == [root / "meta/annotations/windows/chunk-000/file-000.parquet"]
    before = pq.read_table(paths[0]).to_pylist()
    assert {row["episode_index"] for row in before} == {0, 1}
    cfg.only_episodes = (0,)
    cfg.plan.emit_memory = False
    run_annotation_pipeline(cfg, root, client_factory=FACTORY)
    after = pq.read_table(paths[0]).to_pylist()
    assert [row for row in after if row["episode_index"] == 1] == [
        row for row in before if row["episode_index"] == 1
    ]
    assert len(after) == len(before)


def test_unexpected_module_errors_still_fail_fast(single_episode_root, monkeypatch):
    from lerobot.annotations import processing

    def bug(*args):
        raise ValueError("unexpected module bug")

    monkeypatch.setattr(processing.PlanSubtasksMemoryModule, "run_episode", bug)
    with pytest.raises(ValueError, match="unexpected module bug"):
        run_annotation_pipeline(config(), single_episode_root, client_factory=FACTORY)


def test_native_episode_fingerprints_survive_row_group_rewrite(tmp_path):
    import pyarrow as pa

    from lerobot.annotations.processing_source import LeRobotEpisodeSource
    from lerobot.datasets.io_utils import write_table_one_row_group_per_episode
    from tests.fixtures.dataset_factories import build_annotation_dataset

    root = build_annotation_dataset(tmp_path / "data", [(0, 1200, "Task"), (1, 700, "Task")], fps=10)
    first = root / "data/chunk-000/file-000.parquet"
    second = first.with_name("file-001.parquet")
    table = pa.concat_tables([pq.read_table(first), pq.read_table(second)])
    pq.write_table(table, first)
    second.unlink()
    before = LeRobotEpisodeSource(root)
    write_table_one_row_group_per_episode(table, first)
    after = LeRobotEpisodeSource(root)
    assert [episode["native_parts"] for episode in before.episodes.values()] == [
        episode["native_parts"] for episode in after.episodes.values()
    ]


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
    rows = pq.read_table(
        single_episode_root / "meta/annotations/windows/chunk-000/file-000.parquet"
    ).to_pylist()
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
