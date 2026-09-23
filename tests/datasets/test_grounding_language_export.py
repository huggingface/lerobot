# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
import copy
import json
import runpy
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot.annotations.steerable_pipeline.reader import iter_episodes
from lerobot.annotations.steerable_pipeline.staging import EpisodeStaging
from lerobot.annotations.steerable_pipeline.validator import StagingValidator
from lerobot.datasets.language import language_events_arrow_type
from lerobot.datasets.language_render import render_sample
from lerobot.datasets.recipe import MessageTurn, TrainingRecipe


@pytest.fixture
def sample(tmp_path):
    api = runpy.run_path(
        str(Path(__file__).parents[2] / "examples/rebot_agent/export_language_annotations.py")
    )
    dataset, extraction = tmp_path / "source", tmp_path / "extract"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "data/chunk-000").mkdir(parents=True)
    directory = extraction / "clip"
    (directory / "frames").mkdir(parents=True)
    (directory / "masks").mkdir()
    source = {"repo_id": "test/rebot", "revision": "pinned"}
    (dataset / "source.json").write_text(json.dumps(source))
    (dataset / "meta/info.json").write_text(
        json.dumps(
            {
                "features": {
                    "observation.images.base": {"shape": [6, 8, 3], "names": ["height", "width", "channels"]}
                }
            }
        )
    )
    persistent = [
        {
            "role": "assistant",
            "style": "subtask",
            "content": "Pick up tape",
            "timestamp": 0.0,
            "camera": None,
            "tool_calls": None,
        }
    ]
    speech = {
        "role": "assistant",
        "content": None,
        "style": None,
        "camera": None,
        "tool_calls": [{"type": "function", "function": {"name": "say", "arguments": {"text": "Hello"}}}],
    }
    table = pa.table(
        {
            "episode_index": [0, 0, 0, 0],
            "frame_index": [0, 1, 2, 3],
            "timestamp": [0.0, 0.5, 1.0, 1.5],
            "action": [[1, 2], [2, 3], [4, 5], [6, 7]],
            "task_index": [0] * 4,
            "language_persistent": [persistent] * 4,
            "language_events": [[speech], [], [], []],
        }
    )
    path = dataset / "data/chunk-000/file-000.parquet"
    pq.write_table(table, path)
    frames = []
    for i in range(3):
        image = directory / "frames" / f"{i:06d}.jpg"
        image.write_bytes(b"fixture image")
        (directory / "masks" / f"{i}.png").write_bytes(b"fixture mask")
        frames.append({"frame_index": i, "timestamp": i / 2, "sha256": api["digest"](image)})
    manifest = {
        "source": source,
        "models": {"point": "pinned", "track": "pinned"},
        "clips": [
            {
                "path": "clip",
                "episode_index": 0,
                "start_frame": 0,
                "end_frame": 3,
                "camera": "observation.images.base",
                "image_size": [8, 6],
                "frames": frames,
            }
        ],
    }
    (extraction / "extraction.json").write_text(json.dumps(manifest))
    (directory / "tracks.json").write_text("{}")
    selected = {
        "source_manifest_sha256": api["digest"](extraction / "extraction.json"),
        "source_tracks_sha256": api["digest"](directory / "tracks.json"),
        "objects": [{"object_id": 1, "name": "tape", "point": [2, 2]}],
        "frames": [
            {
                **f,
                "objects": [
                    {
                        "object_id": 1,
                        "mask_present": i != 1,
                        "centroid": [2.0, 2.0] if i != 1 else None,
                        "bbox_xyxy": [1, 1, 4, 4] if i != 1 else None,
                        "mask_path": f"masks/{i}.png",
                        "area_fraction": 0.2 if i != 1 else 0.0,
                    }
                ],
            }
            for i, f in enumerate(frames)
        ],
    }
    (directory / "task_objects.json").write_text(json.dumps(selected))
    return api, dataset, extraction, tmp_path / "out", path


def test_native_seed_correction_retains_model_attribution_only_at_anchor(sample):
    api, dataset, extraction, output, path = sample
    selected_path = extraction / "clip/task_objects.json"
    selected = json.loads(selected_path.read_text())
    review = {"reviewer": {"kind": "model", "id": "test"}, "accepted_training_labels": False}
    selected["objects"][0].update(point_source="model_review", seed_review=review)
    selected_path.write_text(json.dumps(selected))
    api["export_dataset"](dataset, [extraction], output)
    events = pq.read_table(output / "data/chunk-000/file-000.parquet")["language_events"].to_pylist()
    for index, rows in enumerate(events[:3]):
        obj = json.loads(
            next(r["content"] for r in rows if r["style"] == "vqa" and r["role"] == "assistant")
        )["detections"][0]
        assert obj["seed_point_source"] == ("model_review" if index == 0 else None)
        assert obj["seed_review"] == (review if index == 0 else None)


def test_native_temporal_prompt_stays_on_its_frame_with_complete_offline_provenance(sample):
    api, dataset, extraction, output, path = sample
    selected_path = extraction / "clip/task_objects.json"
    selected = json.loads(selected_path.read_text())
    prompt = {
        "frame_index": 2,
        "point": [2, 2],
        "review": {"reviewer": {"kind": "model", "id": "test"}, "accepted_training_labels": False},
    }
    selected["objects"][0]["tracking_prompts"] = [prompt]
    selected_path.write_text(json.dumps(selected))
    _, provenance = api["collect_candidates"]([extraction], json.loads((dataset / "source.json").read_text()))
    assert provenance[0]["clips"][0]["tracking_prompts"] == {"1": [prompt]}
    api["export_dataset"](dataset, [extraction], output)
    events = pq.read_table(output / "data/chunk-000/file-000.parquet")["language_events"].to_pylist()
    for index, rows in enumerate(events[:3]):
        obj = json.loads(
            next(r["content"] for r in rows if r["style"] == "vqa" and r["role"] == "assistant")
        )["detections"][0]
        assert obj["tracking_prompt"] == (prompt if index == 2 else None)
        assert obj["review"] == "pending"
        assert obj["mask_present"] == (index != 1)


@pytest.fixture
def point_bindings(sample):
    api, dataset, extraction, output, path = sample
    selected_path = extraction / "clip/task_objects.json"
    selected = json.loads(selected_path.read_text())
    selected["objects"].append({"object_id": 2, "name": "bin", "point": [7, 3]})
    for i, frame in enumerate(selected["frames"]):
        frame["objects"][0].update(mask_present=True, centroid=[2.2 + i, 2.6], bbox_xyxy=[1, 1, 6, 5])
        frame["objects"].append(
            {
                "object_id": 2,
                "mask_present": True,
                "centroid": [7.75, 3.0],
                "bbox_xyxy": [7, 2, 8, 4],
                "mask_path": f"masks/{i}.png",
                "area_fraction": 0.1,
            }
        )
    selected_path.write_text(json.dumps(selected))
    api["export_dataset"](dataset, [extraction], output)
    prefix = api["digest"](extraction / "extraction.json") + ":clip:"
    bindings = {
        "source": json.loads((output / "source.json").read_text()),
        "grounding_provenance_sha256": api["digest"](output / "meta/grounding_provenance.json"),
        "data_sha256": {
            str(p.relative_to(output)): api["digest"](p) for p in (output / "data").rglob("*.parquet")
        },
        "segments": [
            {
                "episode_index": 0,
                "start_frame": 0,
                "end_frame": 3,
                "subtask": "Put tape in the bin",
                "subtask_evidence": {"source": "fixture instruction"},
                "camera": "observation.images.base",
                "image_size": [8, 6],
                "objects": [
                    {"role": "pick", "object_id": prefix + "1", "name": "tape"},
                    {"role": "place", "object_id": prefix + "2", "name": "bin"},
                ],
                "role_review": {
                    "verdict": "accepted",
                    "reviewer": {"kind": "model", "id": "test"},
                    "notes": "Role-only fixture review",
                },
            }
        ],
    }
    compiler = runpy.run_path(
        str(Path(__file__).parents[2] / "examples/rebot_agent/compile_point_features.py")
    )
    return compiler, output, bindings


def test_native_points_compile_in_reviewed_order_without_approving_or_mutating(point_bindings):
    compiler, root, bindings = point_bindings
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    features, report = compiler["compile_features"](root, bindings)
    assert all(p.read_bytes() == content for p, content in before.items())
    assert not features["accepted_training_labels"] and not features["human_verified"]
    assert report["segments"][0]["status"] == "candidate_ready_for_command_review"
    segment = features["segments"][0]
    target = segment["views"][0]["targets"][0]
    assert target["points_by_frame"] == {"0": [[2, 3], [7, 3]], "1": [[3, 3], [7, 3]], "2": [[4, 3], [7, 3]]}
    assert target["evidence"]["role_review"] == bindings["segments"][0]["role_review"]
    assert "motions" not in segment and "traces" not in segment["views"][0]
    # The normal compiler still needs a separate command review before the training index accepts it.
    compose = runpy.run_path(str(Path(__file__).parents[2] / "examples/rebot_agent/prepare_steering.py"))[
        "compose_segment"
    ]
    from lerobot.datasets.steering_commands import SteeringCommands

    commands = compose(segment)
    with pytest.raises(ValueError, match="accepted, attributed review"):
        SteeringCommands({"version": 1, "source": features["source"], "segments": [commands]})


@pytest.mark.parametrize("fault", ["mask", "identity", "camera"])
def test_missing_native_geometry_reports_gap_without_shortening_interval(point_bindings, fault):
    compiler, root, bindings = point_bindings
    path = next((root / "data").rglob("*.parquet"))
    table = pq.read_table(path)
    rows = table.to_pylist()
    event = next(r for r in rows[1]["language_events"] if r["style"] == "vqa" and r["role"] == "assistant")
    content = json.loads(event["content"])
    if fault == "mask":
        content["detections"][0].update(mask_present=False, point=None, bbox=None)
    elif fault == "identity":
        content["detections"].pop(0)
    else:
        event["camera"] = "observation.images.other"
    event["content"] = json.dumps(content)
    event_array = runpy.run_path(
        str(Path(__file__).parents[2] / "examples/rebot_agent/export_language_annotations.py")
    )["event_array"]
    table = table.set_column(
        table.column_names.index("language_events"),
        "language_events",
        event_array([row["language_events"] for row in rows]),
    )
    pq.write_table(table, path)
    bindings["data_sha256"][str(path.relative_to(root))] = compiler["digest"](path)
    features, report = compiler["compile_features"](root, bindings)
    assert not features["segments"]
    gap = report["segments"][0]
    assert (gap["start_frame"], gap["end_frame"]) == (0, 3)
    assert gap["missing"][0]["frame_index"] == 1 and gap["status"] == "missing_geometry"


@pytest.mark.parametrize(
    "fault", ["source", "provenance", "shards", "role_order", "review", "camera", "overlap", "absent_frame"]
)
def test_invalid_native_role_bindings_fail(point_bindings, fault):
    compiler, root, bindings = point_bindings
    segment = bindings["segments"][0]
    if fault == "source":
        bindings["source"]["revision"] = "wrong"
    elif fault == "provenance":
        bindings["grounding_provenance_sha256"] = "stale"
    elif fault == "shards":
        bindings["data_sha256"] = {}
    elif fault == "role_order":
        segment["objects"].reverse()
    elif fault == "review":
        segment["role_review"]["verdict"] = "uncertain"
    elif fault == "camera":
        segment["image_size"] = [80, 60]
    elif fault == "overlap":
        bindings["segments"].append(copy.deepcopy(segment))
    else:
        segment["end_frame"] = 5
    with pytest.raises(ValueError):
        compiler["compile_features"](root, bindings)


def test_native_export_binds_reviewed_identity_to_original_model_evidence(sample):
    api, dataset, extraction, output, path = sample
    identification = {
        "objects": ["tape"],
        "error": None,
        "review": "pending",
        "objects_source": "model_review",
        "identification_review": {
            "reviewer": {"kind": "model", "id": "test-reviewer"},
            "previous_result": {"raw": "original response", "error": "parse failure"},
            "accepted_training_labels": False,
        },
    }
    identity_path = extraction / "clip/identify.json"
    identity_path.write_text(json.dumps(identification))
    api["export_dataset"](dataset, [extraction], output)
    provenance = json.loads((output / "meta/grounding_provenance.json").read_text())
    clip = provenance["extractions"][0]["clips"][0]
    assert clip["identification"] == identification
    assert clip["identification_sha256"] == api["digest"](identity_path)
    events = pq.read_table(output / "data/chunk-000/file-000.parquet")["language_events"].to_pylist()
    obj = json.loads(
        next(r["content"] for r in events[0] if r["style"] == "vqa" and r["role"] == "assistant")
    )["detections"][0]
    assert obj["identity_source"] == "model_review"
    assert obj["evidence"]["identification_sha256"] == clip["identification_sha256"]
    assert obj["review"] == "pending"


def test_export_roundtrip_native_rows_preserves_source_and_temporal_causality(sample, tmp_path):
    api, dataset, extraction, output, path = sample
    before = path.read_bytes()
    report = api["export_dataset"](dataset, [extraction], output)
    assert report["language_rows_added"] == 9
    assert path.read_bytes() == before
    table = pq.read_table(output / "data/chunk-000/file-000.parquet")
    original = pq.read_table(path)
    assert table.drop(["language_events"]).equals(original.drop(["language_events"]))
    assert table.schema.field("language_events").type == language_events_arrow_type()
    events = table["language_events"].to_pylist()
    assert json.loads(events[0][0]["tool_calls"][0])["function"]["name"] == "say"
    for i in range(3):
        answer = json.loads(
            next(r["content"] for r in events[i] if r["style"] == "vqa" and r["role"] == "assistant")
        )
        assert answer["accepted_training_labels"] is False
        obj = answer["detections"][0]
        assert obj["bbox"] == ([1, 1, 4, 4] if i != 1 else None)
        assert obj["seed_point"] == ([2, 2] if i == 0 else None)
        trace = json.loads(next(r["content"] for r in events[i] if r["style"] == "trace"))
        samples = trace["trajectories"][0]["samples"]
        assert len(samples) == i + 1
        assert all(s["frame_index"] <= i for s in samples)
    assert json.loads(events[2][-1]["content"])["trajectories"][0]["samples"][1]["point"] is None
    assert events[3] == []
    # The existing staging validator and recipe resolver consume these rows without new styles.
    staging = EpisodeStaging(tmp_path / "staging", 0)
    staging.write(
        "vqa",
        [
            {**r, "timestamp": i / 2}
            for i, rows in enumerate(events)
            for r in rows
            if r["style"] in {"vqa", "trace"}
        ],
    )
    assert (
        StagingValidator(dataset_camera_keys=("observation.images.base",))
        .validate(list(iter_episodes(output)), staging.root)
        .ok
    )
    recipe = TrainingRecipe(
        bindings={"geometry": "emitted_at(t, style=vqa, role=assistant, camera=observation.images.base)"},
        messages=[MessageTurn(role="assistant", content="${geometry}", stream="high_level", target=True)],
    )
    rendered = render_sample(recipe=recipe, persistent=[], events=events[0], t=0.0, sample_idx=0)
    assert json.loads(rendered["messages_rendered"][0]["content"])["detections"][0]["point"] == [2.0, 2.0]


@pytest.mark.parametrize(
    "failure", ["timestamp", "camera", "stale", "coordinates", "collision", "missing_frame"]
)
def test_export_rejects_invalid_input_before_creating_dataset(sample, failure):
    api, dataset, extraction, output, path = sample
    selected_path = extraction / "clip/task_objects.json"
    selected = json.loads(selected_path.read_text())
    if failure == "stale":
        selected["source_tracks_sha256"] = "stale"
    elif failure == "coordinates":
        selected["frames"][0]["objects"][0]["centroid"] = [8, 2]
    elif failure == "camera":
        (dataset / "meta/info.json").write_text('{"features": {}}')
    else:
        table = pq.read_table(path)
        if failure == "timestamp":
            table = table.set_column(2, "timestamp", pa.array([0.1, 0.5, 1.0, 1.5]))
        elif failure == "missing_frame":
            table = table.slice(1)
        else:
            rows = table["language_events"].to_pylist()
            rows[0].append(
                {
                    "role": "user",
                    "content": "existing",
                    "style": "vqa",
                    "camera": "observation.images.base",
                    "tool_calls": None,
                }
            )
            table = table.set_column(
                table.column_names.index("language_events"), "language_events", pa.array(rows)
            )
        pq.write_table(table, path)
    selected_path.write_text(json.dumps(selected))
    with pytest.raises(ValueError):
        api["export_dataset"](dataset, [extraction], output)
    assert not output.exists()


def test_export_retains_unlocalized_objects_with_explicit_null_geometry(sample):
    api, dataset, extraction, output, path = sample
    selected_path = extraction / "clip/task_objects.json"
    selected = json.loads(selected_path.read_text())
    selected["objects"][0]["point"] = None
    for frame in selected["frames"]:
        frame["objects"][0].update(
            mask_present=False,
            centroid=None,
            bbox_xyxy=None,
            mask_path=None,
            missing_reason="no_point_seed",
            area_fraction=0.0,
        )
    selected_path.write_text(json.dumps(selected))
    api["export_dataset"](dataset, [extraction], output)
    events = pq.read_table(output / "data/chunk-000/file-000.parquet")["language_events"].to_pylist()
    for frame in events[:3]:
        answer = json.loads(
            next(r["content"] for r in frame if r["style"] == "vqa" and r["role"] == "assistant")
        )
        obj = answer["detections"][0]
        assert obj["bbox"] is None and obj["point"] is None and obj["seed_point"] is None
        assert obj["missing_reason"] == "no_point_seed"
        assert obj["evidence"]["mask_sha256"] is None


@pytest.fixture
def gripper_predictions(sample, tmp_path):
    api, _, extraction, _, _ = sample
    visual = json.loads((extraction / "extraction.json").read_text())
    clip = visual["clips"][0]
    root = tmp_path / "grippers"
    root.mkdir()
    prediction = {
        "source": visual["source"],
        "episode_index": 0,
        "camera": clip["camera"],
        "image_size": clip["image_size"],
        "checkpoint_hashes": {"model.safetensors": "a" * 64},
        "frames": [
            {
                **frame,
                "arms": {
                    "left_gripper": {
                        "status": "detected" if i != 1 else "ambiguous",
                        "score": 0.9,
                        "bbox_xyxy": [1, 1, 5, 3] if i != 1 else None,
                    },
                    "right_gripper": {"status": "missing", "score": None, "bbox_xyxy": None},
                },
            }
            for i, frame in enumerate(clip["frames"])
        ],
    }
    manifest = {
        "kind": "gripper_predictions",
        "source": visual["source"],
        "visual_manifest_sha256": api["digest"](extraction / "extraction.json"),
        "checkpoint_hashes": prediction["checkpoint_hashes"],
        "review": "pending",
        "clips": [{"path": "clip", "sha256": ""}],
    }

    def save():
        (root / "clip.json").write_text(json.dumps(prediction))
        manifest["clips"][0]["sha256"] = api["digest"](root / "clip.json")
        (root / "gripper_manifest.json").write_text(json.dumps(manifest))

    save()
    return root, manifest, prediction, save


def test_gripper_native_export_keeps_arm_identity_missing_samples_and_object_paths_separate(
    sample, gripper_predictions
):
    api, dataset, extraction, output, path = sample
    root, _, _, _ = gripper_predictions
    before = path.read_bytes()
    api["export_dataset"](dataset, [extraction], output, [root])
    table = pq.read_table(output / "data/chunk-000/file-000.parquet")
    assert table.schema.field("language_events").type == language_events_arrow_type()
    assert path.read_bytes() == before
    assert table.drop(["language_events"]).equals(pq.read_table(path).drop(["language_events"]))
    for i, rows in enumerate(table["language_events"].to_pylist()[:3]):
        answer = json.loads(
            next(r["content"] for r in rows if r["style"] == "vqa" and r["role"] == "assistant")
        )
        assert len(answer["detections"]) == 3
        left = next(d for d in answer["detections"] if d["label"] == "left_gripper")
        assert left["entity"] == "gripper" and left["arm"] == "left"
        assert left["point"] == ([3.0, 2.0] if i != 1 else None)
        assert left["point_source"] == "detector_box_center"
        assert left["review"] == "pending" and left["visibility"] == "unknown"
        assert answer["accepted_training_labels"] is False
        traces = json.loads(next(r["content"] for r in rows if r["style"] == "trace"))["trajectories"]
        assert [t["entity"] for t in traces] == ["object", "gripper", "gripper"]
        assert all(max(s["frame_index"] for s in t["samples"]) == i for t in traces)
        assert all(s["point"] is None for s in traces[2]["samples"])
        if i >= 1:
            assert traces[1]["samples"][1]["point"] is None
    provenance = json.loads((output / "meta/grounding_provenance.json").read_text())
    assert provenance["gripper_extractions"][0]["manifest_sha256"] == api["digest"](
        root / "gripper_manifest.json"
    )


@pytest.mark.parametrize(
    "case",
    ["manifest", "source", "frame", "arm", "bounds", "status", "score", "checkpoint", "camera", "duplicate"],
)
def test_gripper_native_export_rejects_misaligned_or_ambiguous_evidence(sample, gripper_predictions, case):
    api, dataset, extraction, output, _ = sample
    root, manifest, prediction, save = gripper_predictions
    left = prediction["frames"][0]["arms"]["left_gripper"]
    if case == "manifest":
        manifest["visual_manifest_sha256"] = "changed"
    elif case == "source":
        prediction["source"] = {"repo_id": "other", "revision": "other"}
    elif case == "frame":
        prediction["frames"][0]["timestamp"] = 0.25
    elif case == "arm":
        del prediction["frames"][0]["arms"]["right_gripper"]
    elif case == "bounds":
        left["bbox_xyxy"][2] = 20
    elif case == "status":
        left["status"] = "ambiguous"
    elif case == "score":
        left["score"] = float("nan")
    elif case == "checkpoint":
        prediction["checkpoint_hashes"] = {"model.safetensors": "b" * 64}
    elif case == "camera":
        prediction["camera"] = "observation.images.left_wrist"
    save()
    with pytest.raises(ValueError):
        api["export_dataset"](dataset, [extraction], output, [root, root] if case == "duplicate" else [root])
    assert not output.exists()


def test_gripper_native_export_detects_changed_prediction_bytes(sample, gripper_predictions):
    api, dataset, extraction, output, _ = sample
    root, _, _, _ = gripper_predictions
    (root / "clip.json").write_text("{}")
    with pytest.raises(ValueError, match="changed after extraction"):
        api["export_dataset"](dataset, [extraction], output, [root])
    assert not output.exists()
