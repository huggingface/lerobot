# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
import json
import runpy
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture
def extractor():
    return runpy.run_path(str(Path(__file__).parents[2] / "examples/rebot_agent/extract_visual.py"))


def test_molmo_percentages_are_scaled_and_ambiguous_points_are_missing(extractor):
    parse = extractor["parse_point"]
    assert parse('<point x="50" y="25" alt="tape">tape</point>', (640, 480)) == [320, 120]
    assert parse('<point y="100" x="100">bin</point>', (640, 480)) == [639, 479]
    for raw in [
        "not visible",
        '<point x="101" y="20">x</point>',
        '<point x="nan" y="20">x</point>',
        '<points x1="10" y1="20" x2="30" y2="40">two objects</points>',
        '<point x="10" y="20">x</point><point x="30" y="40">y</point>',
        '<point x="10" x="30" y="20">x</point>',
    ]:
        assert parse(raw, (640, 480)) is None


def test_object_selection_does_not_silently_truncate_or_invent_identities(extractor):
    parse = extractor["parse_objects"]
    assert parse('```json\n["tape", "black bin"]\n```') == ["tape", "black bin"]
    assert parse('Here is the list: {"taskDescription": "pick", "objects": ["tape", "bin"]}\nDone.') == [
        "tape",
        "bin",
    ]
    assert parse(" [remote, tape, cracker, screwdriver]") == ["remote", "tape", "cracker", "screwdriver"]
    for raw in [
        '["tape", "tape"]',
        '["1", "2", "3", "4", "5"]',
        '{"names": []}',
        "[null]",
        '["tape"] ["bin"]',
        '[remote, "tape"]',
    ]:
        with pytest.raises(ValueError):
            parse(raw)


def test_reparse_preserves_model_response_and_records_recovery(extractor, tmp_path):
    directory = tmp_path / "clip"
    directory.mkdir()
    target = directory / "identify.json"
    result = {
        "model": extractor["MODELS"]["identify"],
        "review": "pending",
        "raw": "[tape, bin]",
        "prompt": "original prompt",
        "objects": [],
        "error": "old parse failure",
    }
    target.write_text(json.dumps(result))
    old_hash = extractor["sha256"](target)
    manifest = {"models": extractor["MODELS"], "clips": [{"path": "clip"}]}
    report = extractor["reparse_identification"](tmp_path, manifest)
    restored = json.loads(target.read_text())
    assert report[0]["status"] == "reparsed"
    assert restored["raw"] == result["raw"] and restored["prompt"] == result["prompt"]
    assert restored["objects"] == ["tape", "bin"] and restored["review"] == "pending"
    assert restored["format_recovery"]["previous_file_sha256"] == old_hash
    # A retry must not overwrite provenance or re-query successful identification.
    assert extractor["reparse_identification"](tmp_path, manifest) == []
    assert json.loads(target.read_text()) == restored
    target.write_text(json.dumps(result))
    (directory / "point.json").write_text("{}")
    with pytest.raises(ValueError, match="dependent extraction"):
        extractor["reparse_identification"](tmp_path, manifest)


def test_tracking_preserves_source_time_identity_and_missing_masks(extractor, tmp_path):
    (tmp_path / "frames").mkdir()
    for i in range(2):
        Image.new("RGB", (8, 6)).save(tmp_path / "frames" / f"{i:06d}.jpg")
    objects = [
        {"object_id": 1, "name": "tape", "point": [3, 2]},
        {"object_id": 2, "name": "bin", "point": None},
    ]
    (tmp_path / "point.json").write_text(json.dumps({"objects": objects}))
    frames = [{"frame_index": 300, "timestamp": 10.0}, {"frame_index": 303, "timestamp": 10.1}]

    class Predictor:
        def init_state(self, **kwargs):
            assert kwargs["offload_video_to_cpu"]
            return {}

        def add_new_points_or_box(self, state, **kwargs):
            assert kwargs["obj_id"] == 1
            assert kwargs["frame_idx"] == 0
            np.testing.assert_equal(kwargs["points"], [[3, 2]])

        def propagate_in_video(self, state):
            first = torch.full((1, 1, 6, 8), -1.0)
            first[0, 0, 1:4, 2:5] = 1
            yield 0, [1], first
            yield 1, [1], torch.full_like(first, -1)

    extractor["track_clip"](tmp_path, {"frames": frames, "image_size": [8, 6]}, Predictor())
    result = json.loads((tmp_path / "tracks.json").read_text())
    assert result["objects"] == objects  # Missing seeds stay missing; identities are not renumbered.
    assert result["frames"][0]["timestamp"] == 10.0
    first = result["frames"][0]["objects"][0]
    assert first["centroid"] == [3, 2]
    assert first["bbox_xyxy"] == [2, 1, 5, 4]
    assert first["area_fraction"] == 9 / 48
    second = result["frames"][1]["objects"][0]
    assert second["centroid"] is None
    assert not second["mask_present"]
    assert result["status"] == "unreviewed"
    for frame in result["frames"]:
        assert len(frame["objects"]) == 2
        assert frame["objects"][1]["object_id"] == 2
        assert frame["objects"][1]["mask_path"] is None
        assert frame["objects"][1]["missing_reason"] == "no_point_seed"
    assert np.asarray(Image.open(tmp_path / second["mask_path"])).sum() == 0


def test_modified_frame_invalidates_resume(extractor, tmp_path):
    frames = tmp_path / "clip/frames"
    frames.mkdir(parents=True)
    path = frames / "000000.jpg"
    Image.new("RGB", (8, 6)).save(path)
    manifest = {"clips": [{"path": "clip", "frames": [{"sha256": extractor["sha256"](path)}]}]}
    extractor["verify_frames"](tmp_path, manifest)
    Image.new("RGB", (8, 6), "white").save(path)
    with pytest.raises(ValueError, match="differs"):
        extractor["verify_frames"](tmp_path, manifest)


def test_molmo_constructor_uses_real_dependency_guard(extractor, monkeypatch):
    pytest.importorskip("transformers")
    constructor = extractor["Molmo"]
    processor, model = Mock(), Mock()
    monkeypatch.setitem(constructor.__init__.__globals__, "AutoProcessor", processor)
    monkeypatch.setitem(constructor.__init__.__globals__, "AutoModelForCausalLM", model)
    instance = constructor(extractor["MODELS"]["identify"], "cpu")
    assert instance.model is model.from_pretrained.return_value.eval.return_value
    assert (
        processor.from_pretrained.call_args.kwargs["revision"] == extractor["MODELS"]["identify"]["revision"]
    )


def test_task_relevance_does_not_match_partial_words_or_invent_synonyms(extractor):
    matches = extractor["object_mentioned"]
    assert matches("Black Bin", "Place the tape in the black bin.")
    assert matches("tape roll", "Pick up the tape-roll.")
    assert not matches("hat", "Put that block away.")
    assert not matches("basket", "Place the tape in the black bin.")
    assert not matches("bin", "Return to Home Position")
    assert not matches("", "Move to the bin")


def test_postfilter_keeps_raw_evidence_ids_and_missing_points(extractor, tmp_path):
    directory = tmp_path / "clip"
    directory.mkdir()
    objects = [
        {"object_id": 1, "name": "wall", "point": [1, 2]},
        {"object_id": 2, "name": "tape", "point": [3, 4]},
        {"object_id": 3, "name": "bin", "point": None},
    ]
    tracks = {
        "objects": objects,
        "frames": [
            {
                "frame_index": 15,
                "timestamp": 0.5,
                "objects": [
                    {"object_id": 1, "mask_present": True},
                    {"object_id": 2, "mask_present": False},
                ],
            }
        ],
    }
    source = directory / "tracks.json"
    source.write_text(json.dumps(tracks))
    source_hash = extractor["sha256"](source)
    manifest = {
        "source": {"repo_id": "test"},
        "clips": [{"path": "clip", "subtask": "Put tape into the bin"}],
    }
    (tmp_path / "extraction.json").write_text(json.dumps(manifest))
    report = extractor["filter_objects"](tmp_path, manifest)
    result = json.loads((directory / "task_objects.json").read_text())
    assert result["status"] == "unreviewed"
    assert result["objects"] == objects[1:]
    assert result["excluded_objects"] == objects[:1]
    assert result["frames"][0]["frame_index"] == 15
    assert result["frames"][0]["objects"] == [{"object_id": 2, "mask_present": False}]
    assert report["clips"][0]["missing_points"] == 1
    assert report["clips"][0]["missing_masks"] == 1
    assert result["source_tracks_sha256"] == source_hash == extractor["sha256"](source)


def test_required_candidates_retry_only_missing_task_objects_without_rewriting_evidence(extractor, tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    manifest = {"version": 1, "source": {"repo_id": "test"}, "models": extractor["MODELS"], "clips": []}
    cases = [
        ("missing", "put the block into the bin", "block", [1, 1]),
        ("present", "put the block into the black bin", "black bin", [2, 2]),
        ("home", "return home", "bin", [2, 2]),
        ("no_point", "put the block into the bin", "bin", None),
    ]
    for name, task, obj, point in cases:
        directory = parent / name
        (directory / "frames").mkdir(parents=True)
        frame = directory / "frames/000000.jpg"
        Image.new("RGB", (8, 6)).save(frame)
        manifest["clips"].append(
            {"path": name, "subtask": task, "frames": [{"sha256": extractor["sha256"](frame)}]}
        )
        (directory / "tracks.json").write_text(
            json.dumps({"objects": [{"object_id": 7, "name": obj, "point": point}], "frames": []})
        )
    (parent / "extraction.json").write_text(json.dumps(manifest))
    extractor["filter_objects"](parent, manifest)
    hashes = {p: extractor["sha256"](p) for p in parent.rglob("*") if p.is_file()}
    output = tmp_path / "retry"
    recovery = extractor["prepare_required"](parent, output, ["bin"])
    assert [c["path"] for c in recovery["clips"]] == ["missing", "no_point"]
    assert recovery["models"]["identify"] is None
    assert recovery["models"]["point"] == extractor["MODELS"]["point"]
    candidate = json.loads((output / "missing/identify.json").read_text())
    assert candidate["objects"] == ["bin"]
    assert candidate["review"] == "pending" and candidate["model"] is None
    assert all(extractor["sha256"](p) == h for p, h in hashes.items())
    extractor["verify_frames"](output, recovery)
    # A filtered result may not silently be reused after the tracks have changed.
    (parent / "missing/tracks.json").write_text("{}")
    with pytest.raises(ValueError, match="stale"):
        extractor["prepare_required"](parent, tmp_path / "stale", ["bin"])
    assert not (tmp_path / "stale").exists()


def test_unlocalized_clip_preserves_every_source_frame_without_invoking_tracker(extractor, tmp_path):
    objects = [{"object_id": 1, "name": "bin", "point": None}]
    (tmp_path / "point.json").write_text(json.dumps({"objects": objects}))
    frames = [{"frame_index": 6, "timestamp": 0.2}, {"frame_index": 9, "timestamp": 0.3}]
    predictor = Mock()
    extractor["track_clip"](tmp_path, {"frames": frames, "image_size": [8, 6]}, predictor)
    assert not predictor.mock_calls
    result = json.loads((tmp_path / "tracks.json").read_text())
    assert result["status"] == "no_grounded_objects"
    assert [{k: r[k] for k in ("frame_index", "timestamp")} for r in result["frames"]] == frames
    for frame in result["frames"]:
        assert frame["objects"][0]["centroid"] is None
        assert frame["objects"][0]["mask_path"] is None
        assert frame["objects"][0]["missing_reason"] == "no_point_seed"
