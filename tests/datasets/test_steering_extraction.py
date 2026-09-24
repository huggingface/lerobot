# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
import copy
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


def test_tracking_plan_selects_disjoint_work_without_changing_manifest(extractor):
    manifest = {"source": {"revision": "pinned"}, "clips": [{"path": "a"}, {"path": "b"}, {"path": "done"}]}
    before = copy.deepcopy(manifest)
    plan = {"manifest_sha256": "hash", "shards": [["b"], ["a"]]}
    select = extractor["select_tracking_shard"]
    assert select(manifest, "hash", plan, 0) == {**manifest, "clips": [{"path": "b"}]}
    assert select(manifest, "hash", plan, 1) == {**manifest, "clips": [{"path": "a"}]}
    assert manifest == before


@pytest.mark.parametrize(
    "change",
    [
        {"manifest_sha256": "stale"},
        {"shards": [["a"], ["a"]]},
        {"shards": [["a", "a"]]},
        {"shards": [["unknown"]]},
        {"shards": [[None]]},
        {"shards": [[]]},
        {"shards": []},
        {"shards": "a"},
        {"extra": "ignored?"},
    ],
)
def test_tracking_plan_rejects_stale_overlapping_or_invalid_work(extractor, change):
    plan = {"manifest_sha256": "hash", "shards": [["a"], ["b"]], **change}
    with pytest.raises(ValueError):
        extractor["select_tracking_shard"]({"clips": [{"path": "a"}, {"path": "b"}]}, "hash", plan, 0)


@pytest.mark.parametrize("index", [-1, 2, True, "0"])
def test_tracking_plan_rejects_invalid_shard_index(extractor, index):
    with pytest.raises(ValueError):
        extractor["select_tracking_shard"](
            {"clips": [{"path": "a"}]}, "hash", {"manifest_sha256": "hash", "shards": [["a"]]}, index
        )


def test_tracking_shards_keep_separate_runtime_records_and_skip_completed_clips(
    extractor, tmp_path, monkeypatch
):
    manifest = {"models": extractor["MODELS"], "clips": []}
    for name in ["done", "a", "b"]:
        directory = tmp_path / name / "frames"
        directory.mkdir(parents=True)
        image = directory / "000000.jpg"
        Image.new("RGB", (8, 6)).save(image)
        manifest["clips"].append({"path": name, "frames": [{"sha256": extractor["sha256"](image)}]})
    manifest_path = tmp_path / "extraction.json"
    manifest_path.write_text(json.dumps(manifest))
    original = manifest_path.read_bytes()
    (tmp_path / "done/tracks.json").write_text('{"unchanged": true}')
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps({"manifest_sha256": extractor["sha256"](manifest_path), "shards": [["done", "a"], ["b"]]})
    )
    track = Mock()
    namespace = extractor["main"].__globals__
    for name, value in {
        "require_package": Mock(),
        "hf_hub_download": Mock(return_value="weights"),
        "build_sam2_video_predictor": Mock(),
        "track_clip": track,
    }.items():
        monkeypatch.setitem(namespace, name, value)
    for index in range(2):
        monkeypatch.setattr(
            "sys.argv",
            [
                "extract_visual.py",
                "track",
                "--output",
                str(tmp_path),
                "--device",
                "cpu",
                "--tracking-plan",
                str(plan_path),
                "--tracking-shard",
                str(index),
            ],
        )
        extractor["main"]()
    assert [call.args[1]["path"] for call in track.call_args_list] == ["a", "b"]
    records = sorted(tmp_path.glob("track_runtime_*.json"))
    assert len(records) == 2 and not (tmp_path / "track_runtime.json").exists()
    assert [json.loads(p.read_text())["tracking_plan"]["shard_index"] for p in records] == [0, 1]
    assert manifest_path.read_bytes() == original
    assert (tmp_path / "done/tracks.json").read_text() == '{"unchanged": true}'


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
    assert parse(
        'Based on the image: {"upcomingTask": "Return to Home Position", '
        '"taskObjects": ["plastic container", "green object", "blue object", "white object"], '
        '"taskStatus": "in progress"} This list includes four objects.'
    ) == ["plastic container", "green object", "blue object", "white object"]
    for raw in [
        '["tape", "tape"]',
        '["1", "2", "3", "4", "5"]',
        '{"names": []}',
        '{"objects": ["tape"], "taskObjects": ["bin"]}',
        "[null]",
        '["tape"] ["bin"]',
        '[remote, "tape"]',
    ]:
        with pytest.raises(ValueError):
            parse(raw)


def test_point_target_resume_preserves_evidence_and_rejects_changed_strategy(extractor, tmp_path):
    directory = tmp_path / "clip"
    (directory / "frames").mkdir(parents=True)
    Image.new("RGB", (8, 6)).save(directory / "frames/000000.jpg")
    (directory / "identify.json").write_text(json.dumps({"objects": ["tape"], "error": None}))
    manifest = {"models": extractor["MODELS"], "clips": [{"path": "clip"}]}
    model = Mock(return_value='<point x="50" y="50">tape</point>')
    run = extractor["run_molmo"]
    run(tmp_path, manifest, "point", model, point_target="material")
    target = directory / "point.json"
    result = json.loads(target.read_text())
    assert result["point_target"] == "material"
    assert result["objects"][0]["prompt"] == model.call_args.args[1]
    assert result["objects"][0]["raw"] == model.return_value
    before = target.read_bytes()
    run(tmp_path, manifest, "point", model, point_target="material")
    assert model.call_count == 1
    with pytest.raises(ValueError, match="Pointing target changed"):
        run(tmp_path, manifest, "point", model)
    assert target.read_bytes() == before
    # Pre-option outputs used the object prompt and remain resumable in that mode.
    del result["point_target"]
    target.write_text(json.dumps(result))
    run(tmp_path, manifest, "point", model)
    assert model.call_count == 1


@pytest.fixture
def seed_review_case(extractor, tmp_path):
    parent, output = tmp_path / "parent", tmp_path / "corrected"
    directory = parent / "clip"
    (directory / "frames").mkdir(parents=True)
    image = directory / "frames/000000.jpg"
    Image.new("RGB", (8, 6)).save(image)
    (directory / "identify.json").write_text(json.dumps({"objects": ["cable"], "error": None}))
    point = directory / "point.json"
    point.write_text(json.dumps({"objects": [{"object_id": 1, "name": "cable", "point": [4, 3]}]}))
    manifest = parent / "extraction.json"
    manifest.write_text(
        json.dumps(
            {
                "source": {"repo_id": "test", "revision": "pinned"},
                "models": extractor["MODELS"],
                "clips": [
                    {
                        "path": "clip",
                        "image_size": [8, 6],
                        "frames": [{"sha256": extractor["sha256"](image)}],
                    }
                ],
            }
        )
    )
    review = {
        "parent_manifest_sha256": extractor["sha256"](manifest),
        "reviewer": {"kind": "model", "id": "test-reviewer"},
        "corrections": [
            {
                "clip": "clip",
                "object_id": 1,
                "name": "cable",
                "source_point_sha256": extractor["sha256"](point),
                "frame_sha256": extractor["sha256"](image),
                "point": [2, 3],
                "reason": "Visible cable material, rather than its empty center",
            }
        ],
    }
    return parent, output, tmp_path / "review.json", review


def test_seed_corrections_preserve_parent_and_model_attribution(extractor, seed_review_case):
    parent, output, path, review = seed_review_case
    path.write_text(json.dumps(review))
    before = {p: p.read_bytes() for p in parent.rglob("*") if p.is_file()}
    manifest = extractor["prepare_reviewed_points"](parent, output, path)
    assert all(p.read_bytes() == data for p, data in before.items())
    obj = json.loads((output / "clip/point.json").read_text())["objects"][0]
    assert obj["point"] == [2, 3] and obj["point_source"] == "model_review"
    assert obj["seed_review"]["original_prediction"]["point"] == [4, 3]
    assert not obj["seed_review"]["accepted_training_labels"]
    assert manifest["preparation"]["reviewer"] == review["reviewer"]
    assert not (output / "clip/tracks.json").exists()
    with pytest.raises(ValueError, match="Pointing target changed"):
        extractor["verify_point_target"](output, manifest, "object")


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("source_point_sha256", "stale", "stale point"),
        ("frame_sha256", "stale", "first source frame"),
        ("name", "different object", "object identity"),
        ("point", [8, 3], "image coordinates"),
        ("point", [True, 3], "image coordinates"),
        ("negative_points", [[6, 4]], "later-frame"),
    ],
)
def test_invalid_seed_review_creates_no_output(extractor, seed_review_case, field, value, error):
    parent, output, path, review = seed_review_case
    review["corrections"][0][field] = value
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError, match=error):
        extractor["prepare_reviewed_points"](parent, output, path)
    assert not output.exists()


def test_uncertain_review_can_withdraw_a_seed_without_inventing_visibility(extractor, seed_review_case):
    parent, output, path, review = seed_review_case
    review["corrections"][0].update(point=None, reason="Object not unambiguously visible")
    path.write_text(json.dumps(review))
    extractor["prepare_reviewed_points"](parent, output, path)
    obj = json.loads((output / "clip/point.json").read_text())["objects"][0]
    assert obj["point"] is None and obj["point_source"] == "model_review"


@pytest.fixture
def temporal_review_case(extractor, seed_review_case):
    parent, output, path, review = seed_review_case
    image = parent / "clip/frames/000001.jpg"
    Image.new("RGB", (8, 6), "red").save(image)
    manifest_path = parent / "extraction.json"
    manifest = json.loads(manifest_path.read_text())
    frames = manifest["clips"][0]["frames"]
    frames[0]["frame_index"] = 300
    frames.append({"frame_index": 303, "sha256": extractor["sha256"](image)})
    manifest_path.write_text(json.dumps(manifest))
    tracks = parent / "clip/tracks.json"
    tracks.write_text('{"review": "pending"}')
    review["parent_manifest_sha256"] = extractor["sha256"](manifest_path)
    review["corrections"][0].update(
        frame_index=303,
        frame_sha256=extractor["sha256"](image),
        source_tracks_sha256=extractor["sha256"](tracks),
        reason="Visible object has reappeared but its original track is missing",
    )
    return parent, output, path, review


@pytest.mark.parametrize("negatives", [[], [[6, 4], [7, 5]]])
def test_temporal_correction_preserves_seed_and_uses_local_sam_index(
    extractor, temporal_review_case, negatives
):
    parent, output, path, review = temporal_review_case
    if negatives:
        review["corrections"][0]["negative_points"] = negatives
    path.write_text(json.dumps(review))
    before = {p: p.read_bytes() for p in parent.rglob("*") if p.is_file()}
    manifest = extractor["prepare_reviewed_points"](parent, output, path)
    assert all(p.read_bytes() == content for p, content in before.items())
    directory = output / "clip"
    obj = json.loads((directory / "point.json").read_text())["objects"][0]
    assert obj["point"] == [4, 3]
    prompt = obj["tracking_prompts"][0]
    assert prompt["frame_index"] == 303 and prompt["point"] == [2, 3]
    assert prompt.get("negative_points", []) == negatives
    assert prompt["review"]["reviewer"] == review["reviewer"]
    assert not prompt["review"]["accepted_training_labels"]
    calls = []

    class Predictor:
        def init_state(self, **kwargs):
            return {}

        def add_new_points_or_box(self, state, **kwargs):
            calls.append(
                (kwargs["frame_idx"], kwargs["obj_id"], kwargs["points"].tolist(), kwargs["labels"].tolist())
            )

        def propagate_in_video(self, state):
            assert calls == [
                (0, 1, [[4, 3]], [1]),
                (1, 1, [[2, 3], *negatives], [1, *([0] * len(negatives))]),
            ]
            for index in range(2):
                yield index, [1], torch.full((1, 1, 6, 8), -1.0)

    extractor["track_clip"](directory, manifest["clips"][0], Predictor())
    tracks = json.loads((directory / "tracks.json").read_text())
    assert tracks["objects"][0] == obj
    assert [f["frame_index"] for f in tracks["frames"]] == [300, 303]
    assert all(not f["objects"][0]["mask_present"] for f in tracks["frames"])


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("frame_index", 301, "exported episode-local frame"),
        ("frame_index", True, "exported episode-local frame"),
        ("frame_sha256", "stale", "selected source frame"),
        ("source_tracks_sha256", "stale", "source tracks hash"),
        ("point", None, "visible material point"),
        ("point", [float("nan"), 1], "image coordinates"),
        ("negative_points", [[2, 3]], "distinct"),
        ("negative_points", [[6, 4], [6, 4]], "distinct"),
        ("negative_points", [[8, 2]], "coordinates"),
        ("negative_points", [[float("nan"), 2]], "coordinates"),
        ("negative_points", [[True, 2]], "coordinates"),
        ("negative_points", None, "list"),
    ],
)
def test_bad_temporal_review_has_no_side_effects(extractor, temporal_review_case, field, value, error):
    parent, output, path, review = temporal_review_case
    review["corrections"][0][field] = value
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError, match=error):
        extractor["prepare_reviewed_points"](parent, output, path)
    assert not output.exists()


def test_negative_points_cannot_change_after_their_review(extractor, temporal_review_case):
    parent, output, path, review = temporal_review_case
    review["corrections"][0]["negative_points"] = [[6, 4]]
    path.write_text(json.dumps(review))
    manifest = extractor["prepare_reviewed_points"](parent, output, path)
    point_path = output / "clip/point.json"
    points = json.loads(point_path.read_text())
    points["objects"][0]["tracking_prompts"][0]["negative_points"] = [[7, 4]]
    point_path.write_text(json.dumps(points))
    predictor = Mock()
    with pytest.raises(ValueError, match="attributed review"):
        extractor["track_clip"](output / "clip", manifest["clips"][0], predictor)
    assert not predictor.mock_calls


def test_temporal_correction_cannot_revive_withdrawn_identity(extractor, temporal_review_case):
    parent, output, path, review = temporal_review_case
    correction = dict(review["corrections"][0])
    frame = json.loads((parent / "extraction.json").read_text())["clips"][0]["frames"][0]
    correction.update(frame_index=300, frame_sha256=frame["sha256"], point=None)
    review["corrections"].append(correction)
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError, match="existing initial seed"):
        extractor["prepare_reviewed_points"](parent, output, path)
    assert not output.exists()


def test_duplicate_temporal_correction_is_rejected(extractor, temporal_review_case):
    parent, output, path, review = temporal_review_case
    review["corrections"].append(dict(review["corrections"][0]))
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError, match="duplicate corrected object"):
        extractor["prepare_reviewed_points"](parent, output, path)
    assert not output.exists()


@pytest.mark.parametrize(
    "raw",
    ["[tape, bin]", '{"taskObjects": ["tape", "bin"], "taskStatus": "completed"}'],
)
def test_reparse_preserves_model_response_and_records_recovery(extractor, tmp_path, raw):
    directory = tmp_path / "clip"
    directory.mkdir()
    target = directory / "identify.json"
    result = {
        "model": extractor["MODELS"]["identify"],
        "review": "pending",
        "raw": raw,
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
    assert "taskStatus" not in restored
    assert restored["format_recovery"]["previous_file_sha256"] == old_hash
    # A retry must not overwrite provenance or re-query successful identification.
    assert extractor["reparse_identification"](tmp_path, manifest) == []
    assert json.loads(target.read_text()) == restored
    target.write_text(json.dumps(result))
    (directory / "point.json").write_text("{}")
    with pytest.raises(ValueError, match="dependent extraction"):
        extractor["reparse_identification"](tmp_path, manifest)


@pytest.mark.parametrize("raw", ['["tape", "bin"]', '["battery", "battery"]'])
def test_identification_retry_preserves_original_and_is_bounded(extractor, tmp_path, raw):
    clips = [{"path": name, "subtask": "pick up tape"} for name in ("failed", "successful")]
    manifest = {"models": extractor["MODELS"], "clips": clips}
    originals = {}
    for clip in clips:
        directory = tmp_path / clip["path"]
        (directory / "frames").mkdir(parents=True)
        Image.new("RGB", (8, 6)).save(directory / "frames/000000.jpg")
        result = {
            "model": extractor["MODELS"]["identify"],
            "review": "pending",
            "raw": '{"ambiguous": "tape"}',
            "prompt": "original prompt",
            "objects": [] if clip["path"] == "failed" else ["bin"],
            "error": "invalid response" if clip["path"] == "failed" else None,
        }
        target = directory / "identify.json"
        target.write_text(json.dumps(result))
        originals[clip["path"]] = (target.read_bytes(), extractor["sha256"](target), result)
    model = Mock(return_value=raw)
    retry = extractor["retry_identification"]
    report = retry(tmp_path, manifest, model)
    assert len(report) == 1 and model.call_count == 1
    target = tmp_path / "failed/identify.json"
    result = json.loads(target.read_text())
    assert result["raw"] == raw and result["review"] == "pending"
    assert result["model_retry"]["previous_result"] == originals["failed"][2]
    assert result["model_retry"]["previous_file_sha256"] == originals["failed"][1]
    assert not result["model_retry"]["accepted_training_labels"]
    assert bool(result["error"]) == (raw == '["battery", "battery"]')
    assert (tmp_path / "successful/identify.json").read_bytes() == originals["successful"][0]
    before = target.read_bytes()
    assert retry(tmp_path, manifest, model) == []
    assert model.call_count == 1 and target.read_bytes() == before


def test_retry_preflights_all_failures_before_model_calls(extractor, tmp_path):
    manifest = {"models": extractor["MODELS"], "clips": [{"path": "a"}, {"path": "b"}]}
    for clip in manifest["clips"]:
        directory = tmp_path / clip["path"]
        directory.mkdir()
        (directory / "identify.json").write_text(
            json.dumps({"error": "failed", "review": "pending", "model": extractor["MODELS"]["identify"]})
        )
    (tmp_path / "b/point.json").write_text("{}")
    before = (tmp_path / "a/identify.json").read_bytes()
    model = Mock()
    with pytest.raises(ValueError, match="dependent extraction"):
        extractor["retry_identification"](tmp_path, manifest, model)
    model.assert_not_called()
    assert (tmp_path / "a/identify.json").read_bytes() == before


@pytest.fixture
def name_review_case(extractor, tmp_path):
    clips = []
    corrections = []
    for name in ("a", "b"):
        directory = tmp_path / name
        (directory / "frames").mkdir(parents=True)
        image = directory / "frames/000000.jpg"
        Image.new("RGB", (8, 6)).save(image)
        target = directory / "identify.json"
        target.write_text(
            json.dumps({"objects": [], "raw": "bad response", "error": "failed", "review": "pending"})
        )
        clips.append({"path": name, "frames": [{"sha256": extractor["sha256"](image)}]})
        corrections.append(
            {
                "clip": name,
                "source_identification_sha256": extractor["sha256"](target),
                "frame_sha256": extractor["sha256"](image),
                "objects": ["pink cable", "black bin"],
                "reason": "Fixture visual review; not actual training evidence",
            }
        )
    manifest = {"clips": clips}
    (tmp_path / "extraction.json").write_text(json.dumps(manifest))
    review = {
        "parent_manifest_sha256": extractor["sha256"](tmp_path / "extraction.json"),
        "reviewer": {"kind": "model", "id": "test-reviewer"},
        "corrections": corrections,
    }
    return manifest, tmp_path / "review.json", review


def test_name_review_retains_original_evidence_without_accepting_labels(
    extractor, tmp_path, name_review_case
):
    manifest, path, review = name_review_case
    original = json.loads((tmp_path / "a/identify.json").read_text())
    path.write_text(json.dumps(review))
    result = extractor["review_identification"](tmp_path, manifest, path)
    assert result["corrected_clips"] == 2 and not result["accepted_training_labels"]
    restored = json.loads((tmp_path / "a/identify.json").read_text())
    assert restored["objects"] == ["pink cable", "black bin"]
    assert restored["objects_source"] == "model_review" and restored["review"] == "pending"
    assert restored["error"] is None and restored["raw"] == original["raw"]
    assert restored["identification_review"]["previous_result"] == original
    assert (
        restored["identification_review"]["correction"]["source_identification_sha256"]
        == review["corrections"][0]["source_identification_sha256"]
    )
    with pytest.raises(ValueError, match="stale"):
        extractor["review_identification"](tmp_path, manifest, path)


@pytest.mark.parametrize("failure", ["stale", "frame", "duplicate_names", "dependent", "reviewer"])
def test_name_review_preflight_preserves_every_original(extractor, tmp_path, name_review_case, failure):
    manifest, path, review = name_review_case
    last = review["corrections"][-1]
    if failure == "stale":
        last["source_identification_sha256"] = "wrong"
    elif failure == "frame":
        (tmp_path / "b/frames/000000.jpg").write_bytes(b"different frame")
    elif failure == "duplicate_names":
        last["objects"] = ["cube", "cube"]
    elif failure == "dependent":
        (tmp_path / "b/point.json").write_text("{}")
    else:
        review["reviewer"]["kind"] = "automatic_ground_truth"
    path.write_text(json.dumps(review))
    before = {name: (tmp_path / name / "identify.json").read_bytes() for name in ("a", "b")}
    with pytest.raises(ValueError):
        extractor["review_identification"](tmp_path, manifest, path)
    assert all((tmp_path / name / "identify.json").read_bytes() == data for name, data in before.items())


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
    assert matches("white_cable", "Pick up the white cable with the left arm.")
    assert matches("black bin", "Put the cable into the black_bin.")
    assert matches("BLUE__BLOCK", "Pick up the blue-block.")
    assert not matches("red_cube", "Pick up the red cubes.")
    assert not matches("tape_roll", "Pick up the roll of tape.")
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
    assert result["filter"] == "normalized_whole_name_in_subtask_v2"
    assert result["objects"] == objects[1:]
    assert result["excluded_objects"] == objects[:1]
    assert result["frames"][0]["frame_index"] == 15
    assert result["frames"][0]["objects"] == [{"object_id": 2, "mask_present": False}]
    assert report["clips"][0]["missing_points"] == 1
    assert report["clips"][0]["missing_masks"] == 1
    assert result["source_tracks_sha256"] == source_hash == extractor["sha256"](source)


@pytest.mark.parametrize("source", ["task-objects", "points"])
def test_required_candidates_retry_only_missing_task_objects_without_rewriting_evidence(
    extractor, tmp_path, source
):
    parent = tmp_path / "parent"
    parent.mkdir()
    manifest = {"version": 1, "source": {"repo_id": "test"}, "models": extractor["MODELS"], "clips": []}
    cases = [
        ("missing", "put the block into the bin", "block", [1, 1]),
        ("present", "put the block into the black bin", "black bin", [2, 2]),
        ("home", "return home", "bin", [2, 2]),
        ("no_point", "put the block into the bin", "bin", None),
        ("alias_filtered_out", "put the block into the bin", "black bin", [2, 2]),
    ]
    for name, task, obj, point in cases:
        directory = parent / name
        (directory / "frames").mkdir(parents=True)
        frame = directory / "frames/000000.jpg"
        Image.new("RGB", (8, 6)).save(frame)
        manifest["clips"].append(
            {"path": name, "subtask": task, "frames": [{"sha256": extractor["sha256"](frame)}]}
        )
        (directory / ("point.json" if source == "points" else "tracks.json")).write_text(
            json.dumps({"objects": [{"object_id": 7, "name": obj, "point": point}], "frames": []})
        )
    (parent / "extraction.json").write_text(json.dumps(manifest))
    if source == "task-objects":
        extractor["filter_objects"](parent, manifest)
    hashes = {p: extractor["sha256"](p) for p in parent.rglob("*") if p.is_file()}
    output = tmp_path / "retry"
    recovery = extractor["prepare_required"](parent, output, ["bin"], source=source)
    assert [c["path"] for c in recovery["clips"]] == ["missing", "no_point", "alias_filtered_out"]
    assert recovery["preparation"]["candidate_source"] == source
    assert recovery["models"]["identify"] is None
    assert recovery["models"]["point"] == extractor["MODELS"]["point"]
    candidate = json.loads((output / "missing/identify.json").read_text())
    assert candidate["objects"] == ["bin"]
    assert candidate["review"] == "pending" and candidate["model"] is None
    original_candidate = parent / "missing" / candidate["parent_candidates_file"]
    assert (output / "missing/parent_candidates.json").read_bytes() == original_candidate.read_bytes()
    assert candidate["parent_candidates_sha256"] == extractor["sha256"](original_candidate)
    assert all(extractor["sha256"](p) == h for p, h in hashes.items())
    extractor["verify_frames"](output, recovery)
    with pytest.raises(ValueError, match="outside"):
        extractor["prepare_required"](parent, parent / "nested", ["bin"], source=source)
    if source == "points":
        # A partial parent cannot silently produce an incomplete recovery manifest.
        (parent / "home/point.json").unlink()
        with pytest.raises(FileNotFoundError):
            extractor["prepare_required"](parent, tmp_path / "incomplete", ["bin"], source=source)
        assert not (tmp_path / "incomplete").exists()
        return
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
