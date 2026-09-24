# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image


@pytest.fixture
def label_pack(tmp_path):
    module = runpy.run_path(str(Path(__file__).parents[2] / "examples/rebot_agent/gripper_labels.py"))
    images, labels = [], []
    for i in range(3):
        path = tmp_path / f"{i}.jpg"
        Image.new("RGB", (16, 12)).save(path)
        images.append(
            {
                "id": i,
                "file_name": path.name,
                "sha256": module["digest"](path),
                "width": 16,
                "height": 12,
                "episode_index": i,
                "split": "validation" if i == 1 else "train",
            }
        )
        labels.append(
            {
                "id": i,
                "arms": {
                    arm: {"visibility": "visible", "bbox_xyxy": [1, 2, 5, 8]} for arm in ("left", "right")
                },
                "review": {
                    "kind": "human",
                    "reviewer": "test fixture",
                    "confirmed": True,
                    "arm_identity_verified": True,
                },
            }
        )
    pack = {
        "version": 1,
        "source": {"repo_id": "fixture", "revision": "fixture-sha"},
        "vla_holdout_episodes": list(range(90, 100)),
        "images": images,
    }
    pack_path = tmp_path / "pack.json"
    pack_path.write_text(json.dumps(pack))
    document = {"version": 1, "pack_sha256": module["digest"](pack_path), "images": labels}
    return module, pack_path, document


def run_export(label_pack, tmp_path, **kwargs):
    module, pack_path, labels = label_pack
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(labels))
    return module["export_coco"](pack_path, path, tmp_path / "export", **kwargs)


def model_review(label_pack):
    _, pack_path, labels = label_pack
    images = json.loads(pack_path.read_text())["images"]
    for row, image in zip(labels["images"], images, strict=True):
        row["review"].update(
            kind="model",
            reviewer="model fixture",
            notes="Fixture: both jaws visible; arm identities checked in synchronized context.",
            image_sha256=image["sha256"],
        )


def test_model_review_is_opt_in_and_never_becomes_human_provenance(label_pack, tmp_path):
    model_review(label_pack)
    with pytest.raises(ValueError, match="human review"):
        run_export(label_pack, tmp_path)
    report = run_export(label_pack, tmp_path, allow_model_review=True)
    assert report["human_verified"] is False
    assert report["review_counts"] == {
        "train": {"human": 0, "model": 2},
        "validation": {"human": 0, "model": 1},
    }
    exported = json.loads((tmp_path / "export/reviewed_labels.json").read_text())
    assert exported == label_pack[2]
    for split in ("train", "validation"):
        dataset = json.loads((tmp_path / f"export/{split}.json").read_text())
        assert dataset["info"]["human_verified"] is False
        assert dataset["info"]["review_counts"] == report["review_counts"]


@pytest.mark.parametrize(
    "field,value",
    [("confirmed", False), ("notes", " "), ("image_sha256", "stale"), ("arm_identity_verified", False)],
)
def test_model_review_opt_in_does_not_bypass_evidence_checks(label_pack, tmp_path, field, value):
    model_review(label_pack)
    label_pack[2]["images"][0]["review"][field] = value
    with pytest.raises(ValueError):
        run_export(label_pack, tmp_path, allow_model_review=True)
    assert not (tmp_path / "export").exists()


def test_uncertain_model_labels_are_excluded_not_negative_targets(label_pack, tmp_path):
    model_review(label_pack)
    label_pack[2]["images"][2]["arms"]["left"] = {"visibility": "uncertain", "bbox_xyxy": None}
    report = run_export(label_pack, tmp_path, allow_model_review=True)
    assert report["excluded_uncertain_images"] == [2]
    assert report["review_counts"]["train"] == {"human": 0, "model": 1}
    assert report["splits"]["train"]["images"] == 1


def test_mixed_review_attribution_is_preserved_per_split(label_pack, tmp_path):
    model_review(label_pack)
    label_pack[2]["images"][1]["review"]["kind"] = "human"
    report = run_export(label_pack, tmp_path, allow_model_review=True)
    assert report["human_verified"] is False
    assert report["review_counts"]["validation"] == {"human": 1, "model": 0}


def test_uncertain_gripper_excludes_whole_image_and_keeps_original_boxes(label_pack, tmp_path):
    label_pack[2]["images"][2]["arms"]["left"] = {"visibility": "uncertain", "bbox_xyxy": None}
    report = run_export(label_pack, tmp_path)
    assert report["excluded_uncertain_images"] == [2]
    assert report["splits"]["train"] == {"images": 1, "boxes": 2, "episodes": [0]}
    dataset = json.loads((tmp_path / "export/train.json").read_text())
    assert dataset["annotations"][0]["bbox"] == [1, 2, 4, 6]
    assert dataset["annotations"][0]["area"] == 24
    assert dataset["categories"] == [{"id": 0, "name": "left_gripper"}, {"id": 1, "name": "right_gripper"}]


@pytest.mark.parametrize(
    "case",
    [
        "unreviewed",
        "arm_identity",
        "unlabeled",
        "source_changed",
        "vla_holdout",
        "split_overlap",
        "out_of_bounds",
    ],
)
def test_invalid_gripper_labels_cannot_enter_training(label_pack, tmp_path, case):
    module, path, labels = label_pack
    row = labels["images"][0]
    if case == "unreviewed":
        row["review"]["confirmed"] = False
    elif case == "arm_identity":
        row["review"]["arm_identity_verified"] = False
    elif case == "unlabeled":
        row["arms"]["left"] = {"visibility": "unlabeled", "bbox_xyxy": None}
    elif case == "source_changed":
        Image.new("RGB", (16, 12), "white").save(tmp_path / "0.jpg")
    elif case in ("vla_holdout", "split_overlap"):
        pack = json.loads(path.read_text())
        pack["images"][0]["episode_index"] = 90 if case == "vla_holdout" else 1
        path.write_text(json.dumps(pack))
        labels["pack_sha256"] = module["digest"](path)
    else:
        row["arms"]["left"]["bbox_xyxy"] = [0, 0, 17, 12]
    with pytest.raises(ValueError):
        run_export(label_pack, tmp_path)
    assert not (tmp_path / "export").exists()


@pytest.fixture
def model_suggestions(label_pack, tmp_path):
    module, pack_path, _ = label_pack
    pack = json.loads(pack_path.read_text())
    suggestions = {
        "kind": "model_proposals",
        "reviewer": "model fixture </script>",
        "pack_sha256": module["digest"](pack_path),
        "images": [
            {
                "id": 0,
                "image_sha256": pack["images"][0]["sha256"],
                "candidates": [{"bbox_xyxy": [1, 2, 5, 8]}],
            },
            {"id": 1, "image_sha256": pack["images"][1]["sha256"], "candidates": []},
        ],
    }
    path = tmp_path / "suggestions.json"
    path.write_text(json.dumps(suggestions))
    return path, suggestions


def test_model_suggestions_render_separately_without_changing_source(label_pack, model_suggestions, tmp_path):
    module, pack_path, labels = label_pack
    source = pack_path.read_bytes()
    labels_before = json.dumps(labels)
    suggestions_path, _ = model_suggestions
    output = tmp_path / "review.html"
    module["render_review"](pack_path, output, suggestions_path)
    html = output.read_text()
    assert "/* MODEL_SUGGESTIONS */ null" not in html
    assert module["digest"](suggestions_path) in html
    assert "model fixture </script>" not in html
    assert "model fixture \\u003c/script>" in html
    assert pack_path.read_bytes() == source
    assert json.dumps(labels) == labels_before
    assert not (tmp_path / "reviewed_labels.json").exists()


@pytest.mark.parametrize(
    "case",
    ["pack", "image", "image_bytes", "duplicate", "unknown", "arm", "bounds", "nan", "bool", "reviewer"],
)
def test_invalid_model_suggestions_fail_before_render(label_pack, model_suggestions, tmp_path, case):
    module, pack_path, _ = label_pack
    path, document = model_suggestions
    row = document["images"][0]
    if case == "pack":
        document["pack_sha256"] = "stale"
    elif case == "image":
        row["image_sha256"] = "stale"
    elif case == "image_bytes":
        Image.new("RGB", (16, 12), "white").save(tmp_path / "0.jpg")
    elif case == "duplicate":
        document["images"].append(row)
    elif case == "unknown":
        row["id"] = 99
    elif case == "arm":
        row["candidates"][0]["arm"] = "left"
    elif case == "reviewer":
        document["reviewer"] = " "
    else:
        row["candidates"][0]["bbox_xyxy"][0] = {"bounds": -1, "nan": float("nan"), "bool": True}[case]
    path.write_text(json.dumps(document))
    output = tmp_path / "review.html"
    with pytest.raises(ValueError):
        module["render_review"](pack_path, output, path)
    assert not output.exists()


def test_model_proposal_does_not_replace_human_confirmation(label_pack, tmp_path):
    row = label_pack[2]["images"][0]
    row["arms"]["left"]["model_proposal"] = {"reviewer": "model fixture", "candidate": [1, 2, 5, 8]}
    row["review"]["confirmed"] = False
    with pytest.raises(ValueError, match="human review"):
        run_export(label_pack, tmp_path)
    assert not (tmp_path / "export").exists()


def test_confirmed_fixture_retains_model_proposal_provenance(label_pack, tmp_path):
    proposal = {"reviewer": "model fixture", "source_sha256": "fixture", "candidate": [1, 2, 5, 8]}
    label_pack[2]["images"][0]["arms"]["left"]["model_proposal"] = proposal
    run_export(label_pack, tmp_path)
    reviewed = json.loads((tmp_path / "export/reviewed_labels.json").read_text())
    assert reviewed["images"][0]["arms"]["left"]["model_proposal"] == proposal


@pytest.fixture
def camera_context(label_pack, tmp_path):
    module, pack_path, _ = label_pack
    pack = json.loads(pack_path.read_text())
    for i, image in enumerate(pack["images"]):
        image.update(frame_index=10 + i, timestamp=(10 + i) / 30, camera=module["CAMERAS"][i])
    pack_path.write_text(json.dumps(pack))
    context = {"version": 1, "pack_sha256": module["digest"](pack_path), "images": []}
    for image in pack["images"]:
        row = {k: image[k] for k in ("id", "episode_index", "frame_index", "timestamp")}
        row.update(image_sha256=image["sha256"], views=[])
        for camera in set(module["CAMERAS"]) - {image["camera"]}:
            path = tmp_path / f"context_{image['id']}_{camera}.jpg"
            Image.new("RGB", (16, 12), "green").save(path)
            row["views"].append(
                {
                    "camera": camera,
                    "file_name": path.name,
                    "sha256": module["digest"](path),
                    "width": 16,
                    "height": 12,
                }
            )
        context["images"].append(row)
    path = tmp_path / "context.json"
    path.write_text(json.dumps(context))
    return path, context


def test_camera_context_preserves_pack_and_label_storage_identity(label_pack, camera_context, tmp_path):
    module, pack_path, labels = label_pack
    original_pack, original_labels = pack_path.read_bytes(), json.dumps(labels)
    path, context = camera_context
    output = tmp_path / "context_review.html"
    module["render_review"](pack_path, output, context_path=path)
    assert pack_path.read_bytes() == original_pack and json.dumps(labels) == original_labels
    html = output.read_text()
    assert "/* CAMERA_CONTEXT */ null" not in html
    assert context["images"][0]["views"][0]["file_name"] in html
    assert module["digest"](path) in html
    assert 'key = "rebot-grippers-" + payload.pack_sha256' in html
    assert not (tmp_path / "reviewed_labels.json").exists()


@pytest.mark.parametrize(
    "case", ["timestamp", "frame", "camera", "bytes", "dimensions", "pack", "missing", "duplicate", "outside"]
)
def test_invalid_camera_context_rejected_before_render(label_pack, camera_context, tmp_path, case):
    module, pack_path, _ = label_pack
    path, context = camera_context
    row = context["images"][0]
    view = row["views"][0]
    if case == "timestamp":
        row["timestamp"] += 1 / 30
    elif case == "frame":
        row["frame_index"] += 1
    elif case == "camera":
        view["camera"] = json.loads(pack_path.read_text())["images"][0]["camera"]
    elif case == "bytes":
        Image.new("RGB", (16, 12), "red").save(tmp_path / view["file_name"])
    elif case == "dimensions":
        view["width"] += 1
    elif case == "pack":
        context["pack_sha256"] = "stale"
    elif case == "missing":
        context["images"].pop()
    elif case == "duplicate":
        context["images"].append(row)
    else:
        view["file_name"] = "../outside.jpg"
    path.write_text(json.dumps(context))
    output = tmp_path / "context_review.html"
    with pytest.raises(ValueError):
        module["render_review"](pack_path, output, context_path=path)
    assert not output.exists()


def test_context_extraction_requests_exact_timestamp_in_other_views(
    label_pack, camera_context, tmp_path, monkeypatch
):
    module, pack_path, _ = label_pack
    pack = json.loads(pack_path.read_text())
    (tmp_path / "source.json").write_text(json.dumps(pack["source"]))
    records = [
        SimpleNamespace(
            episode_index=im["episode_index"],
            frame_indices=(im["frame_index"],),
            frame_timestamps=(im["timestamp"],),
        )
        for im in pack["images"]
    ]
    calls = []

    class Provider:
        def __init__(self, *args, **kwargs):
            pass

        def frames_at(self, record, timestamps, camera):
            calls.append((record.episode_index, timestamps, camera))
            return [Image.new("RGB", (16, 12), "blue")]

    globals_ = module["prepare_context"].__globals__
    monkeypatch.setitem(globals_, "iter_episodes", lambda *a, **kw: iter(records))
    monkeypatch.setitem(globals_, "VideoFrameProvider", Provider)
    before = pack_path.read_bytes()
    output = tmp_path / "extracted_context.json"
    module["prepare_context"](tmp_path, pack_path, output)
    assert pack_path.read_bytes() == before
    assert calls == [
        (im["episode_index"], [im["timestamp"]], camera)
        for im in pack["images"]
        for camera in module["CAMERAS"]
        if camera != im["camera"]
    ]
    assert len(module["read_context"](pack_path, output)["images"]) == 3
    with pytest.raises(ValueError, match="fresh context"):
        module["prepare_context"](tmp_path, pack_path, output)
