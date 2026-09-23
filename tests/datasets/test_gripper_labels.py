# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
import json
import runpy
from pathlib import Path

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


def run_export(label_pack, tmp_path):
    module, pack_path, labels = label_pack
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(labels))
    return module["export_coco"](pack_path, path, tmp_path / "export")


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
