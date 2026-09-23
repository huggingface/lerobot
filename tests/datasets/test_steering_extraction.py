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
    for raw in ['["tape", "tape"]', '["1", "2", "3", "4", "5"]', '{"objects": []}', "[null]"]:
        with pytest.raises(ValueError):
            parse(raw)


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
