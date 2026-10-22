# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Projection, provenance and native interpreter contracts, without model inference."""

import json
import sys
from types import SimpleNamespace

import av
import numpy as np
import pytest
import torch

from lerobot.annotations.hands.config import HandBackendConfig
from lerobot.annotations.hands.intrinsics import (
    METHOD,
    estimate_video,
    resolve_calibration,
    summarize_intrinsics,
)


def normalized(focal, width=3840, height=2160):
    return [[focal / width, 0, 0.5], [0, focal / height, 0.5], [0, 0, 1]]


def test_estimate_preserves_projection_and_records_assumptions():
    matrices = [normalized(f) for f in (4000, 4100, 4200, 4300, 9900)]
    result = summarize_intrinsics(matrices, [0, 187, 375, 562, 750], 3840, 2160)
    assert result["fx"] == result["fy"] == 4200
    assert result["measured"] is result["metric_accuracy_validated"] is False
    assert result["source"] == "moge-estimated" and result["distortion"] == []
    assert result["relative_focal_mad"][0] == pytest.approx(100 / 4200)
    # A camera-space ray must project to the same normalized pixel at either source resolution.
    doubled = summarize_intrinsics(matrices, result["sample_frames"], 7680, 4320)
    for name, resolution in (("fx", "width"), ("fy", "height"), ("cx", "width"), ("cy", "height")):
        assert result[name] / result[resolution] == doubled[name] / doubled[resolution]


@pytest.mark.parametrize("change", ["nan", "offcenter", "aspect", "bottom", "missing"])
def test_invalid_or_changed_camera_conventions_are_rejected(change):
    matrix = np.array(normalized(4000))
    if change == "nan":
        matrix[0, 0] = np.nan
    elif change == "offcenter":
        matrix[0, 2] = 0.4
    elif change == "aspect":
        matrix[1, 1] *= 2
    elif change == "bottom":
        matrix[2, 2] = 2
    with pytest.raises(ValueError):
        summarize_intrinsics([matrix.tolist()] if change != "missing" else [], [0], 3840, 2160)


def test_dataset_calibration_never_starts_estimator(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "subprocess.run", lambda *a, **kw: pytest.fail("No estimator for dataset calibration")
    )
    calibration = {"model": "pinhole", "fx": 50, "fy": 50}
    assert (
        resolve_calibration(calibration, tmp_path / "video.mp4", tmp_path, HandBackendConfig()) is calibration
    )
    supplied = summarize_intrinsics([normalized(4200)], [0], 3840, 2160)
    assert resolve_calibration(supplied, tmp_path / "video.mp4", tmp_path, HandBackendConfig()) is supplied


def test_estimator_uses_owned_camera_interpreter_and_bound_resolution(tmp_path, monkeypatch):
    calibration = {"method": METHOD, "width": 3840, "height": 2160}
    predicted = summarize_intrinsics([normalized(4200)], [0], 3840, 2160)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        (tmp_path / "estimated-intrinsics.json").write_text(json.dumps(predicted))

    monkeypatch.setattr("subprocess.run", run)
    config = HandBackendConfig(
        droid_python="/camera/python", droid_moge_checkpoint="/assets/small.pt", timeout_s=20
    )
    result = resolve_calibration(calibration, tmp_path / "video.mp4", tmp_path, config)
    assert result == predicted
    assert calls[0][0][:3] == ["/camera/python", "-m", "lerobot.annotations.hands.intrinsics"]
    assert calls[0][1] == {"check": True, "timeout": 20}
    with pytest.raises(ValueError, match="resolution"):
        resolve_calibration({**calibration, "width": 1920}, tmp_path / "video.mp4", tmp_path, config)


def test_real_video_sampling_passes_no_fov_and_loads_only_local_checkpoint(tmp_path, monkeypatch):
    video = tmp_path / "video.mp4"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 36, "yuv420p"
        for i in range(9):
            frame = av.VideoFrame.from_ndarray(np.full((36, 64, 3), i * 20, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    checkpoint = tmp_path / "small.pt"
    checkpoint.write_bytes(b"local checkpoint fixture; no inference")
    calls = []

    class Model:
        scale_head = True

        @classmethod
        def from_pretrained(cls, path):
            assert path == str(checkpoint)
            return cls()

        def to(self, device):
            assert device == "cuda"
            return self

        def eval(self):
            return self

        def infer(self, image, **kwargs):
            calls.append((image.shape, kwargs))
            return {"intrinsics": torch.tensor(normalized(50, 64, 36))}

    monkeypatch.setitem(sys.modules, "moge.model.v2", SimpleNamespace(MoGeModel=Model))
    original = torch.Tensor.to
    monkeypatch.setattr(
        torch.Tensor,
        "to",
        lambda self, *a, **kw: original(self, "cpu") if a == ("cuda",) else original(self, *a, **kw),
    )
    destination = tmp_path / "intrinsics.json"
    estimate_video(video, checkpoint, destination)
    report = json.loads(destination.read_text())
    assert report["sample_frames"] == [0, 2, 4, 6, 8]
    assert len(calls) == 5 and report["fx"] == pytest.approx(50)
    assert all(shape == (3, 36, 64) for shape, _ in calls)
    assert all(options == {"fov_x": None, "num_tokens": 1200, "use_fp16": False} for _, options in calls)
