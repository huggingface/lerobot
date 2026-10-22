# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Frame preservation, asset isolation and native hand-process contracts."""

import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pyarrow as pa
import pytest

from lerobot.annotations.hands.assets import (
    MANO_FILES,
    checkpoint_guard,
    download_mano,
    file_checksum,
    resolve_mano,
)
from lerobot.annotations.hands.config import COMPUTE, estimate
from lerobot.annotations.hands.writer import ACTION_NAMES, attach_actions, validate_calibration


@pytest.fixture
def source_table():
    return pa.table(
        {
            "episode_index": pa.array([8] * 40, type=pa.int64()),
            "frame_index": pa.array(range(40), type=pa.int64()),
            # A long float32 clock has quantization larger than a fixed 10 us
            # tolerance; native elapsed times still align with these same rows.
            "timestamp": pa.array(1000 + np.arange(40) / 10, type=pa.float32()),
            "action": pa.array([[float(i), 2.0] for i in range(40)], type=pa.list_(pa.float32(), 2)),
            "observation.state": pa.array([[float(i)] for i in range(40)], type=pa.list_(pa.float32(), 1)),
        }
    )


def native_report(directory, *, observed=True):
    segments = []
    if observed:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "segment.npz"
        np.savez(
            path,
            source_frame_index=np.arange(5, 35),
            timestamp_s=np.arange(5, 35) / 10,
            action=np.ones((30, 102), dtype=np.float32),
            confidence=np.full((30, 2), 0.9),
        )
        segments.append(str(path))
    return {
        "rectified_frames": 40,
        "rectified_fps": 10.0,
        "accepted_frames": 30 if observed else 0,
        "segments": segments,
    }


@pytest.mark.parametrize("observed", [True, False])
def test_attach_keeps_every_frame_and_original_action(source_table, tmp_path, observed):
    report = native_report(tmp_path, observed=observed)
    annotated, summary = attach_actions(source_table, report, directory=tmp_path, fps=10)
    assert len(annotated) == len(source_table) == 40
    for name in ("episode_index", "frame_index", "timestamp", "observation.state"):
        assert annotated[name].equals(source_table[name])
    assert annotated["annotation.original_action"].equals(source_table["action"])
    valid = np.array(annotated["annotation.hand_action_valid"].to_pylist()).ravel()
    actions = np.array(annotated["action"].to_pylist())
    assert valid.sum() == (30 if observed else 0)
    assert not actions[~valid].any()
    assert np.isfinite(actions).all() and actions.shape == (40, 102)
    reference = np.array(annotated["annotation.hand_reference_frame"].to_pylist()).ravel()
    assert np.all(reference[~valid] == -1)
    assert np.all(reference[valid] == 5)
    assert summary["unmeasured_frames"] == (10 if observed else 40)
    assert len(ACTION_NAMES) == len(set(ACTION_NAMES)) == 102


@pytest.mark.parametrize("failure", ["overlap", "clock", "length", "outside", "nonfinite", "source_action"])
def test_reject_bad_native_alignment(source_table, tmp_path, failure):
    report = native_report(tmp_path)
    if failure == "overlap":
        report["segments"] *= 2
    elif failure == "clock":
        report["rectified_fps"] = 15
    elif failure == "length":
        report["rectified_frames"] = 39
    elif failure == "outside":
        report["segments"] = [str(tmp_path.parent / "escape.npz")]
    elif failure == "source_action":
        source_table = source_table.append_column("annotation.original_action", source_table["action"])
    else:
        path = tmp_path / "segment.npz"
        with np.load(path) as data:
            values = dict(data)
        values["action"][0, 0] = np.nan
        np.savez(path, **values)
    with pytest.raises(ValueError):
        attach_actions(source_table, report, directory=tmp_path, fps=10)


def model_api(private=True, owner="alice", revision="a" * 40):
    api = Mock()
    api.whoami.return_value = {"name": owner}
    api.model_info.return_value = SimpleNamespace(private=private, sha=revision)
    api.list_repo_files.return_value = list(MANO_FILES.values())
    return api


@pytest.mark.parametrize(
    "repo,accepted,private",
    [("alice/mano", False, True), ("mirror/mano", True, True), ("alice/mano", True, False)],
)
def test_mano_rights_privacy_and_ownership_required(repo, accepted, private):
    with pytest.raises(ValueError):
        resolve_mano(repo, "private-token", accepted, api_factory=lambda **kw: model_api(private))


def test_private_asset_pin_download_and_no_mirror_fallback(tmp_path):
    api = model_api()
    source = resolve_mano("alice/mano", "private-token", True, api_factory=lambda **kw: api)
    files = {}
    for filename in MANO_FILES.values():
        path = tmp_path / filename
        path.write_bytes(filename.encode())
        files[filename] = path
    download = Mock(side_effect=lambda repo, filename, **kw: files[filename])
    receipt = download_mano(
        source, "private-token", tmp_path / "private", api_factory=lambda **kw: api, download=download
    )
    assert all(
        call.kwargs["revision"] == "a" * 40 and call.kwargs["token"] == "private-token"
        for call in download.call_args_list
    )
    guard = checkpoint_guard(tmp_path / "guard", receipt)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import huggingface_hub as h; print(h.hf_hub_download('public/mirror','MANO_RIGHT.safetensors')); h.hf_hub_download('public/mirror','other_MANO.safetensors')",
        ],
        env={**os.environ, "PYTHONPATH": str(guard), "HF_HUB_OFFLINE": "1"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert str(receipt["right"]["path"]) in result.stdout
    assert result.returncode != 0 and "Unapproved MANO" in result.stderr
    assert "private-token" not in (guard / "sitecustomize.py").read_text()
    files["MANO_LEFT.safetensors"].write_bytes(b"changed source")
    # The private copy is checksum-bound independently of mutable source-cache handles.
    assert file_checksum(tmp_path / "private/MANO_LEFT.safetensors") == (
        receipt["left"]["sha256"],
        receipt["left"]["size"],
    )
    (tmp_path / "private/MANO_LEFT.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        checkpoint_guard(tmp_path / "guard", receipt)


def test_changed_assets_fail_before_model_load(tmp_path):
    api = model_api()
    source = resolve_mano("alice/mano", "t", True, api_factory=lambda **kw: api)
    api.model_info.return_value.sha = "b" * 40
    download = Mock()
    with pytest.raises(ValueError, match="changed"):
        download_mano(source, "t", tmp_path, api_factory=lambda **kw: api, download=download)
    download.assert_not_called()


def test_no_fabricated_hf_gpu_benchmarks():
    assert set(COMPUTE) == {"h200", "a100-large", "rtx-pro-6000"}
    assert estimate(3000)["seconds"] is None
    assert estimate(3000, 50)["seconds"] == 60
    assert estimate(3000, 50)["basis"] == "user planning assumption"
    with pytest.raises(ValueError):
        estimate(3000, float("nan"))


@pytest.mark.parametrize(
    "change",
    [
        {"fx": 0},
        {"fy": 51},
        {"width": 32},
        {"cx": float("nan")},
        {"model": "opencv_fisheye", "distortion": []},
    ],
)
def test_calibration_is_measured_and_resolution_bound(change):
    calibration = {
        "model": "pinhole",
        "width": 64,
        "height": 64,
        "fx": 50.0,
        "fy": 50.0,
        "cx": 32.0,
        "cy": 32.0,
        "distortion": [],
    }
    assert validate_calibration(calibration, 64, 64) == calibration
    with pytest.raises(ValueError):
        validate_calibration({**calibration, **change}, 64, 64)


def test_native_infiller_predictions_are_retained_but_not_mislabeled(tmp_path, source_table):
    from lerobot.annotations.hands.bridge import preserve_infilling

    n = 40
    observed = np.ones((n, 2), dtype=bool)
    infilled = np.zeros((n, 2), dtype=bool)
    observed[30:35, 1] = False
    infilled[30:35, 1] = True
    observed[39] = False
    joints = np.zeros((n, 2, 21, 3))
    joints[39] = np.nan  # Unrecoverable; the infiller did not produce a pose.
    archive = tmp_path / "native.npz"
    np.savez(
        archive,
        joints_camera=joints,
        packaged_joints_camera=joints,
        infilled=infilled,
        raw_direct_valid=observed,
        mano_pose=np.ones((n, 2, 96)),
        mano_shape=np.zeros((n, 2, 10)),
        wrist_rotation_camera=np.tile(np.eye(3), (n, 2, 1, 1)),
        world_from_camera=np.tile(np.eye(4), (n, 1, 1)),
        camera_tracked=np.ones(n, dtype=bool),
        confidence=np.ones((n, 2)),
        source_frame_index=np.arange(n),
        timestamp_s=np.arange(n) / 10,
    )
    report = native_report(tmp_path)
    # A separate native observed-only span: valid strict training frames 0–29.
    strict = tmp_path / "strict.npz"
    np.savez(strict, source_frame_index=np.arange(30))
    report.update(hand_prediction=str(archive), segments=[str(strict)])
    geometry = Mock()
    geometry.fuse_segment_to_flu.side_effect = lambda j, w, c: (np.ones((len(j), 12)), j)
    geometry.policy_action_with_mano.side_effect = lambda w, p: np.ones((len(w), 102))
    report = preserve_infilling(report, tmp_path, geometry)
    table, summary = attach_actions(source_table, report, directory=tmp_path, fps=10)
    assert summary["valid_frames"] == 39
    assert summary["observed_training_frames"] == 30
    assert summary["infilled_frames"] == 5
    assert summary["unmeasured_frames"] == 1
    assert table["annotation.hand_action_valid"][32].as_py() is True
    assert table["annotation.hand_infilled"][32].as_py() == [False, True]
    assert table["annotation.hand_action_observed_valid"][32].as_py() is False
    assert table["annotation.hand_action_valid"][39].as_py() is False
    assert table["annotation.hand_reference_frame"][32].as_py() == 0
    geometry.fuse_segment_to_flu.assert_called_once()
    # Reject a corrupted export trying to count infilled poses as observed.
    dense_path = report["dense_segments"][0]
    with np.load(dense_path) as data:
        values = dict(data)
    values["observed_valid"][32] = True
    np.savez(dense_path, **values)
    with pytest.raises(ValueError, match="Infilled predictions"):
        attach_actions(source_table, report, directory=tmp_path, fps=10)


def test_droid_native_bridge_reuses_one_service_across_requests(tmp_path, monkeypatch):
    from pathlib import Path

    from lerobot.annotations.hands.config import HandBackendConfig
    from lerobot.annotations.hands.processing import HandAnnotator

    package = tmp_path / "runtime/research_runtime"
    (package / "egocentric").mkdir(parents=True)
    (package / "infra").mkdir()
    for folder in (package, package / "egocentric", package / "infra"):
        (folder / "__init__.py").write_text("")
    (package / "egocentric/config.py").write_text("""class EgocentricConfig:
    def __init__(self, **kwargs): self.__dict__.update(kwargs)
    def validate(self):
        assert self.camera_backend == "droid-w"
        assert self.hand_calibration == "calibrated"
        assert self.hand_fp16 and not self.sparse_hand_recovery
        assert self.joint_refinement == "world11-half"
        assert self.wrist_rotation_target_gamma == 1.0
        assert self.wrist_orientation_refinement == "articulated-world5-half"
        assert self.wrist_position_refinement == "raw-plane-world5-sixteenth"
""")
    (package / "egocentric/backends.py").write_text("""class MacrodataService:
    def __init__(self, config, gpu): self.calls = 0
    def close(self): pass
""")
    (package / "egocentric/pipeline.py").write_text("""import numpy as np

def _process_video(config, service, video, work_root):
    service.calls += 1
    work_root.mkdir(parents=True)
    path = work_root / "hands.npz"
    n = 40
    np.savez(path, joints_camera=np.zeros((n,2,21,3)), infilled=np.zeros((n,2),dtype=bool), raw_direct_valid=np.ones((n,2),dtype=bool), mano_pose=np.ones((n,2,96)), mano_shape=np.zeros((n,2,10)), wrist_rotation_camera=np.tile(np.eye(3),(n,2,1,1)), world_from_camera=np.tile(np.eye(4),(n,1,1)), camera_tracked=np.ones(n,dtype=bool), confidence=np.ones((n,2)), source_frame_index=np.arange(n), timestamp_s=np.arange(n)/10)
    return dict(hand_prediction=str(path), rectified_frames=n, rectified_fps=10, segments=[], accepted_frames=0, service_calls=service.calls)
""")
    (package / "egocentric/geometry.py").write_text("""import numpy as np

def fuse_segment_to_flu(j,w,c): return np.zeros((len(j),12)),j

def policy_action_with_mano(w,p): return np.zeros((len(w),102))
""")
    (package / "infra/droidw_worker.py").write_text("def close_workers(): pass\n")
    config = HandBackendConfig(root=str(package.parent), module="research_runtime", python=sys.executable)
    monkeypatch.setattr(
        "lerobot.annotations.hands.processing.provisioned_environment",
        lambda *args: {**os.environ, "PYTHONPATH": str(package.parent)},
    )
    video = tmp_path / "source.mp4"
    video.write_bytes(b"transport fixture; no model inference")
    worker = HandAnnotator(config, {}, tmp_path / "warm")
    try:
        first = worker.annotate(video, {}, tmp_path / "first")
        second = worker.annotate(video, {}, tmp_path / "second")
        assert (first["service_calls"], second["service_calls"]) == (1, 2)
        assert first["recoverable_frames"] == second["recoverable_frames"] == 40
        assert len(first["dense_segments"]) == 1
    finally:
        worker.close()
    assert worker.process.poll() is not None
    assert Path(first["dense_segments"][0]).is_file()
