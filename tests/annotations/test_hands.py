# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Native representation, frame alignment, immutable assets and process lifecycle."""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pyarrow as pa
import pytest
from scipy.spatial.transform import Rotation

from lerobot.annotations.hands.assets import download_release, file_checksum, verify_assets
from lerobot.annotations.hands.config import (
    ACTION_SCHEMA,
    HandBackendConfig,
)
from lerobot.annotations.hands.egohand.geometry import optical_wrist, pack_action
from lerobot.annotations.hands.egohand.output import NATIVE_FIELDS, export_actions, observed_spans
from lerobot.annotations.hands.processing import RequestProcess
from lerobot.annotations.hands.writer import ACTION_NAMES, PARAMETERS, attach_actions, validate_calibration


@pytest.fixture
def source_table():
    return pa.table(
        {
            "episode_index": [8] * 40,
            "frame_index": list(range(40)),
            "timestamp": pa.array(1000 + np.arange(40) / 10, type=pa.float32()),
            "action": pa.array([[float(i), 2.0] for i in range(40)], type=pa.list_(pa.float32(), 2)),
            "observation.state": [[float(i)] for i in range(40)],
        }
    )


def native_report(directory, observed=True):
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    if observed:
        path = directory / "segment.npz"
        fields = {name: np.zeros((30, *shape), dtype=np.float32) for name, shape in PARAMETERS.items()}
        np.savez(
            path,
            source_frame_index=np.arange(5, 35),
            timestamp_s=np.arange(5, 35) / 10,
            action=np.ones((30, 156), dtype=np.float32),
            **fields,
        )
        paths.append(str(path))
    return {
        "action_schema": ACTION_SCHEMA,
        "gt_scale_alignment": False,
        "motion_infilling": False,
        "rectified_frames": 40,
        "rectified_fps": 10.0,
        "accepted_frames": 30 if observed else 0,
        "segments": paths,
    }


@pytest.mark.parametrize("observed", [True, False])
def test_attach_preserves_frames_native_fields_and_source_actions(source_table, tmp_path, observed):
    annotated, summary = attach_actions(
        source_table, native_report(tmp_path, observed), directory=tmp_path, fps=10
    )
    assert len(annotated) == len(source_table) == 40
    for name in ("episode_index", "frame_index", "timestamp", "observation.state"):
        assert annotated[name].equals(source_table[name])
    assert annotated["annotation.original_action"].equals(source_table["action"])
    valid = np.asarray(annotated["annotation.hand_action_valid"].to_pylist())
    assert valid.sum() == (30 if observed else 0)
    assert annotated["annotation.hand_action_valid"].equals(
        annotated["annotation.hand_action_observed_valid"]
    )
    actions = np.asarray(annotated["action"].to_pylist())
    assert actions.shape == (40, 156) and not actions[~valid].any()
    reference = np.asarray(annotated["annotation.hand_reference_frame"].to_pylist())
    assert (reference[~valid] == -1).all() and (reference[valid] == 5).all()
    for name, shape in PARAMETERS.items():
        assert np.asarray(annotated["annotation." + name].to_pylist()).shape == (40, np.prod(shape))
    assert summary["infilled_frames"] == 0 and summary["unmeasured_frames"] == (10 if observed else 40)
    assert len(ACTION_NAMES) == len(set(ACTION_NAMES)) == 156
    assert not any("mano" in name or "confidence" in name for name in annotated.column_names)


@pytest.mark.parametrize(
    "failure",
    [
        "overlap",
        "clock",
        "length",
        "outside",
        "nonfinite",
        "source_action",
        "wrong_schema",
        "infilling",
        "gt_scale",
        "short_span",
        "missing_native",
    ],
)
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
    elif failure == "wrong_schema":
        report["action_schema"] = "mano102"
    elif failure == "infilling":
        report["motion_infilling"] = True
    elif failure == "gt_scale":
        report["gt_scale_alignment"] = True
    else:
        path = tmp_path / "segment.npz"
        with np.load(path) as data:
            values = dict(data)
        if failure == "nonfinite":
            values["action"][0, 0] = np.nan
        elif failure == "short_span":
            values = {k: v[:29] for k, v in values.items()}
        else:
            values["soma_hand_pose"] = np.zeros((30, 2, 15, 3, 3))
        np.savez(path, **values)
    with pytest.raises(ValueError):
        attach_actions(source_table, report, directory=tmp_path, fps=10)


def test_native_articulation_retains_every_dof_and_order():
    pose = Rotation.from_rotvec(np.arange(144).reshape(48, 3) * 0.001).as_matrix().reshape(1, 2, 24, 3, 3)
    wrist = np.arange(12, dtype=np.float32)[None]
    output = pack_action(wrist, pose)
    assert output.shape == (1, 156)
    np.testing.assert_array_equal(output[:, :12], wrist)
    np.testing.assert_allclose(output[:, 12:], np.arange(144)[None] * 0.001, atol=1e-7)


@pytest.mark.parametrize(
    "bad",
    [
        np.full((1, 2, 24, 3, 3), np.nan),
        np.zeros((1, 2, 24, 3, 3)),
        np.broadcast_to(np.diag([-1.0, 1.0, 1.0]), (1, 2, 24, 3, 3)),
    ],
)
def test_reject_invalid_rotations(bad):
    with pytest.raises(ValueError):
        pack_action(np.zeros((1, 12)), bad)


def test_rig_basis_camera_equivariance():
    orientation, wrist, camera = Rotation.random(3, random_state=np.random.default_rng(3)).as_matrix()
    transformed = orientation @ camera @ orientation.T @ wrist
    np.testing.assert_allclose(
        optical_wrist(transformed, orientation), camera @ optical_wrist(wrist, orientation), atol=1e-12
    )


def test_direct_support_never_bridges_missing_or_single_hand_frames():
    direct = np.ones((100, 2), dtype=bool)
    direct[35, 1] = False
    direct[70:99] = False
    camera = np.ones(100, dtype=bool)
    assert observed_spans(direct, camera) == [(0, 35), (36, 70)]
    # Isolated last-frame detections are never accepted, even with export minimum 1.
    assert observed_spans(direct, camera, minimum_frames=1) == [(0, 35), (36, 70)]


def test_export_resets_origin_and_preserves_metric_scale(tmp_path):
    n = 70
    fields = {name: np.zeros((n, *shape)) for name, shape in NATIVE_FIELDS.items()}
    for name in ("wrist_rotation_camera", "soma_wrist_orient", "soma_hand_pose"):
        fields[name][...] = np.eye(3)
    joints = np.broadcast_to([0.2, 0.3, 0.6], (n, 2, 21, 3)).copy()
    direct = np.ones((n, 2), dtype=bool)
    direct[35] = False
    prediction = tmp_path / "hands.npz"
    camera = tmp_path / "camera.npz"
    poses = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    poses[:, 0, 3] = 3 + np.arange(n) * 0.01
    np.savez(prediction, direct_valid=direct, joints_camera=joints, **fields)
    np.savez(camera, camera_pose=poses, source_frame_index=np.arange(n), timestamp_s=np.arange(n) / 30)
    report = export_actions(prediction, camera, tmp_path / "actions", frames=n, fps=30)
    assert report["accepted_frames"] == 69 and len(report["segments"]) == 2
    with np.load(report["segments"][1]) as segment:
        np.testing.assert_allclose(segment["action"][0, :3], [0.6, -0.2, -0.3], atol=1e-7)
        np.testing.assert_allclose(segment["action"][1, :3], [0.6, -0.21, -0.3], atol=1e-7)
        assert segment["soma_hand_pose"].shape == (34, 2, 24, 3, 3)
    joints *= 2
    np.savez(prediction, direct_valid=direct, joints_camera=joints, **fields)
    doubled = export_actions(prediction, camera, tmp_path / "doubled", frames=n, fps=30)
    with np.load(doubled["segments"][0]) as segment:
        np.testing.assert_allclose(segment["action"][0, :3], [1.2, -0.4, -0.6], atol=1e-7)


@pytest.mark.parametrize(
    "change", [{"fx": 0}, {"width": 32}, {"cx": float("nan")}, {"model": "opencv_fisheye", "distortion": []}]
)
def test_calibration_is_measured_and_resolution_bound(change):
    calibration = {
        "model": "pinhole",
        "width": 64,
        "height": 64,
        "fx": 50.0,
        "fy": 51.0,
        "cx": 32.0,
        "cy": 32.0,
        "distortion": [],
    }
    assert validate_calibration(calibration, 64, 64) == calibration
    with pytest.raises(ValueError):
        validate_calibration({**calibration, **change}, 64, 64)


def asset_fixture(tmp_path):
    files = []
    for name in ("checkpoint", "detector", "moge"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        sha, size = file_checksum(path)
        files.append({"path": name, "sha256": sha, "size": size})
    document = {"files": files, "identity_backend": "mhr", "gt_scale_alignment": False}
    manifest = tmp_path / "assets.json"
    manifest.write_text(json.dumps(document))
    config = HandBackendConfig(
        checkpoint=str(tmp_path / "checkpoint"),
        detector=str(tmp_path / "detector"),
        droid_moge_checkpoint=str(tmp_path / "moge"),
        assets_manifest=str(manifest),
        assets_sha256=file_checksum(manifest)[0],
    )
    return document, config


@pytest.mark.parametrize("repo_type", ["model", "space"])
def test_asset_download_is_allowlisted_pinned_and_anonymous(tmp_path, monkeypatch, repo_type):
    document, config = asset_fixture(tmp_path)
    download = Mock(return_value=config.assets_manifest)
    snapshot = Mock()
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)
    profile = {
        "repo": "public/hand-runtime",
        "repo_type": repo_type,
        "revision": "a" * 40,
        "manifest_sha256": config.assets_sha256,
    }
    receipt = download_release(profile, "token", tmp_path / "download")
    assert receipt["files"] == 3
    assert snapshot.call_args.kwargs["allow_patterns"] == [x["path"] for x in document["files"]]
    assert snapshot.call_args.kwargs["revision"] == "a" * 40
    assert download.call_args.kwargs["repo_type"] == snapshot.call_args.kwargs["repo_type"] == repo_type
    assert download.call_args.kwargs["token"] is snapshot.call_args.kwargs["token"] is False
    document["files"][0]["path"] = "../escape"
    Path(config.assets_manifest).write_text(json.dumps(document))
    profile["manifest_sha256"] = file_checksum(Path(config.assets_manifest))[0]
    with pytest.raises(ValueError, match="Unexpected"):
        download_release(profile, "token", tmp_path / "download")
    assert snapshot.call_count == 1


def test_corrupt_models_fail_before_loading(tmp_path, monkeypatch):
    document, config = asset_fixture(tmp_path)
    monkeypatch.setattr("lerobot.annotations.hands.assets.CHECKPOINT_SHA256", document["files"][0]["sha256"])
    assert verify_assets(config)["identity_backend"] == "mhr"
    (tmp_path / "detector").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        verify_assets(config)


def test_request_process_supports_out_of_order_replies_and_owned_cleanup(tmp_path):
    program = tmp_path / "worker.py"
    program.write_text("""import concurrent.futures,json,sys,time
pool=concurrent.futures.ThreadPoolExecutor(2)
def reply(row):
 time.sleep(row['delay'])
 print(json.dumps({'id':row['id'],'status':'ok','result':row['value']}),flush=True)
for line in sys.stdin:pool.submit(reply,json.loads(line))
""")
    client = RequestProcess(
        [sys.executable, str(program)],
        cwd=tmp_path,
        environment=os.environ.copy(),
        log=tmp_path / "worker.log",
    )
    first = client.submit({"value": "first", "delay": 0.2})
    second = client.submit({"value": "second", "delay": 0})
    assert second.result(timeout=5) == "second" and first.result(timeout=5) == "first"
    client.close()
    client.close()
    assert client.process.poll() is not None
    with pytest.raises(RuntimeError):
        client.submit({"value": "closed", "delay": 0}).result(timeout=5)


def test_dead_worker_fails_pending_requests_instead_of_hanging(tmp_path):
    program = tmp_path / "worker.py"
    program.write_text(
        "import sys\nsys.stdin.readline()\nprint('fixture model load failed',file=sys.stderr,flush=True)\nsys.exit(1)\n"
    )
    client = RequestProcess(
        [sys.executable, str(program)],
        cwd=tmp_path,
        environment=os.environ.copy(),
        log=tmp_path / "worker.log",
    )
    try:
        with pytest.raises(RuntimeError, match="fixture model load failed"):
            client.submit({}).result(timeout=5)
    finally:
        client.close()


def test_coordinator_pairs_concurrent_episodes_with_real_native_export(tmp_path, monkeypatch):
    """Exercise the actual isolated bridge with CPU fixture model/camera endpoints."""
    from concurrent.futures import ThreadPoolExecutor

    from lerobot.annotations.hands.processing import HandAnnotator

    root = tmp_path / "runtime"
    package = root / "fixture_camera"
    (package / "egocentric").mkdir(parents=True)
    (package / "infra").mkdir()
    for name in ("egocentric", "infra"):
        (package / name / "__init__.py").write_text("")
    # The production target is Linux. Supply its affinity interface to the
    # isolated fixture process when this integration test runs on macOS.
    (package / "__init__.py").write_text(
        "import os\nif not hasattr(os,'sched_getaffinity'): os.sched_getaffinity=lambda pid: set(range(4))\n"
    )
    (package / "egocentric/config.py").write_text("""class EgocentricConfig:
 def __init__(self,**kwargs): self.__dict__.update(kwargs)
 def validate(self):
  assert self.camera_backend=='droid-w' and self.hand_calibration=='calibrated'
  assert self.droid_sample_fps==10. and self.droid_resolution==320
  assert self.droid_moge_checkpoint.is_file()
""")
    (package / "egocentric/intrinsics.py").write_text("""class CameraIntrinsics:
 def __init__(self,width,height,matrix,distortion,model,source): self.matrix=matrix
""")
    (package / "egocentric/video.py").write_text("""from types import SimpleNamespace
import numpy as np
def rectify_video(video,intrinsics,directory):
 directory.mkdir(parents=True)
 path=directory/'frames.rgb'
 values=np.memmap(path,mode='w+',dtype=np.uint8,shape=(40,32,32,3));values[:]=0;values.flush()
 return SimpleNamespace(frames_path=path,frame_count=40,height=32,width=32,matrix=intrinsics.matrix,fps=10.)
""")
    (package / "egocentric/droidw_camera.py").write_text("""import json
import numpy as np
def estimate_rectified_camera(video,directory,settings):
 directory.mkdir(parents=True)
 archive=directory/'camera.npz'
 np.savez(archive,camera_pose=np.tile(np.eye(4),(40,1,1)),source_frame_index=np.arange(40),timestamp_s=np.arange(40)/10)
 receipt=directory/'worker/output/camera/moge-ablation.json';receipt.parent.mkdir(parents=True)
 receipt.write_text(json.dumps(dict(mode='on',metric_depth_reg=True,gt_scale_fit=False,counts=dict(depth_predictions=1))))
 return archive,{'backend':'droid-w'}
""")
    (package / "infra/droidw_worker.py").write_text("_WORKERS={}\ndef close_workers(): pass\n")
    model = root / "hand-python"
    model.write_text(
        f"#!{sys.executable}\n"
        + """import json,sys
from pathlib import Path
import numpy as np
from lerobot.annotations.hands.egohand.output import NATIVE_FIELDS
calls=0
for line in sys.stdin:
 row=json.loads(line);calls+=1
 assert row['row']['frame_count']==40 and row['orientation']=='sensor'
 fields={name:np.zeros((40,*shape)) for name,shape in NATIVE_FIELDS.items()}
 for name in ('soma_hand_pose','soma_wrist_orient','wrist_rotation_camera'): fields[name][...]=np.eye(3)
 path=Path(row['directory'])/'hands.npz'
 np.savez(path,joints_camera=np.broadcast_to([.2,.3,.6],(40,2,21,3)),direct_valid=np.ones((40,2),dtype=bool),**fields)
 print(json.dumps({'id':row['id'],'status':'ok','result':{'prediction':str(path),'components':{'warm_calls':calls}}}),flush=True)
"""
    )
    model.chmod(0o700)
    checkpoint = root / "moge.pt"
    checkpoint.write_bytes(b"CPU transport fixture; no model loaded")
    config = HandBackendConfig(
        root=str(root),
        module="fixture_camera",
        python=sys.executable,
        egohand_python=str(model),
        gem_root=str(root),
        trt_cache=str(root / "cache"),
        droid_moge_checkpoint=str(checkpoint),
    )
    source = str(Path(__file__).resolve().parents[2] / "src")
    monkeypatch.setattr(
        "lerobot.annotations.hands.processing.provisioned_environment",
        lambda config: {**os.environ, "PYTHONPATH": os.pathsep.join([source, str(root)])},
    )
    calibration = {
        "width": 32,
        "height": 32,
        "fx": 25.0,
        "fy": 25.0,
        "cx": 16.0,
        "cy": 16.0,
        "model": "pinhole",
    }
    annotator = HandAnnotator(config, root / "coordinator")
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = [
                pool.submit(annotator.annotate, root / f"episode-{i}.mp4", calibration, root / f"work-{i}")
                for i in range(2)
            ]
            reports = [future.result(timeout=30) for future in pending]
        assert sorted(r["hand_components"]["warm_calls"] for r in reports) == [1, 2]
        for i, report in enumerate(reports):
            assert report["accepted_frames"] == 40 and report["action_schema"] == ACTION_SCHEMA
            assert Path(report["segments"][0]).is_relative_to(root / f"work-{i}")
            with np.load(report["segments"][0]) as segment:
                np.testing.assert_allclose(segment["action"][0, :3], [0.6, -0.2, -0.3], atol=1e-7)
                assert segment["action"].shape == (40, 156)
    finally:
        annotator.close()
    assert annotator.process.poll() is not None


@pytest.mark.parametrize("tf32_override", [None, "1"])
def test_native_interpreters_isolate_cuda_libraries_and_strip_hub_credentials(
    tmp_path, monkeypatch, tf32_override
):
    from lerobot.annotations.hands.config import RUNTIME_REVISION
    from lerobot.annotations.hands.processing import provisioned_environment

    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    source = tmp_path / "source.py"
    source.write_text("# frozen source\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "source.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.test",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-m",
            "Fixture",
        ],
        check=True,
        capture_output=True,
    )
    revision = subprocess.check_output(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True).strip()
    monkeypatch.setattr("lerobot.annotations.hands.processing.GEM_REVISION", revision)
    monkeypatch.setenv("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1")
    untrusted = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], capture_output=True)
    assert untrusted.returncode != 0 and b"dubious ownership" in untrusted.stderr
    receipt = tmp_path / "runtime-source.json"
    receipt.write_text(json.dumps({"revision": RUNTIME_REVISION, "files": []}))
    checkpoint = tmp_path / "moge.pt"
    checkpoint.write_bytes(b"CPU transport fixture")
    config = HandBackendConfig(
        root=str(tmp_path),
        runtime_source_sha256=file_checksum(receipt)[0],
        module="camera_runtime",
        gem_root=str(tmp_path),
        python=sys.executable,
        egohand_python=sys.executable,
        droid_python=sys.executable,
        droid_moge_checkpoint=str(checkpoint),
    )
    monkeypatch.setenv("HF_TOKEN", "must-not-reach-native-code")
    monkeypatch.setenv("ANNOTATION_MANIFEST", "must-not-reach-native-code")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/controller/lib")
    monkeypatch.setenv("DROID_LIBRARY_PATH", "/camera/lib")
    monkeypatch.setenv("EGO_LIBRARY_PATH", "/hand/lib")
    if tf32_override is None:
        monkeypatch.delenv("NVIDIA_TF32_OVERRIDE", raising=False)
    else:
        monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", tf32_override)
    monkeypatch.setattr(
        "lerobot.annotations.hands.processing.verify_assets",
        lambda config: {"environment": {"HF_HOME": str(tmp_path / "offline-cache")}},
    )
    environment = provisioned_environment(config)
    assert "HF_TOKEN" not in environment and "ANNOTATION_MANIFEST" not in environment
    assert environment["HF_HUB_OFFLINE"] == environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["LD_LIBRARY_PATH"] == "/camera/lib"
    assert environment["EGO_LIBRARY_PATH"] == "/hand/lib"
    assert environment["NVIDIA_TF32_OVERRIDE"] == "0"
    assert os.environ["LD_LIBRARY_PATH"] == "/controller/lib"
    assert os.environ.get("NVIDIA_TF32_OVERRIDE") == tf32_override
    source.write_text("# changed source\n")
    with pytest.raises(subprocess.CalledProcessError):
        provisioned_environment(config)
