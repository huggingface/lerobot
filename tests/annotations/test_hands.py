# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""CPU contract tests: real checkpoints, source binding and external-process I/O."""

import io
import json
import os
import sys
import tarfile
import time
from dataclasses import asdict, replace
from pathlib import Path

import draccus
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("datasets")

from lerobot.annotations.hands.config import (
    GROUNDED_REVISION,
    HandBackendConfig,
    HandPipelineConfig,
    estimate,
)
from lerobot.annotations.hands.pipeline import extract_dataset, materialize, run_hand_pipeline
from lerobot.annotations.hands.processing import (
    HandReconstructionStage,
    provisioned_environment,
    validate_dataset,
)
from lerobot.annotations.hands.source import RawHandVideoSource
from lerobot.annotations.runtime.artifacts import ArtifactStore, file_checksum
from lerobot.annotations.runtime.configs import StageConfig
from lerobot.annotations.runtime.pipeline import run_pipeline
from lerobot.annotations.runtime.planner import seal_plan
from lerobot.annotations.runtime.types import Outcome, Resources
from lerobot.annotations.runtime.worker import accepted_items


class CPUHandStage(HandReconstructionStage):
    """Only model inference is replaced; production orchestration stays real."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.spec = replace(self.spec, resources=Resources())

    def setup(self, context):
        self.environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / "raw"
    root.mkdir()
    calibration = {
        "model": "pinhole",
        "width": 64,
        "height": 64,
        "fx": 50,
        "fy": 50,
        "cx": 32,
        "cy": 32,
        "distortion": [],
    }
    for index in range(3):
        (root / f"video{index}.mp4").write_bytes(b"video" + bytes([index]))
        (root / f"video{index}.intrinsics.json").write_text(json.dumps(calibration))
    monkeypatch.setattr(
        "lerobot.annotations.hands.source.probe_video",
        lambda p: {
            "width": 64,
            "height": 64,
            "fps": 30,
            "frames": 150,
            "seconds": 5,
        },
    )
    return RawHandVideoSource(root, videos_per_group=2)


def test_estimate_hardware_and_group_scheduling(source):
    cfg = HandPipelineConfig()
    cfg.runtime.workers = 2
    result = estimate(cfg, source.groups)
    assert result["source_frames"] == 450
    assert result["estimated_seconds"] == pytest.approx(300 / 53.9)
    cfg.compute = "h200"
    assert estimate(cfg, source.groups)["estimated_fps_per_gpu"] is None
    cfg.planning_fps = 40
    assert estimate(cfg, source.groups)["estimated_seconds"] == 7.5
    assert estimate(cfg, source.groups)["basis"] == "user planning assumption"


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_invalid_planning_fps(value):
    with pytest.raises(ValueError, match="planning_fps"):
        HandPipelineConfig(planning_fps=value).validate()


def test_inputs_bind_calibration_content_and_mutations(source, tmp_path):
    store = ArtifactStore(tmp_path / "store")
    factory = "lerobot.annotations.hands.processing:HandReconstructionStage"
    stage = StageConfig("hands", factory, {"backend": {}})
    plan = seal_plan(
        store, source.dataset_ref, factory, stage.config, source.discover(stage, store, {}), shard_size=1
    )
    assert not store.exists(source.videos[0]["video"]["path"])
    calibration = source.root / "video0.intrinsics.json"
    calibration.write_text(calibration.read_text() + "\n")
    with pytest.raises(ValueError, match="Source changed"):
        source.stage_inputs(store, plan)
    updated = RawHandVideoSource(source.root, videos_per_group=2)
    new_plan = seal_plan(
        store, updated.dataset_ref, factory, stage.config, updated.discover(stage, store, {}), shard_size=1
    )
    assert plan.read_shard(store, 0)[0].item_id != new_plan.read_shard(store, 0)[0].item_id
    assert plan.read_shard(store, 1)[0].item_id == new_plan.read_shard(store, 1)[0].item_id


def test_missing_bad_calibration_fails_before_compute(source):
    calibration = source.root / "video0.intrinsics.json"
    calibration.unlink()
    with pytest.raises(ValueError, match="Missing per-video"):
        RawHandVideoSource(source.root)
    calibration.write_text('{"model":"pinhole","width":32,"height":64}')
    with pytest.raises(ValueError, match="resolution"):
        RawHandVideoSource(source.root)


def test_plan_starts_no_models_or_uploads(source, tmp_path, monkeypatch):
    cfg = HandPipelineConfig(input_root=source.root, output_root=tmp_path / "out")
    cfg.runtime.mode = "plan"
    monkeypatch.setattr(HandReconstructionStage, "setup", lambda *a: pytest.fail("model startup during plan"))
    result = run_hand_pipeline(cfg)
    assert result["status"] == "planned"
    assert not list((tmp_path / "out.run").glob("hand-inputs/*"))
    assert not (tmp_path / "out").exists()


def test_requires_research_rights_before_any_execution(source, tmp_path):
    cfg = HandPipelineConfig(input_root=source.root, output_root=tmp_path / "out")
    with pytest.raises(ValueError, match="upstream model rights"):
        run_hand_pipeline(cfg)


def test_config_round_trip():
    cfg = draccus.decode(HandPipelineConfig, {"compute": "h100", "runtime": {"workers": 2}})
    assert cfg.input_root == Path("raw-egocentric")
    # Nested defaults still disable automatic inference retries.
    assert cfg.runtime.max_retries == 0


def make_dataset(root, *, invalid=False):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        name: {"dtype": dtype, "shape": (size,), "names": None}
        for name, dtype, size in [
            ("action", "float32", 102),
            ("annotation.hand_confidence", "float32", 2),
            ("annotation.source_video_id", "int64", 1),
            ("annotation.source_frame_index", "int64", 1),
            ("annotation.source_timestamp_s", "float64", 1),
        ]
    }
    dataset = LeRobotDataset.create(
        "local/hand-fixture", fps=30, features=features, root=root, use_videos=False
    )
    for index in range(2):
        dataset.add_frame(
            {
                "action": np.full(102, np.nan if invalid else 0, dtype=np.float32),
                "annotation.hand_confidence": np.array([0.9, 0.8], dtype=np.float32),
                "annotation.source_video_id": np.array([0], dtype=np.int64),
                "annotation.source_frame_index": np.array([index], dtype=np.int64),
                "annotation.source_timestamp_s": np.array([index / 30], dtype=np.float64),
                "task": "research test",
            }
        )
    dataset.save_episode()
    dataset.finalize()


def test_nonfinite_articulation_rejected(tmp_path):
    make_dataset(tmp_path / "dataset", invalid=True)
    with pytest.raises(ValueError, match="articulation"):
        validate_dataset(tmp_path / "dataset")


@pytest.mark.parametrize("mode", ["normal", "omitted", "backend-failed", "rejected"])
def test_external_group_checkpoint_and_resume(source, tmp_path, monkeypatch, mode):
    # The stub exports a schema-valid hand dataset and accounts for every MP4.
    # It is an executable process, not a mock returning invented ItemResults.
    fixture = tmp_path / "fixture"
    make_dataset(fixture)
    monkeypatch.setenv("HAND_TEST_FIXTURE", str(fixture))
    monkeypatch.setenv("HAND_TEST_MODE", mode)
    backend = tmp_path / "backend"
    (backend / "scripts").mkdir(parents=True)
    script = backend / "scripts/process_egocentric_videos.py"
    script.write_text("""import json, os, shutil, sys
from pathlib import Path
args = sys.argv
assert args[args.index('--camera-backend') + 1] == 'droid-w'
assert args[args.index('--wrist-position-refinement') + 1] == 'raw-plane-world5-sixteenth'
inputs = Path(args[args.index('--input') + 1])
output = Path(args[args.index('--output') + 1])
videos = sorted(inputs.glob('*.mp4'))
assert all(v.with_suffix('.intrinsics.json').is_file() for v in videos)
shutil.copytree(os.environ['HAND_TEST_FIXTURE'], output)
(output / 'meta/egocentric_workflow.json').write_text(json.dumps({
 'source_video_index': {'0': videos[0].name},
 'episode_sources': [{'episode_index':0,'source_video_id':0,'start_frame':0,'end_frame':2}]
}))
report = {'videos': [{'video':v.name,'status':'accepted' if i==0 else 'rejected'} for i,v in enumerate(videos)],
 'summary':{'failed':0,'accepted':1,'accepted_two_hand_frames':2,'rectified_frames':150*len(videos)}}
mode = os.environ['HAND_TEST_MODE']
if mode == 'omitted': report['videos'] = []
if mode == 'rejected':
 for video in report['videos']: video['status'] = 'rejected'
 report['summary'].update(accepted=0,accepted_two_hand_frames=0)
output.with_name('dataset.manifest.json').write_text(json.dumps(report))
if mode == 'backend-failed': sys.exit(2)
""")
    config = HandBackendConfig(root=str(backend), python=sys.executable, research_terms_accepted=True)
    runtime = HandPipelineConfig().runtime
    runtime.run_uri = str(tmp_path / "store")
    stage = StageConfig(
        "hands",
        "tests.annotations.test_hands:CPUHandStage",
        {"backend": asdict(config), "minimum_segment_frames": 2},
    )
    if mode in {"omitted", "backend-failed"}:
        with pytest.raises((RuntimeError, ValueError), match="omitted a video|backend failed"):
            run_pipeline(source, [stage], runtime, before_execute=source.stage_inputs)
        assert not list((tmp_path / "store").rglob("accepted/*.parquet"))
        return
    store, completed = run_pipeline(source, [stage], runtime, before_execute=source.stage_inputs)
    plan, summary = completed["hands"]
    if mode == "rejected":
        assert summary.completed == 0 and summary.rejected == 2
        config = HandPipelineConfig(input_root=source.root, output_root=tmp_path / "rejected")
        report = materialize(store, plan, config, time.perf_counter(), {})
        assert report["status"] == "rejected" and report["episodes"] == 0
        assert report["rejected_video_ids"] == [0, 1, 2]
        assert not (config.output_root / "meta/info.json").exists()
        return
    assert summary.completed == 2
    outputs = list(accepted_items(store, plan))
    assert all(result.outcome == Outcome.COMPLETED for _, result in outputs)
    monkeypatch.setattr(CPUHandStage, "setup", lambda *a: pytest.fail("cached resume reran inference"))
    run_pipeline(source, [stage], runtime, before_execute=source.stage_inputs)
    config = HandPipelineConfig(input_root=source.root, output_root=tmp_path / "final")
    report = materialize(
        store, plan, config, time.perf_counter(), estimate(config, source.groups), fresh=True
    )
    assert report["episodes"] == 2 and report["accepted_frames"] == 4
    assert report["rejected_video_ids"] == [1]
    assert [entry["source_video_id"] for entry in report["episode_sources"]] == [0, 2]
    merged = pa.concat_tables(
        [pq.read_table(path) for path in sorted((config.output_root / "data").rglob("*.parquet"))]
    )
    assert merged["annotation.source_video_id"].to_pylist() == [0, 0, 2, 2]
    reused = materialize(store, plan, config, time.perf_counter(), {}, fresh=False)
    assert reused["reused"] and reused["episodes"] == 2
    # Materialize each group to exercise source-ID remapping without running
    # the full dataset aggregator on intentionally tiny metadata fixtures.
    from lerobot.annotations.hands.pipeline import remap_source_ids

    for index, (_, result) in enumerate(outputs):
        artifact = next(a for a in result.artifacts if a.name == "dataset")
        bundle = store.download(artifact, tmp_path / f"group{index}.tar")
        root = tmp_path / f"group{index}"
        extract_dataset(bundle, root)
        lineage = remap_source_ids(root)
        assert lineage[0]["source_video_id"] == (0 if index == 0 else 2)
        assert (
            pq.read_table(root / "data/chunk-000/file-000.parquet")["annotation.source_video_id"].to_pylist()
            == [index * 2] * 2
        )


@pytest.mark.parametrize("name", ["../escape", "/absolute", "weights/MANO.pkl"])
def test_bundle_path_rejected(tmp_path, name):
    bundle = tmp_path / "bad.tar"
    with tarfile.open(bundle, "w") as archive:
        info = tarfile.TarInfo(name)
        info.size = 1
        archive.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="bundle member"):
        extract_dataset(bundle, tmp_path / "out")


def test_asset_manifest_binds_actual_files(tmp_path, monkeypatch):
    cfg = HandBackendConfig(root=str(tmp_path), research_terms_accepted=True)
    for name in ("python", "macrodata_python", "droid_python", "droid_moge_checkpoint"):
        path = tmp_path / name
        path.touch()
        setattr(cfg, name, str(path))
    cfg.droid_root = cfg.droid_model_cache = str(tmp_path)
    (tmp_path / "grounded_worker.py").touch()
    asset = tmp_path / "asset"
    asset.write_bytes(b"model")
    manifest = tmp_path / "assets.json"
    manifest.write_text(
        json.dumps(
            {
                "files": [{"path": str(asset), "sha256": file_checksum(asset)[0]}],
                "environment": {"HF_HOME": str(tmp_path), "TORCH_HOME": str(tmp_path)},
            }
        )
    )
    cfg.assets_manifest, cfg.assets_sha256 = str(manifest), file_checksum(manifest)[0]
    monkeypatch.setattr(
        "lerobot.annotations.hands.processing.subprocess.check_output", lambda *a, **k: GROUNDED_REVISION
    )
    monkeypatch.setattr("lerobot.annotations.hands.processing.subprocess.run", lambda *a, **k: None)
    assert provisioned_environment(cfg)["HF_HUB_OFFLINE"] == "1"
    asset.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Provisioned asset changed"):
        provisioned_environment(cfg)
