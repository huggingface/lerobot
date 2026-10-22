# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""A process boundary to the independently installed, licensed research runtime."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tarfile
from pathlib import Path

import numpy as np

from lerobot.utils.import_utils import require_package

from ..runtime.artifacts import file_checksum
from ..runtime.types import Artifact, ItemResult, ModuleSpec, Outcome, Resources
from .config import GROUNDED_REVISION, RECIPE, HandBackendConfig

RECIPE_ARGUMENTS = (
    "--camera-backend",
    "droid-w",
    "--hand-calibration",
    "calibrated",
    "--joint-refinement",
    "world11-half",
    "--wrist-rotation-target-gamma",
    "1",
    "--wrist-orientation-refinement",
    "articulated-world5-half",
    "--wrist-position-refinement",
    "raw-plane-world5-sixteenth",
)


def provisioned_environment(config: HandBackendConfig) -> dict[str, str]:
    if not config.research_terms_accepted:
        raise ValueError("Accept the upstream research terms and obtain your own MANO/model rights first")
    root = Path(config.root)
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("Install git in the provisioned research runtime")
    head = subprocess.check_output([git, "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != GROUNDED_REVISION:
        raise ValueError(f"Grounded runtime must be pinned to {GROUNDED_REVISION}")
    subprocess.run([git, "-C", str(root), "diff", "--quiet", "HEAD", "--"], check=True)
    for name in ("python", "macrodata_python", "droid_python", "droid_moge_checkpoint"):
        if not Path(getattr(config, name)).is_file():
            raise ValueError(f"Provision the worker's {name} before execution")
    if not (Path(config.droid_root) / "grounded_worker.py").is_file():
        raise ValueError("Provision the optimized DROID-W worker")
    if not Path(config.droid_model_cache).is_dir():
        raise ValueError("Provision the offline FiT3D/model cache")
    manifest = Path(config.assets_manifest)
    if file_checksum(manifest)[0] != config.assets_sha256:
        raise ValueError("Worker asset-manifest digest differs from the planned runtime")
    assets = json.loads(manifest.read_text())
    if not assets.get("files"):
        raise ValueError("An asset manifest must list the pre-provisioned model and runtime files")
    for asset in assets["files"]:
        if file_checksum(Path(asset["path"]))[0] != asset["sha256"]:
            raise ValueError(f"Provisioned asset changed: {asset['path']}")
    environment = os.environ.copy()
    for name in ("HF_HOME", "TORCH_HOME"):
        value = assets.get("environment", {}).get(name)
        if not value or not Path(value).is_dir():
            raise ValueError(f"Asset manifest must name the provisioned {name} directory")
        environment[name] = value
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    # Import the pinned external checkout in its own interpreter, never in the
    # LeRobot process. No external source/weights are copied into this package.
    environment["PYTHONPATH"] = str(root / "src")
    return environment


def run_backend(command: list[str], *, config: HandBackendConfig, environment: dict, log: Path) -> int:
    with log.open("w") as stream:
        process = subprocess.Popen(
            command,
            cwd=config.root,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=config.timeout_s)
        except BaseException:
            # Stop the camera and hand descendants too, including on timeout.
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise


def validate_dataset(root: Path, minimum_segment_frames: int = 2) -> dict:
    require_package("pyarrow", "dataset")
    import pyarrow.parquet as pq

    info = json.loads((root / "meta/info.json").read_text())
    if not str(info.get("codebase_version", "")).startswith("v3."):
        raise ValueError("The research backend must export LeRobot v3")
    if info["features"]["action"]["shape"] != [102]:
        raise ValueError("Expected the 102D bimanual wrist/articulation action")
    frames = 0
    episodes: dict[int, int] = {}
    for path in sorted((root / "data").glob("chunk-*/file-*.parquet")):
        columns = [
            "action",
            "annotation.hand_confidence",
            "annotation.source_frame_index",
            "annotation.source_timestamp_s",
            "episode_index",
            "frame_index",
        ]
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=columns):
            values = batch.to_pydict()
            action = np.asarray(values["action"])
            confidence = np.asarray(values["annotation.hand_confidence"])
            if action.shape != (batch.num_rows, 102) or not np.isfinite(action).all():
                raise ValueError("Invalid or missing hand articulation in exported actions")
            if (
                confidence.shape != (batch.num_rows, 2)
                or not np.isfinite(confidence).all()
                or np.any(confidence <= 0)
            ):
                raise ValueError("Both hands must have positive, finite observation confidence")
            for episode, frame in zip(values["episode_index"], values["frame_index"], strict=True):
                expected = episodes.get(episode, 0)
                if frame != expected:
                    raise ValueError("Hand episodes must have dense, ordered frame indexes")
                episodes[episode] = expected + 1
            frames += batch.num_rows
    if frames <= 0 or frames != info["total_frames"]:
        raise ValueError("Exported frame counts differ from metadata")
    if (
        len(episodes) != info["total_episodes"]
        or sorted(episodes) != list(range(len(episodes)))
        or any(length < minimum_segment_frames for length in episodes.values())
    ):
        raise ValueError("Invalid hand segment lengths or episode metadata")
    return info


class HandReconstructionStage:
    """Checkpoint a warm group of videos; keep every per-video filtering result."""

    def __init__(self, backend: dict, minimum_segment_frames: int = 30):
        self.config = HandBackendConfig(**backend)
        self.minimum_segment_frames = minimum_segment_frames
        self.spec = ModuleSpec(
            "human_pov_hands",
            "1",
            "asset",
            {"dataset": None, "report": None, "log": None},
            resources=Resources(cpus=2, gpus=1, memory_gb=32),
            hf_jobs_compatible=True,
        )

    def scientific_config(self):
        return {
            "recipe": RECIPE,
            "runtime_revision": GROUNDED_REVISION,
            "assets_sha256": self.config.assets_sha256,
            "minimum_segment_frames": self.minimum_segment_frames,
        }

    def setup(self, context):
        self.environment = provisioned_environment(self.config)
        if "CUDA_VISIBLE_DEVICES" not in self.environment:
            nvidia_smi = shutil.which("nvidia-smi")
            if nvidia_smi is None:
                raise RuntimeError("Assign a CUDA GPU before hand reconstruction")
            devices = subprocess.check_output(
                [nvidia_smi, "--query-gpu=index", "--format=csv,noheader"], text=True
            ).split()
            if len(devices) != 1:
                raise ValueError("Assign exactly one GPU per hand worker using CUDA_VISIBLE_DEVICES")
            self.environment["CUDA_VISIBLE_DEVICES"] = devices[0]

    def process_batch(self, items, context):
        results = []
        for item in items:
            directory = context.scratch / item.item_id
            inputs = directory / "input"
            inputs.mkdir(parents=True)
            for video in item.payload["videos"]:
                stem = f"{video['source_video_id']:08d}"
                for kind, suffix in (("video", ".mp4"), ("calibration", ".intrinsics.json")):
                    context.store.download(Artifact(**video[kind]), inputs / (stem + suffix))
            output = directory / "dataset"
            log = directory / "backend.log"
            config = self.config
            devices = self.environment.get("CUDA_VISIBLE_DEVICES", "")
            if not devices or devices == "-1":
                raise ValueError("A GPU must be assigned by the processing runtime")
            command = [
                config.python,
                str(Path(config.root) / "scripts/process_egocentric_videos.py"),
                "--input",
                str(inputs),
                "--output",
                str(output),
                "--gpus",
                devices,
                "--macrodata-python",
                config.macrodata_python,
                "--droid-root",
                config.droid_root,
                "--droid-python",
                config.droid_python,
                "--droid-moge-checkpoint",
                config.droid_moge_checkpoint,
                "--droid-model-cache",
                config.droid_model_cache,
                "--minimum-segment-frames",
                str(self.minimum_segment_frames),
                "--no-resume",
                *RECIPE_ARGUMENTS,
            ]
            try:
                code = run_backend(command, config=config, environment=self.environment, log=log)
            finally:
                if log.is_file():
                    log_artifact = context.write_asset(item, "log", log)
            report_path = output.with_name("dataset.manifest.json")
            if code or not report_path.is_file():
                raise RuntimeError(
                    f"Hand backend failed (exit {code}); retained log under {context.attempt}/outputs/{item.item_id}/log"
                )
            report = json.loads(report_path.read_text())
            expected = {f"{v['source_video_id']:08d}.mp4" for v in item.payload["videos"]}
            actual = report.get("videos", [])
            if (
                len(actual) != len(expected)
                or {v["video"] for v in actual} != expected
                or any(v["status"] not in {"accepted", "rejected"} for v in actual)
                or report["summary"].get("failed") != 0
            ):
                raise ValueError("The backend omitted a video or reported incomplete work")
            report["lerobot_hand_recipe"] = self.scientific_config()
            report["source_videos"] = item.payload["videos"]
            report_path.write_text(json.dumps(report, allow_nan=False))
            artifacts = [context.write_asset(item, "report", report_path), log_artifact]
            if not report["summary"]["accepted"]:
                results.append(
                    ItemResult(
                        item.item_id,
                        Outcome.REJECTED,
                        tuple(artifacts),
                        "No uninterrupted, directly observed two-hand segments",
                    )
                )
                continue
            info = validate_dataset(output, self.minimum_segment_frames)
            if info["total_frames"] != report["summary"]["accepted_two_hand_frames"]:
                raise ValueError("Backend report and exported frame counts differ")
            bundle = directory / "dataset.tar"
            with tarfile.open(bundle, "w") as archive:
                for path in sorted(output.rglob("*")):
                    if path.is_file():
                        if path.is_symlink():
                            raise ValueError("Dataset exports must not contain symlinks")
                        archive.add(path, arcname=path.relative_to(output).as_posix())
            artifacts.append(context.write_asset(item, "dataset", bundle))
            results.append(ItemResult(item.item_id, Outcome.COMPLETED, tuple(artifacts)))
        return results

    def teardown(self):
        pass
