# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Standalone JSON-lines bridge to an independently installed research runtime.

Only orchestration lives here; upstream inference/refinement code remains in its
own interpreter and retains its own license. No filtered dataset is exported.
"""

import contextlib
import importlib
import json
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np


def preserve_infilling(result, directory, geometry):
    """Retain native infiller output; coordinate/action conversion stays upstream.

    The strict native export mask is kept separately. No new missing-frame
    interpolation, confidence fabrication, or motion estimator is introduced.
    """
    with np.load(result["hand_prediction"], allow_pickle=False) as archive:
        values = dict(archive)
    count = result["rectified_frames"]
    joints = values["joints_camera"]
    # The observed-only refinement can leave unsupported frames unavailable.
    # On infilled frames retain the packaged infiller's joints, without applying
    # corrections whose support protocol explicitly excludes those frames.
    infilled = values["infilled"]
    if "packaged_joints_camera" in values:
        joints = joints.copy()
        joints[infilled] = values["packaged_joints_camera"][infilled]
    observed = values["raw_direct_valid"]
    pose, shape = values["mano_pose"], values["mano_shape"]
    wrist, camera = values["wrist_rotation_camera"], values["world_from_camera"]
    available = values["camera_tracked"] & np.isfinite(camera).all(axis=(1, 2))
    available &= ((observed | infilled) & np.isfinite(values["confidence"]) & (values["confidence"] > 0)).all(
        axis=1
    )
    for array in (joints, wrist, pose, shape):
        available &= np.isfinite(array).reshape(count, -1).all(axis=1)
    strict = np.zeros(count, dtype=bool)
    for path in result["segments"]:
        with np.load(path, allow_pickle=False) as segment:
            strict[segment["source_frame_index"]] = True
    if int(strict.sum()) != result["accepted_frames"]:
        raise ValueError("Native observed-action report is inconsistent")
    # Conversion happens per continuous recoverable span, keeping the same FLU
    # frame across an occlusion bridged by the native motion infiller.
    edges = np.flatnonzero(np.diff(np.pad(available.astype(np.int8), (1, 1))))
    spans = []
    for index, (start, end) in enumerate(edges.reshape(-1, 2)):
        wrist_action, _ = geometry.fuse_segment_to_flu(joints[start:end], wrist[start:end], camera[start:end])
        action = geometry.policy_action_with_mano(wrist_action, pose[start:end])
        path = directory / f"dense-actions-{index:04d}.npz"
        np.savez_compressed(
            path,
            action=action,
            source_frame_index=values["source_frame_index"][start:end],
            timestamp_s=values["timestamp_s"][start:end],
            confidence=values["confidence"][start:end],
            observed=observed[start:end],
            infilled=infilled[start:end],
            observed_valid=strict[start:end],
        )
        spans.append(str(path))
    result.update(
        dense_segments=spans,
        recoverable_frames=int(available.sum()),
        export_policy="native-infilling-with-observation-provenance-v1",
    )
    return result


def main():
    options = json.loads(sys.argv[1])
    prefix = options["module"]
    service = None
    try:
        with contextlib.redirect_stdout(sys.stderr):
            configuration = importlib.import_module(prefix + ".egocentric.config").EgocentricConfig
            backend = importlib.import_module(prefix + ".egocentric.backends").MacrodataService
            process_video = importlib.import_module(prefix + ".egocentric.pipeline")._process_video
            geometry = importlib.import_module(prefix + ".egocentric.geometry")
        for line in sys.stdin:
            try:
                request = json.loads(line)
                directory = Path(request["directory"])
                inputs = directory / "input"
                inputs.mkdir(parents=True, exist_ok=True)
                video = inputs / "episode.mp4"
                shutil.copyfile(request["video"], video)
                video.with_suffix(".intrinsics.json").write_text(json.dumps(request["calibration"]))
                with contextlib.redirect_stdout(sys.stderr):
                    config = configuration(
                        input_root=inputs,
                        output_root=directory / "unused",
                        repo_id="local/hand-annotation",
                        macrodata_python=Path(options["macrodata_python"]),
                        gpus=("0",),
                        camera_backend="droid-w",
                        hand_fp16=True,
                        detector_mode="vendored_track_local",
                        sparse_hand_recovery=False,
                        hand_calibration="calibrated",
                        joint_refinement="world11-half",
                        wrist_rotation_target_gamma=1.0,
                        wrist_orientation_refinement="articulated-world5-half",
                        wrist_position_refinement="raw-plane-world5-sixteenth",
                        minimum_segment_frames=30,
                        droid_root=Path(options["droid_root"]),
                        droid_python=Path(options["droid_python"]),
                        droid_moge_checkpoint=Path(options["droid_moge_checkpoint"]),
                        droid_model_cache=Path(options["droid_model_cache"]),
                    )
                    config.validate()
                    if service is None:
                        service = backend(config, "0")
                    result = process_video(config, service, video, directory / "native")
                    result = preserve_infilling(result, directory, geometry)
                print(json.dumps({"status": "ok", "result": result}), flush=True)
            except Exception:
                traceback.print_exc(file=sys.stderr)
                print(json.dumps({"status": "failed"}), flush=True)
    finally:
        if service is not None:
            service.close()
        with contextlib.redirect_stdout(sys.stderr):
            importlib.import_module(prefix + ".infra.droidw_worker").close_workers()


if __name__ == "__main__":
    main()
