# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Fresh calibrated DROID-W+MoGe, warm EgoHand/SOMA, bounded two-clip scheduling."""

import importlib
import json
import os
import signal
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from .config import HandBackendConfig
from .egohand.output import export_actions
from .intrinsics import resolve_calibration
from .processing import RequestProcess


def main():
    protocol, sys.stdout = sys.stdout, sys.stderr
    config = HandBackendConfig(**json.loads(sys.argv[1]))
    camera_config = importlib.import_module(config.module + ".egocentric.config").EgocentricConfig
    calibration_type = importlib.import_module(config.module + ".egocentric.intrinsics").CameraIntrinsics
    rectify = importlib.import_module(config.module + ".egocentric.video").rectify_video
    estimate_camera = importlib.import_module(
        config.module + ".egocentric.droidw_camera"
    ).estimate_rectified_camera
    camera_workers = importlib.import_module(config.module + ".infra.droidw_worker")
    import cv2
    import numpy as np

    Path(config.trt_cache).mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(min(24, len(os.sched_getaffinity(0))))

    environment = dict(os.environ, NVIDIA_TF32_OVERRIDE="0")
    environment["PYTHONPATH"] = os.pathsep.join([config.gem_root, environment["PYTHONPATH"]])
    environment.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    if environment.get("EGO_LIBRARY_PATH"):
        environment["LD_LIBRARY_PATH"] = environment["EGO_LIBRARY_PATH"]
    hand = RequestProcess(
        [config.egohand_python, "-m", "lerobot.annotations.hands.egohand.worker", json.dumps(asdict(config))],
        cwd=config.gem_root,
        environment=environment,
        log=Path(config.trt_cache) / "hand-worker.log",
    )
    camera_pool = ThreadPoolExecutor(max_workers=1)
    exports = ThreadPoolExecutor(max_workers=1)
    reply_lock = threading.Lock()
    slots = threading.BoundedSemaphore(2)
    stopping = threading.Event()

    def stop_owned_workers():
        stopping.set()
        # These process records belong to this coordinator, never GPU-wide PIDs.
        for worker in list(camera_workers._WORKERS.values()):
            if worker.process.poll() is None:
                os.killpg(worker.process.pid, signal.SIGTERM)
        hand.close()

    def interrupted(signum, frame):
        stop_owned_workers()
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def camera(request):
        directory = Path(request["directory"])
        directory.mkdir(parents=True, exist_ok=False)
        calibration = resolve_calibration(request["calibration"], Path(request["video"]), directory, config)
        matrix = np.array(
            [[calibration["fx"], 0, calibration["cx"]], [0, calibration["fy"], calibration["cy"]], [0, 0, 1]],
            dtype=float,
        )
        intrinsics = calibration_type(
            calibration["width"],
            calibration["height"],
            matrix,
            np.asarray(calibration.get("distortion", [])),
            calibration["model"],
            Path(request["video"]),
        )
        video = rectify(Path(request["video"]), intrinsics, directory / "rgb")
        row = {
            "source": request["video"],
            "frames_path": str(video.frames_path),
            "frame_count": video.frame_count,
            "height": video.height,
            "width": video.width,
            "matrix": video.matrix.tolist(),
            "fps": video.fps,
        }
        hands = hand.submit({"row": row, "directory": str(directory), "orientation": request["orientation"]})
        settings = camera_config(
            input_root=Path(request["video"]).parent,
            output_root=directory / "unused",
            repo_id="local/soma-annotation",
            macrodata_python=Path(config.egohand_python),
            gpus=("0",),
            camera_backend="droid-w",
            hand_calibration="calibrated",
            droid_root=Path(config.droid_root),
            droid_python=Path(config.droid_python),
            droid_moge_checkpoint=Path(config.droid_moge_checkpoint),
            droid_model_cache=Path(config.droid_model_cache),
            droid_resolution=320,
            droid_sample_fps=10.0,
        )
        settings.validate()
        archive, statistics = estimate_camera(video, directory / "camera", settings)
        # Preserve the MoGe-on arm: never silently substitute a camera without its depth prior.
        receipt = directory / "camera/worker/output/camera/moge-ablation.json"
        evidence = json.loads(receipt.read_text())
        if (
            evidence.get("mode") != "on"
            or evidence.get("metric_depth_reg") is not True
            or evidence.get("gt_scale_fit") is not False
            or evidence.get("counts", {}).get("depth_predictions", 0) < 1
        ):
            raise ValueError("The camera worker did not verify DROID-W+MoGe without GT scale fitting")
        return row, hands, archive, statistics, calibration, evidence

    def finish(request, pending):
        try:
            row, hands, archive, statistics, calibration, evidence = pending.result()
            hand_result = hands.result(timeout=config.timeout_s)
            report = export_actions(
                Path(hand_result["prediction"]),
                archive,
                Path(request["directory"]) / "actions",
                frames=row["frame_count"],
                fps=row["fps"],
            )
            report.update(
                camera_statistics=statistics,
                hand_components=hand_result["components"],
                orientation=request["orientation"],
                camera_intrinsics=calibration,
                camera_depth_prior=evidence,
            )
            response = {"id": request["id"], "status": "ok", "result": report}
        except Exception as error:
            traceback.print_exc()
            response = {"id": request["id"], "status": "failed", "error": str(error)}
        finally:
            slots.release()
        with reply_lock:
            protocol.write(json.dumps(response) + "\n")
            protocol.flush()

    try:
        for line in sys.stdin:
            if stopping.is_set():
                break
            request = json.loads(line)
            slots.acquire()
            exports.submit(finish, request, camera_pool.submit(camera, request))
    finally:
        stop_owned_workers()
        camera_pool.shutdown(wait=True, cancel_futures=True)
        exports.shutdown(wait=True, cancel_futures=True)
        camera_workers.close_workers()


if __name__ == "__main__":
    main()
