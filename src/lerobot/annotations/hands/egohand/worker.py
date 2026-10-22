# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Warm EgoHand worker with one complete clip of detection ahead of pose."""

import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..assets import file_checksum
from ..config import CHECKPOINT_SHA256, HandBackendConfig
from .adapter import normalize_hands, verify_joint_layout
from .orientation import UprightFrames, restore_predictions, upright_intrinsics


def configure_predictor(config):
    import cv2
    import torch
    from gem.inference.pose import load_predictor

    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    before = file_checksum(Path(config.checkpoint))
    if before[0] != CHECKPOINT_SHA256:
        raise ValueError("The inference-only checkpoint differs from the validated release")
    verify_joint_layout(Path(config.soma_assets) / "SOMAHand.npz")
    predictor, release = load_predictor(
        Path(config.checkpoint),
        task="hand",
        release="hand",
        soma_assets=Path(config.soma_assets),
        device="cuda",
    )
    if file_checksum(Path(config.checkpoint)) != before:
        raise ValueError("EgoHand checkpoint changed during model loading")

    class Backbone(torch.nn.Module):
        def __init__(self, original):
            super().__init__()
            self.original = original

        def forward(self, image):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return self.original(image).float()

    predictor.model.backbone = Backbone(predictor.model.backbone).eval()
    return predictor, release


def prepare(request, detector):
    from .detector import detect_annotations

    started = time.perf_counter()
    row = request["row"]
    frames = np.memmap(
        row["frames_path"],
        mode="r",
        dtype=np.uint8,
        shape=(row["frame_count"], row["height"], row["width"], 3),
    )
    matrix = np.asarray(row["matrix"], dtype=np.float64)
    if request["orientation"] == "aria-gen1-cw90":
        frames = UprightFrames(frames)
        matrix = upright_intrinsics(matrix, row["height"])
    elif request["orientation"] != "sensor":
        raise ValueError("Choose an explicit supported hand-input orientation")
    # Pixel reads stay lazy to bound host RAM on ordinary long recordings.
    # Association still sees the entire clip, in source order, with unchanged crops.
    annotations, provider = detect_annotations(detector, frames, matrix, row["fps"])
    if set(annotations) != set(range(row["frame_count"])):
        raise ValueError("Detection did not preserve the complete source frame grid")
    return request, frames, annotations, provider, time.perf_counter() - started


def infer(predictor, item):
    from gem.pose.postprocess import smooth_hand_predictions

    from .batching import predict_frames_batched
    from .geometry import native_parameters

    request, frames, annotations, provider, prepared_s = item
    row = request["row"]
    started = time.perf_counter()
    parts = {}
    output, topology = predict_frames_batched(
        predictor, frames, annotations, batch_hands=64, timings=parts, parallel_prepare=True
    )
    result = {
        "format": "gem_pose_predictions_v1",
        "task": "hand",
        "mode": "framewise_video",
        "source": row["source"],
        "source_fps": row["fps"],
        "source_frame_count": len(frames),
        "coordinates": {"geometry": "camera_meters"},
        "topology": topology,
        "frames": output,
        "hand_view": "ego",
        "box_provider": provider,
    }
    result = smooth_hand_predictions(result, geometry=predictor.geometry)
    if request["orientation"] == "aria-gen1-cw90":
        result = restore_predictions(result, predictor.geometry, row["matrix"], row["width"], row["height"])
    values = normalize_hands(result, row["matrix"], "smoothed")
    values.update(native_parameters(result, predictor.geometry, "smoothed"))
    destination = Path(request["directory"]) / "soma-predictions.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **values)
    return {
        "prediction": str(destination),
        "components": {
            "detector_association_s": prepared_s,
            "pose_smoothing_export_s": time.perf_counter() - started,
            "pose_batches": parts,
            "note": "Overlapping components; do not sum for end-to-end FPS",
        },
    }


def main():
    protocol, sys.stdout = sys.stdout, sys.stderr
    config = HandBackendConfig(**json.loads(sys.argv[1]))
    import torch

    from .detector import load_detector

    predictor, _ = configure_predictor(config)
    detector = load_detector(Path(config.detector), Path(config.trt_cache) / "hand-detector")

    def next_clip():
        line = sys.stdin.readline()
        if not line:
            return None
        request = json.loads(line)
        try:
            with torch.inference_mode():
                return prepare(request, detector)
        except Exception as error:
            traceback.print_exc()
            return request, error

    def reply(value):
        protocol.write(json.dumps(value) + "\n")
        protocol.flush()

    with torch.inference_mode(), ThreadPoolExecutor(max_workers=1) as producer:
        pending = producer.submit(next_clip)
        while (item := pending.result()) is not None:
            pending = producer.submit(next_clip)
            request = item[0]
            try:
                if len(item) == 2:
                    raise item[1]
                result = infer(predictor, item)
                reply({"id": request["id"], "status": "ok", "result": result})
            except Exception as error:
                traceback.print_exc()
                reply({"id": request["id"], "status": "failed", "error": str(error)})
            del item


if __name__ == "__main__":
    main()
