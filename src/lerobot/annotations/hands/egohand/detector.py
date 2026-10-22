# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""The qualified FP32 TensorRT detector and complete-clip v3 association."""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .adapter import validate_intrinsics
from .association import annotations_from_candidates


def load_detector(checkpoint: Path, cache: Path):
    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise ValueError("The validated detector requires NVIDIA_TF32_OVERRIDE=0")
    import onnxruntime as ort
    import torch  # noqa: F401 - load CUDA shared libraries before ONNX Runtime
    from gem.common.detectors.hand import RTDETRHandONNX

    cache.mkdir(parents=True, exist_ok=True)
    options_class, session_class = ort.SessionOptions, ort.InferenceSession

    def options():
        value = options_class()
        value.intra_op_num_threads = 4
        value.inter_op_num_threads = 1
        value.add_session_config_entry("session.intra_op.allow_spinning", "0")
        return value

    def session(*args, **kwargs):
        kwargs["providers"] = [
            (
                "TensorrtExecutionProvider",
                {
                    "device_id": 0,
                    "trt_fp16_enable": False,
                    "trt_int8_enable": False,
                    "trt_max_workspace_size": 2 * 1024**3,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(cache),
                    "trt_timing_cache_enable": True,
                    "trt_timing_cache_path": str(cache),
                    "trt_profile_min_shapes": "images:1x3x640x640,orig_target_sizes:1x2",
                    "trt_profile_opt_shapes": "images:8x3x640x640,orig_target_sizes:8x2",
                    "trt_profile_max_shapes": "images:8x3x640x640,orig_target_sizes:8x2",
                },
            ),
            ("CUDAExecutionProvider", {"device_id": 0, "use_tf32": 0, "cudnn_conv_algo_search": "HEURISTIC"}),
            "CPUExecutionProvider",
        ]
        return session_class(*args, **kwargs)

    with patch.object(ort, "SessionOptions", options), patch.object(ort, "InferenceSession", session):
        model = RTDETRHandONNX(
            checkpoint,
            device="cuda",
            score_thr=0.15,  # spellchecker:disable-line
            top1_per_class=False,
            pad=0.0,
            square=False,
            clip=True,
        )
    if "TensorrtExecutionProvider" not in model.execution_providers:
        raise RuntimeError("TensorRT detector provider is unavailable; no silent precision/backend fallback")
    model.experimental_precision = "TensorRT FP32 detector, FP16/INT8/TF32 disabled"
    return model


def detect_annotations(model, frames, matrix, fps, *, batch_size=8):
    from gem.common.camera_adapters import SquarePixelTransform

    matrix = validate_intrinsics(matrix)
    height, width = frames[0].shape[:2]
    transform = SquarePixelTransform.from_intrinsics(matrix, width, height)
    candidates = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0, len(frames), batch_size):
            images = [
                transform.resize_image(frames[i]) for i in range(start, min(start + batch_size, len(frames)))
            ]
            pixels = np.concatenate(list(pool.map(model._preprocess, images)))
            sizes = np.asarray([[image.shape[1], image.shape[0]] for image in images], dtype=np.int64)
            candidates.extend(model._run(pixels, sizes))
    raw = [[asdict(d) for d in frame] for frame in candidates]
    detected = time.perf_counter()
    annotations, diagnostic = annotations_from_candidates(raw, transform.intrinsics, transform.width, fps)
    for record in annotations.values():
        record["intrinsics"] = matrix.tolist()
        for row in record["instances"]:
            row["bbox"] = transform.inverse_boxes(row["bbox"]).tolist()
    diagnostic.update(
        device="cuda",
        precision=model.experimental_precision,
        execution_providers=model.execution_providers,
        detection_image_size=[transform.width, transform.height],
        detector_wall_s=detected - started,
        association_wall_s=time.perf_counter() - detected,
    )
    return annotations, diagnostic
