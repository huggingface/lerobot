# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""One-batch-ahead CPU crop preparation with unchanged crops/model/serialization."""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch

from .adapter import validate_intrinsics

FIELDS = (
    "camera_translation",
    "joints_camera",
    "vertices_camera",
    "keypoints_2d",
    "keypoints_crop",
    "projection_valid",
    "faces",
    "intrinsics",
)


def _row_view(prediction, index):
    """Minimal one-row view consumed only by upstream predict_frame serialization."""
    values = {key: getattr(prediction, key)[index : index + 1] for key in FIELDS}
    return SimpleNamespace(
        **values,
        parameters={key: value[index : index + 1] for key, value in prediction.parameters.items()},
        head=SimpleNamespace(token=prediction.head.token[index : index + 1]),
    )


def _host_prediction(prediction):
    """Transfer each batched result once rather than synchronizing per hand/field."""
    return SimpleNamespace(
        **{key: getattr(prediction, key).detach().cpu() for key in FIELDS},
        parameters={key: value.detach().cpu() for key, value in prediction.parameters.items()},
        head=SimpleNamespace(token=prediction.head.token.detach().cpu()),
    )


def _restore_raster(prediction, transforms, source_boxes):
    if all(transform.is_identity for transform in transforms):
        return prediction
    pixels = prediction.keypoints_2d
    scales = pixels.new_tensor([[t.scale_x, t.scale_y] for t in transforms])
    pixels = (pixels + 0.5) / scales[:, None] - 0.5
    boxes = source_boxes.to(pixels.device)
    crop = (pixels - boxes[:, None, :2]) / boxes[:, None, 2:]
    inverse = prediction.intrinsics.new_tensor(
        [
            [[1 / t.scale_x, 0, 0.5 / t.scale_x - 0.5], [0, 1 / t.scale_y, 0.5 / t.scale_y - 0.5], [0, 0, 1]]
            for t in transforms
        ]
    )
    with torch.autocast(device_type=prediction.intrinsics.device.type, enabled=False):
        intrinsics = inverse @ prediction.intrinsics
    return replace(prediction, keypoints_2d=pixels, keypoints_crop=crop, intrinsics=intrinsics)


@torch.inference_mode()
def predict_frames_batched(
    predictor, frames, annotations, *, batch_hands=64, square_pixels=True, timings=None, parallel_prepare=True
):
    from gem.common.camera_adapters import SquarePixelTransform
    from gem.inference.pose import predict_frame
    from gem.pose.data.crop import prepare_pose_batch

    if batch_hands < 1 or set(annotations) != set(range(len(frames))):
        raise ValueError("Invalid batch size/frame grid")
    output, topology, work = [], {}, []
    if not len(frames):
        return output, topology
    for i in range(len(frames)):
        record = annotations[i]
        validate_intrinsics(record["intrinsics"])
        output.append(
            predict_frame(predictor, frames[0], i, dict(record, instances=[]), task="hand", topology=topology)
        )
        work.extend((i, row) for row in record["instances"])
    chunks = [work[i : i + batch_hands] for i in range(0, len(work), batch_hands)]
    parts = {"prepare_wait": 0.0, "predict_and_transfer": 0.0, "serialize": 0.0}
    if not chunks:
        return output, topology
    with ThreadPoolExecutor(max_workers=4) as crop_pool, ThreadPoolExecutor(max_workers=1) as producer:

        @torch.inference_mode()
        def prepare(items):
            images, boxes, cameras, right, transforms = [], [], [], [], []
            for i, row in items:
                image = frames[i]
                camera = np.asarray(annotations[i]["intrinsics"], dtype=np.float32)
                if square_pixels:
                    transform = SquarePixelTransform.from_intrinsics(camera, image.shape[1], image.shape[0])
                    image = transform.resize_image(image)
                    box = transform.transform_boxes(np.asarray(row["bbox"], dtype=np.float32)[None])[0]
                    camera = transform.intrinsics
                    transforms.append(transform)
                else:
                    box = row["bbox"]
                images.append(image)
                boxes.append(box)
                cameras.append(camera)
                right.append(row["side"] == "right")

            def one(item):
                image, box, camera, side = item
                return prepare_pose_batch(
                    image,
                    np.asarray(box)[None],
                    camera,
                    right_hand=[side],
                    image_size=predictor.config.image_size,
                    hand_mirror=predictor.config.hand_mirror,
                    device="cpu",
                )

            prepared = list(crop_pool.map(one, zip(images, boxes, cameras, right, strict=False)))
            batch = {key: torch.cat([p[key] for p in prepared]) for key in prepared[0]}
            return batch, transforms

        pending = producer.submit(prepare, chunks[0])
        for position, items in enumerate(chunks):
            tick = time.perf_counter()
            cpu, transforms = pending.result()
            if position + 1 < len(chunks):
                pending = producer.submit(prepare, chunks[position + 1])
            batch = {key: value.to(predictor.device) for key, value in cpu.items()}
            batch["img"] = batch["img"].to(dtype=next(predictor.model.parameters()).dtype)
            begin = time.perf_counter()
            parts["prepare_wait"] += begin - tick
            prediction = predictor.predict_batch(batch)
            if square_pixels:
                prediction = _restore_raster(
                    prediction,
                    transforms,
                    torch.tensor([row["bbox"] for _, row in items], dtype=torch.float32),
                )
            prediction = _host_prediction(prediction)
            serialized = time.perf_counter()
            parts["predict_and_transfer"] += serialized - begin
            for index, (i, row) in enumerate(items):
                view = _row_view(prediction, index)
                replay = SimpleNamespace(predict=lambda *args, _view=view, **kwargs: _view)
                record = dict(annotations[i], instances=[row])
                serialized_row = predict_frame(
                    replay, frames[i], i, record, task="hand", topology=topology, square_pixels=False
                )
                output[i]["instances"].extend(serialized_row["instances"])
            parts["serialize"] += time.perf_counter() - serialized
    if timings is not None:
        timings.update(parts)
    return output, topology
