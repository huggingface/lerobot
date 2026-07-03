#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Isaac-GR00T N1.7 train-time color-jitter contract."""

import hashlib
import random

import numpy as np
import torch

from lerobot.policies.groot import processor_groot
from lerobot.policies.groot.processor_groot import (
    GrootN17VLMEncodeStep,
    _apply_n1_7_color_jitter,
    _sample_n1_7_color_jitter_params,
)

# Generated with Isaac-GR00T's albumentations==1.4.18 A.ColorJitter using
# random.seed(1337), np.random.seed(1337), and the deterministic input below.
OSS_COLOR_JITTER_MAGNITUDES = {
    "brightness": 0.3,
    "contrast": 0.4,
    "saturation": 0.5,
    "hue": 0.08,
}
OSS_COLOR_JITTER_PARAMS = {
    "brightness": 1.0706517141708825,
    "contrast": 1.0266124588840007,
    "saturation": 0.8658483592493755,
    "hue": 0.01372597662436345,
    "order": [0, 3, 1, 2],
}
OSS_INPUT_SHA256 = "df4bf2710fd2cafea9ca517db1a16850b31ac3a7b225da50e95e81f09b81b0bb"
OSS_OUTPUT_SHA256 = "27775d8567ebb38f764821f789c57e19247b8d35d19abd3be4f65d76f493b663"


def _make_oss_golden_input() -> np.ndarray:
    shape = (2, 3, 37, 53)
    values = np.arange(np.prod(shape), dtype=np.int64)
    chw = ((values * 37 + 11) % 256).astype(np.uint8).reshape(shape)
    return chw.transpose(0, 2, 3, 1)


def _sha256_chw(images: np.ndarray) -> str:
    chw = np.ascontiguousarray(images.transpose(0, 3, 1, 2))
    return hashlib.sha256(chw.tobytes()).hexdigest()


def test_n1_7_opencv_color_jitter_matches_oss_golden_hash():
    images = _make_oss_golden_input()
    assert _sha256_chw(images) == OSS_INPUT_SHA256

    actual = np.stack([_apply_n1_7_color_jitter(image, OSS_COLOR_JITTER_PARAMS) for image in images])

    assert actual.shape == images.shape
    assert actual.dtype == np.uint8
    assert _sha256_chw(actual) == OSS_OUTPUT_SHA256


def test_n1_7_color_jitter_sampling_matches_oss_seed_sequence():
    random.seed(1337)

    assert _sample_n1_7_color_jitter_params(OSS_COLOR_JITTER_MAGNITUDES) == OSS_COLOR_JITTER_PARAMS


def test_training_color_jitter_runs_in_vlm_preprocessor_and_eval_is_stable(monkeypatch):
    images = _make_oss_golden_input()
    video = images.reshape(1, 1, 2, *images.shape[1:])
    monkeypatch.setattr(
        processor_groot,
        "_sample_n1_7_color_jitter_params",
        lambda _: OSS_COLOR_JITTER_PARAMS,
    )

    train_step = GrootN17VLMEncodeStep(
        use_albumentations=True,
        color_jitter_params=OSS_COLOR_JITTER_MAGNITUDES,
        training=True,
    )
    train_frames = train_step._build_sample_images(video, batch_size=1, target_device=None)[0]
    assert _sha256_chw(np.stack(train_frames)) == OSS_OUTPUT_SHA256

    eval_step = GrootN17VLMEncodeStep(
        use_albumentations=True,
        color_jitter_params=OSS_COLOR_JITTER_MAGNITUDES,
        training=False,
    )
    eval_frames = eval_step._build_sample_images(video, batch_size=1, target_device=None)[0]
    assert _sha256_chw(np.stack(eval_frames)) == OSS_INPUT_SHA256

    with torch.no_grad():
        no_grad_frames = train_step._build_sample_images(video, batch_size=1, target_device=None)[0]
    assert _sha256_chw(np.stack(no_grad_frames)) == OSS_INPUT_SHA256


def test_color_jitter_config_round_trips_but_training_mode_does_not():
    step = GrootN17VLMEncodeStep(
        color_jitter_params=OSS_COLOR_JITTER_MAGNITUDES,
        training=True,
    )

    serialized = step.get_config()
    assert serialized["color_jitter_params"] == OSS_COLOR_JITTER_MAGNITUDES
    assert "training" not in serialized

    restored = GrootN17VLMEncodeStep(**serialized)
    assert restored.color_jitter_params == OSS_COLOR_JITTER_MAGNITUDES
    assert restored.training is False
