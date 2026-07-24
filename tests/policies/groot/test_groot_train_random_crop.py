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

"""Isaac-GR00T N1.7 train-time random crop contract (crop geometry only).

Isaac-GR00T crops a random ``crop_fraction`` window during training and the
deterministic center window at eval, replaying the sampled window across all
camera views of a sample (gr00t/data/transform/video.py, n1.5-release onward:
"If mode is 'train', return a random crop transform. If mode is 'eval', return
a center crop transform."). This mirrors LeRobot's own Diffusion/VQBeT
``crop_is_random`` pattern. Color jitter is intentionally out of scope here.
"""

import random

import numpy as np
import torch

from lerobot.policies.groot.processor_groot import (
    GrootN17VLMEncodeStep,
    _transform_n1_7_image_for_vlm_albumentations,
)


def _structured_image(h=480, w=640):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.stack([(xx * 255 / w), (yy * 255 / h), ((xx + yy) * 255 / (h + w))], axis=-1).astype(np.uint8)


def test_crop_position_none_is_bitexact_center_crop():
    """crop_position=None must remain byte-identical to the pre-change eval path."""
    img = _structured_image()
    ref = _transform_n1_7_image_for_vlm_albumentations(
        img,
        image_crop_size=None,
        image_target_size=[256, 256],
        shortest_image_edge=256,
        crop_fraction=0.95,
    )
    out = _transform_n1_7_image_for_vlm_albumentations(
        img,
        image_crop_size=None,
        image_target_size=[256, 256],
        shortest_image_edge=256,
        crop_fraction=0.95,
        crop_position=None,
    )
    np.testing.assert_array_equal(ref, out)


def test_crop_position_center_matches_center_crop():
    img = _structured_image()
    center = _transform_n1_7_image_for_vlm_albumentations(
        img,
        image_crop_size=None,
        image_target_size=[256, 256],
        shortest_image_edge=256,
        crop_fraction=0.95,
        crop_position=None,
    )
    explicit = _transform_n1_7_image_for_vlm_albumentations(
        img,
        image_crop_size=None,
        image_target_size=[256, 256],
        shortest_image_edge=256,
        crop_fraction=0.95,
        crop_position=(0.5, 0.5),
    )
    # int-floor center vs rounded positional center may differ by <=1 px of grid
    assert center.shape == explicit.shape
    diff = np.abs(center.astype(np.int16) - explicit.astype(np.int16))
    assert diff.mean() < 3.0


def test_crop_position_corners_differ_from_center():
    img = _structured_image()

    def crop_at(position):
        return _transform_n1_7_image_for_vlm_albumentations(
            img,
            image_crop_size=None,
            image_target_size=[256, 256],
            shortest_image_edge=256,
            crop_fraction=0.95,
            crop_position=position,
        )

    center = crop_at(None)
    tl = crop_at((0.0, 0.0))
    br = crop_at((1.0, 1.0))
    assert not np.array_equal(center, tl)
    assert not np.array_equal(tl, br)


def _video(img, views=2):
    return np.stack([img] * views, axis=0).reshape(1, 1, views, *img.shape)


def _step(training):
    return GrootN17VLMEncodeStep(
        image_target_size=[256, 256],
        shortest_image_edge=256,
        crop_fraction=0.95,
        use_albumentations=True,
        training=training,
    )


def test_training_crop_replays_one_window_across_views():
    video = _video(_structured_image())
    frames = _step(training=True)._build_sample_images(video, batch_size=1, target_device=None)[0]
    np.testing.assert_array_equal(np.asarray(frames[0]), np.asarray(frames[1]))


def test_training_crop_differs_from_eval_center_crop():
    video = _video(_structured_image())
    random.seed(3)  # a draw that is not the exact center
    train_frame = np.asarray(
        _step(training=True)._build_sample_images(video, batch_size=1, target_device=None)[0][0]
    )
    eval_frame = np.asarray(
        _step(training=False)._build_sample_images(video, batch_size=1, target_device=None)[0][0]
    )
    assert not np.array_equal(train_frame, eval_frame)


def test_training_crop_is_disabled_under_no_grad():
    video = _video(_structured_image())
    with torch.no_grad():
        no_grad_frame = np.asarray(
            _step(training=True)._build_sample_images(video, batch_size=1, target_device=None)[0][0]
        )
    eval_frame = np.asarray(
        _step(training=False)._build_sample_images(video, batch_size=1, target_device=None)[0][0]
    )
    np.testing.assert_array_equal(no_grad_frame, eval_frame)


def test_training_mode_is_not_serialized():
    step = _step(training=True)
    serialized = step.get_config()
    assert "training" not in serialized
    restored = GrootN17VLMEncodeStep(**serialized)
    assert restored.training is False


def test_training_crop_respects_global_seed():
    video = _video(_structured_image())

    def draw():
        random.seed(11)
        return np.asarray(
            _step(training=True)._build_sample_images(video, batch_size=1, target_device=None)[0][0]
        )

    np.testing.assert_array_equal(draw(), draw())
