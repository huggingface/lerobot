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

import json
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

from lerobot.processor import (
    ImageUndistortProcessorStep,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_robot_observation_processor,
    observation_to_transition,
    transition_to_observation,
)


@pytest.fixture
def calibration() -> dict[str, Any]:
    return {
        "camera_key": "front",
        "camera_matrix": [[80.0, 0.0, 64.0], [0.0, 82.0, 48.0], [0.0, 0.0, 1.0]],
        "distortion_coefficients": [-0.25, 0.06, 0.002, -0.003, 0.01],
        "image_size": [128, 96],
    }


@pytest.fixture
def image() -> np.ndarray:
    # Coordinate ramps make the remapped source coordinates observable in RGB.
    y, x = np.indices((96, 128), dtype=np.uint8)
    return np.stack([x, y, np.full_like(x, 179)], axis=-1)


def make_pipeline(*steps: ImageUndistortProcessorStep) -> RobotProcessorPipeline:
    return RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=list(steps),
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )


def test_zero_distortion_preserves_pixels_and_input(calibration, image):
    calibration["distortion_coefficients"] = [0.0] * 5
    pipeline = make_pipeline(ImageUndistortProcessorStep(**calibration))
    original = image.copy()
    other_image = image.copy()
    observation = {"front": image, "wrist": other_image, "shoulder.pos": 1.5}

    result = pipeline(observation)

    np.testing.assert_array_equal(result["front"], original)
    assert result["front"] is not image
    assert result["front"].dtype == np.uint8
    assert result["front"].shape == image.shape
    assert result["wrist"] is other_image
    assert result["shoulder.pos"] == 1.5
    assert observation["front"] is image
    np.testing.assert_array_equal(image, original)


def test_nonzero_distortion_matches_pinhole_geometry(calibration, image):
    result = make_pipeline(ImageUndistortProcessorStep(**calibration))({"front": image})["front"]
    fx, fy, cx, cy = 80.0, 82.0, 64.0, 48.0
    k1, k2, p1, p2, k3 = calibration["distortion_coefficients"]

    # Independent Brown-Conrady equations: each output pixel samples the
    # corresponding distorted source pixel. No OpenCV remapping in the reference.
    for u, v in [(20, 20), (108, 72), (32, 80), (80, 24)]:
        x, y = (u - cx) / fx, (v - cy) / fy
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        source_x = fx * (x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)) + cx
        source_y = fy * (y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y) + cy
        np.testing.assert_allclose(result[v, u], [source_x, source_y, 179], atol=0.6)
    assert not np.array_equal(result, image)


@pytest.mark.parametrize("coefficient_count", [4, 5, 8, 12, 14])
def test_supported_standard_models_match_opencv(calibration, image, coefficient_count):
    coefficients = (calibration["distortion_coefficients"] + [0.001] * 9)[:coefficient_count]
    calibration["distortion_coefficients"] = coefficients
    result = make_pipeline(ImageUndistortProcessorStep(**calibration))({"front": image})["front"]
    matrix = np.array(calibration["camera_matrix"])
    expected = cv2.undistort(image, matrix, np.array(coefficients), None, matrix)
    np.testing.assert_allclose(result, expected, atol=1)


def test_black_borders_keep_image_size(calibration, image):
    calibration["distortion_coefficients"] = [0.8, 0.0, 0.0, 0.0, 0.0]
    result = make_pipeline(ImageUndistortProcessorStep(**calibration))({"front": image})["front"]
    assert result.shape == image.shape
    np.testing.assert_array_equal(result[0, 0], [0, 0, 0])
    np.testing.assert_array_equal(result[48, 64], image[48, 64])


def test_two_cameras_use_separate_calibrations(calibration, image):
    wrist_calibration = {**calibration, "camera_key": "wrist", "distortion_coefficients": [0.0] * 5}
    pipeline = make_pipeline(
        ImageUndistortProcessorStep(**calibration), ImageUndistortProcessorStep(**wrist_calibration)
    )
    result = pipeline({"front": image, "wrist": image})
    assert not np.array_equal(result["front"], image)
    np.testing.assert_array_equal(result["wrist"], image)


def test_maps_are_cached_and_calibration_is_copied(calibration, image):
    expected_calibration = deepcopy(calibration)
    step = ImageUndistortProcessorStep(**calibration)
    pipeline = make_pipeline(step)
    calibration["camera_matrix"][0][0] = 1.0
    calibration["distortion_coefficients"][0] = 0.0
    calibration["image_size"][0] = 1
    exported = step.get_config()
    exported["camera_matrix"][0][0] = 2.0

    with patch(
        "lerobot.processor.image_undistort_processor.cv2.initUndistortRectifyMap",
        wraps=cv2.initUndistortRectifyMap,
    ) as build_maps:
        result = pipeline({"front": image})
        pipeline.reset()
        repeated = pipeline({"front": image.copy()})
        assert build_maps.call_count == 1

    assert step.get_config() == expected_calibration
    np.testing.assert_array_equal(result["front"], repeated["front"])


def test_save_and_reload_raw_observation_pipeline(calibration, image, tmp_path: Path):
    pipeline = make_pipeline(ImageUndistortProcessorStep(**calibration))
    expected = pipeline({"front": image, "shoulder.pos": 2.0})
    filename = "robot_observation_processor.json"
    pipeline.save_pretrained(tmp_path, config_filename=filename)

    saved = json.loads((tmp_path / filename).read_text())
    assert saved["steps"][0]["config"] == calibration
    assert sorted(path.name for path in tmp_path.iterdir()) == [filename]
    reloaded = RobotProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename=filename,
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    assert isinstance(reloaded.steps[0], ImageUndistortProcessorStep)
    result = reloaded({"front": image, "shoulder.pos": 2.0})
    np.testing.assert_array_equal(result["front"], expected["front"])
    assert result["shoulder.pos"] == 2.0


def test_recording_feature_contract_is_unchanged(calibration):
    pytest.importorskip("datasets", reason="Feature aggregation requires lerobot[dataset].")
    from lerobot.datasets.pipeline_features import (
        aggregate_pipeline_dataset_features,
        create_initial_features,
    )

    features = create_initial_features(observation={"front": (96, 128, 3), "shoulder.pos": float})
    before = deepcopy(features)
    pipeline = make_pipeline(ImageUndistortProcessorStep(**calibration))
    actual = aggregate_pipeline_dataset_features(pipeline, features)
    expected = aggregate_pipeline_dataset_features(make_default_robot_observation_processor(), features)

    assert actual == expected
    assert actual["observation.images.front"]["shape"] == (96, 128, 3)
    assert features == before


@pytest.mark.parametrize(
    "invalid",
    [
        {"camera_key": ""},
        {"camera_key": "  "},
        {"camera_key": None},
        {"image_size": [0, 96]},
        {"image_size": [128, -1]},
        {"image_size": [128.0, 96]},
        {"image_size": [True, 96]},
        {"image_size": [128]},
        {"camera_matrix": [[1.0, 0.0], [0.0, 1.0]]},
        {"camera_matrix": [[0.0, 0.0, 64.0], [0.0, 82.0, 48.0], [0.0, 0.0, 1.0]]},
        {"camera_matrix": [[80.0, 0.1, 64.0], [0.0, 82.0, 48.0], [0.0, 0.0, 1.0]]},
        {"camera_matrix": [[80.0, 0.0, 64.0], [0.0, 82.0, 48.0], [0.0, 1.0, 1.0]]},
        {"camera_matrix": [[80.0, 0.0, np.nan], [0.0, 82.0, 48.0], [0.0, 0.0, 1.0]]},
        {"distortion_coefficients": []},
        {"distortion_coefficients": [0.0] * 6},
        {"distortion_coefficients": [[0.0] * 5]},
        {"distortion_coefficients": [np.inf, 0.0, 0.0, 0.0]},
    ],
)
def test_rejects_invalid_calibration(calibration, invalid):
    with pytest.raises(ValueError):
        ImageUndistortProcessorStep(**{**calibration, **invalid})


@pytest.mark.parametrize(
    "invalid_image",
    [
        np.zeros((48, 64, 3), dtype=np.uint8),
        np.zeros((128, 96, 3), dtype=np.uint8),
        np.zeros((96, 128), dtype=np.uint8),
        np.zeros((3, 96, 128), dtype=np.uint8),
        np.zeros((96, 128, 4), dtype=np.uint8),
        np.zeros((1, 96, 128, 3), dtype=np.uint8),
        np.zeros((96, 128, 3), dtype=np.float32),
    ],
)
def test_rejects_images_incompatible_with_calibration(calibration, invalid_image):
    pipeline = make_pipeline(ImageUndistortProcessorStep(**calibration))
    with pytest.raises(ValueError, match="matching its calibration"):
        pipeline({"front": invalid_image})


def test_rejects_policy_tensors_and_missing_camera(calibration):
    pipeline = make_pipeline(ImageUndistortProcessorStep(**calibration))
    with pytest.raises(TypeError, match="HWC uint8 NumPy image"):
        pipeline({"front": torch.zeros(1, 3, 96, 128)})
    with pytest.raises(KeyError, match="front.*missing"):
        pipeline({"wrist": np.zeros((96, 128, 3), dtype=np.uint8)})
