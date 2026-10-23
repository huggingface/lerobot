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

from numbers import Integral
from typing import Any

import cv2
import numpy as np

from lerobot.configs import PipelineFeatureType, PolicyFeature

from .pipeline import ObservationProcessorStep, ProcessorStepRegistry


@ProcessorStepRegistry.register(name="image_undistort_processor")
class ImageUndistortProcessorStep(ObservationProcessorStep):
    """Undistort one raw camera image using OpenCV's standard pinhole model.

    Use this step in the robot observation pipeline, before images become policy
    tensors. The input must be an HWC uint8 NumPy image with three channels. Its
    shape, dtype and channel order are preserved. A separate step is needed for
    each camera.

    The output uses the input camera matrix, without automatic cropping or
    scaling. Pixels outside the source image become black. Remapping tables are
    built on the first frame and reused; only calibration parameters are saved.

    Args:
        camera_key: Raw observation key, for example ``"front"`` or ``"wrist"``.
        camera_matrix: Calibrated 3x3 intrinsic matrix with zero skew.
        distortion_coefficients: Flat vector of 4, 5, 8, 12 or 14 coefficients in
            OpenCV order. These belong to the standard model, not ``cv2.fisheye``.
        image_size: Calibration image size as ``(width, height)``. Calibration
            must also match the incoming image orientation and camera settings.
    """

    def __init__(
        self,
        camera_key: str,
        camera_matrix: list[list[float]],
        distortion_coefficients: list[float],
        image_size: tuple[int, int] | list[int],
    ) -> None:
        if not isinstance(camera_key, str) or not camera_key.strip():
            raise ValueError("camera_key must be a non-empty string.")
        if (
            not isinstance(image_size, (list, tuple))
            or len(image_size) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, Integral) or value <= 0
                for value in image_size
            )
        ):
            raise ValueError("image_size must contain two positive integers: (width, height).")

        matrix = np.array(camera_matrix, dtype=np.float64, copy=True)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("camera_matrix must be a finite 3x3 intrinsic matrix.")
        if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
            raise ValueError("camera_matrix focal lengths fx and fy must be positive.")
        if not np.allclose(matrix[[0, 1, 2, 2, 2], [1, 0, 0, 1, 2]], [0, 0, 0, 0, 1], rtol=0, atol=1e-12):
            raise ValueError("camera_matrix must have zero skew and last row [0, 0, 1].")

        coefficients = np.array(distortion_coefficients, dtype=np.float64, copy=True)
        if (
            coefficients.ndim != 1
            or coefficients.size not in (4, 5, 8, 12, 14)
            or not np.isfinite(coefficients).all()
        ):
            raise ValueError(
                "distortion_coefficients must be a finite flat vector of length 4, 5, 8, 12 or 14."
            )

        # Copy calibration so caller mutations cannot invalidate cached maps.
        self._camera_key = camera_key
        self._camera_matrix = matrix
        self._distortion_coefficients = coefficients
        self._image_size = (int(image_size[0]), int(image_size[1]))
        self._maps: tuple[np.ndarray, np.ndarray] | None = None

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self._camera_key not in observation:
            raise KeyError(f"Camera {self._camera_key!r} is missing from the observation.")
        image = observation[self._camera_key]
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Camera {self._camera_key!r} must provide an HWC uint8 NumPy image.")
        width, height = self._image_size
        expected_shape = (height, width, 3)
        if image.dtype != np.uint8 or image.shape != expected_shape:
            raise ValueError(
                f"Camera {self._camera_key!r} must provide a uint8 image of shape {expected_shape}, "
                f"matching its calibration; got {image.dtype} with shape {image.shape}."
            )

        if self._maps is None:
            self._maps = cv2.initUndistortRectifyMap(
                self._camera_matrix,
                self._distortion_coefficients,
                None,
                self._camera_matrix,
                self._image_size,
                cv2.CV_32FC1,
            )
        undistorted = cv2.remap(
            image,
            self._maps[0],
            self._maps[1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return {**observation, self._camera_key: undistorted}

    def get_config(self) -> dict[str, Any]:
        return {
            "camera_key": self._camera_key,
            "camera_matrix": self._camera_matrix.tolist(),
            "distortion_coefficients": self._distortion_coefficients.tolist(),
            "image_size": list(self._image_size),
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # Image keys, shapes and channel order are unchanged.
        return features
