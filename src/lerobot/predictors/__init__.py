# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Overhead cube-position predictors for time-advanced observations.

The predictor estimates the red cube's image-plane position/velocity from the
overhead camera; the RTC engine advances the cube by the inference latency (the
"PE gap") and feeds the time-advanced frame to the policy. :func:`shift_cube_in_frame`
is the training-free image-edit baseline used to synthesise that frame.
"""

from .base import PredictorOutput
from .config import CubePredictorConfig, PredictorConfig, PredictorMode
from .cube_predictor import CubePredictor
from .latent_warp import (
    make_flow_token_warp_fn,
    make_token_warp_fn,
    mask_to_token_grid,
    warp_token_grid,
    warp_token_grid_by_flow,
)
from .optical_flow import DenseFlowEstimator, FlowOutput
from .shift import shift_cube_in_frame

__all__ = [
    "CubePredictor",
    "CubePredictorConfig",
    "DenseFlowEstimator",
    "FlowOutput",
    "PredictorConfig",
    "PredictorMode",
    "PredictorOutput",
    "make_flow_token_warp_fn",
    "make_token_warp_fn",
    "mask_to_token_grid",
    "shift_cube_in_frame",
    "warp_token_grid",
    "warp_token_grid_by_flow",
]
