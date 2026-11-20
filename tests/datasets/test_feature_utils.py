#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import numpy as np

from lerobot.datasets.feature_utils import validate_feature_numpy_array


def test_validate_feature_numpy_array_accepts_list_declared_shape():
    """The declared type of expected_shape is list[int] while np.ndarray.shape is always a
    tuple: the comparison must normalize the declared shape, otherwise any
    JSON-declared (list) shape fails validation (repro of #4928)."""
    frame = np.zeros(6, dtype=np.float32)
    assert validate_feature_numpy_array("state", "float32", [6], frame) == ""
    # tuple-declared shapes keep working unchanged
    assert validate_feature_numpy_array("state", "float32", (6,), frame) == ""


def test_validate_feature_numpy_array_still_rejects_wrong_shape_and_dtype():
    frame = np.zeros(6, dtype=np.float32)
    error = validate_feature_numpy_array("state", "float32", [8], frame)
    assert "does not have the expected shape" in error
    error = validate_feature_numpy_array("state", "float32", [6], frame.astype(np.int64))
    assert "not of the expected dtype" in error
