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

"""Feature schemas and the validation of values against them."""

import numpy as np
import pytest

from lerobot.utils.feature_spec import FeatureSpec, validate_array


@pytest.mark.parametrize(
    "value",
    [
        None,
        [1, 2],
        np.zeros((2,), np.float32),
        np.zeros((3, 2), np.float64),
        np.full((3, 2), np.nan, np.float32),
    ],
)
def test_array_validation_checks_batch_dtype_and_finiteness(value):
    feature = FeatureSpec("state", (2,), "float32", semantics="state-v1")
    validate_array(np.zeros((3, 2), np.float32), feature, leading_shape=(3,))
    with pytest.raises(ValueError):
        validate_array(value, feature, leading_shape=(3,))
