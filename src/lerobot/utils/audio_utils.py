#!/usr/bin/env python

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

import numpy as np


def rolling_vstack(buffer: np.ndarray, new_data: np.ndarray) -> np.ndarray:
    """
    Rolling implementation of numpy.vstack to add new data in at the end of a fixed shape buffer in a rolling fashion.

    Args:
        buffer: The *fixed* shape buffer to update.
        new_data: The new data to add to the buffer.

    Returns:
        The updated buffer.
    """

    buffer_size = buffer.shape[0]
    # Remove as many old audio samples as needed
    buffer[: -len(new_data)] = buffer[len(new_data) :]
    # Add new audio samples, only the newest if the buffer is already full
    buffer[-len(new_data) :] = new_data[-buffer_size:]
    return buffer
