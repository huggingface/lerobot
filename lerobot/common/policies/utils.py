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
from collections import deque

import torch
from torch import nn


def populate_queues(queues, batch):
    for key in batch:
        # Ignore keys not in the queues already (leaving the responsibility to the caller to make sure the
        # queues have the keys they want).
        if key not in queues:
            continue
        if len(queues[key]) != queues[key].maxlen:
            # initialize by copying the first observation several times until the queue is full
            while len(queues[key]) != queues[key].maxlen:
                queues[key].append(batch[key])
        else:
            # add latest observation to the queue
            queues[key].append(batch[key])
    return queues


def get_device_from_parameters(module: nn.Module) -> torch.device:
    """Get a module's device by checking one of its parameters.

    Note: assumes that all parameters have the same device
    """
    return next(iter(module.parameters())).device


def get_dtype_from_parameters(module: nn.Module) -> torch.dtype:
    """Get a module's parameter dtype by checking one of its parameters.

    Note: assumes that all parameters have the same dtype.
    """
    return next(iter(module.parameters())).dtype


class TemporalQueue:
    def __init__(self, maxlen):
        # TODO(rcadene): set proper maxlen
        self.items = deque(maxlen=maxlen)
        self.timestamps = deque(maxlen=maxlen)

    def add(self, item, timestamp):
        self.items.append(item)
        self.timestamps.append(timestamp)

    def get_latest(self):
        return self.items[-1], self.timestamps[-1]

    def get(self, timestamp):
        import numpy as np

        timestamps = np.array(list(self.timestamps))
        distances = np.abs(timestamps - timestamp)
        nearest_idx = distances.argmin()

        # print(float(distances[nearest_idx]))
        if float(distances[nearest_idx]) > 1 / 30:
            raise ValueError()

        return self.items[nearest_idx], self.timestamps[nearest_idx]

    def __len__(self):
        return len(self.items)
