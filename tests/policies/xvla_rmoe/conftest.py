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

"""`torch.manual_seed` alone does not make multi-threaded CPU matmul/attention reductions
bit-reproducible: floating-point addition isn't associative, and which thread sums which
partial products is scheduling-dependent, not seed-dependent (see
https://pytorch.org/docs/stable/notes/randomness.html). This test package's gradient-flow
and exact-equality assertions (e.g. `expert_symmetry_breaking_std=0` must give *exactly*
0.0 router/GRU gradient, `use_moe=False` must give a bit-exact match with plain X-VLA) are
sensitive to that jitter given how tiny the synthetic models are. Pin to a single thread for
the duration of this package's tests only, and restore whatever the session had before.
"""

import pytest
import torch


@pytest.fixture(autouse=True, scope="package")
def _single_threaded_cpu_math():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)
