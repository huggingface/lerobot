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

"""The helpers a control loop shares with `lerobot-rollout` load without PyTorch."""

import pytest

from tests.torch_free import assert_runs_without_torch


@pytest.mark.parametrize("module", ["lerobot.utils.action_interpolator", "lerobot.utils.action_dispatch"])
def test_the_helper_loads_without_torch(module):
    assert_runs_without_torch(f"import {module}")


def test_a_numpy_tick_runs_without_torch():
    assert_runs_without_torch(
        """
        import numpy as np

        from lerobot.utils.action_dispatch import dispatch_tick
        from lerobot.utils.action_interpolator import ActionInterpolator

        actions = [np.array([0.0]), np.array([2.0])]
        sent = []


        class Source:
            def dispatch_allowed(self):
                return True

            def acknowledge_hold(self):
                pass

            def get_action(self, obs_frame):
                return actions.pop(0) if actions else None

            def record_dispatch(self):
                pass


        source, interpolator = Source(), ActionInterpolator(multiplier=2)
        for _ in range(3):
            dispatch_tick(source, interpolator, send=lambda target: sent.append(float(target[0])))
        assert sent == [0.0, 1.0, 2.0], sent
        """
    )
