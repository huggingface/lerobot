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

import pytest

pytest.importorskip("grpc")

from lerobot.async_inference.configs import PolicyServerConfig, RobotClientConfig  # noqa: E402
from tests.mocks.mock_robot import MockRobotConfig  # noqa: E402


def client(**overrides):
    return RobotClientConfig(
        policy_type="act",
        pretrained_name_or_path="local",
        robot=MockRobotConfig(),
        actions_per_chunk=10,
        **overrides,
    )


@pytest.mark.parametrize("factory", [PolicyServerConfig, client])
@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf"), -float("inf")])
def test_invalid_fps_is_rejected(factory, fps):
    with pytest.raises(ValueError, match="fps"):
        factory(fps=fps)


@pytest.mark.parametrize("field", ["inference_latency", "obs_queue_timeout"])
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), -float("inf")])
def test_invalid_server_timing_is_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        PolicyServerConfig(**{field: value})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -0.1, 1.1])
def test_invalid_chunk_threshold_is_rejected(value):
    with pytest.raises(ValueError, match="chunk_size_threshold"):
        client(chunk_size_threshold=value)


@pytest.mark.parametrize("fps", [1, 30, 120])
def test_valid_timing_and_threshold_boundaries(fps):
    server = PolicyServerConfig(fps=fps, inference_latency=0, obs_queue_timeout=0)
    assert server.environment_dt == 1 / fps
    for threshold in (0, 0.5, 1):
        assert client(fps=fps, chunk_size_threshold=threshold).environment_dt == 1 / fps
