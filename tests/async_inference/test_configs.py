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

from lerobot.async_inference.configs import PolicyServerConfig


@pytest.mark.parametrize("timeout", [0.0, 7.5])
def test_policy_server_config_round_trip(timeout):
    config = PolicyServerConfig(
        host="127.0.0.1", port=9000, fps=20, inference_latency=0.2, obs_queue_timeout=timeout
    )
    exported = config.to_dict()
    assert exported["obs_queue_timeout"] == timeout
    assert PolicyServerConfig.from_dict(exported) == config


def test_policy_server_config_loads_legacy_export_without_mutation():
    exported = {
        "host": "localhost",
        "port": 8080,
        "fps": 20,
        "environment_dt": 0.05,
        "inference_latency": 0.1,
    }
    original = exported.copy()
    config = PolicyServerConfig.from_dict(exported)
    assert config == PolicyServerConfig(fps=20, inference_latency=0.1)
    assert exported == original


def test_policy_server_config_derives_environment_dt_from_fps():
    config = PolicyServerConfig.from_dict({"fps": 20, "environment_dt": 999})
    assert config.environment_dt == 0.05


def test_policy_server_config_rejects_unknown_fields():
    with pytest.raises(TypeError, match="unknown_option"):
        PolicyServerConfig.from_dict({"unknown_option": 7.5})


def test_policy_server_config_validates_imported_timeout():
    with pytest.raises(ValueError, match="obs_queue_timeout"):
        PolicyServerConfig.from_dict({"environment_dt": 0.05, "obs_queue_timeout": -1})
