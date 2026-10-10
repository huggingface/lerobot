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

from lerobot.policies import make_policy_config
from lerobot.processor.migrate_policy_normalization import drop_unknown_config_fields


def test_drop_unknown_config_fields_removes_fields_no_longer_declared():
    # `lerobot/vqbet_pusht` was saved with `mlp_hidden_dim`, which VQBeTConfig no longer declares.
    config = {"n_obs_steps": 3, "mlp_hidden_dim": 1024}

    cleaned = drop_unknown_config_fields("vqbet", config)

    assert cleaned == {"n_obs_steps": 3}
    assert make_policy_config("vqbet", **cleaned).n_obs_steps == 3


def test_drop_unknown_config_fields_keeps_config_of_unknown_policy_type():
    config = {"n_obs_steps": 3, "mlp_hidden_dim": 1024}

    assert drop_unknown_config_fields("not_a_policy", config) == config
