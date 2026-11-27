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

from lerobot.configs.types import Deltas, FeatureType
from lerobot.utils.constants import ACTION, OBS_ACTION, OBS_ENV_STATE, OBS_STATE, REWARD


def test_deltas_route_each_key_to_its_offsets():
    deltas = Deltas(
        observation={FeatureType.VISUAL: range(-2, 1), FeatureType.ACTION: (-2, -1)},
        action=range(3),
        reward=[0],
    )
    assert deltas.of("observation.images.top", FeatureType.VISUAL) == (-2, -1, 0)
    assert deltas.of(OBS_ACTION) == (-2, -1)
    assert deltas.of(ACTION) == (0, 1, 2)
    assert deltas.of(REWARD) == (0,)
    # Undeclared observations arrive as the current frame alone.
    assert deltas.of(OBS_STATE, FeatureType.STATE) is None
    assert deltas.of(OBS_ENV_STATE, FeatureType.ENV) is None


def test_an_observation_key_overrides_its_type():
    deltas = Deltas(observation={FeatureType.VISUAL: [0, 1], "observation.images.top": range(-50, 1, 25)})
    assert deltas.of("observation.images.top", FeatureType.VISUAL) == (-50, -25, 0)
    assert deltas.of("observation.images.wrist", FeatureType.VISUAL) == (0, 1)
    # Only the named camera gets a window when no type is declared.
    single = Deltas(observation={"observation.images.top": [-1, 0]})
    assert single.of("observation.images.top", FeatureType.VISUAL) == (-1, 0)
    assert single.of("observation.images.wrist", FeatureType.VISUAL) is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"observation": {FeatureType.LANGUAGE: [0]}},
        {"observation": {"action": [0]}},
        {"observation": {OBS_ACTION: [-1]}},
        {"observation": {FeatureType.VISUAL: []}},
        {"action": [0.5]},
    ],
)
def test_deltas_reject_invalid_declarations(kwargs):
    with pytest.raises(ValueError):
        Deltas(**kwargs)


def test_configs_declaring_removed_delta_indices_fail_at_definition():
    from lerobot.configs.policies import PreTrainedConfig

    with pytest.raises(TypeError, match="training_sampler"):

        class _LegacyConfig(PreTrainedConfig):
            @property
            def action_delta_indices(self) -> list[int]:
                return [0]
