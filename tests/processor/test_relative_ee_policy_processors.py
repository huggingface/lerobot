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
import torch

pytest.importorskip("transformers")

from lerobot.configs import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402
from lerobot.policies.act.processor_act import make_act_pre_post_processors  # noqa: E402
from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: E402
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors  # noqa: E402
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: E402
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors  # noqa: E402
from lerobot.processor import (  # noqa: E402
    AbsoluteEEActionsStep,
    RelativeEEActionsStep,
    RelativeEEDeriveStateStep,
    RelativeEEStateStep,
    tokenizer_processor,  # noqa: E402
)
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402


def test_relative_ee_policy_pipelines(monkeypatch):
    monkeypatch.setattr(
        tokenizer_processor.AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: object()
    )
    stats = {
        OBS_STATE: {"min": torch.zeros(20), "max": torch.ones(20)},
        ACTION: {"min": torch.zeros(10), "max": torch.ones(10)},
    }
    configs_and_builders = [
        (ACTConfig(use_relative_ee=True, device="cpu"), make_act_pre_post_processors),
        (SmolVLAConfig(use_relative_ee=True, device="cpu"), make_smolvla_pre_post_processors),
        (PI05Config(use_relative_ee=True, device="cpu"), make_pi05_pre_post_processors),
    ]

    for config, builder in configs_and_builders:
        config.input_features = {OBS_STATE: PolicyFeature(FeatureType.STATE, (20,))}
        config.output_features = {ACTION: PolicyFeature(FeatureType.ACTION, (10,))}
        preprocessor, postprocessor = builder(config, stats)

        assert config.action_delta_indices == [-1, *range(config.chunk_size)]
        assert any(isinstance(step, RelativeEEDeriveStateStep) for step in preprocessor.steps)
        assert any(isinstance(step, RelativeEEActionsStep) for step in preprocessor.steps)
        assert any(isinstance(step, RelativeEEStateStep) for step in preprocessor.steps)
        assert any(isinstance(step, AbsoluteEEActionsStep) for step in postprocessor.steps)
