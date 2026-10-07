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

"""Compare the PI0 processor pipeline against the vendored OpenPI reference processors."""

import os

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.configs import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.pi0 import PI0Policy  # noqa: E402
from lerobot.policies.pi0.configuration_pi0 import PI0Config  # noqa: E402
from lerobot.policies.pi0.processor_pi0 import make_pi0_pre_post_processors  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402
from tests.policies.pi0_pi05.utils.openpi_parity import (  # noqa: E402
    IMAGE_KEYS,
    assert_processor_inputs_match_lerobot,
    clone_batch,
    make_openpi_observation_from_raw,
    openpi_model_actions_from_raw,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true",
    reason="OpenPI processor parity uses the PaliGemma tokenizer; run manually outside CI.",
)

DUMMY_ACTION_DIM = 32
DUMMY_STATE_DIM = 32
DUMMY_ACTION_HORIZON = 50
DUMMY_MAX_TOKEN_LEN = 48
DEVICE = torch.device("cpu")

DUMMY_DATASET_STATS = {
    OBS_STATE: {
        "mean": torch.zeros(DUMMY_STATE_DIM),
        "std": torch.ones(DUMMY_STATE_DIM),
        "q01": torch.zeros(DUMMY_STATE_DIM),
        "q99": torch.ones(DUMMY_STATE_DIM),
    },
    ACTION: {
        "mean": torch.zeros(DUMMY_ACTION_DIM),
        "std": torch.ones(DUMMY_ACTION_DIM),
        "q01": torch.zeros(DUMMY_ACTION_DIM),
        "q99": torch.ones(DUMMY_ACTION_DIM),
    },
    "images": {
        key: {
            "mean": torch.zeros(3, 224, 224),
            "std": torch.ones(3, 224, 224),
            "q01": torch.zeros(3, 224, 224),
            "q99": torch.ones(3, 224, 224),
        }
        for key in IMAGE_KEYS
    },
}


class PI0PolicyInputAdapter(torch.nn.Module):
    """Minimal adapter exposing PI0 policy input-preparation helpers without loading model weights."""

    _preprocess_images = PI0Policy._preprocess_images
    prepare_state = PI0Policy.prepare_state

    def __init__(self, config: PI0Config) -> None:
        super().__init__()
        self.config = config
        self._device_anchor = torch.nn.Parameter(torch.empty((), device=config.device), requires_grad=False)


def create_pi0_config() -> PI0Config:
    config = PI0Config(device=str(DEVICE))
    config.max_state_dim = DUMMY_STATE_DIM
    config.max_action_dim = DUMMY_ACTION_DIM
    config.chunk_size = DUMMY_ACTION_HORIZON
    config.n_action_steps = DUMMY_ACTION_HORIZON
    config.tokenizer_max_length = DUMMY_MAX_TOKEN_LEN
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(DUMMY_STATE_DIM,)),
        **{
            f"observation.images.{key}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
            for key in IMAGE_KEYS
        },
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(DUMMY_ACTION_DIM,)),
    }
    return config


def create_dummy_data() -> dict:
    batch_size = 2
    prompt = "Pick up the red block and place it in the bin"
    return {
        OBS_STATE: torch.randn(batch_size, DUMMY_STATE_DIM, dtype=torch.float32, device=DEVICE),
        ACTION: torch.randn(
            batch_size, DUMMY_ACTION_HORIZON, DUMMY_ACTION_DIM, dtype=torch.float32, device=DEVICE
        ),
        **{
            f"observation.images.{key}": torch.rand(
                batch_size, 3, 224, 224, dtype=torch.float32, device=DEVICE
            )
            for key in IMAGE_KEYS
        },
        "task": [prompt for _ in range(batch_size)],
    }


def test_pi0_processor_inputs_match_openpi_reference():
    torch.manual_seed(0)
    config = create_pi0_config()
    preprocessor, _ = make_pi0_pre_post_processors(config=config, dataset_stats=DUMMY_DATASET_STATS)

    raw_batch = create_dummy_data()
    lerobot_batch = preprocessor(clone_batch(raw_batch))
    openpi_observation = make_openpi_observation_from_raw(
        raw_batch,
        action_dim=DUMMY_ACTION_DIM,
        max_token_len=DUMMY_MAX_TOKEN_LEN,
        dataset_stats=DUMMY_DATASET_STATS,
        pi05=False,
    )

    assert_processor_inputs_match_lerobot(
        PI0PolicyInputAdapter(config),
        lerobot_batch,
        openpi_observation,
        compare_state=True,
    )
    torch.testing.assert_close(
        lerobot_batch[ACTION],
        openpi_model_actions_from_raw(
            raw_batch,
            action_dim=DUMMY_ACTION_DIM,
            dataset_stats=DUMMY_DATASET_STATS,
            pi05=False,
        ),
        rtol=0,
        atol=0,
    )
