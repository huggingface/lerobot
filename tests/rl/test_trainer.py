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

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

import torch  # noqa: E402
from torch import Tensor  # noqa: E402

from lerobot.rl.algorithms.base import RLAlgorithm  # noqa: E402
from lerobot.rl.algorithms.configs import TrainingStats  # noqa: E402
from lerobot.rl.trainer import RLTrainer  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402


class _DummyRLAlgorithmConfig:
    """Dummy config for testing."""


class _DummyRLAlgorithm(RLAlgorithm):
    config_class = _DummyRLAlgorithmConfig
    name = "dummy_rl_algorithm"

    def __init__(self):
        self.configure_calls = 0
        self.update_calls = 0

    def select_action(self, observation: dict[str, Tensor]) -> Tensor:
        return torch.zeros(1)

    def configure_data_iterator(
        self,
        data_mixer,
        batch_size: int,
        *,
        async_prefetch: bool = True,
        queue_size: int = 2,
    ):
        self.configure_calls += 1
        return data_mixer.get_iterator(
            batch_size=batch_size,
            async_prefetch=async_prefetch,
            queue_size=queue_size,
        )

    def make_optimizers_and_scheduler(self):
        return {}

    def update(self, batch_iterator):
        self.update_calls += 1
        _ = next(batch_iterator)
        return TrainingStats(losses={"dummy": 1.0})

    def load_weights(self, weights, device="cpu") -> None:
        _ = (weights, device)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {}

    def load_state_dict(self, state_dict, device="cpu") -> None:
        _ = (state_dict, device)


class _SimpleMixer:
    def get_iterator(self, batch_size: int, async_prefetch: bool = True, queue_size: int = 2):
        _ = (async_prefetch, queue_size)
        while True:
            yield {
                "state": {OBS_STATE: torch.randn(batch_size, 3)},
                ACTION: torch.randn(batch_size, 2),
                "reward": torch.randn(batch_size),
                "next_state": {OBS_STATE: torch.randn(batch_size, 3)},
                "done": torch.zeros(batch_size),
                "truncated": torch.zeros(batch_size),
                "complementary_info": None,
            }


def test_trainer_lazy_iterator_lifecycle_and_reset():
    algo = _DummyRLAlgorithm()
    mixer = _SimpleMixer()
    trainer = RLTrainer(algorithm=algo, data_mixer=mixer, batch_size=4)

    # First call builds iterator once.
    trainer.training_step()
    assert algo.configure_calls == 1
    assert algo.update_calls == 1

    # Second call reuses existing iterator.
    trainer.training_step()
    assert algo.configure_calls == 1
    assert algo.update_calls == 2

    # Explicit reset forces lazy rebuild on next step.
    trainer.reset_data_iterator()
    trainer.training_step()
    assert algo.configure_calls == 2
    assert algo.update_calls == 3


def test_trainer_set_data_mixer_resets_by_default():
    algo = _DummyRLAlgorithm()
    mixer_a = _SimpleMixer()
    mixer_b = _SimpleMixer()
    trainer = RLTrainer(algorithm=algo, data_mixer=mixer_a, batch_size=2)

    trainer.training_step()
    assert algo.configure_calls == 1

    trainer.set_data_mixer(mixer_b, reset=True)
    trainer.training_step()
    assert algo.configure_calls == 2


def test_algorithm_optimization_step_contract_defaults():
    algo = _DummyRLAlgorithm()
    assert algo.optimization_step == 0
    algo.optimization_step = 11
    assert algo.optimization_step == 11
