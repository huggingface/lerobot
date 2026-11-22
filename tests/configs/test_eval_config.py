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

import draccus
import pytest

from lerobot.configs.default import EvalConfig


class TestEvalConfigValidation:
    @pytest.mark.parametrize("n_episodes", [0, -1, -10])
    def test_non_positive_n_episodes_rejected(self, n_episodes: int) -> None:
        with pytest.raises(ValueError, match="eval.n_episodes must be at least 1"):
            EvalConfig(n_episodes=n_episodes)

    @pytest.mark.parametrize("batch_size", [-1, -5])
    def test_negative_batch_size_rejected(self, batch_size: int) -> None:
        with pytest.raises(ValueError, match="eval.batch_size must be non-negative"):
            EvalConfig(n_episodes=10, batch_size=batch_size)

    def test_auto_batch_size_sentinel_remains_valid(self) -> None:
        cfg = EvalConfig(n_episodes=10, batch_size=0)
        assert 1 <= cfg.batch_size <= 10

    def test_batch_size_clamped_to_n_episodes(self) -> None:
        cfg = EvalConfig(n_episodes=4, batch_size=16)
        assert cfg.batch_size == 4

    def test_valid_explicit_values_preserved(self) -> None:
        cfg = EvalConfig(n_episodes=10, batch_size=3, max_episode_steps=25)
        assert cfg.n_episodes == 10
        assert cfg.batch_size == 3
        assert cfg.max_episode_steps == 25

    def test_auto_batch_size_with_single_episode(self) -> None:
        cfg = EvalConfig(n_episodes=1, batch_size=0)
        assert cfg.batch_size == 1

    def test_draccus_decode_rejects_invalid_bounds(self) -> None:
        with pytest.raises(draccus.ParsingError) as exc_info:
            draccus.decode(EvalConfig, {"n_episodes": 0})
        assert isinstance(exc_info.value.__cause__, ValueError)
        assert "eval.n_episodes must be at least 1" in str(exc_info.value.__cause__)

        with pytest.raises(draccus.ParsingError) as exc_info:
            draccus.decode(EvalConfig, {"n_episodes": 5, "batch_size": -1})
        assert isinstance(exc_info.value.__cause__, ValueError)
        assert "eval.batch_size must be non-negative" in str(exc_info.value.__cause__)

    def test_draccus_round_trip(self) -> None:
        cfg = EvalConfig(n_episodes=8, batch_size=3, use_async_envs=False)
        decoded = draccus.decode(EvalConfig, draccus.encode(cfg))
        assert decoded == cfg
