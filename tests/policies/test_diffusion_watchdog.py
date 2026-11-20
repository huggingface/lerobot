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

"""Unit tests for Diffusion Policy Adaptive Discrepancy Watchdog & Reactive Re-planning."""

import unittest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE


def _create_test_policy(
    horizon: int = 16,
    n_action_steps: int = 8,
    watchdog_tracking_threshold: float | None = None,
) -> DiffusionPolicy:
    config = DiffusionConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(4,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(4,))},
        n_obs_steps=1,
        horizon=horizon,
        n_action_steps=n_action_steps,
        drop_n_last_frames=0,
        down_dims=(16, 32),
        diffusion_step_embed_dim=16,
        num_train_timesteps=5,
        num_inference_steps=2,
        watchdog_tracking_threshold=watchdog_tracking_threshold,
    )
    return DiffusionPolicy(config)


class TestDiffusionWatchdog(unittest.TestCase):
    def test_config_validation(self):
        """Verify parameter validation for Diffusion watchdog configuration."""
        with self.assertRaises(ValueError):
            _create_test_policy(watchdog_tracking_threshold=-0.1)

        with self.assertRaises(ValueError):
            _create_test_policy(watchdog_tracking_threshold=0.0)

        policy = _create_test_policy(watchdog_tracking_threshold=0.25)
        self.assertEqual(policy.config.watchdog_tracking_threshold, 0.25)
        self.assertEqual(policy.config.watchdog_replan_mode, "flush")

    def test_watchdog_disabled_behavior(self):
        """When watchdog is disabled, diffusion action queue is consumed open-loop even if state drifts."""
        policy = _create_test_policy(horizon=16, n_action_steps=6, watchdog_tracking_threshold=None)
        policy.reset()

        state = torch.zeros(1, 4)
        env_state = torch.zeros(1, 4)
        batch = {OBS_STATE: state, OBS_ENV_STATE: env_state}

        # Step 1: Populates action queue (6 actions, 1 popped -> 5 remain)
        _ = policy.select_action(batch)
        self.assertEqual(len(policy._queues[ACTION]), 5)
        self.assertEqual(policy.watchdog_stats["triggers"], 0)

        # Step 2: Large disturbance injected without watchdog
        diverged_batch = {OBS_STATE: state + 10.0, OBS_ENV_STATE: env_state}
        _ = policy.select_action(diverged_batch)

        # Stale actions continue to be consumed
        self.assertEqual(len(policy._queues[ACTION]), 4)
        self.assertEqual(policy.watchdog_stats["triggers"], 0)

    def test_watchdog_triggers_and_flushes_queue_on_drift(self):
        """When state diverges beyond threshold, watchdog flushes stale queue and forces diffusion re-plan."""
        threshold = 0.25
        policy = _create_test_policy(horizon=16, n_action_steps=6, watchdog_tracking_threshold=threshold)
        policy.reset()

        state = torch.zeros(1, 4)
        env_state = torch.zeros(1, 4)
        batch = {OBS_STATE: state, OBS_ENV_STATE: env_state}

        action1 = policy.select_action(batch)
        self.assertEqual(len(policy._queues[ACTION]), 5)
        self.assertEqual(policy.watchdog_stats["triggers"], 0)

        # Step 2: Normal follow-up within tracking tolerance
        normal_batch = {OBS_STATE: action1.clone(), OBS_ENV_STATE: env_state}
        _ = policy.select_action(normal_batch)
        self.assertEqual(len(policy._queues[ACTION]), 4)
        self.assertEqual(policy.watchdog_stats["triggers"], 0)

        # Step 3: Physical disturbance occurs (delta >> threshold)
        disturbed_state = action1 + 2.0
        disturbed_batch = {OBS_STATE: disturbed_state, OBS_ENV_STATE: env_state}

        _ = policy.select_action(disturbed_batch)

        # Watchdog should have triggered
        self.assertEqual(policy.watchdog_stats["triggers"], 1)
        self.assertGreater(policy.watchdog_stats["last_discrepancy"], threshold)

        # Stale queue was flushed and freshly re-planned from disturbed state
        self.assertEqual(len(policy._queues[ACTION]), 5)


if __name__ == "__main__":
    unittest.main()
