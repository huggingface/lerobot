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

"""Unit tests for ACT Adaptive Discrepancy Watchdog & Reactive Re-planning."""

import unittest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE


def _create_test_policy(
    chunk_size: int = 8,
    n_action_steps: int = 8,
    temporal_ensemble_coeff: float | None = None,
    watchdog_tracking_threshold: float | None = None,
) -> ACTPolicy:
    config = ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(4,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(4,))},
        chunk_size=chunk_size,
        n_action_steps=n_action_steps,
        use_vae=False,
        dim_model=32,
        n_heads=2,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        temporal_ensemble_coeff=temporal_ensemble_coeff,
        watchdog_tracking_threshold=watchdog_tracking_threshold,
    )
    return ACTPolicy(config)


class TestACTWatchdog(unittest.TestCase):
    def test_config_validation(self):
        """Verify parameter validation for the watchdog configuration."""
        # Non-positive threshold should raise ValueError
        with self.assertRaises(ValueError):
            _create_test_policy(watchdog_tracking_threshold=-0.1)

        with self.assertRaises(ValueError):
            _create_test_policy(watchdog_tracking_threshold=0.0)

        # Valid positive threshold
        policy = _create_test_policy(watchdog_tracking_threshold=0.2)
        self.assertEqual(policy.config.watchdog_tracking_threshold, 0.2)
        self.assertEqual(policy.config.watchdog_replan_mode, "flush")

    def test_watchdog_disabled_behavior(self):
        """When watchdog is disabled, action queue is consumed open-loop even if state drifts."""
        policy = _create_test_policy(chunk_size=6, n_action_steps=6, watchdog_tracking_threshold=None)
        policy.reset()

        state = torch.zeros(1, 4)
        env_state = torch.zeros(1, 4)
        batch = {OBS_STATE: state, OBS_ENV_STATE: env_state}

        # Step 1: Populates action queue (6 actions generated, 1 popped -> 5 remain)
        action1 = policy.select_action(batch)
        self.assertEqual(len(policy._action_queue), 5)
        self.assertEqual(policy.watchdog_stats["triggers"], 0)

        # Step 2: Inject large state divergence
        diverged_batch = {OBS_STATE: state + 10.0, OBS_ENV_STATE: env_state}
        _ = policy.select_action(diverged_batch)

        # Without watchdog, queue is not flushed; next action is simply popped
        self.assertEqual(len(policy._action_queue), 4)
        self.assertEqual(policy.watchdog_stats["triggers"], 0)

    def test_watchdog_triggers_and_flushes_queue_on_drift(self):
        """When state diverges beyond threshold, watchdog flushes stale queue and forces re-plan."""
        threshold = 0.25
        policy = _create_test_policy(chunk_size=6, n_action_steps=6, watchdog_tracking_threshold=threshold)
        policy.reset()

        # Step 1: Initial observation at state [0, 0, 0, 0]
        state = torch.zeros(1, 4)
        env_state = torch.zeros(1, 4)
        batch = {OBS_STATE: state, OBS_ENV_STATE: env_state}

        action1 = policy.select_action(batch)
        self.assertEqual(len(policy._action_queue), 5)
        self.assertEqual(policy.watchdog_stats["triggers"], 0)

        # Step 2: Normal follow-up within tracking tolerance (robot followed action1 closely)
        normal_batch = {OBS_STATE: action1.clone(), OBS_ENV_STATE: env_state}
        _ = policy.select_action(normal_batch)
        self.assertEqual(len(policy._action_queue), 4)
        self.assertEqual(policy.watchdog_stats["triggers"], 0)

        # Step 3: Physical disturbance occurs! Measured state is far away from commanded action
        disturbed_state = action1 + 1.5  # Delta = 1.5 >> threshold of 0.25
        disturbed_batch = {OBS_STATE: disturbed_state, OBS_ENV_STATE: env_state}

        action_recovered = policy.select_action(disturbed_batch)

        # Watchdog should have triggered
        self.assertEqual(policy.watchdog_stats["triggers"], 1)
        self.assertGreater(policy.watchdog_stats["last_discrepancy"], threshold)

        # Because the queue was flushed and a fresh chunk predicted for the disturbed state,
        # the queue now has (n_action_steps - 1) = 5 actions remaining, rather than 3!
        self.assertEqual(len(policy._action_queue), 5)

    def test_watchdog_temporal_ensembler_reset(self):
        """When temporal ensembling is active, watchdog flushes stale history on discrepancy."""
        threshold = 0.3
        policy = _create_test_policy(
            chunk_size=6,
            n_action_steps=1,
            temporal_ensemble_coeff=0.01,
            watchdog_tracking_threshold=threshold,
        )
        policy.reset()

        state = torch.zeros(1, 4)
        env_state = torch.zeros(1, 4)
        batch = {OBS_STATE: state, OBS_ENV_STATE: env_state}

        action1 = policy.select_action(batch)
        self.assertIsNotNone(policy.temporal_ensembler.ensembled_actions)
        self.assertEqual(policy.watchdog_stats["triggers"], 0)

        # Step 2: Follow closely (no trigger)
        normal_batch = {OBS_STATE: action1.clone(), OBS_ENV_STATE: env_state}
        action2 = policy.select_action(normal_batch)
        self.assertEqual(policy.watchdog_stats["triggers"], 0)

        # Step 3: Induce severe tracking anomaly
        disturbed_state = action2 + 2.0
        disturbed_batch = {OBS_STATE: disturbed_state, OBS_ENV_STATE: env_state}

        _ = policy.select_action(disturbed_batch)
        self.assertEqual(policy.watchdog_stats["triggers"], 1)
        self.assertGreater(policy.watchdog_stats["last_discrepancy"], threshold)


if __name__ == "__main__":
    unittest.main()
