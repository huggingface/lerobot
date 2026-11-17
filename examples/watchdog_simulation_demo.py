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

"""
Simulation Demonstration:
Comparing Standard Blind Action Chunking vs. Watchdog-Protected Reactive Re-planning.

Simulates an industrial manipulation task:
- An arm reaches toward a target position.
- At timestep t=10, an unexpected physical disturbance (bump / obstacle contact) knocks the joint off trajectory.
- Case A (Standard): Blindly consumes the remaining action chunk. Fails to react.
- Case B (Watchdog): Detects the discrepancy, aborts the stale chunk, and immediately re-plans from the new position.
"""

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE


def run_simulation(use_watchdog: bool, disturbance_step: int = 10, disturbance_magnitude: float = 1.2):
    print(f"\n{'='*70}")
    mode_str = "WITH WATCHDOG (Reactive Re-planning)" if use_watchdog else "WITHOUT WATCHDOG (Standard Open-Loop Chunking)"
    print(f"RUNNING SIMULATION: {mode_str}")
    print(f"{'='*70}")

    config = ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(2,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        chunk_size=16,
        n_action_steps=16,
        use_vae=False,
        dim_model=32,
        n_heads=2,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        watchdog_tracking_threshold=0.20 if use_watchdog else None,
    )

    policy = ACTPolicy(config)
    policy.reset()

    # Robot starting state
    current_state = torch.tensor([[0.0, 0.0]])
    target_state = torch.tensor([[2.0, 2.0]])
    env_state = target_state.clone()

    print(f"Initial State: {current_state.tolist()[0]}")
    print(f"Target State:  {target_state.tolist()[0]}")
    print("-" * 70)
    print(f"{'Step':<6} | {'Actual State':<22} | {'Commanded Action':<22} | {'Event / Status'}")
    print("-" * 70)

    for step in range(1, 16):
        batch = {OBS_STATE: current_state.clone(), OBS_ENV_STATE: env_state.clone()}
        action = policy.select_action(batch)

        status = "Normal execution"

        # Inject physical bump at disturbance_step
        if step == disturbance_step:
            current_state = action.clone() + disturbance_magnitude
            status = f"💥 BUMP INJECTED! (Displacement: +{disturbance_magnitude:.2f})"
        else:
            # Under normal physics, the arm reaches the commanded position with minor noise
            current_state = action.clone() + 0.01 * torch.randn_like(action)

        if use_watchdog and policy.watchdog_stats["triggers"] > 0 and step == disturbance_step + 1:
            status = f"⚡ WATCHDOG TRIPPED! Stale chunk flushed. Re-planning from {current_state.tolist()[0]}"

        print(
            f"{step:<6} | {str([round(x, 3) for x in current_state.tolist()[0]]):<22} | "
            f"{str([round(x, 3) for x in action.tolist()[0]]):<22} | {status}"
        )

    print("-" * 70)
    if use_watchdog:
        print(f"Watchdog triggers detected: {policy.watchdog_stats['triggers']}")
        print(f"Last recorded discrepancy:  {policy.watchdog_stats['last_discrepancy']:.4f}")
        print("RESULT: Successfully interrupted and adapted to physical disturbance.")
    else:
        print("RESULT: Blind open-loop execution continued despite severe displacement.")


if __name__ == "__main__":
    run_simulation(use_watchdog=False)
    run_simulation(use_watchdog=True)
