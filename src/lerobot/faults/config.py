#!/usr/bin/env python

# Copyright 2026 Gangelia and The HuggingFace Inc. team. All rights reserved.
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

"""Configuration for evaluation-time fault injection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class FaultInjectionConfig:
    """Controls optional fault injection during ``lerobot-eval`` rollouts.

    When ``enabled`` is False (the default), evaluation behavior is unchanged:
    proposed actions reach ``env.step`` without modification and no fault events
    are logged.

    Supported ``type`` values:
        - ``action_hold``: at ``trigger_step``, repeat the previous valid action
          for ``duration`` environment steps, then resume policy actions.
    """

    enabled: bool = False
    type: str = "action_hold"
    # Episode step (0-indexed) at which the fault begins. Must be >= 1 so a
    # previous valid executed action exists to hold.
    trigger_step: int = 55
    # Number of environment steps to hold the previous action.
    duration: int = 8
    # Probability in [0, 1] that the fault activates when the trigger is reached.
    probability: float = 1.0
    # Dedicated RNG seed for activation decisions. Independent of eval seed.
    seed: int | None = 42
    # Vector-env indices to apply the fault to. None means all environments.
    env_ids: list[int] | None = None
    # JSONL path for fault events. Relative paths are resolved against output_dir
    # by the eval entrypoint when provided.
    log_path: Path | None = None

    def __post_init__(self) -> None:
        if isinstance(self.log_path, str):
            self.log_path = Path(self.log_path)
        if self.env_ids is not None:
            self.env_ids = list(self.env_ids)
        # Always validate fields so misconfiguration is caught even before enable.
        self.validate()

    def validate(self, num_envs: int | None = None) -> None:
        """Raise ``ValueError`` if configuration fields are invalid."""
        if self.type != "action_hold":
            raise ValueError(
                f"Unsupported fault type {self.type!r}. Currently supported: 'action_hold'."
            )
        if self.trigger_step < 1:
            raise ValueError(
                f"trigger_step must be >= 1 so a previous valid action exists to hold "
                f"(got {self.trigger_step})."
            )
        if self.duration < 1:
            raise ValueError(f"duration must be >= 1 (got {self.duration}).")
        if not (0.0 <= self.probability <= 1.0):
            raise ValueError(f"probability must be in [0.0, 1.0] (got {self.probability}).")
        if self.env_ids is not None:
            if len(self.env_ids) == 0:
                raise ValueError("env_ids must be non-empty when provided (or leave as None for all).")
            if any(i < 0 for i in self.env_ids):
                raise ValueError(f"env_ids must be non-negative (got {self.env_ids}).")
            if len(self.env_ids) != len(set(self.env_ids)):
                raise ValueError(f"env_ids contain duplicates: {self.env_ids}.")
            if num_envs is not None and any(i >= num_envs for i in self.env_ids):
                raise ValueError(
                    f"env_ids out of range for num_envs={num_envs}: {self.env_ids}."
                )


def default_fault_config() -> FaultInjectionConfig:
    """Factory for ``EvalPipelineConfig.fault`` (disabled by default)."""
    return FaultInjectionConfig()
