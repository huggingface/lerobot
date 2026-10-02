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

"""Adapter from one scalar Gymnasium environment to ``TaskEndpoint``."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol

import gymnasium as gym
import numpy as np
import torch

from lerobot.utils.constants import ACTION

from .endpoint import StepResult

if TYPE_CHECKING:
    from lerobot.envs.configs import EnvConfig


class GymEndpointAdapter(Protocol):
    """Conversion between one provider's values and canonical LeRobot values."""

    def reset(self) -> None:
        """Reset episode-scoped conversion state."""
        ...

    def to_canonical_observation(self, observation: Any) -> dict[str, Any]:
        """Map a provider observation to canonical LeRobot features."""
        ...

    def to_native_action(self, action: torch.Tensor) -> Any:
        """Map a canonical action to the provider action space."""
        ...

    def to_canonical_applied_action(self, action: Any, info: dict[str, Any]) -> torch.Tensor:
        """Report what the provider applied in canonical action space."""
        ...


class IdentityGymAdapter:
    """Adapter for environments that already use canonical observation keys."""

    def reset(self) -> None:
        """Keep no episode-scoped state."""

    def to_canonical_observation(self, observation: Any) -> dict[str, Any]:
        """Return an already-canonical dictionary observation."""
        if not isinstance(observation, dict):
            raise TypeError("IdentityGymAdapter requires dictionary observations")
        return observation

    def to_native_action(self, action: torch.Tensor) -> np.ndarray:
        """Convert a scalar-runtime tensor to an unbatched NumPy action."""
        native = action.detach().cpu().numpy()
        if native.ndim == 2 and native.shape[0] == 1:
            native = native[0]
        return native

    def to_canonical_applied_action(self, action: Any, info: dict[str, Any]) -> torch.Tensor:
        """Convert an unchanged native action back to a tensor."""
        del info
        return torch.as_tensor(action)


class FeatureMapGymAdapter:
    """Rename active Gym features with an ``EnvConfig.features_map``."""

    def __init__(
        self,
        features_map: Mapping[str, str],
        *,
        active_features: Collection[str] | None = None,
        to_native_action: Callable[[torch.Tensor], Any] | None = None,
        to_canonical_applied_action: Callable[[Any, dict[str, Any]], torch.Tensor] | None = None,
    ) -> None:
        """Bind active feature names and optional provider-specific conversions."""
        active = set(active_features) if active_features is not None else None
        self.features_map = {
            source: target
            for source, target in features_map.items()
            if source != ACTION and target != ACTION and (active is None or source in active)
        }
        self._to_native_action = to_native_action or IdentityGymAdapter().to_native_action
        self._to_canonical_applied_action = (
            to_canonical_applied_action or IdentityGymAdapter().to_canonical_applied_action
        )

    @classmethod
    def from_env_config(
        cls,
        cfg: EnvConfig,
        *,
        to_native_action: Callable[[torch.Tensor], Any] | None = None,
        to_canonical_applied_action: Callable[[Any, dict[str, Any]], torch.Tensor] | None = None,
    ) -> FeatureMapGymAdapter:
        """Build an adapter from only the features enabled by ``cfg``.

        Environment configs retain mappings for alternate observation modes. Filtering
        here avoids asking a PushT pixels observation for an inactive environment state,
        for example. Environments that require their own processor pipelines must use
        an explicit adapter until those pipelines are part of the scalar runtime.
        """
        env_preprocessor, env_postprocessor = cfg.get_env_processors()
        if env_preprocessor.steps or env_postprocessor.steps:
            raise NotImplementedError(
                f"{cfg.type!r} requires environment-specific processors that the scalar runtime "
                "does not compose yet; pass an explicit GymEndpointAdapter"
            )
        return cls(
            cfg.features_map,
            active_features=cfg.features,
            to_native_action=to_native_action,
            to_canonical_applied_action=to_canonical_applied_action,
        )

    def reset(self) -> None:
        """Keep no episode-scoped state."""

    @staticmethod
    def _read_observation(observation: Mapping[str, Any], source_key: str) -> Any:
        if source_key in observation:
            return observation[source_key]
        value: Any = observation
        for component in source_key.split("/"):
            if not isinstance(value, Mapping) or component not in value:
                raise KeyError(f"Gym observation is missing mapped feature {source_key!r}")
            value = value[component]
        return value

    def to_canonical_observation(self, observation: Any) -> dict[str, Any]:
        """Rename flat or slash-delimited provider observation keys."""
        if not isinstance(observation, Mapping):
            raise TypeError("FeatureMapGymAdapter requires a mapping observation")
        return {
            target: self._read_observation(observation, source)
            for source, target in self.features_map.items()
        }

    def to_native_action(self, action: torch.Tensor) -> Any:
        """Convert one canonical action with the configured callback."""
        return self._to_native_action(action)

    def to_canonical_applied_action(self, action: Any, info: dict[str, Any]) -> torch.Tensor:
        """Map a provider action back to canonical space."""
        return self._to_canonical_applied_action(action, info)


class GymEndpoint:
    """A scalar Gymnasium environment behind the shared endpoint contract."""

    def __init__(
        self,
        env: gym.Env,
        *,
        adapter: GymEndpointAdapter,
        action_names: Sequence[str] | None = None,
        success_key: str = "is_success",
        close_env: bool = True,
    ) -> None:
        """Wrap one scalar environment and declare its canonical action order."""
        if isinstance(env, gym.vector.VectorEnv):
            raise TypeError("GymEndpoint accepts one scalar environment, not a vector environment")
        self.env = env
        self.adapter = adapter
        self._action_names = tuple(action_names) if action_names is not None else None
        if self._action_names == ():
            raise ValueError("action_names must be non-empty when provided")
        self.success_key = success_key
        self.close_env = close_env
        self._has_reset = False
        self._reset_info: dict[str, Any] = {}

    @property
    def action_names(self) -> tuple[str, ...] | None:
        """Return the simulator adapter's canonical action order when declared."""
        return self._action_names

    @property
    def reset_info(self) -> dict[str, Any]:
        """Metadata produced by the latest reset."""
        return self._reset_info.copy()

    def reset(self, *, seed: int | None = None) -> dict[str, Any]:
        """Reset adapter and environment, then return the canonical observation."""
        self.adapter.reset()
        native_observation, info = self.env.reset(seed=seed)
        self._has_reset = True
        self._reset_info = dict(info)
        return self.adapter.to_canonical_observation(native_observation)

    def step(self, action: torch.Tensor) -> StepResult:
        """Apply one action and preserve Gym's native outcome in ``StepResult``."""
        if not self._has_reset:
            raise RuntimeError("Call reset() before stepping a GymEndpoint")
        native_action = self.adapter.to_native_action(action)
        observation, reward, terminated, truncated, info = self.env.step(native_action)
        info = dict(info)
        success = info.get(self.success_key)
        return StepResult(
            observation=self.adapter.to_canonical_observation(observation),
            applied_action=self.adapter.to_canonical_applied_action(native_action, info),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            success=bool(success) if success is not None else None,
            info=info,
        )

    def start_step(self) -> None:
        """Run simulation without wall-clock pacing."""

    def stop(self) -> None:
        """Require no extra stop command for a stepped simulator."""

    def close(self) -> None:
        """Close the environment when this adapter owns it."""
        if self.close_env:
            self.env.close()
