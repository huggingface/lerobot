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

"""Execution record and history sampling shared by every chunk execution path.

The side that sees every control tick (the robot client, or the evaluation loop)
records what each tick observed and which policy action it dispatched. A chunk
request then carries the history its policy needs (:func:`policy_history`):

* time-anchored windows sample the tick captured closest to ``capture + offset x
  interval``, so late or skipped ticks do not stretch the window the way counting
  calls would. Targets before the first recorded tick are padding, like a dataset
  window crossing the episode start;
* the command window holds the command in effect at each target: the last action
  dispatched at or before it.

Sampling never fails on a timing gap; the largest distance between a target and
the tick used for it is reported as ``max_skew_s``.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import Deltas, FeatureType
from lerobot.utils.constants import OBS_ACTION


@dataclass(frozen=True)
class FeatureHistory:
    """Past values of one observation feature, as the policy was trained on them.

    ``offsets`` are training-dataset frames relative to the current one, ascending and
    ending at ``0`` (the current frame), as the config's ``inference_deltas`` declare. At
    inference a frame is one action interval, and each offset is sampled by capture time.
    """

    feature: str
    offsets: tuple[int, ...]

    def __post_init__(self) -> None:
        """Require ascending past offsets that end at the current frame."""
        offsets = tuple(self.offsets)
        object.__setattr__(self, "offsets", offsets)
        if not self.feature or not offsets or offsets[-1] != 0:
            raise ValueError("A feature history needs a name and offsets ending at the current frame (0).")
        if any(type(offset) is not int for offset in offsets) or list(offsets) != sorted(set(offsets)):
            raise ValueError("Feature history offsets must be distinct ascending integers.")


@dataclass(frozen=True)
class ChunkHistory:
    """Everything a chunk call needs besides the current observation, as runners sample it.

    Derived from a policy by :func:`policy_history`. ``observations`` and ``actions`` are
    windows anchored to time: ``actions`` are the offsets (negative, ascending) of the
    commands in effect, i.e. the dataset's ``action`` column.
    """

    observations: tuple[FeatureHistory, ...] = ()
    actions: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Reject duplicate or overlapping declarations."""
        observations = tuple(self.observations)
        actions = tuple(self.actions)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "actions", actions)
        names = [window.feature for window in observations]
        if len(set(names)) != len(names):
            raise ValueError("Each feature may declare one history window.")
        if any(type(offset) is not int or offset >= 0 for offset in actions) or list(actions) != sorted(
            set(actions)
        ):
            raise ValueError("Action history offsets must be distinct ascending negative integers.")

    @property
    def empty(self) -> bool:
        """Whether the current observation suffices."""
        return not self.observations and not self.actions

    @property
    def lookback_frames(self) -> int:
        """How many frames before the current one the time-anchored windows reach."""
        offsets = [offset for window in self.observations for offset in window.offsets] + list(self.actions)
        return -min(offsets, default=0)

    @property
    def recorded_features(self) -> tuple[str, ...]:
        """Observation features a client must keep for later requests."""
        return tuple(window.feature for window in self.observations)


def policy_history(config: PreTrainedConfig) -> ChunkHistory:
    """What a policy's chunk calls need besides the current observation.

    Observation windows and past commands come from the config's ``inference_deltas``.

    Raises:
        ValueError: If the inference deltas declare targets, or windows a runner cannot serve.
    """
    deltas = getattr(config, "inference_deltas", None) or Deltas()
    if deltas.action is not None or deltas.reward is not None:
        raise ValueError("Inference deltas declare inputs only: use observation ACTION for past commands.")
    windows = []
    for key, feature in (getattr(config, "input_features", None) or {}).items():
        if (offsets := deltas.of(key, feature.type)) is not None:
            windows.append(FeatureHistory(key, offsets))
    actions = deltas.observation.get(FeatureType.ACTION, ())
    return ChunkHistory(observations=tuple(windows), actions=tuple(actions))


@dataclass(frozen=True)
class HistoryInputs:
    """History sampled for one chunk request, oldest first, excluding the current frame.

    Values keep the representation the recording side stored: canonical wire arrays
    for robot clients, batched tensors for the evaluation loop.
    """

    observations: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)
    observation_padding: Mapping[str, tuple[bool, ...]] = field(default_factory=dict)
    actions: tuple[Any, ...] = ()
    action_padding: tuple[bool, ...] = ()
    max_skew_s: float = 0.0


@dataclass
class _Tick:
    time: float
    values: dict[str, Any]
    action: Any = None


class ExecutionRecord:
    """Per-tick observations and dispatched actions since the last reset.

    Only the features a history declares are stored, for ``lookback`` plus two
    intervals of ticks.
    Not thread-safe: engines guard it with the lock that orders their captures.
    """

    def __init__(
        self,
        history: ChunkHistory,
        *,
        interval: float,
        action_padding_value: Callable[[], Any] | None = None,
    ) -> None:
        """Bind a declared history to the control interval it is sampled at.

        Args:
            history: The policy's declared history.
            interval: Seconds per policy action, i.e. per training frame.
            action_padding_value: Builds the value of a padded command slot.
        """
        if interval <= 0:
            raise ValueError("History interval must be positive.")
        self.history = history
        self.interval = interval
        self._features = set(history.recorded_features)
        self._keep_seconds = (history.lookback_frames + 2) * interval
        self._pad_action = action_padding_value
        self._ticks: deque[_Tick] = deque()
        self._start: float | None = None
        self._first: dict[str, Any] = {}

    @property
    def active(self) -> bool:
        """Whether this record has anything to keep."""
        return not self.history.empty

    def clear(self) -> None:
        """Start a new episode: earlier ticks become padding."""
        self._ticks.clear()
        self._start = None
        self._first = {}

    def observe(self, time: float, values: Mapping[str, Any]) -> None:
        """Record one control tick's capture. Ticks must arrive in capture order."""
        if not self.active:
            return
        if self._ticks and time <= self._ticks[-1].time:
            if time == self._ticks[-1].time:
                return  # the same capture published again
            raise ValueError("Execution record ticks must have increasing capture times.")
        kept = {name: values[name] for name in self._features if name in values}
        if missing := self._features - kept.keys():
            raise ValueError(f"Observation lacks recorded history features: {sorted(missing)}.")
        if self._start is None:
            self._start, self._first = time, kept
        self._ticks.append(_Tick(time, kept))
        while len(self._ticks) > 1 and self._ticks[1].time < self._ticks[-1].time - self._keep_seconds:
            self._ticks.popleft()

    def dispatched(self, action: Any) -> None:
        """Attach the action the latest tick dispatched."""
        if self.active and self._ticks and self._ticks[-1].action is None:
            self._ticks[-1].action = action

    def sample(self, time: float) -> HistoryInputs:
        """Sample the declared history for a request whose observation was captured at ``time``.

        Raises:
            ValueError: If the request's observation was not recorded.
        """
        if not self.active:
            return HistoryInputs()
        times = [tick.time for tick in self._ticks]
        query = bisect_right(times, time) - 1
        if query < 0 or times[query] != time or self._start is None:
            raise ValueError("The request's observation is not in the execution record.")
        skew = 0.0
        observations: dict[str, tuple[Any, ...]] = {}
        padding: dict[str, tuple[bool, ...]] = {}
        for window in self.history.observations:
            values, pads = [], []
            for offset in window.offsets[:-1]:
                tick, distance = self._nearest(times, query, time + offset * self.interval)
                pads.append(tick is None)
                values.append(self._first[window.feature] if tick is None else tick.values[window.feature])
                skew = max(skew, distance)
            observations[window.feature], padding[window.feature] = tuple(values), tuple(pads)
        actions, action_pads = [], []
        for offset in self.history.actions:
            # The query tick's own command is not yet part of the past.
            tick, distance = self._nearest(times, query - 1, time + offset * self.interval)
            command = None
            if tick is not None:
                skew = max(skew, distance)
                position = self._ticks.index(tick)
                command = next(
                    (
                        self._ticks[i].action
                        for i in range(position, -1, -1)
                        if self._ticks[i].action is not None
                    ),
                    None,
                )
            if command is None:
                if self._pad_action is None:
                    raise ValueError("Command history padding needs an action_padding_value.")
                command = self._pad_action()
                action_pads.append(True)
            else:
                action_pads.append(False)
            actions.append(command)
        return HistoryInputs(
            observations=observations,
            observation_padding=padding,
            actions=tuple(actions),
            action_padding=tuple(action_pads),
            max_skew_s=skew,
        )

    def _nearest(self, times: list[float], last: int, target: float) -> tuple[_Tick | None, float]:
        """The tick at or before ``last`` closest to ``target``; ``None`` before the episode."""
        assert self._start is not None
        if target < self._start - self.interval / 2 or last < 0:
            return None, 0.0
        right = min(bisect_right(times, target, hi=last + 1), last)
        candidates = [i for i in (right - 1, right) if 0 <= i <= last]
        best = min(candidates, key=lambda i: abs(times[i] - target))
        return self._ticks[best], abs(times[best] - target)


def recorded_names(keys: Iterable[str], rename_maps: Iterable[Mapping[str, str]]) -> dict[str, str]:
    """Map each policy feature name to the observation key it is recorded under.

    ``rename_maps`` are the preprocessor's rename steps in order; a recorder sees keys
    before any of them applies.
    """
    names = {key: key for key in keys}
    for rename_map in rename_maps:
        names = {key: rename_map.get(name, name) for key, name in names.items()}
    return {policy_name: key for key, policy_name in names.items()}


def rename_history(history: ChunkHistory, names: Mapping[str, str]) -> ChunkHistory:
    """Re-key a declared history from policy feature names to the names a recorder sees.

    Raises:
        ValueError: If a declared feature has no recorded counterpart.
    """
    if missing := set(history.recorded_features) - set(names):
        raise ValueError(f"History features have no recorded observation: {sorted(missing)}.")
    return ChunkHistory(
        observations=tuple(FeatureHistory(names[w.feature], w.offsets) for w in history.observations),
        actions=history.actions,
    )


def apply_history(
    batch: dict[str, Any],
    history: HistoryInputs,
    to_tensor: Callable[[str, Any], torch.Tensor],
) -> dict[str, Any]:
    """Lay out sampled history the way training batches carry it.

    ``to_tensor(name, value)`` converts one stored value to the batched tensor the
    current observation has under ``name`` (``[B, ...]``); commands use ``"action"``.
    Windows become ``[B, T, ...]`` ending with the current value and gain
    ``{name}_is_pad``; commands become ``observation.action`` ``[B, T, A]`` with
    ``observation.action_is_pad``.
    """
    for name, values in history.observations.items():
        current = batch[name]
        past = [to_tensor(name, value).to(current.device, current.dtype) for value in values]
        batch[name] = torch.stack([*past, current], dim=1)
        pads = [*history.observation_padding[name], False]
        batch[f"{name}_is_pad"] = torch.tensor(pads, device=current.device).expand(current.shape[0], -1)
    if history.actions:
        commands = torch.stack([to_tensor("action", value) for value in history.actions], dim=1)
        batch[OBS_ACTION] = commands
        batch[f"{OBS_ACTION}_is_pad"] = torch.tensor(history.action_padding, device=commands.device).expand(
            commands.shape[0], -1
        )
    return batch
