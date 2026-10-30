# Copyright 2026 Shaohua Pan.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
# See the repository root LICENSE for the full license.
# Adapted from Streaming-WAM, Xuyao Huang et al.
# Source revision: ff36d6e66514afd65bf957a06a0a2410f182b0e3.

"""Bounded H32/s16/d8 controller for a serial, acknowledged control loop."""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


def snapshot(value: Any) -> Any:
    """Detach observation buffers before the control loop can reuse them."""
    if isinstance(value, Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(snapshot(item) for item in value)
    return copy.deepcopy(value)


def prefix_target(chunk: Tensor, cursor: int) -> Tensor:
    """Align the old normalized chunk with the observation at the launch cursor."""
    if chunk.ndim != 3 or chunk.shape[:2] != (1, 32) or not 0 <= cursor <= 32:
        raise ValueError("Expected [1,32,D] chunk and cursor in [0,32].")
    target = torch.zeros_like(chunk)
    count = min(16, 32 - cursor)
    target[:, :count] = chunk[:, cursor : cursor + count]
    return target.detach()


@dataclass(frozen=True)
class Completion:
    generation: int
    actions: Tensor
    started_ns: int
    completed_ns: int


class StreamingController:
    """Keep normalized actions and overlap one inference with acknowledged execution.

    Call ``next_action(observation)``, execute the returned action, then call
    ``mark_action_executed()``. Calls must be serial. Reset and close drain the
    worker, including errors, before another episode can use the model.
    """

    def __init__(
        self,
        predict: Callable[[dict[str, Any], Tensor | None], Tensor],
        *,
        executor: Executor | None = None,
        clock: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        self._predict = predict
        self._executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="streaming_wam")
        self._clock = clock
        self._generation = 0
        self._closed = False
        self._future: Future[Completion] | None = None
        self._chunk: Tensor | None = None
        self._cursor = 0
        self._boundary = 16
        self._launch_after = 8
        self._issued = False
        self._boundary_ns: int | None = None
        self.inference_ms: list[float] = []
        self.wait_ms: list[float] = []
        self.deadline_misses = 0

    def _submit(self, observation: dict[str, Any], previous: Tensor | None) -> Future[Completion]:
        observation = snapshot(observation)
        generation = self._generation

        def run() -> Completion:
            started = self._clock()
            actions = self._predict(observation, previous)
            if actions.ndim != 3 or actions.shape[:2] != (1, 32):
                raise ValueError("Streaming inference must return [1,32,D] normalized actions.")
            if previous is not None and not torch.equal(actions[:, :8], previous[:, :8].to(actions)):
                raise RuntimeError("Streaming inference changed the hard prefix.")
            return Completion(generation, actions.detach().clone(), started, self._clock())

        return self._executor.submit(run)

    def _install(self, future: Future[Completion], cursor: int) -> None:
        before = self._clock()
        result = future.result()
        after = self._clock()
        if result.generation != self._generation:
            raise RuntimeError("Stale streaming episode result.")
        self.inference_ms.append((result.completed_ns - result.started_ns) / 1e6)
        if self._boundary_ns is not None:
            self.wait_ms.append(max(0, after - before) / 1e6)
            self.deadline_misses += int(result.completed_ns > self._boundary_ns)
        self._chunk = result.actions
        self._cursor = cursor
        self._boundary = cursor + 16
        self._launch_after = cursor + 8
        self._boundary_ns = None

    def next_action(self, observation: dict[str, Any]) -> Tensor:
        if self._closed:
            raise RuntimeError("Streaming controller is closed.")
        if self._issued:
            raise RuntimeError("Call mark_action_executed before requesting another action.")
        if self._chunk is None:
            self._install(self._submit(observation, None), 0)
        if self._cursor >= self._boundary:
            if self._future is None:
                raise RuntimeError("Streaming boundary has no continuation.")
            pending, self._future = self._future, None
            self._install(pending, 8)
        if self._cursor == self._launch_after:
            if self._chunk is None:
                raise RuntimeError("Streaming launch has no current chunk.")
            self._future = self._submit(observation, prefix_target(self._chunk, self._cursor))
        if self._chunk is None:
            raise RuntimeError("Streaming installation did not produce a chunk.")
        self._issued = True
        return self._chunk[:, self._cursor].clone()

    def mark_action_executed(self) -> None:
        """Acknowledge exactly one returned action after execution completes."""
        if not self._issued:
            raise RuntimeError("No returned action is awaiting execution acknowledgement.")
        self._issued = False
        self._cursor += 1
        if self._cursor == self._boundary:
            self._boundary_ns = self._clock()

    def count_queued_actions(self) -> int:
        """Count current actions still available before the continuation boundary."""
        if self._chunk is None:
            return 0
        return max(0, self._boundary - self._cursor - int(self._issued))

    def reset(self) -> None:
        """Drain pending work, invalidate its generation, and clear the episode."""
        self._generation += 1
        pending, self._future = self._future, None
        try:
            if pending is not None and not pending.cancel():
                pending.result()
        finally:
            self._chunk = None
            self._cursor, self._boundary, self._launch_after = 0, 16, 8
            self._issued = False
            self._boundary_ns = None
            self.inference_ms.clear()
            self.wait_ms.clear()
            self.deadline_misses = 0

    def close(self) -> None:
        """Drain the worker and release its thread; safe to call more than once."""
        if self._closed:
            return
        try:
            self.reset()
        finally:
            self._closed = True
            self._executor.shutdown(wait=True, cancel_futures=True)
