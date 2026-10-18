# Adapted from NVlabs/LongLive at 1b9e348897cafa206fc6dd6feb80b59e2a8b1aa9.
# Changes: LeRobot package imports and reference-only runtime integration.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Long-WAM contributor infrastructure, integrated and adapted from the source snapshot.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: src/longwam/runtime/execution.py
# License texts and provenance: see the Long-WAM _core/NOTICE.md.
# Licensed under the Apache License, Version 2.0 for NVIDIA changes; upstream terms are retained.

"""Environment-independent action-chunk scheduling with one request in flight."""

import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Schedule:
    horizon: int
    execute: int
    stride: int

    def __post_init__(self):
        if any(type(x) is not int for x in (self.horizon, self.execute, self.stride)):
            raise TypeError("H/R/S must be integers")
        if not 0 < self.stride < self.execute <= self.horizon or 2 * self.stride < self.execute:
            raise ValueError("Async scheduling requires R/2 <= S < R <= H")


class AsyncPolicy:
    """Return one action per call; the caller alone advances the environment.

    Requests and results carry episode and control-step identities. At each
    handoff, the elapsed prefix of the new chunk is discarded. A late result
    pauses control at the boundary rather than extending the previous chunk.
    """

    def __init__(self, worker, schedule, *, timeout=600.0):
        self.worker, self.schedule, self.timeout = worker, schedule, float(timeout)
        self.episode = 0
        self._pending = None
        self._active = None
        self.step = 0
        self.instruction = ""
        self.metrics = {"requests": 0, "late_handoffs": 0, "wait_seconds": 0.0}
        self._closed = False

    def reset(self, instruction=""):
        if self._closed:
            raise RuntimeError("Policy is closed")
        if self._pending is not None:
            self._pending.result(timeout=self.timeout)
        self.episode += 1
        self.worker.call("reset", {"episode": self.episode, "instruction": instruction})
        self.step, self.instruction = 0, str(instruction)
        self._active, self._pending = None, None
        self.metrics = {"requests": 0, "late_handoffs": 0, "wait_seconds": 0.0}

    def _submit(self):
        self._anchor = self.step
        self._pending = self.worker.submit("infer", {"episode": self.episode, "anchor": self.step})
        self.metrics["requests"] += 1

    def _commit(self, *, handoff):
        if self._pending is None:
            raise RuntimeError("No action candidate at the handoff boundary")
        late = not self._pending.done()
        start = time.monotonic()
        result = self._pending.result(timeout=self.timeout)
        self.metrics["wait_seconds"] += time.monotonic() - start
        self.metrics["late_handoffs"] += int(handoff and late)
        if result["episode"] != self.episode or result["anchor"] != self._anchor:
            raise RuntimeError("Stale or misaligned inference result")
        actions = np.asarray(result["actions"])
        if actions.ndim != 2 or actions.shape[0] != self.schedule.horizon or actions.shape[1] == 0:
            raise ValueError("Action candidate must have shape [H, D]")
        if not np.isfinite(actions).all():
            raise ValueError("Action candidate contains non-finite values")
        if not self._anchor <= self.step < self._anchor + self.schedule.execute:
            raise RuntimeError("The candidate has no eligible suffix at this boundary")
        self._active = (self._anchor, actions)
        self._pending = None

    def act(self, observation):
        if self._closed:
            raise RuntimeError("Policy is closed")
        self.worker.notify(
            "observe", {"episode": self.episode, "step": self.step, "observation": observation}
        )
        if self._active is None:
            self._submit()
            self._commit(handoff=False)
        elif self.step == self._active[0] + self.schedule.execute:
            self._commit(handoff=True)
        anchor, actions = self._active
        if self.step == anchor + self.schedule.stride:
            self._submit()
        action = actions[self.step - anchor].copy()
        self.step += 1
        return action

    def close(self):
        if not self._closed:
            self.worker.close(force=self._pending is not None)
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@dataclass(frozen=True)
class SyncSchedule:
    horizon: int
    execute: int

    def __post_init__(self):
        if any(type(x) is not int for x in (self.horizon, self.execute)):
            raise TypeError("H/R must be integers")
        if not 0 < self.execute <= self.horizon:
            raise ValueError("Sync scheduling requires 0 < R <= H")


class SyncPolicy(AsyncPolicy):
    """Predict at the boundary and execute an unblended prefix before replanning."""

    def act(self, observation):
        if self._closed:
            raise RuntimeError("Policy is closed")
        self.worker.notify(
            "observe", {"episode": self.episode, "step": self.step, "observation": observation}
        )
        if self._active is None or self.step == self._active[0] + self.schedule.execute:
            self._submit()
            self._commit(handoff=False)
        anchor, actions = self._active
        action = actions[self.step - anchor].copy()
        self.step += 1
        return action
