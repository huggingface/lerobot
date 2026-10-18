# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional comparison against an independently checked-out release scheduler."""

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from lerobot.policies.longwam import scheduling_longwam as native


class TraceFuture:
    def __init__(self, value, events, late):
        self.value, self.events, self.late = value, events, late

    def done(self):
        return not self.late

    def result(self, **kwargs):
        self.events.append(("result", self.value["episode"], self.value["anchor"], self.late))
        self.late = False
        return self.value


class TraceWorker:
    def __init__(self, late):
        self.events, self.observations = [], []
        self.late = late

    def call(self, operation, payload):
        assert operation == "reset"
        self.observations.clear()
        self.events.append(("reset", payload["episode"], payload["instruction"]))

    def notify(self, operation, payload):
        assert operation == "observe"
        self.observations.append(payload["observation"])

    def submit(self, operation, payload):
        assert operation == "infer"
        self.events.append(("anchor", payload["episode"], payload["anchor"], tuple(self.observations)))
        actions = np.array([[payload["episode"], payload["anchor"], offset] for offset in range(8)])
        return TraceFuture({**payload, "actions": actions}, self.events, self.late and payload["anchor"] == 4)


def trace(module, execution, late, reset_pending):
    worker = TraceWorker(late)
    policy = (
        module.AsyncPolicy(worker, module.Schedule(8, 6, 4))
        if execution == "async"
        else module.SyncPolicy(worker, module.SyncSchedule(8, 6))
    )
    policy.reset("first")
    actions = [tuple(policy.act(step)) for step in range(5 if reset_pending else 15)]
    policy.reset("second")
    actions.extend(tuple(policy.act(step)) for step in range(9))
    return actions, worker.events


@pytest.fixture(scope="module")
def reference():
    root = os.environ.get("LONGWAM_REFERENCE_ROOT")
    if root is None:
        pytest.skip("Set LONGWAM_REFERENCE_ROOT to the pinned Long-WAM source checkout")
    path = Path(root) / "src/longwam/runtime/execution.py"
    spec = importlib.util.spec_from_file_location("longwam_reference_scheduler", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


@pytest.mark.parametrize("execution", ["sync", "async"])
@pytest.mark.parametrize("late", [False, True])
@pytest.mark.parametrize("reset_pending", [False, True])
def test_reference_agreement_on_chunk_identity_observation_anchors_and_reset(
    reference, execution, late, reset_pending
):
    assert trace(native, execution, late, reset_pending) == trace(reference, execution, late, reset_pending)
