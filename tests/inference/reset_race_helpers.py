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

"""Independent two-joint evidence for asynchronous reset/dispatch cases."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from threading import Condition, Event, Lock, local
from types import SimpleNamespace

import torch

from lerobot.inference import rtc
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.rollout.configs import BaseStrategyConfig
from lerobot.rollout.robot_wrapper import ThreadSafeRobot
from lerobot.rollout.strategies.base import BaseStrategy
from lerobot.rollout.strategies.core import send_next_action
from lerobot.utils.action_interpolator import ActionInterpolator

from .rtc_helpers import ControlledPolicy, make_engine

JOINTS = ("j1.pos", "j2.pos")
OLD_INPUT = (65.0, 81.0)
FRESH_INPUT = (129.0, 145.0)
# Independent, exactly representable float32 stimuli, never copied from a returned tensor.
ACTIONS = {
    OLD_INPUT: (
        (4100.0, 4124.0),
        (4164.0, 4188.0),
        (4228.0, 4252.0),
        (4292.0, 4316.0),
        (4356.0, 4380.0),
        (4420.0, 4444.0),
    ),
    FRESH_INPUT: (
        (8196.0, 8220.0),
        (8260.0, 8284.0),
        (8324.0, 8348.0),
        (8388.0, 8412.0),
        (8452.0, 8476.0),
        (8516.0, 8540.0),
    ),
}
WAIT_SECONDS = 5.0
# Independent reference: elementary arithmetic on declared endpoints, never runtime buffers.
INTERPOLATED_ACTIONS = {
    observed: (endpoints[0],)
    + tuple(
        tuple(a + (b - a) * quarter / 4 for a, b in zip(left, right, strict=True))
        for left, right in zip(endpoints, endpoints[1:], strict=False)
        for quarter in range(1, 5)
    )
    for observed, endpoints in ACTIONS.items()
}


@dataclass
class Call:
    number: int
    consumed: torch.Tensor
    released: Event = field(default_factory=Event)
    returned: Event = field(default_factory=Event)
    handled: Event = field(default_factory=Event)
    accepted: bool | None = None
    admission_id: int | None = None


@dataclass(frozen=True)
class Command:
    kind: str
    requested: tuple[tuple[str, float], ...]
    applied: tuple[tuple[str, float], ...]
    measured: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class FreshnessWitness:
    consumed: tuple[float, float]
    expected: tuple[float, float]
    old_capture_id: int
    fresh_capture_id: int
    invocation: int
    gate: str
    ordering: tuple[tuple[int, str], ...]
    prediction_released: bool
    prediction_returned: bool


@dataclass(frozen=True)
class BufferWitness:
    multiplier: int
    index: int
    remaining: tuple[tuple[float, ...], ...]
    needs_new_action: bool
    old_call: int
    old_admission: int
    commands: int
    dispatches: int
    cached_observation: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class InterpolationWitness:
    before_reset: BufferWitness
    pre_boundary_admissions: tuple[int, ...]
    fresh_call: int
    fresh_admission: int
    consumed: tuple[float, float]
    expected: tuple[float, float]
    returned: bool
    handled: bool
    accepted_diagnostic: bool
    queued: int
    dispatches: int
    commands: int
    permissions: int
    cached_observation_cleared: bool
    ordering: tuple[tuple[int, str], ...]


class CopyGate:
    """One selected frame call, outside engine locks; all other calls delegate unchanged."""

    def __init__(self, features: dict, events: list[dict]):
        self.features = features
        self.events = events
        self.identity = "copied-observation-1"
        self.entered, self.released = Event(), Event()
        self.lock = Lock()
        self.claimed = self.closing = False
        self.observation: tuple[tuple[str, float], ...] = ()
        self.original = rtc.build_dataset_frame
        rtc.build_dataset_frame = self

    def __call__(self, *args, **kwargs):
        with self.lock:
            selected = args[0] is self.features and not self.claimed and not self.closing
            if selected:
                self.claimed = True
                self.observation = tuple(args[1].items())
                self.events.append(
                    {"event": "copy_gate_entered", "gate": self.identity, "observation": self.observation}
                )
                self.entered.set()
        # In the pinned CHUNK loop, observation/task locks were released before this call.
        if selected and not self.released.wait(2 * WAIT_SECONDS):
            raise RuntimeError("SETUP: copied-observation gate watchdog")
        return self.original(*args, **kwargs)

    def release(self) -> None:
        self.events.append({"event": "copy_gate_released", "gate": self.identity})
        self.released.set()

    def close(self) -> None:
        with self.lock:
            self.closing = True
            self.released.set()

    def restore(self) -> None:
        # The rig calls this only after its retained worker has terminated.
        rtc.build_dataset_frame = self.original


def vector(items: tuple[tuple[str, float], ...]) -> tuple[float, float]:
    assert len(items) == len(JOINTS) and set(dict(items)) == set(JOINTS), (
        f"PAYLOAD_INTEGRITY: expected exactly {JOINTS}, got {items}"
    )
    values = tuple(dict(items)[key] for key in JOINTS)
    assert all(math.isfinite(value) for value in values), f"PAYLOAD_INTEGRITY: nonfinite {items}"
    return values


def consumed_input(call: Call) -> tuple[float, float]:
    assert tuple(call.consumed.shape) == (1, 2), "OBSERVATION_FRESHNESS: invalid consumed shape"
    values = tuple(call.consumed[0].tolist())
    assert all(math.isfinite(value) for value in values), "OBSERVATION_FRESHNESS: nonfinite input"
    return values


def check_trace(
    commands: list[Command],
    calls: list[Call],
    *,
    boundary: int,
    fresh_input: tuple[float, float],
    require_recovery: bool,
    policy_vectors: dict | None = None,
) -> None:
    """Check independent driver/input records on the control thread, never runtime tags."""
    issued = {consumed_input(call) for call in calls if call.returned.is_set()}
    previous_applied = None
    recovered = False
    declarations = ACTIONS if policy_vectors is None else policy_vectors
    for index, command in enumerate(commands):
        requested, applied = vector(command.requested), vector(command.applied)
        if command.kind == "hold":
            expected = previous_applied if previous_applied is not None else vector(command.measured)
            assert requested == expected, f"HOLD_TARGET: requested {requested}, expected applied {expected}"
        else:
            assert command.kind == "policy", f"PAYLOAD_INTEGRITY: unknown operation {command.kind}"
            origins = {observed for observed in issued if requested in declarations.get(observed, ())}
            assert origins, f"PAYLOAD_INTEGRITY: unissued or corrupt policy vector {requested}"
            if index >= boundary:
                assert origins == {fresh_input}, f"NO_STALE_POLICY: {requested} came from {origins}"
                recovered = True
        previous_applied = applied
    if require_recovery:
        assert recovered, "RECOVERY: no valid fresh policy command (holds do not qualify)"


def check_interpolation_trace(
    commands: list[Command],
    calls: list[Call],
    *,
    boundary: int,
    fresh_input: tuple[float, float],
    require_recovery: bool,
) -> None:
    """Attribute complete declared interpolation vectors before checking exact restart."""
    check_trace(
        commands,
        calls,
        boundary=boundary,
        fresh_input=fresh_input,
        require_recovery=require_recovery,
        policy_vectors=INTERPOLATED_ACTIONS,
    )
    if require_recovery:
        first = commands[boundary]
        assert first.kind == "policy" and vector(first.requested) == ACTIONS[fresh_input][0], (
            "FIRST_RESUMED_ENDPOINT: first resumed command must be the fresh first endpoint"
        )


class GatedPolicy(ControlledPolicy):
    """One release and result-handled event per actual serial invocation."""

    def __init__(self, events: list[dict]):
        super().__init__()
        self.config.chunk_size = self.config.n_action_steps = 6
        self.entries: list[Call] = []
        self.condition = Condition()
        self.closing = False
        self.last_returned: Call | None = None
        self.events = events
        self.admission_id: int | None = None

    def predict_action_chunk(self, batch, **kwargs):
        consumed = batch["observation.state"].detach().cpu().clone()
        with self.condition:
            call = Call(len(self.entries) + 1, consumed, admission_id=self.admission_id)
            self.entries.append(call)
            self.events.append({"event": "policy_entered", "call": call.number, "input": consumed.tolist()})
            if self.closing:
                call.released.set()
            self.condition.notify_all()
        if not call.released.wait(2 * WAIT_SECONDS):
            raise RuntimeError(f"SETUP: prediction gate watchdog for call {call.number}")
        # Stimuli come from a fixed table; the oracle never reads this returned tensor.
        result = torch.tensor([ACTIONS[tuple(consumed[0].tolist())]], dtype=torch.float32)
        self.last_returned = call
        call.returned.set()
        return result

    def close_gates(self) -> None:
        with self.condition:
            self.closing = True
            for call in self.entries:
                call.released.set()


class RecordingRobot:
    """Fake driver with independent immutable requested/applied/capture records."""

    robot_type = "mock"
    name = "reset-recorder"
    supports_position_hold = True
    action_features = dict.fromkeys(JOINTS, float)
    observation_features = dict.fromkeys(JOINTS, float)

    def __init__(self, *, clipped: bool = False):
        self.is_connected = True
        self.clipped = clipped
        self.operation = local()
        self.pose = OLD_INPUT
        self.captures: list[tuple[tuple[str, float], ...]] = []
        self.commands: list[Command] = []

    def get_observation(self) -> dict:
        observation = dict(zip(JOINTS, self.pose, strict=True))
        self.captures.append(tuple(observation.items()))
        return observation

    def send_action(self, action: dict) -> dict:
        applied = action.copy()
        if self.clipped:
            limits = {"j1.pos": 1000.0, "j2.pos": 2000.0}
            applied = {key: min(value, limits[key]) for key, value in action.items()}
        # No wrapper calls here: its non-reentrant I/O lock is held by the caller.
        self.commands.append(
            Command(
                getattr(self.operation, "kind", "policy"),
                tuple(action.items()),
                tuple(applied.items()),
                self.captures[-1],
            )
        )
        return applied

    def disconnect(self) -> None:
        self.is_connected = False


class RecordingWrapper(ThreadSafeRobot):
    def hold(self) -> None:
        previous = getattr(self.inner.operation, "kind", "policy")
        self.inner.operation.kind = "hold"
        try:
            super().hold()
        finally:
            self.inner.operation.kind = previous


class ResetDispatchRig:
    """Real engine, queue, interpolation and dispatch with independently recorded inputs."""

    def __init__(self, monkeypatch):
        self.events: list[dict] = []
        self.policy = GatedPolicy(self.events)
        self.driver = RecordingRobot()
        self.wrapper = RecordingWrapper(self.driver)
        self.engine = make_engine(
            policy=self.policy,
            robot=self.wrapper,
            rtc_config=RTCConfig(enabled=False),
            rtc_queue_threshold=2,
            max_observation_age_s=60,
            action_timeout_s=60,
            startup_timeout_s=60,
            action_starvation_grace_s=60,
        )
        self.interpolator = ActionInterpolator(multiplier=1)
        self.context = SimpleNamespace(
            policy=SimpleNamespace(inference=self.engine),
            hardware=SimpleNamespace(robot_wrapper=self.wrapper),
            data=SimpleNamespace(dataset_features=self.engine._obs_features, ordered_action_keys=JOINTS),
            processors=SimpleNamespace(robot_action_processor=lambda pair: pair[0]),
        )
        self.observation: dict = {}
        self.worker = None
        self.cleanup: dict = {}
        self.copy_gate: CopyGate | None = None
        self.freshness_witness: FreshnessWitness | None = None
        self.buffer_witness: BufferWitness | None = None
        self.interpolation_witness: InterpolationWitness | None = None
        self.strategy: BaseStrategy | None = None
        self.dispatch_count = 0
        self.permissions: list[dict] = []
        self.monkeypatch = monkeypatch
        self.admissions: list[dict] = []
        selected_begin = self.engine._runtime.begin

        def observed_begin(observation):
            request = selected_begin(observation)
            if request is not None:
                admission = {
                    "id": len(self.admissions) + 1,
                    "input": observation.features["observation.state"].tolist(),
                    "request_id_diagnostic": request.request_id,
                }
                self.admissions.append(admission)
                self.policy.admission_id = admission["id"]
                self.events.append({"event": "request_admitted", **admission})
            return request

        # Caller holds _obs_lock: delegate and record only, never wait or re-enter engine APIs.
        monkeypatch.setattr(self.engine._runtime, "begin", observed_begin)
        # Any other wrapper around accept must already be installed before this observer.
        selected_accept = self.engine._runtime.accept

        def observed_accept(*args, **kwargs):
            accepted = selected_accept(*args, **kwargs)
            call = self.policy.last_returned
            call.accepted = accepted  # Diagnostics only; never the correctness oracle.
            self.events.append({"event": "result_handled", "call": call.number, "accepted": accepted})
            call.handled.set()
            return accepted

        monkeypatch.setattr(self.engine._runtime, "accept", observed_accept)

    def configure_interpolation_owner(self) -> BaseStrategy:
        assert self.worker is None, "SETUP: configure owner before worker start"
        self.interpolator = ActionInterpolator(multiplier=4)
        self.strategy = BaseStrategy(BaseStrategyConfig())
        self.strategy._engine = self.engine
        self.strategy._interpolator = self.interpolator
        selected_permission = self.engine.dispatch_allowed

        def observed_permission():
            allowed = selected_permission()
            self.permissions.append({"tick": self.dispatch_count, "allowed": allowed})
            return allowed

        self.monkeypatch.setattr(self.engine, "dispatch_allowed", observed_permission)
        return self.strategy

    def arm_copy_gate(self) -> CopyGate:
        assert self.worker is None and self.copy_gate is None, "SETUP: copy gate must arm before start"
        self.copy_gate = CopyGate(self.engine._obs_features, self.events)
        return self.copy_gate

    def start(self) -> None:
        self.engine.start()
        self.worker = self.engine._rtc_thread  # Retain the object, even after stop clears the engine field.
        self.engine.resume()

    def capture(self, values: tuple[float, float]) -> tuple[float, float]:
        self.driver.pose = values
        self.observation = self.wrapper.get_observation()
        captured = vector(self.driver.captures[-1])
        self.engine.notify_observation(self.observation)
        self.events.append({"event": "capture_published", "input": captured})
        return captured

    def next_call(self, after: Call | None = None) -> Call:
        index = after.number if after is not None else 0
        with self.policy.condition:
            entered = self.policy.condition.wait_for(lambda: len(self.policy.entries) > index, WAIT_SECONDS)
        self.assert_healthy()
        assert entered, "SETUP: no next policy invocation acknowledged"
        return self.policy.entries[index]

    def release(self, call: Call) -> None:
        self.events.append({"event": "release", "call": call.number})
        call.released.set()

    def wait_handled(self, call: Call) -> None:
        handled = call.handled.wait(WAIT_SECONDS)
        self.assert_healthy()
        assert handled, f"SETUP: result handling not acknowledged for call {call.number}"

    def dispatch(self) -> None:
        self.dispatch_count += 1
        send_next_action(self.observation, self.observation, self.context, self.interpolator)
        self.events.append({"event": "dispatch_finished", "command_count": len(self.driver.commands)})
        self.assert_healthy()

    def assert_healthy(self) -> None:
        assert not self.engine.failed, f"ENGINE_FAILURE: {self.engine.failure_traceback}"

    def assert_trace(
        self, boundary: int, fresh_input: tuple[float, float], *, require_recovery: bool
    ) -> None:
        with self.policy.condition:
            calls = list(self.policy.entries)
        check_trace(
            list(self.driver.commands),
            calls,
            boundary=boundary,
            fresh_input=fresh_input,
            require_recovery=require_recovery,
        )
        self.assert_healthy()

    def close(self) -> None:
        if self.copy_gate is not None:
            self.copy_gate.close()
        self.policy.close_gates()  # Also bypass any successor gate entered during shutdown.
        try:
            self.engine.stop()
        finally:
            self.driver.disconnect()
            self.cleanup = {
                "worker_alive": self.worker is not None and self.worker.is_alive(),
                "disconnected": not self.driver.is_connected,
                "engine_failure": self.engine.failure_traceback,
            }
            if self.copy_gate is not None:
                self.cleanup["copy_gate_restored"] = False
                if not self.cleanup["worker_alive"]:
                    self.copy_gate.restore()
                    self.cleanup["copy_gate_restored"] = rtc.build_dataset_frame is self.copy_gate.original
        assert not self.cleanup["worker_alive"], "WORKER_CLEANUP: retained RTC worker is alive"
        self.assert_healthy()
