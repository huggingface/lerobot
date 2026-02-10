"""Thread-safe last-write-wins register keyed by control step.

The control step t is the monotone logical clock in DRTC: it increments
every tick of the robot control loop.  This register implements a
monotone join:

    state := state ⊔ incoming

where ⊔ keeps the state with the larger control step.

The system uses two clocks:
- control_step (t): monotone per control-loop tick; used for LWW / watermarks.
- action_step (j): execution index; incremented when an action executes.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class LWWState(Generic[T]):
    """Last-write-wins state element."""

    control_step: int
    value: T

    def __or__(self, other: "LWWState[T]") -> "LWWState[T]":
        """Join (⊔): keep the state with the larger control_step.

        Tie-breaking is intentionally stable: if control_step is equal, keep `self`.
        """

        if other.control_step > self.control_step:
            return other
        return self


@dataclass(frozen=True)
class LWWCursor:
    """Monotone consumer cursor (watermark) for read-once semantics."""

    watermark: int

    def __or__(self, other: "LWWCursor") -> "LWWCursor":
        return self if self.watermark >= other.watermark else other


class LWWReader(Generic[T]):
    """Per-consumer read-once view of an `LWWRegister`.

    The cursor (watermark) is stored inside the reader, so call sites don't need
    to carry `_last_*` or explicit cursor arguments.
    """

    def __init__(self, register: "LWWRegister[T]", *, initial_watermark: int):
        self._register = register
        self._cursor = LWWCursor(watermark=initial_watermark)

    @property
    def cursor(self) -> LWWCursor:
        return self._cursor

    def read_if_newer(self) -> tuple[LWWState[T], LWWCursor, bool]:
        state = self._register.read()
        is_new = state.control_step > self._cursor.watermark
        if is_new:
            self._cursor = self._cursor | LWWCursor(watermark=state.control_step)
        return state, self._cursor, is_new


class LWWRegister(Generic[T]):
    """A thread-safe LWW register holding a single `LWWState`.

    Notes:
    - This register has no "consume" semantics. Consumers must track a watermark
      (via LWWReader) to avoid re-processing the same state repeatedly.
    - Updates are monotone w.r.t. control_step: stale (or equal) updates cannot overwrite.
    """

    def __init__(self, *, initial_control_step: int, initial_value: T):
        self._lock = threading.Lock()
        self._state: LWWState[T] = LWWState(control_step=initial_control_step, value=initial_value)

    def reader(self, *, initial_watermark: int = -1) -> LWWReader[T]:
        """Create a per-consumer reader with an internal monotone cursor."""

        return LWWReader(self, initial_watermark=initial_watermark)

    def read(self) -> LWWState[T]:
        with self._lock:
            return self._state

    def update(self, control_step: int, value: T) -> LWWState[T]:
        state, _ = self.update_if_newer(control_step, value)
        return state

    def update_if_newer(self, control_step: int, value: T) -> tuple[LWWState[T], bool]:
        """Update the register iff the incoming control_step is strictly newer.

        Returns:
            (state, did_update)
        """

        incoming = LWWState(control_step=control_step, value=value)
        with self._lock:
            prev = self._state
            new = prev | incoming
            did_update = new is not prev
            self._state = new
            return new, did_update
