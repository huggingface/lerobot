"""Thread-safe last-write-wins register keyed by an integer timestep.

This is a tiny primitive used to replace the "overwrite-if-full" one-slot Queue mailboxes
in async inference codepaths with a monotone join:

    state := state ⊔ incoming

where ⊔ keeps the state with the larger timestep (k).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class LWWState(Generic[T]):
    """Last-write-wins state element."""

    k: int
    value: T

    def __or__(self, other: "LWWState[T]") -> "LWWState[T]":
        """Join (⊔): keep the state with the larger k.

        Tie-breaking is intentionally stable: if k is equal, keep `self`.
        """

        if other.k > self.k:
            return other
        return self


@dataclass(frozen=True)
class LWWCursor:
    """Monotone consumer cursor (watermark) for read-once semantics."""

    w: int

    def __or__(self, other: "LWWCursor") -> "LWWCursor":
        return self if self.w >= other.w else other


class LWWReader(Generic[T]):
    """Per-consumer read-once view of an `LWWRegister`.

    The cursor (watermark) is stored inside the reader, so call sites don't need
    to carry `_last_*` or explicit cursor arguments.
    """

    def __init__(self, register: "LWWRegister[T]", *, initial_w: int):
        self._register = register
        self._cursor = LWWCursor(w=initial_w)

    @property
    def cursor(self) -> LWWCursor:
        return self._cursor

    def read_if_newer(self) -> tuple[LWWState[T], LWWCursor, bool]:
        state = self._register.read()
        is_new = state.k > self._cursor.w
        if is_new:
            self._cursor = self._cursor | LWWCursor(w=state.k)
        return state, self._cursor, is_new


class LWWRegister(Generic[T]):
    """A thread-safe LWW register holding a single `LWWState`.

    Notes:
    - This register has no "consume" semantics. Consumers must track a `last_seen_k` to
      avoid re-processing the same state repeatedly.
    - Updates are monotone w.r.t. k: stale (or equal-k) updates cannot overwrite.
    """

    def __init__(self, *, initial_k: int, initial_value: T):
        self._lock = threading.Lock()
        self._state: LWWState[T] = LWWState(k=initial_k, value=initial_value)

    def reader(self, *, initial_w: int = -1) -> LWWReader[T]:
        """Create a per-consumer reader with an internal monotone cursor."""

        return LWWReader(self, initial_w=initial_w)

    def read(self) -> LWWState[T]:
        with self._lock:
            return self._state

    def update(self, k: int, value: T) -> LWWState[T]:
        state, _ = self.update_if_newer(k, value)
        return state

    def update_if_newer(self, k: int, value: T) -> tuple[LWWState[T], bool]:
        """Update the register iff the incoming k is strictly newer.

        Returns:
            (state, did_update)
        """

        incoming = LWWState(k=k, value=value)
        with self._lock:
            prev = self._state
            new = prev | incoming
            did_update = new is not prev
            self._state = new
            return new, did_update

