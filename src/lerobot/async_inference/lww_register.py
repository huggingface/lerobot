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

    def read(self) -> LWWState[T]:
        with self._lock:
            return self._state

    def update(self, k: int, value: T) -> LWWState[T]:
        incoming = LWWState(k=k, value=value)
        with self._lock:
            self._state = self._state | incoming
            return self._state

