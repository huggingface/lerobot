"""Toy, notebook-friendly LWW (last-writer-wins) schedule join.

This is a single-file playground that mirrors the *semilattice / join* story used by
DRTC async inference:

- Each value is tagged with an integer logical clock (here: `action_step`).
- Join (⊔) keeps the value with the larger clock (stable tie-breaking keeps the left).
- A schedule is a map: execution_step -> LWWRegister[action], where the register clock
  is the *provenance* (`src_action_step`) of the action chunk that produced that slot.

Everything is intentionally lock-free for interactive experimentation.
This toy uses explicit `join(...)` methods (no operator overloading).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Generic, Iterable, Iterator, Mapping, Optional, Tuple, TypeVar

T = TypeVar("T")
A = TypeVar("A")


@dataclass(frozen=True)
class LWWState(Generic[T]):
    """Last-write-wins state element tagged by a logical clock."""

    action_step: int
    value: T

    def __or__(self, other: "LWWState[T]") -> "LWWState[T]":
        """Join (⊔): keep the state with the larger action_step.

        Stable tie-breaking: if action_step is equal, keep `self`.
        """

        if other.action_step > self.action_step:
            return other
        return self


class LWWRegister(Generic[T]):
    """Lock-free toy LWW register (single-slot).

    This is *not* thread-safe. It is meant for notebook experiments where you want:
    - `reg.update_if_newer(step, value)` monotone updates
    - `reg.join(other)` to compute the LWW join result
    """

    __slots__ = ("_state",)

    def __init__(self, *, initial_action_step: int, initial_value: T):
        self._state: LWWState[T] = LWWState(action_step=initial_action_step, value=initial_value)

    def read(self) -> LWWState[T]:
        return self._state

    @property
    def state(self) -> LWWState[T]:
        return self._state

    def update_if_newer(self, action_step: int, value: T) -> bool:
        incoming = LWWState(action_step=action_step, value=value)
        new_state = self._state.join(incoming)
        did_update = new_state is not self._state
        self._state = new_state
        return did_update

    def join(self, other: "LWWRegister[T]") -> "LWWRegister[T]":
        joined = self._state.join(other._state)
        return LWWRegister(initial_action_step=joined.action_step, initial_value=joined.value)

    def __repr__(self) -> str:
        return f"LWWRegister(step={self._state.action_step}, value={self._state.value!r})"


class RegisterChain3(Generic[T]):
    """A 3-stage chain of LWW registers.

    Two patterns are useful to experiment with:
    - Pipeline propagation (handoff): r2 = r2.join(r1); r3 = r3.join(r2)
    - Replica convergence: repeated joins between independent replicas
    """

    def __init__(self, *, initial_action_step: int, initial_value: T):
        self.r1 = LWWRegister(initial_action_step=initial_action_step, initial_value=initial_value)
        self.r2 = LWWRegister(initial_action_step=initial_action_step, initial_value=initial_value)
        self.r3 = LWWRegister(initial_action_step=initial_action_step, initial_value=initial_value)

    def publish(self, action_step: int, value: T) -> bool:
        return self.r1.update_if_newer(action_step, value)

    def propagate(self) -> None:
        self.r2 = self.r2.join(self.r1)
        self.r3 = self.r3.join(self.r2)

    def snapshot(self) -> Tuple[LWWState[T], LWWState[T], LWWState[T]]:
        return self.r1.read(), self.r2.read(), self.r3.read()


class LWWSchedule(Generic[A]):
    """Map-CRDT where each key maps to an LWW register.

    - key: execution_step (when the action is executed)
    - per-key register clock: src_action_step (when the source observation was captured)
    """

    def __init__(self, slots: Optional[Mapping[int, LWWRegister[A]]] = None):
        self.slots: Dict[int, LWWRegister[A]] = dict(slots) if slots is not None else {}

    def put(self, execution_step: int, *, src_action_step: int, action: A) -> bool:
        """Insert/update a slot using freshest-observation-wins (by src_action_step)."""

        reg = self.slots.get(execution_step)
        if reg is None:
            reg = LWWRegister(initial_action_step=-1, initial_value=action)
            self.slots[execution_step] = reg
        return reg.update_if_newer(src_action_step, action)

    def get(self, execution_step: int) -> Optional[LWWState[A]]:
        reg = self.slots.get(execution_step)
        return None if reg is None else reg.read()

    def items(self) -> Iterator[Tuple[int, LWWState[A]]]:
        for step, reg in self.slots.items():
            yield step, reg.read()

    def restrict_active(self, current_action_step: int) -> "LWWSchedule[A]":
        """Return schedule restricted to the active region: {i : i > current_action_step}."""

        return LWWSchedule({k: v for k, v in self.slots.items() if k > current_action_step})

    def join(self, other: "LWWSchedule[A]") -> "LWWSchedule[A]":
        out: Dict[int, LWWRegister[A]] = {}
        keys = set(self.slots.keys()) | set(other.slots.keys())
        for k in keys:
            left = self.slots.get(k)
            right = other.slots.get(k)
            if left is None:
                if right is None:
                    raise RuntimeError("unreachable: key in union but missing on both sides")
                out[k] = right
            elif right is None:
                out[k] = left
            else:
                out[k] = left.join(right)
        return LWWSchedule(out)

    def __repr__(self) -> str:
        rows = ", ".join(
            f"{k}: (src={v.action_step}, action={v.value!r})"
            for k, v in sorted(((k, reg.read()) for k, reg in self.slots.items()), key=lambda x: x[0])
        )
        return f"LWWSchedule({{{rows}}})"


def chunk_as_schedule(chunk: Iterable[Tuple[int, A]], *, src_action_step: int) -> LWWSchedule[A]:
    """Convert a list of (execution_step, action) pairs into an LWWSchedule."""

    s = LWWSchedule[A]()
    for execution_step, action in chunk:
        s.put(execution_step, src_action_step=src_action_step, action=action)
    return s


def merge_active(
    schedule: LWWSchedule[A],
    incoming: LWWSchedule[A],
    *,
    current_action_step: int,
) -> LWWSchedule[A]:
    """Merge as pointwise join on the active region (i > current_action_step)."""

    return schedule.restrict_active(current_action_step).join(
        incoming.restrict_active(current_action_step)
    )


# ---------------------------------------------------------------------------
# Notebook demos
# ---------------------------------------------------------------------------


def demo_pipeline_chain3() -> Dict[str, object]:
    """Demo: monotone handoff through a 3-stage pipeline."""

    chain = RegisterChain3[str](initial_action_step=-1, initial_value="⊥")
    chain.publish(1, "obs@1")
    before = chain.snapshot()
    chain.propagate()
    after1 = chain.snapshot()
    chain.publish(2, "obs@2")
    chain.propagate()
    after2 = chain.snapshot()
    return {"chain": chain, "before": before, "after1": after1, "after2": after2}


def demo_three_replicas_converge() -> Dict[str, object]:
    """Demo: 3 replicas converge under repeated joins (CRDT-style)."""

    a = LWWRegister[str](initial_action_step=-1, initial_value="⊥")
    b = LWWRegister[str](initial_action_step=-1, initial_value="⊥")
    c = LWWRegister[str](initial_action_step=-1, initial_value="⊥")

    # Different delivery orders / partial knowledge
    a.update_if_newer(1, "v1")
    b.update_if_newer(2, "v2")
    c.update_if_newer(3, "v3")

    # Gossip / join rounds (no in-place join)
    a = a.join(b)
    b = b.join(c)
    c = c.join(a)
    a = a.join(c)
    b = b.join(a)
    c = c.join(b)

    return {
        "a": a,
        "b": b,
        "c": c,
        "final_states": (a.read(), b.read(), c.read()),
    }


def demo_schedule_join_and_consumption() -> Dict[str, object]:
    """Demo: schedule join + active-region restriction models 'no past modification'."""

    current_action_step = 5

    # Existing schedule with some past and some future actions.
    s0 = LWWSchedule[str]()
    s0.put(4, src_action_step=10, action="past_should_not_matter")
    s0.put(6, src_action_step=1, action="old_future")

    # Incoming chunk tries to overwrite both a past slot and a future slot.
    incoming = chunk_as_schedule([(4, "overwrite_past"), (6, "overwrite_future")], src_action_step=2)

    merged = merge_active(s0, incoming, current_action_step=current_action_step)
    # After restricting to active region, key 4 is absent; key 6 should take src=2.

    return {
        "current_action_step": current_action_step,
        "s0": s0,
        "incoming": incoming,
        "merged_active_only": merged,
        "merged_step6": merged.get(6),
        "merged_step4": merged.get(4),
    }


__all__ = [
    "A",
    "LWWRegister",
    "LWWState",
    "LWWSchedule",
    "RegisterChain3",
    "T",
    "chunk_as_schedule",
    "demo_pipeline_chain3",
    "demo_schedule_join_and_consumption",
    "demo_three_replicas_converge",
    "merge_active",
]

