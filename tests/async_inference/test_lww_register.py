"""Tests for the LWW register primitive.

The LWW (Last-Write-Wins) register uses a single logical clock (action step)
for all causality relationships. See robot_client_drtc.py for the full
causality model documentation.
"""

import pytest
from hypothesis import assume, given, settings, strategies as st

from lerobot.async_inference.lww_register import LWWCursor, LWWRegister, LWWState

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

control_steps = st.integers(min_value=0, max_value=1000)
watermarks = st.integers(min_value=0, max_value=1000)
values = st.one_of(st.integers(), st.text(max_size=10), st.none())

lww_states = st.builds(LWWState, control_step=control_steps, value=values)
lww_cursors = st.builds(LWWCursor, watermark=watermarks)


def test_lww_state_idempotent() -> None:
    s = LWWState(action_step=3, value="x")
    assert (s | s) == s


def test_lww_state_commutative_distinct_action_step() -> None:
    a = LWWState(action_step=1, value="a")
    b = LWWState(action_step=2, value="b")
    assert (a | b) == (b | a)
    assert (a | b).action_step == 2


def test_lww_state_associative_distinct_action_step() -> None:
    a = LWWState(action_step=1, value="a")
    b = LWWState(action_step=2, value="b")
    c = LWWState(action_step=3, value="c")
    assert ((a | b) | c) == (a | (b | c))
    assert (a | (b | c)).action_step == 3


def test_lww_state_equal_action_step_stability_when_values_equal() -> None:
    # Our join is stable on ties: equal-action_step should not introduce changes.
    a1 = LWWState(action_step=7, value={"x": 1})
    a2 = LWWState(action_step=7, value={"x": 1})
    assert (a1 | a2) == a1
    assert (a2 | a1) == a2


def test_register_starts_at_initial_state() -> None:
    reg: LWWRegister[str | None] = LWWRegister(initial_action_step=-1, initial_value=None)
    s = reg.read()
    assert s.action_step == -1
    assert s.value is None


def test_register_monotone_ignores_stale_and_equal_action_step_updates() -> None:
    reg: LWWRegister[str | None] = LWWRegister(initial_action_step=-1, initial_value=None)

    reg.update(1, "v1")
    assert reg.read() == LWWState(action_step=1, value="v1")

    # stale
    reg.update(0, "stale")
    assert reg.read() == LWWState(action_step=1, value="v1")

    # equal-action_step (should not overwrite)
    reg.update(1, "v1-duplicate")
    assert reg.read() == LWWState(action_step=1, value="v1")


def test_register_out_of_order_tolerates_gaps() -> None:
    reg: LWWRegister[str] = LWWRegister(initial_action_step=-1, initial_value="")
    reg.update(10, "ten")
    reg.update(5, "five")  # out-of-order
    assert reg.read() == LWWState(action_step=10, value="ten")


def test_register_action_step_never_decreases_under_updates() -> None:
    reg: LWWRegister[int] = LWWRegister(initial_action_step=-1, initial_value=0)
    steps = [3, 1, 5, 5, 2, 9, 4]
    last_step = reg.read().action_step
    for step in steps:
        reg.update(step, step)
        new_step = reg.read().action_step
        assert new_step >= last_step
        last_step = new_step


def test_register_read_returns_latest_state() -> None:
    reg: LWWRegister[str | None] = LWWRegister(initial_action_step=-1, initial_value=None)
    assert reg.read().value is None
    reg.update(0, "zero")
    reg.update(2, "two")
    assert reg.read() == LWWState(action_step=2, value="two")


def test_cursor_is_monotone_semilattice() -> None:
    c1 = LWWCursor(watermark=1)
    c2 = LWWCursor(watermark=2)
    c3 = LWWCursor(watermark=3)
    assert (c1 | c1) == c1
    assert (c1 | c2) == (c2 | c1) == c2
    assert ((c1 | c2) | c3) == (c1 | (c2 | c3)) == c3


def test_read_if_newer_returns_is_new_once() -> None:
    reg: LWWRegister[str | None] = LWWRegister(initial_action_step=-1, initial_value=None)
    reader = reg.reader()

    # Nothing newer than cursor (action_step=-1)
    state, cursor2, is_new = reader.read_if_newer()
    assert is_new is False
    assert cursor2 == LWWCursor(watermark=-1)
    assert state.action_step == -1

    # Update to action_step=0; first read should be new
    reg.update_if_newer(0, "zero")
    state, cursor, is_new = reader.read_if_newer()
    assert is_new is True
    assert cursor.watermark == 0
    assert state == LWWState(action_step=0, value="zero")

    # Second read without update should not be new
    state2, cursor2, is_new2 = reader.read_if_newer()
    assert is_new2 is False
    assert cursor2 == cursor
    assert state2 == state


def test_update_if_newer_reports_rejection_on_stale_or_equal_action_step() -> None:
    reg: LWWRegister[str] = LWWRegister(initial_action_step=-1, initial_value="")
    _, did_update1 = reg.update_if_newer(1, "one")
    assert did_update1 is True
    _, did_update_stale = reg.update_if_newer(0, "stale")
    assert did_update_stale is False
    _, did_update_equal = reg.update_if_newer(1, "equal")
    assert did_update_equal is False


# ---------------------------------------------------------------------------
# Property-based tests (Hypothesis)
# ---------------------------------------------------------------------------


# -- LWWState semilattice laws --


@given(s=lww_states)
def test_prop_lww_state_idempotent(s: LWWState) -> None:
    assert (s | s) == s


@given(a=lww_states, b=lww_states)
def test_prop_lww_state_commutative_on_distinct_control_step(
    a: LWWState, b: LWWState
) -> None:
    assume(a.control_step != b.control_step)
    assert (a | b) == (b | a)


@given(a=lww_states, b=lww_states, c=lww_states)
def test_prop_lww_state_associative(a: LWWState, b: LWWState, c: LWWState) -> None:
    assert ((a | b) | c) == (a | (b | c))


@given(a=lww_states, b=lww_states)
def test_prop_lww_state_join_is_monotone(a: LWWState, b: LWWState) -> None:
    joined = a | b
    assert joined.control_step >= a.control_step
    assert joined.control_step >= b.control_step


# -- LWWCursor semilattice laws --


@given(c=lww_cursors)
def test_prop_lww_cursor_idempotent(c: LWWCursor) -> None:
    assert (c | c) == c


@given(a=lww_cursors, b=lww_cursors)
def test_prop_lww_cursor_commutative(a: LWWCursor, b: LWWCursor) -> None:
    assert (a | b) == (b | a)


@given(a=lww_cursors, b=lww_cursors, c=lww_cursors)
def test_prop_lww_cursor_associative(a: LWWCursor, b: LWWCursor, c: LWWCursor) -> None:
    assert ((a | b) | c) == (a | (b | c))


# -- LWWRegister monotonicity --


@given(updates=st.lists(st.tuples(control_steps, values), min_size=1, max_size=50))
def test_prop_register_control_step_never_decreases(updates: list) -> None:
    reg: LWWRegister = LWWRegister(initial_control_step=-1, initial_value=None)
    prev_step = reg.read().control_step
    for step, val in updates:
        reg.update(step, val)
        cur_step = reg.read().control_step
        assert cur_step >= prev_step
        prev_step = cur_step


# -- LWWRegister.update_if_newer consistency --


@given(updates=st.lists(st.tuples(control_steps, values), min_size=1, max_size=50))
def test_prop_update_if_newer_reports_correctly(updates: list) -> None:
    reg: LWWRegister = LWWRegister(initial_control_step=-1, initial_value=None)
    for step, val in updates:
        prev_step = reg.read().control_step
        _, did_update = reg.update_if_newer(step, val)
        if step > prev_step:
            assert did_update is True
        else:
            assert did_update is False


# -- LWWReader cursor monotonicity & is_new correctness --


@given(updates=st.lists(st.tuples(control_steps, values), min_size=1, max_size=50))
def test_prop_reader_cursor_monotone_and_is_new_correct(updates: list) -> None:
    reg: LWWRegister = LWWRegister(initial_control_step=-1, initial_value=None)
    reader = reg.reader(initial_watermark=-1)

    prev_watermark = reader.cursor.watermark
    for step, val in updates:
        reg.update(step, val)
        state, cursor, is_new = reader.read_if_newer()
        assert cursor.watermark >= prev_watermark
        if is_new:
            assert state.control_step > prev_watermark
        prev_watermark = cursor.watermark
