import pytest

from lerobot.async_inference.lww_register import LWWRegister, LWWState


def test_lww_state_idempotent() -> None:
    s = LWWState(k=3, value="x")
    assert (s | s) == s


def test_lww_state_commutative_distinct_k() -> None:
    a = LWWState(k=1, value="a")
    b = LWWState(k=2, value="b")
    assert (a | b) == (b | a)
    assert (a | b).k == 2


def test_lww_state_associative_distinct_k() -> None:
    a = LWWState(k=1, value="a")
    b = LWWState(k=2, value="b")
    c = LWWState(k=3, value="c")
    assert ((a | b) | c) == (a | (b | c))
    assert (a | (b | c)).k == 3


def test_lww_state_equal_k_stability_when_values_equal() -> None:
    # Our join is stable on ties: equal-k should not introduce changes.
    a1 = LWWState(k=7, value={"x": 1})
    a2 = LWWState(k=7, value={"x": 1})
    assert (a1 | a2) == a1
    assert (a2 | a1) == a2


def test_register_starts_at_initial_state() -> None:
    reg: LWWRegister[str | None] = LWWRegister(initial_k=-1, initial_value=None)
    s = reg.read()
    assert s.k == -1
    assert s.value is None


def test_register_monotone_ignores_stale_and_equal_k_updates() -> None:
    reg: LWWRegister[str | None] = LWWRegister(initial_k=-1, initial_value=None)

    reg.update(1, "v1")
    assert reg.read() == LWWState(k=1, value="v1")

    # stale
    reg.update(0, "stale")
    assert reg.read() == LWWState(k=1, value="v1")

    # equal-k (should not overwrite)
    reg.update(1, "v1-duplicate")
    assert reg.read() == LWWState(k=1, value="v1")


def test_register_out_of_order_tolerates_gaps() -> None:
    reg: LWWRegister[str] = LWWRegister(initial_k=-1, initial_value="")
    reg.update(10, "ten")
    reg.update(5, "five")  # out-of-order
    assert reg.read() == LWWState(k=10, value="ten")


def test_register_k_never_decreases_under_updates() -> None:
    reg: LWWRegister[int] = LWWRegister(initial_k=-1, initial_value=0)
    ks = [3, 1, 5, 5, 2, 9, 4]
    last_k = reg.read().k
    for k in ks:
        reg.update(k, k)
        new_k = reg.read().k
        assert new_k >= last_k
        last_k = new_k


def test_register_read_returns_latest_state() -> None:
    reg: LWWRegister[str | None] = LWWRegister(initial_k=-1, initial_value=None)
    assert reg.read().value is None
    reg.update(0, "zero")
    reg.update(2, "two")
    assert reg.read() == LWWState(k=2, value="two")

