from queue import Queue
from types import SimpleNamespace

from lerobot.teleoperators.keyboard import teleop_keyboard
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardEndEffectorTeleop
from lerobot.teleoperators.utils import TeleopEvents


class FakeListener:
    def is_alive(self):
        return True


class FakeKey:
    up = object()
    down = object()
    left = object()
    right = object()
    shift = object()
    shift_l = object()
    shift_r = object()
    ctrl = object()
    ctrl_l = object()
    ctrl_r = object()
    esc = object()


class FakeKeyboard:
    Listener = FakeListener
    Key = FakeKey


def make_keyboard_ee(monkeypatch):
    monkeypatch.setattr(teleop_keyboard, "PYNPUT_AVAILABLE", True)
    monkeypatch.setattr(teleop_keyboard, "keyboard", FakeKeyboard)

    teleop = KeyboardEndEffectorTeleop.__new__(KeyboardEndEffectorTeleop)
    teleop.config = SimpleNamespace(use_gripper=True)
    teleop.event_queue = Queue()
    teleop.current_pressed = {}
    teleop.listener = FakeListener()
    teleop.logs = {}
    teleop.misc_keys_queue = Queue()
    return teleop


def test_special_keys_with_no_char_are_preserved(monkeypatch):
    teleop = make_keyboard_ee(monkeypatch)

    class SpecialKey:
        char = None

    special_key = SpecialKey()
    teleop._on_press(special_key)

    key, is_pressed = teleop.event_queue.get_nowait()

    assert key is special_key
    assert is_pressed is True


def test_releasing_modifier_keys_clears_left_right_aliases(monkeypatch):
    teleop = make_keyboard_ee(monkeypatch)

    teleop._on_release(FakeKey.ctrl)
    ctrl_releases = {teleop.event_queue.get_nowait() for _ in range(3)}

    assert ctrl_releases == {
        (FakeKey.ctrl, False),
        (FakeKey.ctrl_l, False),
        (FakeKey.ctrl_r, False),
    }

    teleop._on_release(FakeKey.shift)
    shift_releases = {teleop.event_queue.get_nowait() for _ in range(3)}

    assert shift_releases == {
        (FakeKey.shift, False),
        (FakeKey.shift_l, False),
        (FakeKey.shift_r, False),
    }


def test_released_gripper_keys_keep_neutral_action(monkeypatch):
    teleop = make_keyboard_ee(monkeypatch)
    teleop.current_pressed = {
        FakeKey.ctrl_l: False,
        FakeKey.ctrl_r: False,
    }

    action = teleop.get_action()

    assert action["gripper"] == 1.0


def test_teleop_events_keep_held_movement_keys(monkeypatch):
    teleop = make_keyboard_ee(monkeypatch)
    teleop.current_pressed = {FakeKey.up: True}

    events = teleop.get_teleop_events()
    action = teleop.get_action()

    assert events[TeleopEvents.IS_INTERVENTION] is True
    assert teleop.current_pressed[FakeKey.up] is True
    assert action["delta_y"] == -1
