#!/usr/bin/env python

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

"""Unit tests for the display-independent keyboard input helpers.

These cover the parts most likely to regress: the environment-detection decision
table (the heart of the Wayland/headless fix), the macOS trust probe, the control
mapping, the terminal escape-sequence parsing, and backend selection. They require
neither ``pynput`` nor a real terminal.
"""

import io
import platform
import sys

import pytest

import lerobot.utils.keyboard_input as ki
from lerobot.utils.keyboard_input import (
    TerminalKeyListener,
    apply_recording_control,
    create_key_listener,
    init_keyboard_listener,
    is_headless,
    is_wayland,
    pynput_can_capture,
    pynput_listener_is_trusted,
)


@pytest.fixture(autouse=True)
def _clear_detection_caches():
    """The detection helpers are ``@cache``-decorated; clear around each test."""
    for fn in (is_headless, is_wayland, pynput_can_capture):
        fn.cache_clear()
    yield
    for fn in (is_headless, is_wayland, pynput_can_capture):
        fn.cache_clear()


def _set_platform(monkeypatch, name):
    monkeypatch.setattr(platform, "system", lambda: name)


def _set_tty(monkeypatch, is_tty):
    stdin = io.StringIO("")
    stdin.isatty = lambda: is_tty
    monkeypatch.setattr(sys, "stdin", stdin)


# --- Environment detection (the core of the fix) ---------------------------
@pytest.mark.parametrize(
    ("system", "env", "expected"),
    [
        ("Linux", {}, True),  # no display server
        ("Linux", {"DISPLAY": ":0"}, False),  # X11
        ("Linux", {"WAYLAND_DISPLAY": "wayland-0"}, False),  # Wayland
        ("Darwin", {}, False),  # display always assumed present
    ],
)
def test_is_headless(monkeypatch, system, env, expected):
    _set_platform(monkeypatch, system)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert is_headless() is expected


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"XDG_SESSION_TYPE": "wayland"}, True),
        ({"WAYLAND_DISPLAY": "wayland-0"}, True),
        ({"XDG_SESSION_TYPE": "x11"}, False),
        ({}, False),
    ],
)
def test_is_wayland(monkeypatch, env, expected):
    monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert is_wayland() is expected


@pytest.mark.parametrize(
    ("system", "env", "pynput_available", "expected"),
    [
        ("Linux", {"DISPLAY": ":0"}, True, True),  # X11
        ("Linux", {"DISPLAY": ":0", "WAYLAND_DISPLAY": "wayland-0"}, True, False),  # Wayland
        ("Linux", {}, True, False),  # headless
        ("Darwin", {}, True, True),
        ("Linux", {"DISPLAY": ":0"}, False, False),  # pynput not installed
    ],
)
def test_pynput_can_capture(monkeypatch, system, env, pynput_available, expected):
    _set_platform(monkeypatch, system)
    monkeypatch.setattr(ki, "_pynput_available", pynput_available)
    for var in ("DISPLAY", "WAYLAND_DISPLAY", "XDG_SESSION_TYPE"):
        monkeypatch.delenv(var, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert pynput_can_capture() is expected


# --- macOS trust probe ------------------------------------------------------
class _FakeListener:
    def __init__(self, is_trusted):
        self.IS_TRUSTED = is_trusted


def test_pynput_listener_is_trusted(monkeypatch):
    _set_platform(monkeypatch, "Linux")
    assert pynput_listener_is_trusted(_FakeListener(False)) is True  # non-macOS: always assumed ok
    _set_platform(monkeypatch, "Darwin")
    assert pynput_listener_is_trusted(_FakeListener(False), timeout_s=0.05) is False


# --- Control mapping --------------------------------------------------------
def test_apply_recording_control():
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    apply_recording_control("left", events)
    assert events == {"exit_early": True, "rerecord_episode": True, "stop_recording": False}
    apply_recording_control("esc", events)
    assert events["stop_recording"] is True
    apply_recording_control("up", events)  # unknown control -> no-op (no error)


# --- Terminal escape-sequence parsing (the tricky bit) ----------------------
def _drive(listener, byte_seq):
    """Run the listener's read loop over a scripted list of bytes (no real terminal)."""
    script = list(byte_seq)

    def fake_read(timeout):
        if script:
            return script.pop(0)
        listener._running = False
        return None

    listener._read_char = fake_read
    listener._running = True
    listener._run()


@pytest.mark.parametrize(
    ("byte_seq", "expected"),
    [
        (["\x1b", "[", "C"], ["right"]),  # CSI arrow
        (["\x1b", "O", "D"], ["left"]),  # SS3 arrow (e.g. over SSH/tmux)
        (["\x1b"], ["esc"]),  # bare ESC
        (["\x1b", "[", "A"], ["up"]),  # decoded even though the record handler ignores it
        (["n"], ["n"]),  # letter passthrough
    ],
)
def test_terminal_parsing(byte_seq, expected):
    collected = []
    _drive(TerminalKeyListener(collected.append), byte_seq)
    assert collected == expected


# --- Backend selection ------------------------------------------------------
def test_init_selects_terminal_when_pynput_cannot_capture(monkeypatch):
    monkeypatch.setattr(ki, "pynput_can_capture", lambda: False)
    _set_tty(monkeypatch, is_tty=True)
    monkeypatch.setattr(TerminalKeyListener, "start", lambda self: None)  # avoid touching termios
    listener, _ = init_keyboard_listener()
    assert isinstance(listener, TerminalKeyListener)


def test_init_returns_none_without_tty(monkeypatch):
    monkeypatch.setattr(ki, "pynput_can_capture", lambda: False)
    _set_tty(monkeypatch, is_tty=False)
    listener, _ = init_keyboard_listener()
    assert listener is None


@pytest.mark.parametrize(
    ("key", "flag"),
    [("right", "exit_early"), ("r", "rerecord_episode"), ("q", "stop_recording")],
)
def test_init_terminal_key_routing(monkeypatch, key, flag):
    """Arrows and their letter equivalents drive the same events (terminal backend)."""
    monkeypatch.setattr(ki, "pynput_can_capture", lambda: False)
    _set_tty(monkeypatch, is_tty=True)
    monkeypatch.setattr(TerminalKeyListener, "start", lambda self: None)
    listener, events = init_keyboard_listener()
    listener._on_key(key)
    assert events[flag] is True


# --- Shared factory + pynput key resolver -----------------------------------
def test_resolve_pynput_key_char_fallback():
    """Unmapped keys fall back to ``.char`` (and yield None when there is none)."""
    assert ki._resolve_pynput_key(type("K", (), {"char": "s"})()) == "s"
    assert ki._resolve_pynput_key(type("K", (), {"char": None})()) is None
    assert ki._resolve_pynput_key(type("K", (), {"char": ""})()) is None  # empty char -> no key


def test_create_key_listener_routes_to_dispatch(monkeypatch):
    """The terminal backend forwards canonical key names straight to ``dispatch``."""
    monkeypatch.setattr(ki, "pynput_can_capture", lambda: False)
    _set_tty(monkeypatch, is_tty=True)
    monkeypatch.setattr(TerminalKeyListener, "start", lambda self: None)
    seen = []
    listener = create_key_listener(seen.append, controls_help="save='s'")
    assert isinstance(listener, TerminalKeyListener)
    listener._on_key("space")
    assert seen == ["space"]


def test_create_key_listener_none_without_tty(monkeypatch):
    monkeypatch.setattr(ki, "pynput_can_capture", lambda: False)
    _set_tty(monkeypatch, is_tty=False)
    assert create_key_listener(lambda name: None) is None
