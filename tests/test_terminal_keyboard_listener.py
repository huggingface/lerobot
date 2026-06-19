#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Unit tests for the Wayland/headless keyboard backend in ``lerobot.common.control_utils``.

Covers ``is_wayland`` session detection, the shared ``handle_key`` mapping, and the
``TerminalKeyListener`` key parsing (ANSI escape sequences + letter fallbacks, with up/down
arrows ignored) and terminal-state restore on teardown.
"""

import sys

import pytest

from lerobot.common.control_utils import (
    TerminalKeyListener,
    handle_key,
    is_wayland,
)


@pytest.fixture
def events():
    return {"exit_early": False, "rerecord_episode": False, "stop_recording": False}


def _drive(listener: TerminalKeyListener, keys: list[str | None]) -> None:
    """Run the listener's read loop synchronously over a scripted list of characters.

    ``None`` entries simulate a read timeout (no character available). The loop stops once the
    script is exhausted, so no real terminal or thread is involved.
    """
    script = list(keys)

    def fake_read(_timeout):
        if script:
            return script.pop(0)
        listener._running = False
        return None

    listener._read_char = fake_read
    listener._running = True
    listener._run()


class TestHandleKey:
    def test_right_sets_exit_early(self, events):
        handle_key("right", events)
        assert events == {"exit_early": True, "rerecord_episode": False, "stop_recording": False}

    def test_left_sets_rerecord_and_exit(self, events):
        handle_key("left", events)
        assert events == {"exit_early": True, "rerecord_episode": True, "stop_recording": False}

    def test_esc_sets_stop_and_exit(self, events):
        handle_key("esc", events)
        assert events == {"exit_early": True, "rerecord_episode": False, "stop_recording": True}

    def test_unknown_kind_is_noop(self, events):
        handle_key("up", events)
        assert not any(events.values())


class TestTerminalKeyParsing:
    def test_right_arrow_sequence(self, events):
        _drive(TerminalKeyListener(events), ["\x1b", "[", "C"])
        assert events["exit_early"] is True
        assert events["rerecord_episode"] is False
        assert events["stop_recording"] is False

    def test_left_arrow_sequence(self, events):
        _drive(TerminalKeyListener(events), ["\x1b", "[", "D"])
        assert events["rerecord_episode"] is True
        assert events["exit_early"] is True

    def test_bare_esc_stops(self, events):
        # No bytes follow the ESC -> treated as the Escape key, not a CSI sequence.
        _drive(TerminalKeyListener(events), ["\x1b"])
        assert events["stop_recording"] is True
        assert events["exit_early"] is True

    @pytest.mark.parametrize("final", ["A", "B"])
    def test_up_down_arrows_are_ignored(self, events, final):
        _drive(TerminalKeyListener(events), ["\x1b", "[", final])
        assert not any(events.values())

    def test_letter_n_is_next(self, events):
        _drive(TerminalKeyListener(events), ["n"])
        assert events["exit_early"] is True
        assert events["rerecord_episode"] is False

    def test_letter_r_is_rerecord(self, events):
        _drive(TerminalKeyListener(events), ["r"])
        assert events["rerecord_episode"] is True
        assert events["exit_early"] is True

    def test_letter_q_is_quit(self, events):
        _drive(TerminalKeyListener(events), ["q"])
        assert events["stop_recording"] is True

    def test_letters_are_case_insensitive(self, events):
        _drive(TerminalKeyListener(events), ["Q"])
        assert events["stop_recording"] is True

    def test_unmapped_letter_is_noop(self, events):
        _drive(TerminalKeyListener(events), ["x"])
        assert not any(events.values())


class TestIsWayland:
    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        # is_wayland is @cache-decorated; clear around each test so env changes take effect.
        is_wayland.cache_clear()
        yield
        is_wayland.cache_clear()

    def test_detects_xdg_session_type(self, monkeypatch):
        monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        assert is_wayland() is True

    def test_xdg_session_type_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("XDG_SESSION_TYPE", "Wayland")
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        assert is_wayland() is True

    def test_detects_wayland_display(self, monkeypatch):
        monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        assert is_wayland() is True

    def test_x11_is_not_wayland(self, monkeypatch):
        monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        assert is_wayland() is False

    def test_no_env_is_not_wayland(self, monkeypatch):
        monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        assert is_wayland() is False


class TestStartStopLifecycle:
    def test_start_is_noop_without_tty(self, events, monkeypatch):
        pytest.importorskip("termios")

        class _FakeStdin:
            def isatty(self):
                return False

        monkeypatch.setattr(sys, "stdin", _FakeStdin())
        listener = TerminalKeyListener(events)
        listener.start()
        assert listener._thread is None
        # stop() must be safe even though start() did nothing.
        listener.stop()

    def test_stop_restores_terminal_attributes(self, events, monkeypatch):
        termios = pytest.importorskip("termios")
        import tty

        class _FakeStdin:
            def isatty(self):
                return True

            def fileno(self):
                return 0

        restored = {}
        monkeypatch.setattr(sys, "stdin", _FakeStdin())
        monkeypatch.setattr(termios, "tcgetattr", lambda fd: "ORIGINAL_ATTRS")
        monkeypatch.setattr(tty, "setcbreak", lambda fd: None)
        monkeypatch.setattr(
            termios,
            "tcsetattr",
            lambda fd, when, attrs: restored.update(fd=fd, attrs=attrs),
        )

        listener = TerminalKeyListener(events)
        # Avoid the reader thread touching real stdin while it runs.
        listener._read_char = lambda _timeout: None

        listener.start()
        assert listener._thread is not None
        listener.stop()

        assert restored["attrs"] == "ORIGINAL_ATTRS"
        assert listener._thread is None
