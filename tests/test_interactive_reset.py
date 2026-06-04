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

"""Unit tests for the Wayland-friendly interactive recording mode.

Covers ``interactive_reset_prompt`` (the between-episodes ``[Y/n/q]`` stdin
prompt) and the SIGQUIT early-exit signal handlers in
``lerobot.common.control_utils``.
"""

import io
import signal
import sys

import pytest

from lerobot.common.control_utils import (
    install_signal_early_exit,
    interactive_reset_prompt,
    restore_signal_early_exit,
)


@pytest.fixture
def events():
    return {"exit_early": False, "rerecord_episode": False, "stop_recording": False}


@pytest.fixture
def fake_tty_stdin(monkeypatch):
    """Replace ``sys.stdin`` with a TTY-reporting buffer holding ``answer``."""

    def _set(answer):
        stdin = io.StringIO(answer + "\n")
        stdin.isatty = lambda: True
        monkeypatch.setattr(sys, "stdin", stdin)

    return _set


class TestInteractiveResetPrompt:
    def test_enter_keeps_scene(self, events, fake_tty_stdin):
        fake_tty_stdin("")
        interactive_reset_prompt(events)
        assert not any(events.values())

    def test_y_keeps_scene(self, events, fake_tty_stdin):
        fake_tty_stdin("y")
        interactive_reset_prompt(events)
        assert not any(events.values())

    def test_yes_keeps_scene(self, events, fake_tty_stdin):
        fake_tty_stdin("YES")  # case-insensitive
        interactive_reset_prompt(events)
        assert not any(events.values())

    def test_n_triggers_rerecord(self, events, fake_tty_stdin):
        fake_tty_stdin("n")
        interactive_reset_prompt(events)
        assert events["rerecord_episode"] is True
        assert events["stop_recording"] is False

    def test_q_triggers_stop(self, events, fake_tty_stdin):
        fake_tty_stdin("q")
        interactive_reset_prompt(events)
        assert events["stop_recording"] is True
        assert events["rerecord_episode"] is False

    def test_unrecognized_then_valid_reprompts(self, events, monkeypatch, capsys):
        # First answer is invalid → reprompt; second answer ("n") is honored.
        stdin = io.StringIO("maybe\nn\n")
        stdin.isatty = lambda: True
        monkeypatch.setattr(sys, "stdin", stdin)
        interactive_reset_prompt(events)
        assert events["rerecord_episode"] is True
        assert "Unrecognized answer" in capsys.readouterr().out

    def test_eof_triggers_stop(self, events, monkeypatch):
        stdin = io.StringIO("")  # empty → EOFError on input()
        stdin.isatty = lambda: True
        monkeypatch.setattr(sys, "stdin", stdin)
        interactive_reset_prompt(events)
        assert events["stop_recording"] is True

    def test_non_tty_is_noop(self, events, monkeypatch):
        stdin = io.StringIO("n\n")  # would discard if read, but non-TTY must skip
        stdin.isatty = lambda: False
        monkeypatch.setattr(sys, "stdin", stdin)
        interactive_reset_prompt(events)
        assert not any(events.values())


class TestSignalEarlyExit:
    def test_install_and_restore(self, events):
        original = install_signal_early_exit(events)
        try:
            assert signal.getsignal(signal.SIGQUIT) is not original
        finally:
            restore_signal_early_exit(original)
        assert signal.getsignal(signal.SIGQUIT) == original

    def test_handler_sets_exit_early(self, events):
        original = install_signal_early_exit(events)
        try:
            handler = signal.getsignal(signal.SIGQUIT)
            # Invoke the handler directly to simulate receiving SIGQUIT.
            handler(signal.SIGQUIT, None)
            assert events["exit_early"] is True
        finally:
            restore_signal_early_exit(original)

    def test_restore_none_is_noop(self, events):
        # Guard branch: restoring a ``None`` original must not raise.
        original = install_signal_early_exit(events)
        try:
            restore_signal_early_exit(None)
            # Our handler is still installed (None restore was a no-op).
            assert signal.getsignal(signal.SIGQUIT) is not original
        finally:
            restore_signal_early_exit(original)
