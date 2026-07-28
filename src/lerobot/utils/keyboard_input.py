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

"""Display-independent keyboard input for interactive controls.

This module centralizes everything related to *discrete* keyboard controls
(end-episode-early, re-record, stop, and the rollout strategies' custom keys):

* environment detection — :func:`is_headless`, :func:`is_wayland`,
  :func:`pynput_can_capture` (the single predicate every call-site should use to
  decide whether ``pynput`` can actually capture keys here);
* a shared key mapping — :func:`apply_recording_control`; and
* two interchangeable backends behind one ``(listener, events)`` contract:
  the ``pynput`` global listener (X11 / trusted-macOS / Windows) and a
  standard-library :class:`TerminalKeyListener` that reads the controlling TTY
  (Wayland / headless-SSH-with-TTY / macOS without Accessibility permission).

NOTE: *continuous* key-state teleoperation ("hold a key to keep moving") is
deliberately NOT served here. A terminal in cbreak mode delivers only key-down
bytes — there is no key-release event — so the held-key model cannot be
reproduced. Those teleoperators stay on ``pynput`` and use
:func:`pynput_can_capture` to warn instead of silently doing nothing.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import platform
import select
import sys
import threading
import time
from collections.abc import Callable
from functools import cache
from typing import TYPE_CHECKING

from .import_utils import _pynput_available

logger = logging.getLogger(__name__)

# POSIX-only terminal modules (absent on Windows, where the pynput backend is used).
if TYPE_CHECKING:
    import termios
    import tty

    _TERMIOS_AVAILABLE = True
else:
    try:
        import termios
        import tty

        _TERMIOS_AVAILABLE = True
    except ImportError:  # POSIX-only modules; unavailable on Windows
        termios = tty = None
        _TERMIOS_AVAILABLE = False

keyboard = None
if _pynput_available:
    try:
        from pynput import keyboard
    except Exception as e:  # e.g. no reachable X display on a headless Linux box
        logger.info("Could not import pynput keyboard backend: %s", e)


@cache
def is_headless() -> bool:
    """Return ``True`` when no display server is available.

    * Linux: headless when neither ``DISPLAY`` (X11) nor ``WAYLAND_DISPLAY`` is set.
    * macOS / Windows: a display is always assumed to be present. A genuinely GUI-less
    Mac/Windows CI host would be misclassified but it doesn't matter, because the
    sys.stdin.isatty() gate returns None there regardless.
    """
    if platform.system() == "Linux":
        return not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return False


@cache
def is_wayland() -> bool:
    """Return ``True`` when running under a Wayland session.

    ``pynput`` relies on an X11 backend. Under Wayland it still imports (XWayland
    is usually present and ``$DISPLAY`` is set) but cannot capture *global*
    hotkeys, so the documented arrow/Esc shortcuts silently do nothing. This case
    is invisible to :func:`is_headless`, hence the dedicated check.
    """
    return os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland" or bool(
        os.environ.get("WAYLAND_DISPLAY")
    )


@cache
def pynput_can_capture() -> bool:
    """Return ``True`` when a ``pynput`` global listener can actually capture keys.

    This is the single predicate every keyboard call-site should use to choose
    between the ``pynput`` backend and a fallback. It is intentionally
    conservative:

    * Linux: only a real X11 session (a display is present *and* it is not Wayland).
    * macOS: ``True`` here — Accessibility / Input-Monitoring permission
      (``IS_TRUSTED``) can only be confirmed at runtime *after* starting a
      listener, so :func:`init_keyboard_listener` refines this with
      :func:`pynput_listener_is_trusted`.
    * Windows: ``True`` (the low-level global hook needs no special permission).

    Always ``False`` when ``pynput`` is not installed.
    """
    if not _pynput_available:
        return False
    if platform.system() == "Linux":
        return not is_headless() and not is_wayland()
    return True


def pynput_listener_is_trusted(listener, timeout_s: float = 1.0) -> bool:
    """Best-effort check that a freshly started ``pynput`` listener can capture.

    On macOS, ``pynput`` sets ``listener.IS_TRUSTED`` on its *listener thread*
    once the Quartz event tap is created; the class default is ``False``. We
    therefore wait for the thread to either flip it ``True`` (trusted) or for a
    short timeout to elapse (untrusted — it stays ``False`` forever). This biases
    toward the common trusted case (returns as soon as the flag flips) and only
    pays the full ``timeout_s`` on an already-broken untrusted machine.

    On non-macOS backends the attribute is absent and capture is assumed to work.
    """
    if platform.system() != "Darwin":
        return True
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if getattr(listener, "IS_TRUSTED", False):
            return True
        time.sleep(0.02)
    return bool(getattr(listener, "IS_TRUSTED", False))


def apply_recording_control(control: str, events: dict) -> None:
    """Apply a recording control-flow key press to the shared ``events`` dict.

    Centralizes the mapping so the ``pynput`` and terminal backends behave
    identically. ``control`` is one of ``"right"`` (end the loop early), ``"left"``
    (re-record the last episode), or ``"esc"`` (stop recording).
    """
    if control == "right":
        print("Right arrow key pressed. Exiting loop...")
        events["exit_early"] = True
    elif control == "left":
        print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
        events["rerecord_episode"] = True
        events["exit_early"] = True
    elif control == "esc":
        print("Escape key pressed. Stopping data recording...")
        events["stop_recording"] = True
        events["exit_early"] = True


# Terminal arrow keys arrive as a 3-byte escape sequence whose *final* byte identifies
# the direction. Two encodings exist depending on the terminal's cursor-key mode — CSI
# ("ESC [ X") and SS3 ("ESC O X", common over SSH/tmux) — but both share the same final
# byte, so this single table decodes either. Looked up by TerminalKeyListener._parse;
# an unknown final byte yields None (sequence ignored).
_ARROW_FINAL_BYTES = {"A": "up", "B": "down", "C": "right", "D": "left"}


class TerminalKeyListener:
    """Display-independent keyboard listener that reads keys from the controlling TTY.

    Used as the Wayland / headless / macOS-untrusted equivalent of the ``pynput``
    listener for *discrete* controls. It puts the terminal into cbreak mode with
    echo disabled and reads bytes on a daemon thread, decoding them into logical
    key names that are passed to ``on_key``:

    * arrow keys (``ESC [ C`` / ``ESC O C`` …) -> ``"right"`` / ``"left"`` / ``"up"`` / ``"down"``
    * a bare ``ESC`` -> ``"esc"``
    * Enter / Tab / Space / Backspace -> ``"enter"`` / ``"tab"`` / ``"space"`` / ``"backspace"``
    * any other printable byte -> that character (e.g. ``"n"``, ``"s"``)

    Only key-down events are produced (terminals have no key-release), so this is
    suitable for discrete commands but NOT for continuous "hold-to-move" teleop.

    The terminal is restored on :meth:`stop` and also via an ``atexit`` hook, so a
    crash or Ctrl-C never leaves the shell in a no-echo cbreak state. POSIX-only
    (``termios`` / ``tty`` / ``select``); those modules are imported lazily so this
    file stays importable on Windows (where ``pynput`` is used instead).
    """

    def __init__(self, on_key: Callable[[str], None]):
        self._on_key = on_key
        self._running = False
        self._thread: threading.Thread | None = None
        self._fd: int | None = None
        self._old_attrs = None

    def _read_char(self, timeout: float) -> str | None:
        """Return one character from stdin within ``timeout`` seconds, or ``None``."""
        if self._fd is None:
            return None
        ready, _, _ = select.select([self._fd], [], [], timeout)
        if not ready:
            return None
        try:
            data = os.read(self._fd, 1)
        except OSError:
            return None
        if not data:
            return None
        return data.decode(errors="ignore")

    def _parse(self, ch: str) -> str | None:
        """Decode one (possibly multi-byte) key starting at ``ch`` into a key name."""
        if ch == "\x1b":
            # Possible CSI / SS3 escape sequence (arrow keys) or a bare ESC. Use
            # short follow-up reads so a lone ESC is not mistaken for a sequence.
            ch2 = self._read_char(timeout=0.02)
            if ch2 is None:
                return "esc"
            if ch2 in ("[", "O"):
                ch3 = self._read_char(timeout=0.02)
                return _ARROW_FINAL_BYTES.get(ch3 or "")
            # Some other escape sequence (e.g. Alt+key); ignore it.
            return None
        if ch in ("\r", "\n"):
            return "enter"
        if ch == "\t":
            return "tab"
        if ch == " ":
            return "space"
        if ch in ("\x7f", "\x08"):
            return "backspace"
        if ch.isprintable():
            return ch
        return None

    def _run(self) -> None:
        while self._running:
            ch = self._read_char(timeout=0.05)
            if ch is None:
                continue
            name = self._parse(ch)
            if name is None:
                continue
            try:
                self._on_key(name)
            except Exception as e:  # never let a handler error kill the reader thread
                logger.debug("Terminal key handler error: %s", e)

    def start(self) -> None:
        """Switch the terminal to cbreak mode (echo off) and read keys on a daemon thread.

        No-op when stdin is not a TTY (piped/redirected input) or on platforms
        without ``termios`` (e.g. Windows), so non-interactive runs are unaffected.
        """
        if not sys.stdin.isatty():
            return
        if not _TERMIOS_AVAILABLE:  # POSIX-only modules (e.g. unavailable on Windows)
            logger.warning("Terminal keyboard input is not supported on this platform.")
            return

        self._fd = sys.stdin.fileno()
        self._old_attrs = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        # Explicitly disable ECHO so arrow-key escape sequences (e.g. ^[[C) are not
        # echoed as garbage into the recording terminal. (Independent of the
        # version-specific behavior of tty.setcbreak.)
        new_attrs = termios.tcgetattr(self._fd)
        new_attrs[3] &= ~termios.ECHO  # index 3 == lflags
        termios.tcsetattr(self._fd, termios.TCSADRAIN, new_attrs)
        # Safety net: restore the terminal even if stop() is never reached (crash).
        atexit.register(self.stop)

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the reader thread and restore the original terminal attributes.

        Idempotent: safe to call multiple times (e.g. explicitly and via atexit).
        """
        self._running = False
        thread = self._thread
        if thread is not None:
            thread.join(timeout=0.5)
            self._thread = None
        if self._fd is not None and self._old_attrs is not None and _TERMIOS_AVAILABLE:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)
            finally:
                self._old_attrs = None
        with contextlib.suppress(Exception):
            atexit.unregister(self.stop)


# Map pynput key objects to the same canonical names TerminalKeyListener emits, so a
# single dispatch works across both backends. Empty when pynput is unavailable.
if keyboard is not None:
    _PYNPUT_KEY_NAMES = {
        keyboard.Key.right: "right",
        keyboard.Key.left: "left",
        keyboard.Key.up: "up",
        keyboard.Key.down: "down",
        keyboard.Key.esc: "esc",
        keyboard.Key.enter: "enter",
        keyboard.Key.tab: "tab",
        keyboard.Key.space: "space",
        keyboard.Key.backspace: "backspace",
    }
else:
    _PYNPUT_KEY_NAMES = {}


def _resolve_pynput_key(key) -> str | None:
    """Resolve a pynput key event to the canonical name TerminalKeyListener also emits.

    Special keys map through :data:`_PYNPUT_KEY_NAMES`; character keys fall back to their
    ``.char`` (e.g. ``"n"``). Returns ``None`` for keys with no mapping and no character.
    """
    name = _PYNPUT_KEY_NAMES.get(key)
    if name is not None:
        return name
    # ``or None`` keeps the historical truthy-char semantics: an empty/None char is "no key".
    return getattr(key, "char", None) or None


def create_key_listener(dispatch: Callable[[str], None], *, controls_help: str = ""):
    """Start a keyboard listener that routes resolved key names to ``dispatch``.

    Shared backend selection used by recording and the rollout strategies:

    * the ``pynput`` global listener on X11 / trusted-macOS / Windows (on macOS the
      listener's ``IS_TRUSTED`` flag is checked after start, and an untrusted listener is
      stopped so the terminal backend is used instead);
    * the stdlib :class:`TerminalKeyListener` on Wayland / headless sessions with a TTY;
    * ``None`` when no backend is usable (non-interactive / piped runs).

    Both backends pass ``dispatch`` the same canonical key names ("right" / "left" / "up" /
    "down" / "esc" / "enter" / "tab" / "space" / "backspace", or a character), so one
    ``dispatch`` works regardless of backend. ``controls_help`` is an optional hint
    appended to the log messages.

    Returns the listener (exposing ``.stop()``) or ``None``.
    """
    suffix = f" ({controls_help})" if controls_help else ""

    if pynput_can_capture() and keyboard is not None:

        def on_press(key):
            with contextlib.suppress(Exception):
                name = _resolve_pynput_key(key)
                if name is not None:
                    dispatch(name)

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        if pynput_listener_is_trusted(listener):
            logger.info("Keyboard listener started%s.", suffix)
            return listener
        # macOS without Accessibility / Input-Monitoring permission: the listener never
        # fires. Stop it and fall through to the terminal backend.
        logger.warning(
            "pynput keyboard listener is not trusted (missing macOS Accessibility / "
            "Input Monitoring permission); falling back to terminal keyboard input."
        )
        listener.stop()

    if sys.stdin.isatty():
        listener = TerminalKeyListener(dispatch)
        listener.start()
        logger.info("Using terminal keyboard input — keep this terminal focused%s.", suffix)
        return listener

    logger.warning(
        "Keyboard controls unavailable: no usable display (Wayland/headless) and stdin is "
        "not an interactive terminal%s.",
        suffix,
    )
    return None


def init_keyboard_listener():
    """Initialize a non-blocking keyboard listener for interactive recording controls.

    Backend selection:

    * ``pynput`` global listener when :func:`pynput_can_capture` is true (real
      X11, macOS, Windows). On macOS the listener's ``IS_TRUSTED`` flag is checked
      after start; if the process lacks Accessibility / Input-Monitoring
      permission, the listener is stopped and the terminal backend is used.
    * a :class:`TerminalKeyListener` reading the controlling TTY when ``pynput``
      cannot capture (Wayland / headless-SSH / macOS-untrusted) *and* stdin is a TTY.
    * otherwise no listener (non-interactive / piped runs) — recording relies on
      the episode/reset timers (or Ctrl+C).

    Both backends accept the same controls: Right/Left/Esc, plus the single-byte letter
    equivalents ``n`` (next), ``r`` (re-record) and ``q`` (quit). The letters are the most
    reliable choice over high-latency SSH/VNC links, where arrow-key escape sequences can
    be split, delayed, or intercepted by the terminal.

    Returns:
        A tuple ``(listener, events)`` where ``listener`` exposes ``.stop()`` or is
        ``None``, and ``events`` is the dict of flags (``exit_early``,
        ``rerecord_episode``, ``stop_recording``) set by key presses.
    """
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
    }

    # Accept the single-byte letter equivalents n/r/q alongside the arrow/Esc keys: the
    # letters are immune to the escape-sequence split/delay/interception that affects arrows
    # over laggy SSH/VNC links. Case-insensitive so Shift+letter still works.
    def on_key(name: str) -> None:
        key = name.lower()
        if key in ("right", "n"):
            apply_recording_control("right", events)
        elif key in ("left", "r"):
            apply_recording_control("left", events)
        elif key in ("esc", "q"):
            apply_recording_control("esc", events)
        # other keys (incl. up/down) are intentionally ignored

    listener = create_key_listener(on_key, controls_help="Right/Left/Esc, or n=next, r=re-record, q=quit")
    return listener, events
