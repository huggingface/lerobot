#!/usr/bin/env python3

"""
Simple keyboard test utility.

It collects all keys pressed on stdin and prints the batch every second.
Press Ctrl+C to exit.
"""

from __future__ import annotations

import select
import sys
import termios
import time
import tty


def _format_keys(keys: list[str]) -> str:
    """Return a readable representation of the collected keys."""
    pretty = []
    for key in keys:
        if key == "\n":
            pretty.append("<ENTER>")
        elif key == "\r":
            continue  # handled with newline on most terminals
        elif key == "\t":
            pretty.append("<TAB>")
        elif key == "\x7f":
            pretty.append("<BACKSPACE>")
        elif key.isprintable():
            pretty.append(key)
        else:
            pretty.append(f"<0x{ord(key):02x}>")
    return " ".join(pretty) if pretty else "<none>"


def main() -> None:
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    print("Keyboard test started. Press keys (Ctrl+C to quit).")

    try:
        tty.setcbreak(fd)
        window: list[str] = []
        last_tick = time.monotonic()

        while True:
            # poll stdin; timeout keeps loop responsive for the 1s ticker
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if ready:
                key = sys.stdin.read(1)
                if key == "\x03":  # Ctrl+C
                    raise KeyboardInterrupt
                window.append(key)

            now = time.monotonic()
            if now - last_tick >= 1.0:
                stamp = time.strftime("%H:%M:%S")
                print(f"[{stamp}] { _format_keys(window) }")
                window.clear()
                last_tick = now

    except KeyboardInterrupt:
        print("\nExiting keyboard test.")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    main()
