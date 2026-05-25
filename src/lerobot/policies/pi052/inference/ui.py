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
"""Rich-based REPL layout for the PI052 runtime.

Two-zone terminal layout:

    [chat scrollback — user messages / robot responses, scrolls naturally]

    ┌── State ──────────────────────────────────────────┐
    │ task     please clean up the kitchen              │
    │ subtask  grasp the handle of the sponge           │
    │ plan     1. grasp sponge  2. wipe  3. tidy        │
    │ memory   sponge picked up; counter still dirty    │
    └───────────────────────────────────────────────────┘
    > _

The state panel re-renders on every state change. Chat lines are
``console.print``'d above the live region so they accumulate naturally
in scrollback. Implemented with :class:`rich.live.Live` plus
:func:`rich.console.Console.input` for the prompt — when an input is
pending, ``rich.Live`` auto-suspends so the input doesn't fight the
panel for cursor position.
"""

from __future__ import annotations

from typing import Any

try:  # rich is optional; only required for the interactive REPL.
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    _HAS_RICH = True
except ImportError:  # pragma: no cover
    _HAS_RICH = False
    Console = Any  # type: ignore[assignment]
    Panel = Any  # type: ignore[assignment]
    Table = Any  # type: ignore[assignment]
    Text = Any  # type: ignore[assignment]


_STATE_KEYS = (
    ("task", "task"),
    ("current_subtask", "subtask"),
    ("current_plan", "plan"),
    ("current_memory", "memory"),
)


def make_state_panel(state: dict[str, Any]) -> Any:
    """Render the persistent state panel for the live region.

    Returns a :class:`rich.panel.Panel`. Caller passes it to
    ``Live.update(panel)`` whenever the state changes.
    """
    if not _HAS_RICH:
        raise RuntimeError(
            "rich is required for the interactive REPL. "
            "`pip install rich` (it's a transitive dep of lerobot)."
        )
    table = Table.grid(padding=(0, 2), expand=True)
    table.add_column(justify="right", style="dim", no_wrap=True, width=10)
    table.add_column(justify="left")
    for key, label in _STATE_KEYS:
        value = state.get(key)
        if value is None:
            rendered = Text("(not set)", style="dim italic")
        else:
            rendered = Text(str(value), style="bold")
        table.add_row(label, rendered)
    queue = state.get("action_queue")
    queue_len = len(queue) if hasattr(queue, "__len__") else 0
    pending = state.get("tool_calls_pending") or []
    footer = Text.assemble(
        ("queued actions: ", "dim"),
        (str(queue_len), "bold cyan"),
        ("    pending tool calls: ", "dim"),
        (str(len(pending)), "bold magenta"),
    )
    table.add_row("", footer)
    run_mode = state.get("mode", "action")
    mode_tag = (
        "[green]action[/]" if run_mode == "action" else "[yellow]paused[/]"
    )
    return Panel(
        table,
        title=f"[bold]PI052 state[/] · mode: {mode_tag}",
        border_style="cyan",
    )


def print_user_line(console: Any, line: str) -> None:
    """Append a user-typed line to the chat scrollback."""
    if not _HAS_RICH:
        print(f"you: {line}", flush=True)
        return
    console.print(f"[bold cyan]you:[/] {line}")


def print_robot_lines(console: Any, lines: list[str]) -> None:
    """Append robot/runtime log lines to the chat scrollback."""
    if not _HAS_RICH:
        for line in lines:
            print(f"robot: {line.lstrip()}", flush=True)
        return
    for line in lines:
        # The runtime uses leading whitespace + "label: text"; render
        # the label in green and the value in default for readability.
        stripped = line.lstrip()
        if ":" in stripped:
            label, _, value = stripped.partition(":")
            console.print(f"[bold green]robot[/] [dim]({label.strip()})[/] {value.strip()}")
        else:
            console.print(f"[bold green]robot:[/] {stripped}")
