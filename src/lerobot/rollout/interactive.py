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

"""Interactive rollout session: chat-style stdin commands for ``lerobot-rollout``.

Enabled with ``--interactive=true``, this module lets the operator control a
rollout from the terminal while hardware and policy stay connected and warm:

    /start           start (or restart) the policy control loop
    /subtask <text>  change the instruction the policy follows, mid-run
    /vqa <text>      ask the policy a question about what it currently sees
    /autosteer <goal>  let the policy pick its own subtasks toward a goal
                     (``/autosteer off`` hands control back)
    /reset           stop movement, return the robot to its initial position,
                     and restore the instruction passed on the command line
    /stop            end the session and run the normal shutdown routines
    /help            show the available commands

This module is only the CLI front-end: stdin reading
(:class:`lerobot.utils.stdin_input.StdinCommandListener`), command parsing,
terminal output, and log muting.  All control logic lives in
:class:`lerobot.rollout.controller.RolloutController`, which is the public
API for driving a rollout programmatically (from an application, a network
server, a notebook, ...) without any of this module's terminal I/O.

Threading model: a daemon stdin-listener thread parses lines and calls the
controller's thread-safe methods (``start``/``reset``/``stop``/``set_task``/
``ask``/``autosteer``); it never touches hardware or policy state.  ``RolloutController.serve()``
runs on the main thread and executes ``strategy.run(ctx)`` in *segments*,
ended through the session's :class:`LinkedEvent` (installed as
``ctx.runtime.shutdown_event``).  Real shutdown signals (SIGINT/SIGTERM)
propagate through the linked event's parent, so Ctrl-C behaves exactly as in
non-interactive runs.

While the session runs, log records below WARNING and Python warnings are
suppressed process-wide (via ``logging.disable``) so routine system output
does not interleave with the chat prompt; WARNING records and above still
reach the console (e.g. the control loop missing its FPS target), and a
fatal inference-engine error is additionally reported with its captured
traceback.  Normal logging resumes when the session ends (so teardown logs
are visible).  Run without ``--interactive`` to see the full live log output.

The command table is intentionally a name → (handler, argument hint, help)
mapping so further commands can be registered without restructuring the
parser, the help output, or the session loop.
"""

from __future__ import annotations

import contextlib
import logging
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import IO, TYPE_CHECKING

from lerobot.utils.stdin_input import StdinCommandListener
from lerobot.utils.utils import log_say

from .controller import AskResult, RolloutController, RolloutEvent
from .inference import QueryAnswer, QueryKind

if TYPE_CHECKING:
    from .context import RolloutContext
    from .strategies import RolloutStrategy

logger = logging.getLogger(__name__)

_BANNER_RULE = "─" * 60


@contextlib.contextmanager
def _mute_system_output() -> Iterator[None]:
    """Suppress log records below WARNING and Python warnings, process-wide.

    System logs (policy, robot, control loop) contend with the chat prompt
    for the terminal.  ``logging.disable`` gates records before any handler
    dispatch, which covers non-propagating library loggers (``transformers``,
    ``datasets``) and loggers created mid-session alike; WARNING and above
    still get through, so control-loop overruns and failures stay visible.
    The gate applies to every handler — including file handlers, which
    therefore also miss INFO/DEBUG records for the duration.  Python warnings
    bypass logging entirely and are silenced separately: they are dominated
    by third-party deprecation notices, not operational signals.
    """
    previous_disable = logging.root.manager.disable
    saved_warning_filters = warnings.filters[:]
    logging.disable(logging.INFO)
    warnings.simplefilter("ignore")
    try:
        yield
    finally:
        logging.disable(previous_disable)
        warnings.filters[:] = saved_warning_filters


@dataclass(frozen=True)
class InteractiveCommand:
    """A parsed ``/name args`` line from the interactive prompt."""

    name: str
    args: str = ""


def _format_task(task: str) -> str:
    """Render a task string for the operator, naming the empty case explicitly."""
    return repr(task) if task else "(none — set one with /subtask <text>)"


def _strip_quotes(text: str) -> str:
    """Drop one layer of matching surrounding quotes from a command argument."""
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1]
    return text


def parse_command(line: str) -> InteractiveCommand | None:
    """Parse an input line into an :class:`InteractiveCommand`.

    Commands are ``/name`` optionally followed by free-text arguments (e.g.
    ``/subtask grab the red cube``).  Returns ``None`` for lines that are not
    commands (no leading ``/`` or a bare ``/``).
    """
    line = line.strip()
    if not line.startswith("/"):
        return None
    head, *rest = line.split(maxsplit=1)
    name = head[1:].lower()
    if not name:
        return None
    return InteractiveCommand(name=name, args=rest[0].strip() if rest else "")


class InteractiveSession:
    """Drive a rollout from chat-style stdin commands.

    A thin terminal front-end over :class:`RolloutController`: the stdin
    listener parses lines into commands, each command calls one of the
    controller's thread-safe methods, and controller events are rendered
    back as terminal output.  The controller is exposed as
    :attr:`controller` for tests and embedders.

    Commands are last-write-wins: ``/reset`` and ``/stop`` cancel a pending
    ``/start`` so the robot never starts moving after the operator's final
    command asked it not to.  End-of-file on the command stream stops the
    session (a closed stdin means there is no way left to command the
    robot), so piped scripts must keep stdin open for the intended session
    duration, e.g. ``(printf '/start\\n'; sleep 60; printf '/stop\\n') |
    lerobot-rollout ... --interactive=true``.  The session works over SSH
    and in headless setups — it reads the terminal (or pipe) directly and
    needs no display server.
    """

    def __init__(
        self,
        strategy: RolloutStrategy,
        ctx: RolloutContext,
        input_stream: IO[str] | None = None,
    ) -> None:
        self.controller = RolloutController(strategy, ctx, on_event=self._on_event)
        self._play_sounds = ctx.runtime.cfg.play_sounds
        self._listener = StdinCommandListener(self._handle_line, on_eof=self._handle_eof, stream=input_stream)

        # name -> (handler, argument hint, help line); /help and the banner
        # render from this table, so future commands stay documented for free.
        self._commands: dict[str, tuple[Callable[[InteractiveCommand], None], str, str]] = {
            "start": (self._cmd_start, "", "start (or restart) the policy control loop"),
            "subtask": (self._cmd_subtask, " <text>", "set the instruction the policy follows"),
            "vqa": (self._cmd_vqa, " <text>", "ask the policy a question about what it sees"),
            "autosteer": (
                self._cmd_autosteer,
                " <goal>|off",
                "let the policy pick its own subtasks toward a high-level goal",
            ),
            "reset": (self._cmd_reset, "", "stop movement, return to initial position, restore the task"),
            "stop": (self._cmd_stop, "", "end the session and shut down"),
            "help": (self._cmd_help, "", "show this help"),
        }

    def run(self) -> None:
        """Run the session until ``/stop``, EOF, engine failure, or a shutdown signal."""
        try:
            with _mute_system_output():
                self._print(self._render_banner())
                self._listener.start()
                try:
                    self.controller.serve()
                finally:
                    self._listener.stop()
        finally:
            # Outside the muting context so the closing announcement and any
            # teardown logs are visible again.
            log_say("Interactive session ended", self._play_sounds)

    # ------------------------------------------------------------------
    # Controller events (fired on the serve thread) -> terminal output
    # ------------------------------------------------------------------

    def _on_event(self, event: RolloutEvent, payload: QueryAnswer | None = None) -> None:
        if event is RolloutEvent.QUERY_ANSWERED and payload is not None:
            self._report_answer(payload)
        elif event is RolloutEvent.SEGMENT_STARTED:
            log_say("Starting rollout", self._play_sounds)
            self._print(
                f"Rollout running — task {_format_task(self.controller.task)}. "
                "/subtask <text> to change it, /reset to return to initial position, /stop to shut down."
            )
        elif event is RolloutEvent.SEGMENT_ENDED:
            self._print(
                "Rollout run ended on its own (duration reached). Robot is holding position — "
                "/start to run again, /reset to return to initial position, /stop to shut down."
            )
        elif event is RolloutEvent.RESET_STARTED:
            log_say("Resetting robot to initial position", self._play_sounds)
            self._print("Resetting — returning the robot to its initial position...")
        elif event is RolloutEvent.RESET_DONE:
            self._print("Robot reset — holding at initial position. /start to run.")
        elif event is RolloutEvent.RESET_SKIPPED:
            self._print("Robot paused — no initial position captured, holding current pose. /start to run.")
        elif event is RolloutEvent.ENGINE_FAILED:
            self._report_engine_failure()

    def _report_answer(self, answer: QueryAnswer) -> None:
        """Render a resolved text query (an operator question or an autosteer turn)."""
        if answer.kind is QueryKind.NEXT_SUBTASK:
            if answer.ok:
                # The engine has already applied it via set_task; just announce.
                self._print(f"Autosteer subtask: {answer.answer!r}")
            else:
                self._print(
                    f"Autosteer stopped — could not plan the next subtask for {answer.question!r}: "
                    f"{answer.error}"
                )
        elif answer.ok:
            self._print(f"Q: {answer.question}\nA: {answer.answer}")
        else:
            self._print(f"Could not answer {answer.question!r} — {answer.error}")

    def _report_engine_failure(self) -> None:
        """Surface a fatal engine error despite the muted console logging."""
        self._print("Inference engine failed — shutting down.")
        failure_traceback = self.controller.failure_traceback
        if failure_traceback:
            self._print(failure_traceback)
        else:
            self._print("Re-run without --interactive=true to see the error output.")

    # ------------------------------------------------------------------
    # Command handlers (called from the listener thread; the controller's
    # methods are thread-safe and only publish state)
    # ------------------------------------------------------------------

    def _handle_line(self, line: str) -> None:
        cmd = parse_command(line)
        if cmd is None:
            self._print("Input not recognized — commands start with '/'. Type /help for the list.")
            return
        entry = self._commands.get(cmd.name)
        if entry is None:
            self._print(f"Unknown command '/{cmd.name}'. Type /help for the list.")
            return
        handler = entry[0]
        handler(cmd)

    def _handle_eof(self) -> None:
        self._print("Input stream closed — stopping the session.")
        self.controller.stop()

    def _cmd_start(self, cmd: InteractiveCommand) -> None:
        if not self.controller.start():
            self._print("Already running — /reset to pause first, or /stop to shut down.")

    def _cmd_subtask(self, cmd: InteractiveCommand) -> None:
        if not cmd.args:
            self._print(f"Current task: {_format_task(self.controller.task)}")
            return
        task = _strip_quotes(cmd.args)
        previous = self.controller.task
        steering = self.controller.autosteer_goal
        if steering is not None:
            self._print(f"Autosteer off (was {steering!r}) — setting the instruction by hand takes over.")
        if self.controller.set_task(task):
            self._print(
                f"Task: {_format_task(previous)} → {_format_task(task)} "
                "(applies from the next policy inference)"
            )
        else:
            self._print(f"Task unchanged: {_format_task(task)}")

    def _cmd_vqa(self, cmd: InteractiveCommand) -> None:
        if not cmd.args:
            self._print("Usage: /vqa <question> — e.g. /vqa is the cube inside the box?")
            return
        question = _strip_quotes(cmd.args)
        result = self.controller.ask(question)
        if result is AskResult.QUEUED:
            self._print(f"Asked: {question!r} — answering from the next observation...")
        elif result is AskResult.UNSUPPORTED:
            self._print("This policy has no text head — it cannot answer questions.")
        elif result is AskResult.NOT_RUNNING:
            self._print("Not running — /start first so the policy has a live view to answer from.")
        else:
            # Could be a previous /vqa or an autosteer subtask query — the
            # channel holds one at a time and does not say which.
            self._print("The policy is busy with another query — try again in a moment.")

    def _cmd_autosteer(self, cmd: InteractiveCommand) -> None:
        goal = _strip_quotes(cmd.args)
        if not goal:
            current = self.controller.autosteer_goal
            self._print(
                f"Autosteer on — goal {current!r}." if current else "Autosteer off. Usage: /autosteer <goal>"
            )
            return
        if goal.lower() == "off":
            stopped = self.controller.stop_autosteer()
            self._print(
                f"Autosteer off (was {stopped!r}). The last subtask stays in effect."
                if stopped
                else "Autosteer was not running."
            )
            return
        result = self.controller.autosteer(goal)
        if result is AskResult.UNSUPPORTED:
            self._print("This policy has no text head — it cannot plan subtasks.")
            return
        if result is AskResult.NOT_RUNNING:
            self._print("Not running — /start first so the policy has a live view to plan from.")
            return
        self._print(
            f"Autosteer on — goal {goal!r}. The policy picks its own subtasks; "
            "each one is announced here. Take over with /subtask <text> or /autosteer off."
        )

    def _cmd_reset(self, cmd: InteractiveCommand) -> None:
        if self.controller.reset():
            self._print(f"Task restored to {_format_task(self.controller.initial_task)}")

    def _cmd_stop(self, cmd: InteractiveCommand) -> None:
        self.controller.stop()

    def _cmd_help(self, cmd: InteractiveCommand) -> None:
        self._print(self._render_help())

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_help(self) -> str:
        usages = {name: f"/{name}{entry[1]}" for name, entry in self._commands.items()}
        width = max(len(usage) for usage in usages.values())
        lines = [f"  {usages[name]:<{width}}   {entry[2]}" for name, entry in self._commands.items()]
        return "Available commands:\n" + "\n".join(lines)

    def _render_banner(self) -> str:
        return (
            f"{_BANNER_RULE}\n"
            "Interactive rollout session — the robot will NOT move until you type /start.\n"
            f"Task: {_format_task(self.controller.initial_task)}\n"
            f"{self._render_help()}\n"
            "Routine system logs are muted during the session (warnings and errors still show).\n"
            f"{_BANNER_RULE}"
        )

    @staticmethod
    def _print(message: str) -> None:
        """User-facing chat output; logging stays on stderr, replies on stdout."""
        print(message, flush=True)
