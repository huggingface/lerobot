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

"""Interactive rollout strategy: autonomous policy execution with a
live-editable language instruction.

A background thread reads full lines of text typed at the terminal and
forwards the latest one to the inference engine between control ticks, so
the VLA policy's instruction can be changed mid-episode without stopping
the rollout. Unlike the single-character hotkeys used by other strategies
(:func:`lerobot.utils.keyboard_input.create_key_listener`), this needs
normal line-buffered/echoing terminal input to let the user type and edit
a full sentence, so it deliberately does not use that utility.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time

from lerobot.utils.robot_utils import precise_sleep

from ..configs import InteractiveStrategyConfig
from ..context import RolloutContext
from .core import RolloutStrategy, send_next_action

logger = logging.getLogger(__name__)


def _drain_latest_instruction(instruction_queue: queue.Queue[str]) -> str | None:
    """Pop all pending instructions non-blockingly; return only the newest.

    Returns ``None`` if nothing was queued since the last call (last-writer-wins
    semantics: intermediate lines typed between ticks are discarded).
    """
    latest = None
    while True:
        try:
            latest = instruction_queue.get_nowait()
        except queue.Empty:
            break
    return latest


class InteractiveStrategy(RolloutStrategy):
    """Autonomous policy rollout whose language instruction can be changed live.

    Identical control loop to :class:`~lerobot.rollout.strategies.base.BaseStrategy`
    (no data recording), except that a background stdin thread lets the
    operator type a new instruction and press Enter at any time; the next
    control tick picks it up and forwards it to the inference engine.
    """

    def __init__(self, config: InteractiveStrategyConfig) -> None:
        super().__init__(config)
        self.config: InteractiveStrategyConfig = config
        self._instruction_queue: queue.Queue[str] = queue.Queue()
        self._input_thread: threading.Thread | None = None
        self._stop_input = threading.Event()
        self._current_task: str = ""

    def setup(self, ctx: RolloutContext) -> None:
        """Initialise the inference engine and start the instruction input thread."""
        self._init_engine(ctx)
        self._current_task = ctx.runtime.cfg.task

        if sys.stdin.isatty():
            self._stop_input.clear()
            self._input_thread = threading.Thread(
                target=self._input_loop,
                daemon=True,
                name="InteractiveInstructionInput",
            )
            self._input_thread.start()
            logger.info("Interactive strategy ready (type a new instruction + Enter to update it live)")
        else:
            self._input_thread = None
            logger.warning(
                "stdin is not a TTY; dynamic instruction updates are disabled. "
                "Running with the fixed --task instruction only."
            )

    def _input_loop(self) -> None:
        """Background thread: block on `input()`, push non-empty lines to the queue."""
        while not self._stop_input.is_set():
            try:
                line = input(f"{self.config.prompt}[{self._current_task}] > ")
            except (EOFError, OSError):
                break
            text = line.strip()
            if text:
                self._instruction_queue.put(text)
                self._current_task = text

    def run(self, ctx: RolloutContext) -> None:
        """Run the autonomous control loop, applying new instructions as they arrive."""
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        interpolator = self._interpolator

        control_interval = interpolator.get_control_interval(cfg.fps)

        start_time = time.perf_counter()
        engine.resume()
        logger.info("Interactive strategy control loop started")

        while not ctx.runtime.shutdown_event.is_set():
            loop_start = time.perf_counter()

            if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                logger.info("Duration limit reached (%.0fs)", cfg.duration)
                break

            obs = robot.get_observation()
            obs_processed = self._process_observation_and_notify(ctx.processors, obs)

            if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                continue

            new_task = _drain_latest_instruction(self._instruction_queue)
            if new_task is not None:
                engine.update_task(new_task)
                if self.config.echo_on_change:
                    logger.info("Instruction updated: '%s'", new_task)

            action_dict = send_next_action(obs_processed, obs, ctx, interpolator)
            self._log_telemetry(obs_processed, action_dict, ctx.runtime)

            dt = time.perf_counter() - loop_start
            if (sleep_t := control_interval - dt) > 0:
                precise_sleep(sleep_t)
            else:
                logger.warning(
                    f"Record loop is running slower ({1 / dt:.1f} Hz) than the target FPS ({cfg.fps} Hz). Dataset frames might be dropped and robot control might be unstable. Common causes are: 1) Camera FPS not keeping up 2) Policy inference taking too long 3) CPU starvation"
                )

    def teardown(self, ctx: RolloutContext) -> None:
        """Stop the input thread and disconnect hardware."""
        self._stop_input.set()
        # The input thread is daemonic and may be blocked on `input()`; it is
        # not joined here and is reaped automatically at process exit.
        self._teardown_hardware(
            ctx.hardware,
            return_to_initial_position=ctx.runtime.cfg.return_to_initial_position,
        )
        logger.info("Interactive strategy teardown complete")
