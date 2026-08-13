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

"""Base rollout strategy: autonomous policy execution with no data recording."""

from __future__ import annotations

import logging
import time

from lerobot.utils.cycle_timer import CycleTimer

from ..context import RolloutContext
from .core import RolloutStrategy, send_next_action

logger = logging.getLogger(__name__)


class BaseStrategy(RolloutStrategy):
    """Autonomous policy rollout with no data recording.

    All actions flow through the ``robot_action_processor`` pipeline
    before reaching the robot.
    """

    def setup(self, ctx: RolloutContext) -> None:
        """Initialise the inference engine."""
        self._init_engine(ctx)
        logger.info("Base strategy ready")

    def run(self, ctx: RolloutContext) -> None:
        """Run the autonomous control loop until shutdown or duration expires."""
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        interpolator = self._interpolator

        timer = CycleTimer(cfg.fps, interpolator.multiplier, records_data=False)

        start_time = time.perf_counter()
        engine.resume()
        logger.info("Base strategy control loop started")

        try:
            while not ctx.runtime.shutdown_event.is_set():
                timer.tick(new_cycle=interpolator.needs_new_action())

                if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                    logger.info("Duration limit reached (%.0fs)", cfg.duration)
                    break

                with timer.section("observe"):
                    obs = robot.get_observation()
                with timer.section("process_obs"):
                    obs_processed = self._process_observation_and_notify(ctx.processors, obs)

                if self._handle_warmup(cfg.use_torch_compile, timer):
                    continue

                action_dict = send_next_action(obs_processed, obs, ctx, interpolator, timer)
                with timer.section("telemetry"):
                    self._log_telemetry(obs_processed, action_dict, ctx.runtime)

                # Service the text-query channel at the end of the tick: /vqa
                # answers and the /autosteer sequencer both advance here.  A sync
                # backend generates here (text generation is far slower than a
                # control tick, so it must not sit inside the action path), and
                # every backend hands ready answers over here so observers fire on
                # this thread.  No-op when nothing is queued.
                with timer.section("query"):
                    served_query = engine.pump_query(obs_processed)
                if served_query:
                    # A tick that generated text inline is *expected* to overrun;
                    # warning on it would teach operators to dismiss the one signal
                    # the interactive session's log muting lets through.  restart()
                    # zeroes the closed-group counter, so the group wait() is about
                    # to close is treated as a start-up group and exempted from
                    # judging — the same idiom used after save_episode and after
                    # DAgger's handover ramps.  Keep this conditional: an
                    # unconditional restart() would exempt *every* group and
                    # silently disable the slow-loop warning for the whole run.
                    timer.restart()

                timer.wait()
        finally:
            logger.info("Base strategy control loop ended")
            timer.log_run_summary()

    def teardown(self, ctx: RolloutContext) -> None:
        """Disconnect hardware and stop inference."""
        self._teardown_hardware(
            ctx.hardware,
            return_to_initial_position=ctx.runtime.cfg.return_to_initial_position,
        )
        logger.info("Base strategy teardown complete")
