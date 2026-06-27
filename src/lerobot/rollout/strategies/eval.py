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

"""Eval rollout strategy: multi-episode autonomous run that scores task success.

No data recording — use ``episodic`` instead (or alongside, in a separate
run) if you also want a dataset/video. Designed primarily for the MuJoCo
sim (``sim_so101``), where ``robot.check_success()`` can read privileged
state (e.g. whether the cube was lifted) that real hardware has no
equivalent for.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import statistics
import time

from lerobot.utils.robot_utils import precise_sleep

from ..configs import EvalStrategyConfig
from ..context import RolloutContext
from .core import RolloutStrategy, send_next_action

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class EpisodeResult:
    episode: int
    success: bool
    success_step: int | None
    num_steps: int


class EvalStrategy(RolloutStrategy):
    """Autonomous multi-episode rollout that tracks per-episode task success.

    Each episode runs the policy for up to ``episode_time_s`` seconds. If the
    robot implements ``check_success()`` (duck-typed — real hardware doesn't),
    it's polled every step; the episode is marked successful on the first
    True, and that step index is kept as "success step". A short hold phase
    between episodes returns the robot to its initial position.
    """

    config: EvalStrategyConfig

    def __init__(self, config: EvalStrategyConfig) -> None:
        super().__init__(config)
        self._results: list[EpisodeResult] = []
        self._warned_no_success_check = False

    def setup(self, ctx: RolloutContext) -> None:
        self._init_engine(ctx)
        logger.info("Eval strategy ready")

    def run(self, ctx: RolloutContext) -> None:
        robot = ctx.hardware.robot_wrapper

        for episode in range(self.config.num_episodes):
            if ctx.runtime.shutdown_event.is_set():
                break

            self._engine.reset()
            self._interpolator.reset()
            self._engine.resume()

            result = self._run_episode(ctx, robot, episode)
            self._results.append(result)
            logger.info(
                "Episode %d/%d: success=%s success_step=%s steps=%d",
                episode + 1,
                self.config.num_episodes,
                result.success,
                result.success_step,
                result.num_steps,
            )

            is_last = episode == self.config.num_episodes - 1
            if not is_last and self.config.reset_to_initial_position:
                self._return_to_initial_position(ctx.hardware, duration_s=1)
                precise_sleep(self.config.reset_time_s)

        self._log_summary()
        if self.config.output_path:
            self._write_summary(self.config.output_path)

    def _run_episode(self, ctx: RolloutContext, robot, episode: int) -> EpisodeResult:
        fps = ctx.runtime.cfg.fps
        interpolator = self._interpolator
        control_interval = interpolator.get_control_interval(fps)

        success = False
        success_step: int | None = None
        step = 0
        timestamp = 0.0
        start_t = time.perf_counter()

        while timestamp < self.config.episode_time_s:
            loop_start = time.perf_counter()

            if ctx.runtime.shutdown_event.is_set():
                break

            obs = robot.get_observation()
            obs_processed = self._process_observation_and_notify(ctx.processors, obs)

            if self._handle_warmup(ctx.runtime.cfg.use_torch_compile, loop_start, control_interval):
                continue

            action_dict = send_next_action(obs_processed, obs, ctx, interpolator)

            if action_dict is not None:
                step += 1
                self._log_telemetry(obs_processed, action_dict, ctx.runtime)
                if not success and self._check_success(robot):
                    success = True
                    success_step = step

            dt = time.perf_counter() - loop_start
            if (sleep_t := control_interval - dt) > 0:
                precise_sleep(sleep_t)
            timestamp = time.perf_counter() - start_t

        return EpisodeResult(episode=episode, success=success, success_step=success_step, num_steps=step)

    def _check_success(self, robot) -> bool:
        check = getattr(robot.inner, "check_success", None)
        if check is None:
            if not self._warned_no_success_check:
                logger.warning(
                    "Robot '%s' has no check_success() — success will be reported as False "
                    "for every episode.",
                    robot.name,
                )
                self._warned_no_success_check = True
            return False
        return bool(check())

    def _log_summary(self) -> None:
        n = len(self._results)
        if n == 0:
            logger.info("Eval summary: no episodes ran")
            return
        successes = [r for r in self._results if r.success]
        success_rate = len(successes) / n
        steps = [r.success_step for r in successes]
        mean_step = statistics.mean(steps) if steps else None
        logger.info(
            "Eval summary: success_rate=%.2f (%d/%d) mean_success_step=%s",
            success_rate,
            len(successes),
            n,
            f"{mean_step:.1f}" if mean_step is not None else "n/a",
        )

    def _write_summary(self, path: str) -> None:
        n = len(self._results)
        successes = [r for r in self._results if r.success]
        steps = [r.success_step for r in successes]
        summary = {
            "num_episodes": n,
            "success_rate": len(successes) / n if n else 0.0,
            "mean_success_step": statistics.mean(steps) if steps else None,
            "episodes": [dataclasses.asdict(r) for r in self._results],
        }
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Wrote eval summary to %s", path)

    def teardown(self, ctx: RolloutContext) -> None:
        self._teardown_hardware(
            ctx.hardware,
            return_to_initial_position=ctx.runtime.cfg.return_to_initial_position,
        )
        logger.info("Eval strategy teardown complete")
