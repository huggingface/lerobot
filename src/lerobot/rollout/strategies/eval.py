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

Recording is optional: pass ``--dataset.repo_id=...`` (must start with
``rollout_``, same convention as ``episodic``) to also save a video/dataset
of every episode alongside the success metrics. Without a dataset config,
no frames are kept. Designed primarily for the MuJoCo sim (``sim_so101``),
where ``robot.check_success()`` can read privileged state (e.g. whether the
cube was lifted) that real hardware has no equivalent for.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import statistics
import time

from lerobot.datasets import VideoEncodingManager
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.robot_utils import precise_sleep

from ..configs import EvalStrategyConfig
from ..context import RolloutContext
from .core import RolloutStrategy, safe_push_to_hub, send_next_action

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
        dataset = ctx.data.dataset
        cfg = ctx.runtime.cfg
        single_task = (cfg.dataset.single_task if cfg.dataset else "") or cfg.task

        encoding_ctx = VideoEncodingManager(dataset) if dataset is not None else contextlib.nullcontext()
        with encoding_ctx:
            try:
                for episode in range(self.config.num_episodes):
                    if ctx.runtime.shutdown_event.is_set():
                        break

                    self._engine.reset()
                    self._interpolator.reset()
                    self._engine.resume()

                    result = self._run_episode(ctx, robot, episode, dataset, single_task)
                    self._results.append(result)
                    logger.info(
                        "Episode %d/%d: success=%s success_step=%s steps=%d",
                        episode + 1,
                        self.config.num_episodes,
                        result.success,
                        result.success_step,
                        result.num_steps,
                    )

                    if dataset is not None:
                        dataset.save_episode()

                    is_last = episode == self.config.num_episodes - 1
                    if not is_last and self.config.reset_to_initial_position:
                        self._return_to_initial_position(ctx.hardware, duration_s=1)
                        precise_sleep(self.config.reset_time_s)
            finally:
                # Safety net: persist a partially-recorded episode left by an
                # unexpected exception or KeyboardInterrupt instead of dropping it.
                if dataset is not None:
                    with contextlib.suppress(Exception):
                        dataset.save_episode()

        self._log_summary()
        if self.config.output_path:
            self._write_summary(self.config.output_path)

    def _run_episode(
        self, ctx: RolloutContext, robot, episode: int, dataset, single_task: str
    ) -> EpisodeResult:
        fps = ctx.runtime.cfg.fps
        features = ctx.data.dataset_features
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
                if dataset is not None:
                    obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
                    action_frame = build_dataset_frame(features, action_dict, prefix=ACTION)
                    dataset.add_frame({**obs_frame, **action_frame, "task": single_task})
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
        cfg = ctx.runtime.cfg
        dataset = ctx.data.dataset

        if dataset is not None:
            logger.info("Finalizing dataset...")
            dataset.finalize()
            if (
                cfg.dataset is not None
                and cfg.dataset.push_to_hub
                and safe_push_to_hub(dataset, tags=cfg.dataset.tags, private=cfg.dataset.private)
            ):
                logger.info("Dataset uploaded to hub")

        self._teardown_hardware(
            ctx.hardware,
            return_to_initial_position=cfg.return_to_initial_position,
        )
        logger.info("Eval strategy teardown complete")
