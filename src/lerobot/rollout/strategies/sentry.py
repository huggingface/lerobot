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

"""Sentry rollout strategy: continuous autonomous recording with auto-upload."""

from __future__ import annotations

import contextlib
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event, Lock

from lerobot.datasets import VideoEncodingManager
from lerobot.datasets.utils import DEFAULT_VIDEO_FILE_SIZE_IN_MB
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

from ..configs import SentryStrategyConfig
from ..context import RolloutContext
from .core import RolloutStrategy, estimate_max_episode_seconds, safe_push_to_hub, send_next_action

logger = logging.getLogger(__name__)


class SentryStrategy(RolloutStrategy):
    """Continuous autonomous rollout with always-on recording.

    Episode duration is derived from camera resolution, FPS, and
    ``DEFAULT_VIDEO_FILE_SIZE_IN_MB`` so that each saved episode
    produces a video file that has crossed the chunk-size boundary.
    This keeps ``push_to_hub`` efficient — it uploads complete video
    files rather than re-uploading a still-growing one.

    The dataset is pushed to the Hub via a bounded single-worker executor
    so no push is ever silently dropped and exactly one push runs at a
    time.

    Policy state (hidden state, RTC queue) intentionally persists across
    episode boundaries — Sentry slices one continuous rollout, the robot
    does not reset between slices.

    Requires ``streaming_encoding=True`` (enforced in config validation)
    to prevent disk I/O from blocking the control loop.
    """

    config: SentryStrategyConfig

    def __init__(self, config: SentryStrategyConfig):
        super().__init__(config)
        self._push_executor: ThreadPoolExecutor | None = None
        self._pending_push: Future | None = None
        self._needs_push = Event()
        self._episode_lock = Lock()

    def setup(self, ctx: RolloutContext) -> None:
        """Initialise the inference engine and background push executor."""
        self._init_engine(ctx)
        self._push_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sentry-push")
        target_mb = self.config.target_video_file_size_mb or DEFAULT_VIDEO_FILE_SIZE_IN_MB
        self._episode_duration_s = estimate_max_episode_seconds(
            ctx.data.dataset_features, ctx.runtime.cfg.fps, target_size_mb=target_mb
        )
        logger.info(
            "Sentry strategy ready (episode_duration=%.0fs, upload_every=%d eps)",
            self._episode_duration_s,
            self.config.upload_every_n_episodes,
        )

    def run(self, ctx: RolloutContext) -> None:
        """Run the continuous recording loop with automatic episode rotation."""
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        dataset = ctx.data.dataset
        interpolator = self._interpolator
        features = ctx.data.dataset_features

        control_interval = interpolator.get_control_interval(cfg.fps)

        engine.resume()
        play_sounds = cfg.play_sounds
        episode_duration_s = self._episode_duration_s

        start_time = time.perf_counter()
        episode_start = time.perf_counter()
        episodes_since_push = 0
        task_str = cfg.dataset.single_task if cfg.dataset else cfg.task
        logger.info("Sentry recording started (episode_duration=%.0fs)", episode_duration_s)

        with VideoEncodingManager(dataset):
            try:
                while not ctx.runtime.shutdown_event.is_set():
                    loop_start = time.perf_counter()

                    if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                        logger.info("Duration limit reached (%.0fs)", cfg.duration)
                        break

                    obs = robot.get_observation()
                    obs_processed = self._process_observation_and_notify(ctx.processors, obs)

                    if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                        continue

                    action_dict = send_next_action(obs_processed, obs, ctx, interpolator)

                    if action_dict is not None:
                        self._log_telemetry(obs_processed, action_dict, ctx.runtime)
                        obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
                        action_frame = build_dataset_frame(features, action_dict, prefix=ACTION)
                        frame = {**obs_frame, **action_frame, "task": task_str}
                        # ``add_frame`` writes to the in-progress episode buffer; the
                        # background pusher only ever touches *finalised* episode
                        # artifacts on disk.  The two operate on disjoint state, so
                        # ``add_frame`` does not need ``_episode_lock``.
                        dataset.add_frame(frame)

                    # Episode rotation derived from video file-size target.
                    # The duration is a conservative estimate so the actual
                    # video has crossed DEFAULT_VIDEO_FILE_SIZE_IN_MB by now,
                    # keeping push_to_hub efficient (uploads complete files).
                    elapsed = time.perf_counter() - episode_start
                    if elapsed >= episode_duration_s:
                        # ``save_episode`` finalises the in-progress episode and
                        # flushes it to disk; ``_episode_lock`` serialises this with
                        # ``push_to_hub`` (run in the background executor) so the
                        # pusher never reads a half-written episode.
                        with self._episode_lock:
                            dataset.save_episode()
                        episodes_since_push += 1
                        self._needs_push.set()
                        logger.info(
                            "Episode saved (total: %d, elapsed: %.1fs)",
                            dataset.num_episodes,
                            elapsed,
                        )
                        log_say(f"Episode {dataset.num_episodes} saved", play_sounds)

                        if episodes_since_push >= self.config.upload_every_n_episodes:
                            self._background_push(dataset, cfg)
                            episodes_since_push = 0

                        episode_start = time.perf_counter()

                    dt = time.perf_counter() - loop_start
                    if (sleep_t := control_interval - dt) > 0:
                        precise_sleep(sleep_t)
                    else:
                        logger.warning(
                            f"Record loop is running slower ({1 / dt:.1f} Hz) than the target FPS ({cfg.fps} Hz). Dataset frames might be dropped and robot control might be unstable. Common causes are: 1) Camera FPS not keeping up 2) Policy inference taking too long 3) CPU starvation"
                        )

            finally:
                logger.info("Sentry control loop ended — saving final episode")
                with contextlib.suppress(Exception):
                    with self._episode_lock:
                        dataset.save_episode()
                    self._needs_push.set()

    def teardown(self, ctx: RolloutContext) -> None:
        """Flush pending pushes, finalise the dataset, and disconnect hardware."""
        play_sounds = ctx.runtime.cfg.play_sounds
        logger.info("Stopping sentry recording")
        log_say("Stopping sentry recording", play_sounds)

        # Flush any queued/running push cleanly.
        if self._push_executor is not None:
            logger.info("Shutting down push executor (waiting for pending pushes)...")
            self._push_executor.shutdown(wait=True)
            self._push_executor = None

        if ctx.data.dataset is not None:
            logger.info("Finalizing dataset...")
            ctx.data.dataset.finalize()
            if self._needs_push.is_set() and ctx.runtime.cfg.dataset and ctx.runtime.cfg.dataset.push_to_hub:
                logger.info("Pushing final dataset to hub...")
                if safe_push_to_hub(
                    ctx.data.dataset,
                    tags=ctx.runtime.cfg.dataset.tags,
                    private=ctx.runtime.cfg.dataset.private,
                ):
                    logger.info("Dataset uploaded to hub")
                    log_say("Dataset uploaded to hub", play_sounds)

        self._teardown_hardware(
            ctx.hardware,
            return_to_initial_position=ctx.runtime.cfg.return_to_initial_position,
        )
        logger.info("Sentry strategy teardown complete")

    def _background_push(self, dataset, cfg) -> None:
        """Queue a Hub push on the single-worker executor.

        The executor's max_workers=1 guarantees at most one push runs at
        a time; submitted tasks are queued rather than dropped.
        """
        if self._push_executor is None:
            return

        if self._pending_push is not None and not self._pending_push.done():
            logger.info("Previous push still in progress; queueing next")

        def _push():
            try:
                with self._episode_lock:
                    if safe_push_to_hub(
                        dataset,
                        tags=cfg.dataset.tags if cfg.dataset else None,
                        private=cfg.dataset.private if cfg.dataset else False,
                    ):
                        self._needs_push.clear()
                        logger.info("Background push to hub complete")
            except Exception as e:
                logger.error("Background push failed: %s", e)

        self._pending_push = self._push_executor.submit(_push)
        logger.info("Background push task submitted")
