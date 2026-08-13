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

from lerobot.datasets.utils import DEFAULT_VIDEO_FILE_SIZE_IN_MB
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.cycle_timer import CycleTimer
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.utils import log_say

from ..configs import SentryStrategyConfig
from ..context import RolloutContext
from .core import (
    RolloutStrategy,
    estimate_max_episode_seconds,
    safe_push_to_hub,
    send_next_action,
)

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

    ``run()`` is restartable, as ``--interactive=true`` requires: each call records
    complete episodes plus one final partial one, and only ``teardown()`` finalizes
    the dataset.
    """

    config: SentryStrategyConfig

    def __init__(self, config: SentryStrategyConfig):
        super().__init__(config)
        self._push_executor: ThreadPoolExecutor | None = None
        self._pending_push: Future | None = None
        self._needs_push = Event()
        self._episode_lock = Lock()
        # Instance state, not run()-local, so the upload cadence survives segments.
        self._episodes_since_push = 0
        # Latched when save_episode fails mid-write: the dataset on disk may then
        # hold committed rows unreachable from metadata, so recording more into it
        # or pushing it would grow/upload corruption.
        self._dataset_poisoned = False

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
        if self._dataset_poisoned:
            raise RuntimeError(
                "Refusing to start a new segment: a previous save_episode failed mid-write, so "
                "the dataset on disk may be partially committed. Inspect it before recording more."
            )
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        dataset = ctx.data.dataset
        interpolator = self._interpolator
        features = ctx.data.dataset_features

        # Per-segment timer, never hoisted onto the instance (see ``RolloutStrategy.run``).
        timer = CycleTimer(cfg.fps, interpolator.multiplier, report=ctx.runtime.cadence_report)

        engine.resume()
        episode_duration_s = self._episode_duration_s

        start_time = time.perf_counter()
        episode_start = time.perf_counter()
        logger.info("Sentry recording started (episode_duration=%.0fs)", episode_duration_s)

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

                if action_dict is not None:
                    with timer.section("telemetry"):
                        self._log_telemetry(obs_processed, action_dict, ctx.runtime)
                    # Record once per interpolation cycle so the dataset cadence
                    # matches its declared fps; interpolated ticks only send
                    # commands to the robot.
                    if interpolator.emitted_policy_action:
                        with timer.section("record"):
                            obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
                            action_frame = build_dataset_frame(features, action_dict, prefix=ACTION)
                            # Label with ``dispatched_task``, the instruction that generated the action just
                            # sent (the live ``engine.task`` would mislabel actions still queued from the
                            # previous one); sound at any multiplier, since no ``get_action`` has run since
                            # the refill tick that produced this action.
                            frame = {**obs_frame, **action_frame, "task": engine.dispatched_task}
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
                    self._checked_save_episode(dataset)
                    logger.info(
                        "Episode saved (total: %d, elapsed: %.1fs)",
                        dataset.num_episodes,
                        elapsed,
                    )
                    # ``save_episode`` blocks for a good fraction of a second
                    # inside the timed loop body.  That is episode finalisation,
                    # not the steady-state cadence, so report the episode and then
                    # drop the partial group and the gap the save opened.
                    timer.log_episode_summary(f"episode {dataset.num_episodes}")
                    timer.restart()

                    self._register_saved_episode(dataset, cfg)

                    episode_start = time.perf_counter()

                # Service the text-query channel after the frame is recorded, so a
                # multi-second generate cannot land between this tick's observation
                # and its ``add_frame``; outside the guard above so starved ticks pump too.
                with timer.section("query"):
                    engine.pump_query(obs_processed)

                timer.wait()

        finally:
            logger.info("Sentry control loop ended")
            # Report before the tail save, which re-raises on a broken save.
            timer.log_run_summary()
            self._save_tail_episode(dataset, cfg)

    def _checked_save_episode(self, dataset) -> None:
        """``save_episode`` under the push lock; a failure poisons the dataset and re-raises.

        A failed ``save_episode`` is *not* recoverable by discarding the buffer:
        rows and counters are committed before the failure-prone steps (video
        encode, metadata commit), so the next segment would reuse the same episode
        index.  Hence the poison latch, which refuses further segments and pushes.
        """
        self._warn_if_push_in_flight()
        try:
            with self._episode_lock:
                dataset.save_episode()
        except Exception:
            self._dataset_poisoned = True
            with contextlib.suppress(Exception):
                dataset.clear_episode_buffer(delete_images=False)
            raise

    def _save_tail_episode(self, dataset, cfg) -> None:
        """Commit the segment's partial tail episode; fail loudly on real errors.

        Runs in ``run()``'s ``finally``.  Returns early on an already-poisoned
        dataset so the original error propagates, and on a segment that recorded
        nothing, so :meth:`_checked_save_episode` only ever fails on a broken save.
        """
        if self._dataset_poisoned:
            return
        if not dataset.has_pending_frames():
            logger.info("No frames pending at segment end — nothing to save")
            return
        logger.info("Saving the segment's final (partial) episode")
        self._checked_save_episode(dataset)
        self._register_saved_episode(dataset, cfg)

    def _register_saved_episode(self, dataset, cfg) -> None:
        """Post-save bookkeeping, shared by the rotation and tail-save sites.

        Tail episodes must count toward ``upload_every_n_episodes`` too, or a session
        of short segments would never background-push.
        """
        self._episodes_since_push += 1
        self._needs_push.set()
        log_say(f"Episode {dataset.num_episodes} saved", cfg.play_sounds)
        if self._episodes_since_push >= self.config.upload_every_n_episodes:
            self._background_push(dataset, cfg)
            self._episodes_since_push = 0

    def _warn_if_push_in_flight(self) -> None:
        """Warn before a save that must wait on a background upload.

        ``save_episode`` contends with a background Hub push for ``_episode_lock``,
        so on a slow uplink a ``/reset`` freezes the robot for minutes.
        """
        if self._pending_push is not None and not self._pending_push.done():
            logger.warning(
                "Waiting for an in-flight Hub upload to finish before saving the episode — "
                "the robot will hold position until it completes..."
            )

    def teardown(self, ctx: RolloutContext) -> None:
        """Disconnect hardware, then flush pending pushes and finalise the dataset."""
        logger.info("Stopping sentry recording")
        log_say("Stopping sentry recording", ctx.runtime.cfg.play_sounds)

        self._teardown(ctx)
        logger.info("Sentry strategy teardown complete")

    def _teardown_dataset(self, ctx: RolloutContext) -> None:
        play_sounds = ctx.runtime.cfg.play_sounds

        # Flush any queued/running push cleanly.
        if self._push_executor is not None:
            logger.info("Shutting down push executor (waiting for pending pushes)...")
            self._push_executor.shutdown(wait=True)
            self._push_executor = None

        if ctx.data.dataset is not None:
            if self._dataset_poisoned:
                logger.error(
                    "The dataset may be partially committed (a save_episode failed mid-write): "
                    "closing it without pushing. Inspect it before using or uploading it."
                )
            logger.info("Finalizing dataset...")
            ctx.data.dataset.finalize()
            if (
                not self._dataset_poisoned
                and self._needs_push.is_set()
                and ctx.runtime.cfg.dataset
                and ctx.runtime.cfg.dataset.push_to_hub
            ):
                logger.info("Pushing final dataset to hub...")
                if safe_push_to_hub(
                    ctx.data.dataset,
                    tags=ctx.runtime.cfg.dataset.tags,
                    private=ctx.runtime.cfg.dataset.private,
                ):
                    logger.info("Dataset uploaded to hub")
                    log_say("Dataset uploaded to hub", play_sounds)

    def _background_push(self, dataset, cfg) -> None:
        """Queue a Hub push on the single-worker executor.

        The executor's max_workers=1 guarantees at most one push runs at
        a time; submitted tasks are queued rather than dropped.
        """
        if self._push_executor is None:
            return
        if self._dataset_poisoned:
            logger.error("Skipping Hub push: a failed save_episode left the dataset possibly corrupt")
            return

        if self._pending_push is not None and not self._pending_push.done():
            logger.info("Previous push still in progress; queueing next")

        def _push():
            try:
                with self._episode_lock:
                    if self._dataset_poisoned:
                        # Poisoned while queued, after the submit-time check passed.
                        logger.error(
                            "Skipping queued Hub push: a failed save_episode left the dataset "
                            "possibly corrupt"
                        )
                        return
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
