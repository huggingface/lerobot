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

"""Highlight Reel strategy: on-demand recording via ring buffer."""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event as ThreadingEvent, Lock

from lerobot.common.control_utils import is_headless
from lerobot.datasets import VideoEncodingManager
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.import_utils import _pynput_available, require_package
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

from ..configs import HighlightStrategyConfig
from ..context import RolloutContext
from ..ring_buffer import RolloutRingBuffer
from .core import RolloutStrategy, safe_push_to_hub, send_next_action

PYNPUT_AVAILABLE = _pynput_available
keyboard = None
if PYNPUT_AVAILABLE:
    try:
        if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
            logging.info("No DISPLAY set. Skipping pynput import.")
            PYNPUT_AVAILABLE = False
        else:
            from pynput import keyboard
    except Exception as e:
        PYNPUT_AVAILABLE = False
        logging.info(f"Could not import pynput: {e}")

logger = logging.getLogger(__name__)


class HighlightStrategy(RolloutStrategy):
    """Autonomous rollout with on-demand recording via ring buffer.

    The robot runs autonomously while a memory-bounded ring buffer
    captures continuous telemetry.  When the user presses the save key:

    1. The ring buffer is flushed to the dataset (last *Z* seconds).
    2. Live recording continues until the save key is pressed again.
    3. The episode is saved and the ring buffer resumes capturing.

    Requires ``streaming_encoding=True`` (enforced in config validation)
    so that ``dataset.add_frame`` is a non-blocking queue put — flushing
    the entire ring buffer in one tick must not stall the control loop.
    """

    config: HighlightStrategyConfig

    def __init__(self, config: HighlightStrategyConfig):
        super().__init__(config)
        require_package("pynput", extra="pynput-dep")
        self._ring: RolloutRingBuffer | None = None
        self._listener = None
        self._save_requested = ThreadingEvent()
        self._recording_live = ThreadingEvent()
        self._push_requested = ThreadingEvent()
        self._push_executor: ThreadPoolExecutor | None = None
        self._pending_push: Future | None = None
        self._episode_lock = Lock()

    def setup(self, ctx: RolloutContext) -> None:
        """Initialise the inference engine, ring buffer, and keyboard listener."""
        self._init_engine(ctx)

        self._ring = RolloutRingBuffer(
            max_seconds=self.config.ring_buffer_seconds,
            max_memory_mb=self.config.ring_buffer_max_memory_mb,
            fps=ctx.runtime.cfg.fps,
        )

        self._push_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="highlight-push")
        logger.info(
            "Ring buffer initialized (max_seconds=%.0f, max_memory=%.0fMB)",
            self.config.ring_buffer_seconds,
            self.config.ring_buffer_max_memory_mb,
        )
        self._setup_keyboard(ctx.runtime.shutdown_event)
        logger.info(
            "Highlight strategy ready (buffer=%.0fs, save='%s', push='%s')",
            self.config.ring_buffer_seconds,
            self.config.save_key,
            self.config.push_key,
        )

    def run(self, ctx: RolloutContext) -> None:
        """Run the autonomous loop, buffering frames and recording on demand."""
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        dataset = ctx.data.dataset
        ring = self._ring
        interpolator = self._interpolator
        features = ctx.data.dataset_features

        control_interval = interpolator.get_control_interval(cfg.fps)

        engine.resume()
        play_sounds = cfg.play_sounds

        start_time = time.perf_counter()
        task_str = cfg.dataset.single_task if cfg.dataset else cfg.task
        logger.info("Highlight strategy recording started (press '%s' to save)", self.config.save_key)

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

                        # NOTE: ``is_set()`` then ``clear()`` is not atomic
                        # against the keyboard thread setting the flag again
                        # in between — but that is benign: we lose at most one
                        # toggle, processed on the next iteration.
                        if self._save_requested.is_set():
                            self._save_requested.clear()
                            if not self._recording_live.is_set():
                                logger.info(
                                    "Flushing ring buffer (%d frames) + starting live recording",
                                    len(ring),
                                )
                                for buffered_frame in ring.drain():
                                    dataset.add_frame(buffered_frame)
                                self._recording_live.set()
                            else:
                                dataset.add_frame(frame)
                                with self._episode_lock:
                                    dataset.save_episode()
                                logger.info("Episode saved (total: %d)", dataset.num_episodes)
                                log_say(
                                    f"Episode {dataset.num_episodes} saved",
                                    play_sounds,
                                )
                                self._recording_live.clear()
                                continue  # frame already consumed — skip ring.append

                        if self._push_requested.is_set():
                            self._push_requested.clear()
                            logger.info("Push requested by user")
                            self._background_push(dataset, cfg)

                        if self._recording_live.is_set():
                            dataset.add_frame(frame)
                        else:
                            ring.append(frame)

                    dt = time.perf_counter() - loop_start
                    if (sleep_t := control_interval - dt) > 0:
                        precise_sleep(sleep_t)
                    else:
                        logger.warning(
                            f"Record loop is running slower ({1 / dt:.1f} Hz) than the target FPS ({cfg.fps} Hz). Dataset frames might be dropped and robot control might be unstable. Common causes are: 1) Camera FPS not keeping up 2) Policy inference taking too long 3) CPU starvation"
                        )

            finally:
                logger.info("Highlight control loop ended")
                if self._recording_live.is_set():
                    logger.info("Saving in-progress live episode")
                    with contextlib.suppress(Exception), self._episode_lock:
                        dataset.save_episode()

    def teardown(self, ctx: RolloutContext) -> None:
        """Stop listeners, finalise the dataset, and disconnect hardware."""
        play_sounds = ctx.runtime.cfg.play_sounds
        logger.info("Stopping highlight recording")
        log_say("Stopping highlight recording", play_sounds)

        if self._listener is not None:
            logger.info("Stopping keyboard listener")
            self._listener.stop()

        if self._push_executor is not None:
            logger.info("Shutting down push executor (waiting for pending pushes)...")
            self._push_executor.shutdown(wait=True)
            self._push_executor = None

        if ctx.data.dataset is not None:
            logger.info("Finalizing dataset...")
            ctx.data.dataset.finalize()
            if ctx.runtime.cfg.dataset and ctx.runtime.cfg.dataset.push_to_hub:
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
        logger.info("Highlight strategy teardown complete")

    def _setup_keyboard(self, shutdown_event: ThreadingEvent) -> None:
        """Set up keyboard listener for save and push keys."""
        if is_headless():
            logger.warning("Headless environment — highlight keys unavailable")
            return

        try:
            save_key = self.config.save_key
            push_key = self.config.push_key

            def on_press(key):
                with contextlib.suppress(Exception):
                    if hasattr(key, "char") and key.char == save_key:
                        self._save_requested.set()
                    elif hasattr(key, "char") and key.char == push_key:
                        self._push_requested.set()
                    elif key == keyboard.Key.esc:
                        self._save_requested.clear()
                        shutdown_event.set()

            self._listener = keyboard.Listener(on_press=on_press)
            self._listener.start()
            logger.info("Keyboard listener started (save='%s', push='%s', ESC=stop)", save_key, push_key)
        except ImportError:
            logger.warning("pynput not available — keyboard listener disabled")

    def _background_push(self, dataset, cfg) -> None:
        """Queue a Hub push on the single-worker executor."""
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
                        logger.info("Background push to hub complete")
            except Exception as e:
                logger.error("Background push failed: %s", e)

        self._pending_push = self._push_executor.submit(_push)
        logger.info("Background push task submitted")
