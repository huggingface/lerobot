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

import json
import logging
import time
from pathlib import Path

from lerobot.utils.robot_utils import precise_sleep

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
        if self.config.seed_start_position:
            self._seed_start_position(ctx)
        self._init_engine(ctx)
        logger.info("Base strategy ready")

    def _seed_start_position(self, ctx: RolloutContext) -> None:
        """Move the robot to the first-frame position of the training dataset."""
        policy_path = Path(ctx.runtime.cfg.policy.pretrained_path)
        train_config_path = policy_path / "train_config.json"
        if not train_config_path.exists():
            logger.warning("train_config.json not found at %s; skipping seed", policy_path)
            return

        with open(train_config_path) as f:
            train_cfg = json.load(f)
        repo_id = train_cfg.get("dataset", {}).get("repo_id")
        if not repo_id:
            logger.warning("No dataset.repo_id in train_config.json; skipping seed")
            return

        episode = self.config.seed_episode
        logger.info("Seeding start position from '%s' episode %d ...", repo_id, episode)

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset(repo_id, episodes=[episode])
        state = ds[0]["observation.state"]  # [state_dim]

        state_feature = ctx.data.dataset_features.get("observation.state")
        if state_feature is None or "names" not in state_feature:
            logger.warning("observation.state names not found in dataset features; skipping seed")
            return

        motor_names = state_feature["names"]
        if len(motor_names) != state.shape[0]:
            logger.warning(
                "Motor name count (%d) != state dim (%d); skipping seed",
                len(motor_names),
                state.shape[0],
            )
            return

        target = {name: state[i].item() for i, name in enumerate(motor_names)}
        logger.info("Moving to dataset start position: %s", {k: f"{v:.3f}" for k, v in target.items()})
        self._move_to_position(ctx.hardware.robot_wrapper, target)

    def run(self, ctx: RolloutContext) -> None:
        """Run the autonomous control loop until shutdown or duration expires."""
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        interpolator = self._interpolator

        control_interval = interpolator.get_control_interval(cfg.fps)

        start_time = time.perf_counter()
        engine.resume()
        logger.info("Base strategy control loop started")

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
            self._log_telemetry(obs_processed, action_dict, ctx.runtime)

            dt = time.perf_counter() - loop_start
            if (sleep_t := control_interval - dt) > 0:
                precise_sleep(sleep_t)
            else:
                logger.warning(
                    f"Record loop is running slower ({1 / dt:.1f} Hz) than the target FPS ({cfg.fps} Hz). Dataset frames might be dropped and robot control might be unstable. Common causes are: 1) Camera FPS not keeping up 2) Policy inference taking too long 3) CPU starvation"
                )

    def teardown(self, ctx: RolloutContext) -> None:
        """Disconnect hardware and stop inference."""
        self._teardown_hardware(
            ctx.hardware,
            return_to_initial_position=ctx.runtime.cfg.return_to_initial_position,
        )
        logger.info("Base strategy teardown complete")
