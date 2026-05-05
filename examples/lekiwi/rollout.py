# !/usr/bin/env python

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

"""Run a trained policy on LeKiwi without recording (base rollout).

Uses the rollout engine's :class:`BaseStrategy` (autonomous execution,
no dataset) with :class:`SyncInferenceConfig` (inline policy call per
control tick).  For a CLI entry point with the same capabilities plus
recording, upload, and human-in-the-loop variants, see ``lerobot-rollout``.
"""

from lerobot.configs import PreTrainedConfig
from lerobot.robots.lekiwi import LeKiwiClientConfig
from lerobot.rollout import BaseStrategyConfig, RolloutConfig, build_rollout_context
from lerobot.rollout.inference import SyncInferenceConfig
from lerobot.rollout.strategies import BaseStrategy
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.utils import init_logging

FPS = 30
DURATION_SEC = 60
TASK_DESCRIPTION = "My task description"
HF_MODEL_ID = "<hf_username>/<model_repo_id>"


def main():
    init_logging()

    # Robot: LeKiwi client — make sure lekiwi_host is already running on the robot.
    robot_config = LeKiwiClientConfig(remote_ip="172.18.134.136", id="lekiwi")

    # Policy: load the pretrained config.  ``pretrained_path`` is read downstream
    # by ``build_rollout_context`` to reload the full model.
    policy_config = PreTrainedConfig.from_pretrained(HF_MODEL_ID)
    policy_config.pretrained_path = HF_MODEL_ID

    # Assemble the rollout config: base strategy (no recording) + sync inference.
    cfg = RolloutConfig(
        robot=robot_config,
        policy=policy_config,
        strategy=BaseStrategyConfig(),
        inference=SyncInferenceConfig(),
        fps=FPS,
        duration=DURATION_SEC,
        task=TASK_DESCRIPTION,
    )

    # Graceful Ctrl-C: the strategy loop exits when shutdown_event is set.
    signal_handler = ProcessSignalHandler(use_threads=True)

    # Build the context (connects robot, loads policy, wires the inference strategy).
    # No custom processors here — LeKiwi runs on raw joint features.
    ctx = build_rollout_context(cfg, signal_handler.shutdown_event)

    strategy = BaseStrategy(cfg.strategy)
    try:
        strategy.setup(ctx)
        strategy.run(ctx)
    finally:
        strategy.teardown(ctx)


if __name__ == "__main__":
    main()
