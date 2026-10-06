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

"""Descriptor-based rollout setup: no policy or checkpoint is loaded on the client."""

from __future__ import annotations

import logging
from threading import Event

from lerobot.inference import FeatureSpec, RemoteInferenceConfig
from lerobot.processor import RobotProcessorPipeline
from lerobot.remote_inference import RemoteClient, RemoteInferenceEngine
from lerobot.utils.feature_utils import hw_to_dataset_features

from .configs import RolloutConfig
from .context import (
    DatasetContext,
    HardwareContext,
    PolicyContext,
    RolloutContext,
    RuntimeContext,
    _aggregate_rollout_features,
    _align_to_checkpoint_order,
    _build_rollout_dataset,
    _connect_rollout_hardware,
    _disconnect_setup_hardware,
    _resolve_robot_processors,
)

logger = logging.getLogger(__name__)


def _feature_spec(name: str, feature: dict, semantics: str) -> FeatureSpec:
    image = feature["dtype"] in {"video", "image"}
    return FeatureSpec(
        name=name,
        shape=tuple(feature["shape"]),
        dtype="uint8" if image else feature["dtype"],
        kind="rgb" if image else "tensor",
        names=() if image else tuple(feature.get("names") or ()),
        semantics=semantics,
    )


def build_remote_rollout_context(
    cfg: RolloutConfig,
    shutdown_event: Event,
    teleop_action_processor: RobotProcessorPipeline | None = None,
    robot_action_processor: RobotProcessorPipeline | None = None,
    robot_observation_processor: RobotProcessorPipeline | None = None,
) -> RolloutContext:
    """Admit a descriptor-based rollout without loading local policy resources.

    Failed setup closes acquired resources; successful setup transfers ownership
    to strategy teardown. Recording uses the ordinary local dataset schema.
    """
    config = cfg.inference
    if not isinstance(config, RemoteInferenceConfig) or cfg.robot is None:
        raise ValueError("Remote rollout requires remote inference and robot configuration")
    if cfg.policy is not None or cfg.device is not None or cfg.use_torch_compile:
        raise ValueError("Remote rollout does not accept local policy/device/compile configuration")
    client = RemoteClient.connect(config)
    hardware: HardwareContext | None = None
    try:
        descriptor = client.descriptor
        task = cfg.dataset.single_task if cfg.dataset else cfg.task
        if len(task) > descriptor["limits"]["max_input_chars"]:
            raise ValueError("Initial instruction exceeds the deployed text limit")
        capabilities = descriptor["capabilities"]
        expected = tuple(FeatureSpec(**feature) for feature in capabilities["features"])
        action = FeatureSpec(**capabilities["action_feature"])
        if not action.names:
            raise ValueError("Remote rollout requires explicit ordered canonical action component names")

        processors = _resolve_robot_processors(
            teleop_action_processor, robot_action_processor, robot_observation_processor
        )
        hardware = _connect_rollout_hardware(cfg, require_hold=True)
        wrapper = hardware.robot_wrapper
        robot = wrapper.inner

        observation_hw: dict[str, type | tuple] = {
            key: value
            for key, value in robot.observation_features.items()
            if isinstance(value, tuple) or value is float
        }
        expected_state = next((feature for feature in expected if feature.name == "observation.state"), None)
        observation_hw = _align_to_checkpoint_order(
            observation_hw,
            list(expected_state.names) if expected_state and expected_state.names else None,
            what="state",
        )
        action_hw = _align_to_checkpoint_order(robot.action_features, list(action.names), what="action")
        if list(action_hw) != list(action.names):
            raise ValueError("Robot action component names differ from the deployment contract")
        features = _aggregate_rollout_features(
            processors,
            action_hw,
            observation_hw,
            use_videos=cfg.dataset.video if cfg.dataset else True,
        )
        mapped = {
            cfg.rename_map.get(name, name): feature
            for name, feature in features.items()
            if name.startswith("observation.")
        }
        if len(mapped) != sum(name.startswith("observation.") for name in features):
            raise ValueError("Observation feature mapping is not one-to-one")
        expected_names = {feature.name for feature in expected}
        if not expected_names.issubset(mapped):
            raise ValueError(
                f"Remote observation features differ: expected {expected_names}, got {set(mapped)}"
            )
        extra_features = set(mapped) - expected_names
        if extra_features:
            logger.info("Observations retained locally for recording only: %s", sorted(extra_features))
        client.admit(
            features=tuple(
                _feature_spec(feature.name, mapped[feature.name], feature.semantics) for feature in expected
            ),
            action_feature=_feature_spec("action", features["action"], action.semantics),
            semantics=config.semantics,
            action_interval=1.0 / cfg.fps,
            mode=config.mode,
        )
        dataset = _build_rollout_dataset(cfg, robot, features)
        engine = RemoteInferenceEngine(
            client=client,
            config=config,
            dataset_features=features,
            rename_map=cfg.rename_map,
            robot_wrapper=wrapper,
            task=task,
            shutdown_event=shutdown_event,
        )
        logger.info(
            "Remote rollout admitted: deployment=%s instance=%s mode=%s",
            config.deployment,
            descriptor["instance_id"],
            config.mode,
        )
        return RolloutContext(
            runtime=RuntimeContext(cfg=cfg, shutdown_event=shutdown_event),
            hardware=hardware,
            policy=PolicyContext(None, None, None, engine),
            processors=processors,
            data=DatasetContext(
                dataset, features, hw_to_dataset_features(observation_hw, "observation"), list(action_hw)
            ),
        )
    except BaseException:
        # Each cleanup is independent; preserve the setup failure even if the
        # server disappeared or one connected device cannot disconnect.
        try:
            client.close()
        except Exception:
            logger.exception("Could not close remote client after rollout setup failed")
        if hardware is not None:
            _disconnect_setup_hardware(hardware.robot_wrapper.inner, hardware.teleop)
        raise
