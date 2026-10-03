# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Descriptor-based rollout setup: no policy or checkpoint is loaded on the client."""

from __future__ import annotations

import logging
from threading import Event

from lerobot.datasets import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.inference.contracts import FeatureSpec
from lerobot.processor import RobotProcessorPipeline, make_default_processors
from lerobot.remote_inference.client import RemoteClient
from lerobot.robots import make_robot_from_config
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.feature_utils import combine_feature_dicts, hw_to_dataset_features

from .configs import RolloutConfig
from .context import (
    DatasetContext,
    HardwareContext,
    PolicyContext,
    ProcessorContext,
    RolloutContext,
    RuntimeContext,
    _align_to_checkpoint_order,
    _build_rollout_dataset,
)
from .inference import RemoteInferenceConfig
from .inference.remote import RemoteInferenceEngine
from .robot_wrapper import ThreadSafeRobot

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
    config = cfg.inference
    if not isinstance(config, RemoteInferenceConfig) or cfg.robot is None:
        raise ValueError("Remote rollout requires remote inference and robot configuration")
    if cfg.policy is not None or cfg.device is not None or cfg.use_torch_compile:
        raise ValueError("Remote rollout does not accept local policy/device/compile configuration")
    client = RemoteClient.connect(config)
    robot = None
    teleop = None
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

        default_teleop, default_action, default_observation = make_default_processors()
        teleop_action_processor = teleop_action_processor or default_teleop
        robot_action_processor = robot_action_processor or default_action
        robot_observation_processor = robot_observation_processor or default_observation

        robot = make_robot_from_config(cfg.robot)
        wrapper = ThreadSafeRobot(robot)
        wrapper.configure_position_hold()
        robot.connect()
        initial_obs = wrapper.get_observation()
        initial_position = {key: value for key, value in initial_obs.items() if key.endswith(".pos")}

        if cfg.teleop is not None:
            teleop = make_teleoperator_from_config(cfg.teleop)
            teleop.connect()

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
        use_videos = cfg.dataset.video if cfg.dataset else True
        features = combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=teleop_action_processor,
                initial_features=create_initial_features(action=action_hw),
                use_videos=use_videos,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=robot_observation_processor,
                initial_features=create_initial_features(observation=observation_hw),
                use_videos=use_videos,
            ),
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
        if dataset is not None:
            engine.configure_event_log(dataset.root / "inference_events" / f"{client.session_id}.jsonl")
        logger.info(
            "Remote rollout admitted: deployment=%s instance=%s mode=%s",
            config.deployment,
            descriptor["instance_id"],
            config.mode,
        )
        return RolloutContext(
            runtime=RuntimeContext(cfg=cfg, shutdown_event=shutdown_event),
            hardware=HardwareContext(wrapper, teleop, initial_position),
            policy=PolicyContext(None, None, None, engine),
            processors=ProcessorContext(
                teleop_action_processor, robot_action_processor, robot_observation_processor
            ),
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
        for name, resource in (("teleoperator", teleop), ("robot", robot)):
            try:
                if resource is not None and resource.is_connected:
                    resource.disconnect()
            except Exception:
                logger.exception("Could not close %s after rollout setup failed", name)
        raise
