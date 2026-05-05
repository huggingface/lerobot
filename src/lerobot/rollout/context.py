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

"""Rollout context: shared state created once before strategy dispatch.

Grouped into five topical sub-contexts — :class:`RuntimeContext`,
:class:`HardwareContext`, :class:`PolicyContext`, :class:`ProcessorContext`,
and :class:`DatasetContext` — assembled into :class:`RolloutContext`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from threading import Event

import torch

from lerobot.configs import FeatureType
from lerobot.datasets import (
    LeRobotDataset,
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import (
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
    rename_stats,
)
from lerobot.processor.relative_action_processor import RelativeActionsProcessorStep
from lerobot.robots import make_robot_from_config
from lerobot.teleoperators import Teleoperator, make_teleoperator_from_config
from lerobot.utils.feature_utils import combine_feature_dicts, hw_to_dataset_features

from .configs import BaseStrategyConfig, DAggerStrategyConfig, RolloutConfig
from .inference import (
    InferenceEngine,
    RTCInferenceConfig,
    SyncInferenceConfig,
    create_inference_engine,
)
from .robot_wrapper import ThreadSafeRobot

logger = logging.getLogger(__name__)


def _resolve_action_key_order(
    policy_action_names: list[str] | None, dataset_action_names: list[str]
) -> list[str]:
    """Choose action name ordering for mapping policy tensor outputs to robot action dicts."""
    if not policy_action_names:
        return dataset_action_names
    policy_action_names = list(policy_action_names)
    if len(policy_action_names) != len(dataset_action_names):
        logger.warning(
            "policy.action_feature_names length (%d) != dataset action dim (%d); using dataset order",
            len(policy_action_names),
            len(dataset_action_names),
        )
        return dataset_action_names
    if set(dataset_action_names) != set(policy_action_names):
        logger.warning("policy.action_feature_names keys don't match dataset; using dataset order")
        return dataset_action_names
    return policy_action_names


# ---------------------------------------------------------------------------
# Sub-contexts
# ---------------------------------------------------------------------------


@dataclass
class RuntimeContext:
    """Runtime knobs shared with every strategy."""

    cfg: RolloutConfig
    shutdown_event: Event


@dataclass
class HardwareContext:
    """Connected hardware.

    The raw robot is available via ``robot_wrapper.inner`` when needed
    (e.g. for disconnect); strategies should otherwise go through the
    thread-safe wrapper.

    ``initial_position`` stores the robot's joint positions at connect
    time.  Strategies use it to return the robot to a safe pose before
    shutting down.
    """

    robot_wrapper: ThreadSafeRobot
    teleop: Teleoperator | None
    initial_position: dict | None = None


@dataclass
class PolicyContext:
    """Loaded policy and its inference engine."""

    policy: PreTrainedPolicy
    preprocessor: PolicyProcessorPipeline
    postprocessor: PolicyProcessorPipeline
    inference: InferenceEngine


@dataclass
class ProcessorContext:
    """Robot-side pipelines (run outside the policy)."""

    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation]


@dataclass
class DatasetContext:
    """Dataset and feature bookkeeping."""

    dataset: LeRobotDataset | None
    dataset_features: dict = field(default_factory=dict)
    hw_features: dict = field(default_factory=dict)
    ordered_action_keys: list[str] = field(default_factory=list)


@dataclass
class RolloutContext:
    """Bundle of sub-contexts passed to every rollout strategy.

    Built once by :func:`build_rollout_context` before strategy dispatch.
    """

    runtime: RuntimeContext
    hardware: HardwareContext
    policy: PolicyContext
    processors: ProcessorContext
    data: DatasetContext


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_rollout_context(
    cfg: RolloutConfig,
    shutdown_event: Event,
    teleop_action_processor: RobotProcessorPipeline | None = None,
    robot_action_processor: RobotProcessorPipeline | None = None,
    robot_observation_processor: RobotProcessorPipeline | None = None,
) -> RolloutContext:
    """Wire up policy, processors, hardware, dataset, and inference engine.

    The order is policy-first / hardware-last so a bad ``--policy.path``
    fails fast without touching the robot.
    """
    is_rtc = isinstance(cfg.inference, RTCInferenceConfig)

    # --- 1. Policy (heavy I/O, but no hardware yet) -------------------
    logger.info("Loading policy from '%s'...", cfg.policy.pretrained_path)
    policy_config = cfg.policy
    policy_class = get_policy_class(policy_config.type)

    if hasattr(policy_config, "compile_model"):
        policy_config.compile_model = cfg.use_torch_compile

    if policy_config.type == "vqbet" and cfg.device == "mps":
        raise NotImplementedError(
            "Current implementation of VQBeT does not support `mps` backend. "
            "Please use `cpu` or `cuda` backend."
        )

    if policy_config.use_peft:
        from peft import PeftConfig, PeftModel

        peft_path = policy_config.pretrained_path
        peft_config = PeftConfig.from_pretrained(peft_path)
        policy = policy_class.from_pretrained(
            pretrained_name_or_path=peft_config.base_model_name_or_path, config=policy_config
        )
        policy = PeftModel.from_pretrained(policy, peft_path, config=peft_config)
    else:
        policy = policy_class.from_pretrained(policy_config.pretrained_path, config=policy_config)

    if is_rtc:
        policy.config.rtc_config = cfg.inference.rtc
        if hasattr(policy, "init_rtc_processor"):
            policy.init_rtc_processor()

    policy = policy.to(cfg.device)
    policy.eval()
    logger.info("Policy loaded: type=%s, device=%s", policy_config.type, cfg.device)

    if cfg.use_torch_compile and policy.type not in ("pi0", "pi05"):
        try:
            if hasattr(torch, "compile"):
                compile_kwargs = {
                    "backend": cfg.torch_compile_backend,
                    "mode": cfg.torch_compile_mode,
                    "options": {"triton.cudagraphs": False},
                }
                policy.predict_action_chunk = torch.compile(policy.predict_action_chunk, **compile_kwargs)
                logger.info("torch.compile applied to predict_action_chunk")
        except Exception as e:
            logger.warning("Failed to apply torch.compile: %s", e)

    # --- 2. Robot-side processors (user-supplied or defaults) --------
    if (
        teleop_action_processor is None
        or robot_action_processor is None
        or robot_observation_processor is None
    ):
        _t, _r, _o = make_default_processors()
        teleop_action_processor = teleop_action_processor or _t
        robot_action_processor = robot_action_processor or _r
        robot_observation_processor = robot_observation_processor or _o

    # --- 3. Hardware (heaviest side-effect, deferred) -----------------
    logger.info("Connecting robot (%s)...", cfg.robot.type if cfg.robot else "?")
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    logger.info("Robot connected: %s", robot.name)

    # Store the initial joint positions so we can return to a safe pose on shutdown.
    initial_obs = robot.get_observation()
    initial_position = {k: v for k, v in initial_obs.items() if k.endswith(".pos")}
    logger.info("Captured initial robot position (%d keys)", len(initial_position))

    robot_wrapper = ThreadSafeRobot(robot)

    teleop = None
    if cfg.teleop is not None:
        logger.info("Connecting teleoperator (%s)...", cfg.teleop.type if cfg.teleop else "?")
        teleop = make_teleoperator_from_config(cfg.teleop)
        teleop.connect()
        logger.info("Teleoperator connected")

    # TODO(Steven): once Teleoperator motor-control methods are standardised
    # (``enable_torque`` / ``disable_torque`` / ``write_goal_positions``), gate
    # the DAgger strategy on their presence here and fail fast with a helpful
    # message instead of relying on the operator to pre-align the leader by
    # hand.  See :func:`DAggerStrategy._apply_transition` for the matching
    # disabled call sites.
    # if isinstance(cfg.strategy, DAggerStrategyConfig) and teleop is not None:
    #     required_teleop_methods = ("enable_torque", "disable_torque", "write_goal_positions")
    #     missing = [m for m in required_teleop_methods if not callable(getattr(teleop, m, None))]
    #     if missing:
    #         teleop.disconnect()
    #         raise ValueError(
    #             f"DAgger strategy requires a teleoperator with motor control methods "
    #             f"{required_teleop_methods}. '{type(teleop).__name__}' is missing: {missing}"
    #         )

    # --- 4. Features + action-key reconciliation ---------------------
    # TODO(Steven):Only ``.pos`` joint features are routed to the policy as state and as the
    # action target; velocity and torque channels (when present) are kept in
    # the raw observation but excluded from the policy-facing tensors.
    all_obs_features = robot.observation_features
    # ``observation_features`` values are either a tuple (camera shape) or the
    # ``float`` type itself used as a sentinel for scalar motor features —
    # see ``dict[str, type | tuple]`` annotation on ``Robot.observation_features``.
    observation_features_hw = {
        k: v
        for k, v in all_obs_features.items()
        if isinstance(v, tuple) or (v is float and k.endswith(".pos"))
    }
    action_features_hw = {k: v for k, v in robot.action_features.items() if k.endswith(".pos")}

    # The action side is always needed: sync inference reads action names from
    # ``dataset_features[ACTION]`` to map policy tensors back to robot actions.
    action_dataset_features = aggregate_pipeline_dataset_features(
        pipeline=teleop_action_processor,
        initial_features=create_initial_features(action=action_features_hw),
        use_videos=cfg.dataset.video if cfg.dataset else True,
    )
    # Observation-side aggregation is needed because of build_dataset_frame
    observation_dataset_features = aggregate_pipeline_dataset_features(
        pipeline=robot_observation_processor,
        initial_features=create_initial_features(observation=observation_features_hw),
        use_videos=cfg.dataset.video if cfg.dataset else True,
    )
    dataset_features = combine_feature_dicts(action_dataset_features, observation_dataset_features)
    hw_features = hw_to_dataset_features(observation_features_hw, "observation")
    raw_action_keys = list(action_features_hw.keys())
    policy_action_names = getattr(policy_config, "action_feature_names", None)
    ordered_action_keys = _resolve_action_key_order(
        list(policy_action_names) if policy_action_names else None,
        raw_action_keys,
    )

    # Validate visual features if no rename_map is active
    rename_map = cfg.rename_map
    if not rename_map:
        expected_visuals = {
            k for k, v in policy_config.input_features.items() if v.type == FeatureType.VISUAL
        }
        provided_visuals = {
            f"observation.images.{k}" for k, v in robot.observation_features.items() if isinstance(v, tuple)
        }
        policy_subset = expected_visuals.issubset(provided_visuals)
        hw_subset = provided_visuals.issubset(expected_visuals)
        if not (policy_subset or hw_subset):
            raise ValueError(
                f"Visual feature mismatch between policy and robot hardware.\n"
                f"Policy expects: {expected_visuals}\n"
                f"Robot provides: {provided_visuals}"
            )

    # --- 5. Dataset -------------
    dataset = None
    if cfg.dataset is not None and not isinstance(cfg.strategy, BaseStrategyConfig):
        logger.info("Setting up dataset (repo_id=%s)...", cfg.dataset.repo_id)
        if cfg.resume:
            dataset = LeRobotDataset.resume(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera
                * len(robot.cameras if hasattr(robot, "cameras") else []),
            )
        else:
            if isinstance(cfg.strategy, DAggerStrategyConfig):
                dataset_features["intervention"] = {
                    "dtype": "bool",
                    "shape": (1,),
                    "names": None,
                }

            repo_name = cfg.dataset.repo_id.split("/", 1)[-1]
            if not repo_name.startswith("rollout_"):
                raise ValueError(
                    "Dataset names for rollout must start with 'rollout_'. "
                    "Use --dataset.repo_id=<user>/rollout_<name> for policy deployment datasets."
                )
            cfg.dataset.stamp_repo_id()
            target_video_mb = getattr(cfg.strategy, "target_video_file_size_mb", None)
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera
                * len(robot.cameras if hasattr(robot, "cameras") else []),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
                video_files_size_in_mb=target_video_mb,
            )

    if dataset is not None:
        logger.info("Dataset ready: %s (%d existing episodes)", dataset.repo_id, dataset.num_episodes)

    # --- 6. Policy pre/post processors (needs dataset stats if any) ---
    dataset_stats = None
    if dataset is not None:
        dataset_stats = rename_stats(
            dataset.meta.stats,
            cfg.rename_map,
        )

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=dataset_stats,
        preprocessor_overrides={
            "device_processor": {"device": cfg.device},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )

    if isinstance(cfg.inference, SyncInferenceConfig) and any(
        isinstance(step, RelativeActionsProcessorStep) and step.enabled
        for step in getattr(preprocessor, "steps", ())
    ):
        raise NotImplementedError(
            "SyncInferenceEngine does not support policies with relative actions for now."
            "Use --inference.type=rtc or remove relative action processor steps from the policy pipeline."
        )

    # --- 7. Inference strategy (needs policy + pre/post + hardware) --
    logger.info(
        "Creating inference engine (type=%s)...",
        cfg.inference.type if hasattr(cfg.inference, "type") else "sync",
    )
    task_str = cfg.dataset.single_task if cfg.dataset else cfg.task
    inference_strategy = create_inference_engine(
        cfg.inference,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        robot_wrapper=robot_wrapper,
        hw_features=hw_features,
        dataset_features=dataset_features,
        ordered_action_keys=ordered_action_keys,
        task=task_str,
        fps=cfg.fps,
        device=cfg.device,
        use_torch_compile=cfg.use_torch_compile,
        compile_warmup_inferences=cfg.compile_warmup_inferences,
        shutdown_event=shutdown_event,
    )

    # --- 8. Assemble ---------------------------------------------------
    logger.info("Rollout context assembled successfully")
    return RolloutContext(
        runtime=RuntimeContext(cfg=cfg, shutdown_event=shutdown_event),
        hardware=HardwareContext(
            robot_wrapper=robot_wrapper, teleop=teleop, initial_position=initial_position
        ),
        policy=PolicyContext(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            inference=inference_strategy,
        ),
        processors=ProcessorContext(
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        ),
        data=DatasetContext(
            dataset=dataset,
            dataset_features=dataset_features,
            hw_features=hw_features,
            ordered_action_keys=ordered_action_keys,
        ),
    )
