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
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass, field
from threading import Event
from typing import TYPE_CHECKING

import torch

from lerobot.configs import FeatureType, PreTrainedConfig
from lerobot.datasets import (
    LeRobotDataset,
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.inference import (
    FeatureSpec,
    InferenceEngine,
    RemoteInferenceConfig,
    RTCInferenceConfig,
    create_inference_engine,
    supports_rtc_inference,
    validate_local_chunk_policy,
)
from lerobot.policies import PreTrainedPolicy, get_policy_class, make_pre_post_processors
from lerobot.policies.rtc.configuration_rtc import validate_trained_rtc_horizon
from lerobot.processor import (
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    bind_relative_anchor,
    make_default_processors,
    rename_stats,
)
from lerobot.remote_inference import RemoteClient, RemoteInferenceEngine
from lerobot.robots import Robot, make_robot_from_config
from lerobot.teleoperators import Teleoperator, make_teleoperator_from_config
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.feature_utils import combine_feature_dicts, hw_to_dataset_features
from lerobot.utils.import_utils import _peft_available, require_package

from .configs import RolloutConfig
from .robot_wrapper import ThreadSafeRobot

if TYPE_CHECKING or _peft_available:
    from peft import PeftConfig, PeftModel
else:
    PeftConfig = None
    PeftModel = None

logger = logging.getLogger(__name__)


def _wrap_predict_action_chunk_with_torch_compile(
    policy: PreTrainedPolicy,
    *,
    backend: str,
    mode: str,
) -> bool:
    """Install the JIT wrapper and report whether it was configured successfully.

    ``torch.compile`` compiles lazily on the first invocation, so success here
    does not guarantee that backend compilation will succeed during warm-up.
    """
    if not hasattr(torch, "compile"):
        logger.warning("torch.compile is not available in this PyTorch build")
        return False

    try:
        policy.predict_action_chunk = torch.compile(  # type: ignore[method-assign]
            policy.predict_action_chunk,
            backend=backend,
            mode=mode,
        )
    except Exception as exc:
        logger.warning("Failed to configure torch.compile: %s", exc)
        return False

    logger.info("torch.compile configured for predict_action_chunk")
    return True


def _validate_trained_rtc_rollout_config(policy_config, inference_config: RTCInferenceConfig) -> None:
    """Fail fast when rollout cannot retain every trained RTC prefix."""
    rtc = inference_config.rtc
    if not rtc.enabled or rtc.mode != "trained":
        return
    training_max_delay = int(getattr(policy_config, "rtc_training_max_delay", 0))
    validate_trained_rtc_horizon(
        rtc.execution_horizon, int(getattr(policy_config, "chunk_size", 0)), training_max_delay
    )
    if inference_config.queue_threshold < training_max_delay:
        raise ValueError(
            f"--inference.queue_threshold ({inference_config.queue_threshold}) must be at least the "
            f"checkpoint's rtc_training_max_delay ({training_max_delay})."
        )


def _align_to_checkpoint_order(
    features: dict[str, type | tuple], policy_action_names: list[str] | None, *, what: str
) -> dict[str, type | tuple]:
    """Order ``features`` so its motor entries follow the checkpoint's joint order.

    One rule for both sides: the policy emits and consumes tensors in the order it was
    trained on, so whichever dict is about to be flattened into a tensor has to match it.
    Only the scalar motor entries take part; anything else (camera shapes, on the state
    side) keeps its relative order at the end.

    A set mismatch is left alone rather than forced: extra ``.vel`` channels or an
    uncommanded base mean the two describe different things, and the robot's own order is
    the only meaningful one. ``what`` names the side being aligned, so the warning says
    which of the two reordered.
    """
    if not policy_action_names:
        return features

    motor_names = [name for name, feature in features.items() if not isinstance(feature, tuple)]
    if set(motor_names) != set(policy_action_names) or motor_names == policy_action_names:
        return features

    logger.warning(
        "Robot %s order %s differs from checkpoint joint order %s; reordering %s",
        what,
        motor_names,
        policy_action_names,
        what,
    )
    reordered = {name: features[name] for name in policy_action_names}
    reordered.update({name: feature for name, feature in features.items() if name not in reordered})
    return reordered


def _assert_state_matches_action_order(dataset_features: dict, ordered_action_keys: list[str]) -> None:
    """Reject a state layout that is a permutation of the action dispatch order.

    ``send_next_action`` labels the action tensor positionally with ``ordered_action_keys``,
    so a permuted ``observation.state`` commands every joint with another joint's value.
    Differing *sets* are legitimate (extra ``.vel`` channels, an uncommanded base) and pass.
    """
    state_ft = dataset_features.get(OBS_STATE)
    if state_ft is None or not ordered_action_keys:
        return
    state_names = list(state_ft.get("names") or [])
    if state_names == ordered_action_keys or set(state_names) != set(ordered_action_keys):
        return
    raise ValueError(
        f"observation.state order {state_names} is a permutation of the action dispatch order "
        f"{ordered_action_keys}; every joint would be commanded with another joint's value. "
        "Check policy.action_feature_names against the checkpoint's training dataset."
    )


# ---------------------------------------------------------------------------
# Sub-contexts
# ---------------------------------------------------------------------------


@dataclass
class RuntimeContext:
    """Runtime knobs shared with every strategy."""

    cfg: RolloutConfig
    shutdown_event: Event
    # Where the control loop's ``CycleTimer`` sends its cadence summaries; None
    # leaves them on ``logger.info``.  A strategy declaring ``supports_interactive``
    # must forward it to the timer it builds in ``run()``, since a session mutes
    # everything below ERROR.
    cadence_report: Callable[[str], None] | None = None


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
    initial_position: dict[str, float] | None = None


@dataclass
class PolicyContext:
    """Inference engine and optional local policy resources.

    Remote inference leaves ``policy``, ``preprocessor`` and ``postprocessor``
    as ``None``: those resources live on the server. Strategies use ``inference``
    for policy work and ``ProcessorContext`` for robot-side processing.
    """

    policy: PreTrainedPolicy | None
    preprocessor: PolicyProcessorPipeline | None
    postprocessor: PolicyProcessorPipeline | None
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


def _load_pretrained_policy(policy_config: PreTrainedConfig) -> PreTrainedPolicy:
    """Load policy weights, keeping adapter and base-model revisions independent."""
    pretrained_path = policy_config.pretrained_path
    if pretrained_path is None:
        raise ValueError("--policy.path is required for rollout")
    pretrained_revision = policy_config.pretrained_revision
    policy_class = get_policy_class(policy_config.type)

    if not policy_config.use_peft:
        return policy_class.from_pretrained(
            pretrained_path,
            config=policy_config,
            revision=pretrained_revision,
        )

    require_package("peft", extra="peft")

    peft_config = PeftConfig.from_pretrained(pretrained_path, revision=pretrained_revision)
    policy = policy_class.from_pretrained(
        pretrained_name_or_path=peft_config.base_model_name_or_path,
        config=policy_config,
        revision=peft_config.revision,
    )
    return PeftModel.from_pretrained(
        policy,
        pretrained_path,
        config=peft_config,
        revision=pretrained_revision,
    )


def _resolve_robot_processors(
    teleop_action: RobotProcessorPipeline | None,
    robot_action: RobotProcessorPipeline | None,
    robot_observation: RobotProcessorPipeline | None,
) -> ProcessorContext:
    if teleop_action is None or robot_action is None or robot_observation is None:
        default_teleop, default_action, default_observation = make_default_processors()
        teleop_action = teleop_action or default_teleop
        robot_action = robot_action or default_action
        robot_observation = robot_observation or default_observation
    return ProcessorContext(teleop_action, robot_action, robot_observation)


def _disconnect_setup_hardware(robot: Robot, teleop: Teleoperator | None) -> None:
    """Release each connected device without masking the original setup failure."""
    for name, resource in (("teleoperator", teleop), ("robot", robot)):
        try:
            if resource is not None and resource.is_connected:
                resource.disconnect()
        except Exception:
            logger.exception("Could not close %s after rollout setup failed", name)


def _connect_rollout_hardware(cfg: RolloutConfig, *, require_hold: bool = False) -> HardwareContext:
    """Connect devices and capture the initial pose after policy/descriptor validation."""
    if cfg.robot is None:
        raise ValueError("--robot.type is required for rollout")
    robot = make_robot_from_config(cfg.robot)
    wrapper = ThreadSafeRobot(robot)
    teleop = None
    try:
        if require_hold:
            wrapper.configure_position_hold()
        logger.info("Connecting robot (%s)...", cfg.robot.type)
        robot.connect()
        logger.info("Robot connected: %s", robot.name)
        initial_obs = wrapper.get_observation()
        initial_position = {key: value for key, value in initial_obs.items() if key.endswith(".pos")}
        logger.info("Captured initial robot position (%d keys)", len(initial_position))
        if cfg.teleop is not None:
            logger.info("Connecting teleoperator (%s)...", cfg.teleop.type)
            teleop = make_teleoperator_from_config(cfg.teleop)
            teleop.connect()
            logger.info("Teleoperator connected")
        return HardwareContext(wrapper, teleop, initial_position)
    except BaseException:
        _disconnect_setup_hardware(robot, teleop)
        raise


def _aggregate_rollout_features(
    processors: ProcessorContext,
    action: dict[str, type | tuple],
    observation: dict[str, type | tuple],
    *,
    use_videos: bool,
) -> dict:
    """Apply the same robot-side feature transformations for local and remote recording."""
    return combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=processors.teleop_action_processor,
            initial_features=create_initial_features(action=action),
            use_videos=use_videos,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=processors.robot_observation_processor,
            initial_features=create_initial_features(observation=observation),
            use_videos=use_videos,
        ),
    )


def _build_rollout_dataset(cfg: RolloutConfig, robot: Robot, dataset_features: dict) -> LeRobotDataset | None:
    """Create the local recording destination independently of policy placement."""
    dataset = None
    if cfg.dataset is not None:
        logger.info("Setting up dataset (repo_id=%s)...", cfg.dataset.repo_id)
        # Strategy-owned columns join the robot/policy features above the resume/create
        # split, so ``ctx.data.dataset_features`` describes the same schema on both paths.
        dataset_features.update(cfg.strategy.extra_dataset_features())
        if cfg.resume:
            dataset = LeRobotDataset.resume(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                rgb_encoder=cfg.dataset.rgb_encoder,
                depth_encoder=cfg.dataset.depth_encoder,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera
                * len(robot.cameras if hasattr(robot, "cameras") else []),
            )
        else:
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
                rgb_encoder=cfg.dataset.rgb_encoder,
                depth_encoder=cfg.dataset.depth_encoder,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
                video_files_size_in_mb=target_video_mb,
            )

    if dataset is not None:
        logger.info("Dataset ready: %s (%d existing episodes)", dataset.repo_id, dataset.num_episodes)

    return dataset


def build_rollout_context(
    cfg: RolloutConfig,
    shutdown_event: Event,
    teleop_action_processor: RobotProcessorPipeline | None = None,
    robot_action_processor: RobotProcessorPipeline | None = None,
    robot_observation_processor: RobotProcessorPipeline | None = None,
) -> RolloutContext:
    """Wire up policy, processors, hardware, dataset, and inference engine.

    The order is policy-first / hardware-last so a bad ``--policy.path``
    fails fast without touching the robot. A missing policy configuration raises
    ``ValueError`` before any policy access.
    """
    if isinstance(cfg.inference, RemoteInferenceConfig):
        return build_remote_rollout_context(
            cfg, shutdown_event, teleop_action_processor, robot_action_processor, robot_observation_processor
        )
    is_rtc = isinstance(cfg.inference, RTCInferenceConfig)

    # --- 1. Policy (heavy I/O, but no hardware yet) -------------------
    policy_config = cfg.policy
    if policy_config is None:
        raise ValueError("--policy.path is required for rollout")
    logger.info("Loading policy from '%s'...", policy_config.pretrained_path)
    # Policy constructors and custom processors must use the resolved rollout device too.
    policy_config.device = cfg.device

    if is_rtc:
        _validate_trained_rtc_rollout_config(policy_config, cfg.inference)

    if hasattr(policy_config, "compile_model"):
        policy_config.compile_model = cfg.use_torch_compile

    if policy_config.type == "vqbet" and cfg.device == "mps":
        raise NotImplementedError(
            "Current implementation of VQBeT does not support `mps` backend. "
            "Please use `cpu` or `cuda` backend."
        )

    policy = _load_pretrained_policy(policy_config)

    if is_rtc:
        validate_local_chunk_policy(policy)

    if is_rtc and cfg.inference.rtc.enabled:
        if not supports_rtc_inference(policy):
            raise ValueError(
                f"RTC inference is not supported by policy type '{policy_config.type}': "
                "the policy must implement RTC semantics and predict_action_chunk must accept "
                "inference_delay and prev_chunk_left_over. Use '--inference.type=sync' instead."
            )
        policy.config.rtc_config = cfg.inference.rtc
        if hasattr(policy, "init_rtc_processor"):
            policy.init_rtc_processor()

    policy = policy.to(cfg.device)
    policy.eval()
    logger.info("Policy loaded: type=%s, device=%s", policy_config.type, cfg.device)

    torch_compile_active = cfg.use_torch_compile
    if cfg.use_torch_compile and policy.type not in ("pi0", "pi05"):
        torch_compile_active = _wrap_predict_action_chunk_with_torch_compile(
            policy,
            backend=cfg.torch_compile_backend,
            mode=cfg.torch_compile_mode,
        )

    if cfg.use_torch_compile and not torch_compile_active:
        # RolloutConfig.__post_init__ reloads the policy configuration, so avoid
        # dataclasses.replace when carrying the effective state downstream.
        cfg = copy(cfg)
        cfg.use_torch_compile = False

    processors = _resolve_robot_processors(
        teleop_action_processor, robot_action_processor, robot_observation_processor
    )
    hardware = _connect_rollout_hardware(cfg)
    robot_wrapper = hardware.robot_wrapper
    robot = robot_wrapper.inner

    try:
        # Retain position and base-velocity channels for mobile manipulators, plus cameras.
        all_obs_features = robot.observation_features
        observation_features_hw: dict[str, type | tuple] = {
            k: v
            for k, v in all_obs_features.items()
            if isinstance(v, tuple) or (v is float and k.endswith((".pos", ".vel")))
        }
        policy_action_names = getattr(policy_config, "action_feature_names", None)
        checkpoint_order = list(policy_action_names) if policy_action_names else None
        observation_features_hw = _align_to_checkpoint_order(
            observation_features_hw, checkpoint_order, what="state"
        )
        action_features_hw = {k: v for k, v in robot.action_features.items() if k.endswith((".pos", ".vel"))}
        action_features_hw = _align_to_checkpoint_order(action_features_hw, checkpoint_order, what="action")

        dataset_features = _aggregate_rollout_features(
            processors,
            action_features_hw,
            observation_features_hw,
            use_videos=cfg.dataset.video if cfg.dataset else True,
        )
        hw_features = hw_to_dataset_features(observation_features_hw, "observation")
        # ``action_features_hw`` is already in checkpoint order, so it *is* the dispatch order.
        ordered_action_keys = list(action_features_hw)
        _assert_state_matches_action_order(dataset_features, ordered_action_keys)

        # Validate visual features if no rename_map is active
        rename_map = cfg.rename_map
        if not rename_map:
            expected_visuals = {
                k for k, v in (policy_config.input_features or {}).items() if v.type == FeatureType.VISUAL
            }
            provided_visuals = {
                f"observation.images.{k}"
                for k, v in robot.observation_features.items()
                if isinstance(v, tuple)
            }
            policy_subset = expected_visuals.issubset(provided_visuals)
            hw_subset = provided_visuals.issubset(expected_visuals)
            if not (policy_subset or hw_subset):
                raise ValueError(
                    f"Visual feature mismatch between policy and robot hardware.\n"
                    f"Policy expects: {expected_visuals}\n"
                    f"Robot provides: {provided_visuals}\n"
                    f"Use --rename_map to map camera names, e.g. "
                    f"""--rename_map='{{"observation.images.top": "observation.images.cam0"}}'"""
                )

        dataset = _build_rollout_dataset(cfg, robot, dataset_features)

        # Policy processors use recording statistics when available.
        dataset_stats = None
        if dataset is not None:
            dataset_stats = rename_stats(
                dataset.meta.stats,
                cfg.rename_map,
            )

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_config,
            pretrained_path=policy_config.pretrained_path,
            pretrained_revision=policy_config.pretrained_revision,
            dataset_stats=dataset_stats,
            preprocessor_overrides={
                "device_processor": {"device": cfg.device},
                "rename_observations_processor": {"rename_map": cfg.rename_map},
            },
        )

        # Keep relative-action anchors fixed while the local policy's queued chunk drains.
        bind_relative_anchor(policy, preprocessor)

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
            dataset_features=dataset_features,
            ordered_action_keys=ordered_action_keys,
            task=task_str,
            fps=cfg.fps,
            device=cfg.device,
            use_torch_compile=torch_compile_active,
            compile_warmup_inferences=cfg.compile_warmup_inferences,
            shutdown_event=shutdown_event,
        )

        logger.info("Rollout context assembled successfully")
        return RolloutContext(
            runtime=RuntimeContext(cfg=cfg, shutdown_event=shutdown_event),
            hardware=hardware,
            policy=PolicyContext(
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                inference=inference_strategy,
            ),
            processors=processors,
            data=DatasetContext(
                dataset=dataset,
                dataset_features=dataset_features,
                hw_features=hw_features,
                ordered_action_keys=ordered_action_keys,
            ),
        )
    except BaseException:
        _disconnect_setup_hardware(robot, hardware.teleop)
        raise


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
