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

"""Configuration dataclasses for the rollout deployment engine."""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field

import draccus

from lerobot.configs import PreTrainedConfig, parser
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.utils.device_utils import auto_select_torch_device, is_torch_device_available

from .inference import InferenceEngineConfig, SyncInferenceConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy configs (polymorphic dispatch via draccus ChoiceRegistry)
# ---------------------------------------------------------------------------


@dataclass
class RolloutStrategyConfig(draccus.ChoiceRegistry, abc.ABC):
    """Abstract base for rollout strategy configurations.

    Use ``--strategy.type=<name>`` on the CLI to select a strategy.
    """

    @property
    def type(self) -> str:
        """The registered name of this strategy (e.g. `"base"`, `"sentry"`, `"dagger"`)."""
        return self.get_choice_name(self.__class__)


@RolloutStrategyConfig.register_subclass("base")
@dataclass
class BaseStrategyConfig(RolloutStrategyConfig):
    """Autonomous rollout with no data recording."""

    pass


@RolloutStrategyConfig.register_subclass("sentry")
@dataclass
class SentryStrategyConfig(RolloutStrategyConfig):
    """Continuous autonomous rollout with always-on recording.

    Episode duration is derived from camera resolution, FPS, and
    ``target_video_file_size_mb`` so that each saved episode produces a
    video file that has crossed the target size.  This aligns episode
    boundaries with the dataset's video file chunking, so each
    ``push_to_hub`` call uploads complete video files rather than
    re-uploading a growing file that hasn't crossed the chunk boundary.

    Args:
        upload_every_n_episodes (`int`, *optional*, defaults to 5):
            Push the dataset to the Hub after every N saved episodes.
        target_video_file_size_mb (`int | None`, *optional*):
            Target video file size in MB for episode rotation. Episodes are saved once the estimated
            video duration would exceed this limit. Defaults to `DEFAULT_VIDEO_FILE_SIZE_IN_MB` when
            `None`.
    """

    upload_every_n_episodes: int = 5
    target_video_file_size_mb: int | None = None


@RolloutStrategyConfig.register_subclass("highlight")
@dataclass
class HighlightStrategyConfig(RolloutStrategyConfig):
    """Autonomous rollout with on-demand recording via ring buffer.

    A memory-bounded ring buffer continuously captures telemetry.  When
    the user presses the save key, the buffer contents are flushed to
    the dataset and live recording continues until the key is pressed
    again.

    Args:
        ring_buffer_seconds (`float`, *optional*, defaults to 10.0):
            Duration, in seconds, of telemetry kept in the ring buffer before it's overwritten.
        ring_buffer_max_memory_mb (`int`, *optional*, defaults to 1024):
            Hard memory cap, in MiB, for the ring buffer. Frames are evicted early if this is reached
            before `ring_buffer_seconds` of telemetry.
        save_key (`str`, *optional*, defaults to `"s"`):
            Keyboard key that flushes the ring buffer and starts (or ends) live recording.
        push_key (`str`, *optional*, defaults to `"h"`):
            Keyboard key that requests an on-demand push of the dataset to the Hub.
    """

    ring_buffer_seconds: float = 10.0
    ring_buffer_max_memory_mb: int = 1024
    save_key: str = "s"
    push_key: str = "h"


@dataclass
class DAggerKeyboardConfig:
    """Keyboard key bindings for DAgger controls.

    Keys are specified as single characters (e.g. ``"c"``, ``"h"``) or
    special key names (``"space"``).

    Args:
        pause_resume (`str`, *optional*, defaults to `"space"`):
            Key that toggles policy execution on/off.
        correction (`str`, *optional*, defaults to `"tab"`):
            Key that toggles human correction recording.
        upload (`str`, *optional*, defaults to `"enter"`):
            Key that pushes the dataset to the Hub on demand (corrections-only mode).
    """

    pause_resume: str = "space"
    correction: str = "tab"
    upload: str = "enter"


@dataclass
class DAggerPedalConfig:
    """Foot pedal configuration for DAgger controls.

    Pedal codes are evdev key code strings (e.g. ``"KEY_A"``).

    Args:
        device_path (`str`, *optional*, defaults to `"/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd"`):
            evdev device path of the foot pedal.
        pause_resume (`str`, *optional*, defaults to `"KEY_A"`):
            evdev key code that toggles policy execution on/off.
        correction (`str`, *optional*, defaults to `"KEY_B"`):
            evdev key code that toggles human correction recording.
        upload (`str`, *optional*, defaults to `"KEY_C"`):
            evdev key code that pushes the dataset to the Hub on demand (corrections-only mode).
    """

    device_path: str = "/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd"
    pause_resume: str = "KEY_A"
    correction: str = "KEY_B"
    upload: str = "KEY_C"


@RolloutStrategyConfig.register_subclass("episodic")
@dataclass
class EpisodicStrategyConfig(RolloutStrategyConfig):
    """Episode-oriented recording that mirrors the behavior of ``lerobot-record``.

    Records ``dataset.num_episodes`` episodes of maximum ``dataset.episode_time_s`` each.
    After each episode, runs ``dataset.reset_time_s`` seconds of reset time.

    Keyboard controls:
        Right arrow  — end current episode or reset phase early
        Left arrow   — discard current episode and re-record
        Escape       — stop recording session

    In between episodes:
    - if there is no teleop leader, the robot is held at its initial joint positions captured at startup.
    - else, the robot is moved smoothly to the position of the teleop leader.

    Args:
        reset_to_initial_position (`bool`, *optional*, defaults to `True`):
            Only applies when there is no teleop leader. When `True`, moves the robot back to the
            joint positions captured at startup during the reset phase. Otherwise, leaves the robot in
            its current position.
        smooth_leader_to_follower_handover (`bool`, *optional*, defaults to `True`):
            Whether to turn on or off the leader -> follower smooth handover behavior. When `False`,
            falls back to follower -> leader handover. Leader -> follower handover is only supported
            when the leader has `send_feedback` capability.
        smooth_handover (`bool`, *optional*, defaults to `True`):
            Whether to turn on or off the smooth handover behavior at the start of the reset phase: the
            leader is driven to the follower position (actuated teleops, see
            `smooth_leader_to_follower_handover`), or the follower is slid to the teleop pose
            (non-actuated teleops). Disable for clutch-style teleoperators (e.g. VR controllers) that
            re-reference at the current robot pose on engage: the handover is already continuous
            there, and the blocking interpolation only delays the start of the reset phase.
    """

    reset_to_initial_position: bool = True
    smooth_leader_to_follower_handover: bool = True
    smooth_handover: bool = True


@RolloutStrategyConfig.register_subclass("dagger")
@dataclass
class DAggerStrategyConfig(RolloutStrategyConfig):
    """Human-in-the-loop data collection (DAgger / RaC).

    Alternates between autonomous policy execution and human intervention.
    Intervention frames are tagged with ``intervention=True``.

    Input is controlled via either a keyboard or foot pedal, selected by
    ``input_device``.  Each device exposes three actions:

    1. **pause_resume** — toggle policy execution on/off.
    2. **correction** — toggle human correction recording.
    3. **upload** — push dataset to hub on demand (corrections-only mode).

    When ``record_autonomous=False`` (default) only human-correction windows
    are recorded — each correction becomes its own episode.  Set to ``True``
    to record both autonomous and correction frames with size-based episode
    rotation (same as Sentry) and background uploading.  ``push_to_hub`` is
    blocked while a correction is in progress.

    Args:
        num_episodes (`int | None`, *optional*):
            Number of correction episodes to collect (corrections-only mode). When `None`, falls back
            to `--dataset.num_episodes`.
        record_autonomous (`bool`, *optional*, defaults to `False`):
            When `False`, only human-correction windows are recorded, each becoming its own episode.
            When `True`, both autonomous and correction frames are recorded with size-based episode
            rotation (same as Sentry) and background uploading.
        upload_every_n_episodes (`int`, *optional*, defaults to 5):
            Push the dataset to the Hub after every N saved episodes (`record_autonomous=True` mode).
        target_video_file_size_mb (`int | None`, *optional*):
            Target video file size in MB for episode rotation (`record_autonomous=True` mode only).
            Defaults to `DEFAULT_VIDEO_FILE_SIZE_IN_MB` when `None`.
        smooth_handover (`bool`, *optional*, defaults to `True`):
            Whether to turn on or off the smooth handover behavior at phase transitions: the leader is
            driven to the follower position on pause (teleops with `send_feedback` capability), and
            the follower is slid to the teleop pose when a correction starts (non-actuated teleops).
            Disable for clutch-style teleoperators (e.g. VR controllers) that re-reference at the
            current robot pose on engage: the handover is already continuous there, and the blocking
            interpolation only delays the start of the correction.
        input_device (`str`, *optional*, defaults to `"keyboard"`):
            Input device used for the pause_resume/correction/upload controls. One of `"keyboard"` or
            `"pedal"`.
        keyboard (`DAggerKeyboardConfig`, *optional*):
            Keyboard key bindings, used when `input_device="keyboard"`.
        pedal (`DAggerPedalConfig`, *optional*):
            Foot pedal configuration, used when `input_device="pedal"`.

    Raises:
        ValueError: If `input_device` is not `"keyboard"` or `"pedal"`.
    """

    num_episodes: int | None = None
    record_autonomous: bool = False
    upload_every_n_episodes: int = 5
    target_video_file_size_mb: int | None = None
    smooth_handover: bool = True
    input_device: str = "keyboard"
    keyboard: DAggerKeyboardConfig = field(default_factory=DAggerKeyboardConfig)
    pedal: DAggerPedalConfig = field(default_factory=DAggerPedalConfig)

    def __post_init__(self):
        """Validate that `input_device` is a supported value.

        Raises:
            ValueError: If `input_device` is not `"keyboard"` or `"pedal"`.
        """
        if self.input_device not in ("keyboard", "pedal"):
            raise ValueError(f"DAgger input_device must be 'keyboard' or 'pedal', got '{self.input_device}'")


# ---------------------------------------------------------------------------
# Top-level rollout config
# ---------------------------------------------------------------------------


@dataclass
class RolloutConfig:
    """Top-level configuration for the ``lerobot-rollout`` CLI.

    Combines hardware, policy, strategy, and runtime settings.  The
    ``__post_init__`` method performs fail-fast validation to reject
    invalid flag combinations early.

    Args:
        robot (`RobotConfig | None`, *optional*):
            Robot hardware configuration. Required — validated in `__post_init__`.
        teleop (`TeleoperatorConfig | None`, *optional*):
            Teleoperator hardware configuration. Required by the `dagger` strategy.
        policy (`PreTrainedConfig | None`, *optional*):
            Loaded automatically from `--policy.path` during `__post_init__`; do not set directly.
        strategy (`RolloutStrategyConfig`, *optional*, defaults to `BaseStrategyConfig()`):
            Polymorphic rollout strategy config, selected via `--strategy.type=base|sentry|highlight|dagger|episodic`.
        inference (`InferenceEngineConfig`, *optional*, defaults to `SyncInferenceConfig()`):
            Polymorphic inference backend config, selected via `--inference.type=sync|rtc`.
        dataset (`DatasetRecordConfig | None`, *optional*):
            Dataset recording configuration. Required for the `sentry`, `highlight`, `dagger`, and
            `episodic` strategies; must be `None` for `base`.
        fps (`float`, *optional*, defaults to 30.0):
            Control loop frequency, in Hz.
        duration (`float`, *optional*, defaults to 0.0):
            Maximum rollout duration, in seconds. `0` means run indefinitely (24/7 mode).
        interpolation_multiplier (`int`, *optional*, defaults to 1):
            Number of interpolated control ticks generated per policy inference.
        device (`str | None`, *optional*):
            Torch device to run the policy on. Resolved from the policy config (or auto-selected) in
            `__post_init__` when unset or unavailable.
        task (`str`, *optional*, defaults to `""`):
            Task description propagated to (or from) `dataset.single_task` in `__post_init__`.
        display_data (`bool`, *optional*, defaults to `False`):
            Whether to stream observation/action telemetry to a visualization backend.
        display_mode (`str`, *optional*, defaults to `"rerun"`):
            Visualization backend used when `display_data` is `True`: `"rerun"` or `"foxglove"`.
        display_ip (`str | None`, *optional*):
            For `"rerun"`: IP of a remote server to send to. For `"foxglove"`: interface to bind the
            WebSocket server to (`127.0.0.1` for local only, `0.0.0.0` for all interfaces).
        display_port (`int | None`, *optional*):
            For `"rerun"`: port of the remote server. For `"foxglove"`: port to bind the WebSocket
            server to.
        display_compressed_images (`bool`, *optional*, defaults to `False`):
            Whether to display compressed (JPEG) images instead of raw frames.
        play_sounds (`bool`, *optional*, defaults to `True`):
            Whether to use vocal synthesis to read out session events.
        resume (`bool`, *optional*, defaults to `False`):
            Whether to resume recording into an existing dataset instead of creating a new one.
        rename_map (`dict[str, str]`, *optional*):
            Mapping of robot/dataset observation keys to the policy's expected feature keys.
        return_to_initial_position (`bool`, *optional*, defaults to `True`):
            When `True`, smoothly interpolates the robot back to the joint positions captured at
            startup before disconnecting. Set to `False` to leave the robot in its final achieved
            pose at shutdown.
        use_torch_compile (`bool`, *optional*, defaults to `False`):
            Whether to wrap the policy's `predict_action_chunk` with `torch.compile`.
        torch_compile_backend (`str`, *optional*, defaults to `"inductor"`):
            Backend passed to `torch.compile`.
        torch_compile_mode (`str`, *optional*, defaults to `"default"`):
            Mode passed to `torch.compile`.
        compile_warmup_inferences (`int`, *optional*, defaults to 2):
            Number of warmup inferences run before `torch.compile`-backed inference is considered
            ready.

    Raises:
        ValueError: If a required flag combination is missing (e.g. `--robot.type`, `--policy.path`,
            `--teleop.type` for DAgger, `--dataset.repo_id` for a recording strategy) or if the
            strategy/dataset combination is invalid (e.g. a dataset passed to the `base` strategy).
    """

    # Hardware
    robot: RobotConfig | None = None
    teleop: TeleoperatorConfig | None = None

    # Policy (loaded from --policy.path via __post_init__)
    policy: PreTrainedConfig | None = None

    # Strategy (polymorphic: --strategy.type=base|sentry|highlight|dagger)
    strategy: RolloutStrategyConfig = field(default_factory=BaseStrategyConfig)

    # Inference backend (polymorphic: --inference.type=sync|rtc)
    inference: InferenceEngineConfig = field(default_factory=SyncInferenceConfig)

    # Dataset (required for sentry, highlight, dagger; None for base)
    dataset: DatasetRecordConfig | None = None

    # Runtime
    fps: float = 30.0
    duration: float = 0.0  # 0 = infinite (24/7 mode)
    interpolation_multiplier: int = 1
    device: str | None = None
    task: str = ""
    display_data: bool = False
    display_mode: str = "rerun"
    display_ip: str | None = None
    display_port: int | None = None
    display_compressed_images: bool = False
    play_sounds: bool = True
    resume: bool = False
    rename_map: dict[str, str] = field(default_factory=dict)

    # Hardware teardown
    return_to_initial_position: bool = True

    # Torch compile
    use_torch_compile: bool = False
    torch_compile_backend: str = "inductor"
    torch_compile_mode: str = "default"
    compile_warmup_inferences: int = 2

    def __post_init__(self):
        """Validate config invariants and load the policy config from ``--policy.path``.

        Raises:
            ValueError: If a required flag combination is missing or the strategy/dataset combination
                is invalid.
        """
        # --- Strategy-specific validation ---
        if isinstance(self.strategy, DAggerStrategyConfig) and self.teleop is None:
            raise ValueError("DAgger strategy requires --teleop.type to be set")

        # TODO(Steven): DAgger shouldn't require a dataset (user may want to just rollout+intervene without recording), but for now we require it to simplify the implementation.
        needs_dataset = isinstance(
            self.strategy,
            (
                SentryStrategyConfig,
                HighlightStrategyConfig,
                DAggerStrategyConfig,
                EpisodicStrategyConfig,
            ),
        )
        if needs_dataset and (self.dataset is None or not self.dataset.repo_id):
            raise ValueError(f"{self.strategy.type} strategy requires --dataset.repo_id to be set")

        if isinstance(self.strategy, BaseStrategyConfig) and self.dataset is not None:
            raise ValueError(
                "Base strategy does not record data. Use sentry, highlight, or dagger for recording."
            )

        # Sentry MUST use streaming encoding to avoid disk I/O blocking the control loop
        if (
            isinstance(self.strategy, SentryStrategyConfig)
            and self.dataset is not None
            and not self.dataset.streaming_encoding
        ):
            logger.warning("Sentry mode forces streaming_encoding=True")
            self.dataset.streaming_encoding = True

        # Highlight writes frames while the policy is still running, so streaming is mandatory.
        if (
            isinstance(self.strategy, HighlightStrategyConfig)
            and self.dataset is not None
            and not self.dataset.streaming_encoding
        ):
            logger.warning("Highlight mode forces streaming_encoding=True")
            self.dataset.streaming_encoding = True

        # DAgger: streaming is mandatory only when the autonomous phase is also recorded.
        if isinstance(self.strategy, DAggerStrategyConfig) and self.dataset is not None:
            if self.strategy.record_autonomous and not self.dataset.streaming_encoding:
                logger.warning("DAgger with record_autonomous=True forces streaming_encoding=True")
                self.dataset.streaming_encoding = True
            elif not self.strategy.record_autonomous and not self.dataset.streaming_encoding:
                logger.info(
                    "Streaming encoding is disabled for DAgger corrections-only mode. "
                    "Consider enabling it for faster episode saving: "
                    "--dataset.streaming_encoding=true --dataset.encoder_threads=2"
                )

        # DAgger: resolve num_episodes from dataset config when not explicitly set.
        if isinstance(self.strategy, DAggerStrategyConfig) and self.strategy.num_episodes is None:
            if self.dataset is not None:
                self.strategy.num_episodes = self.dataset.num_episodes
                logger.info(
                    "DAgger num_episodes not set — using --dataset.num_episodes=%d",
                    self.strategy.num_episodes,
                )
            else:
                raise ValueError(
                    "DAgger num_episodes must be set either via --strategy.num_episodes or --dataset.num_episodes"
                )

        # --- Policy loading ---
        if self.robot is None:
            raise ValueError("--robot.type is required for rollout")

        policy_path = parser.get_path_arg("policy")
        if policy_path:
            yaml_overrides = parser.get_yaml_overrides("policy")
            cli_overrides = parser.get_cli_overrides("policy") or []
            policy_overrides = yaml_overrides + cli_overrides
            pretrained_revision = parser.parse_arg("pretrained_revision", cli_overrides)
            if pretrained_revision is None:
                pretrained_revision = parser.parse_arg("pretrained_revision", yaml_overrides)
            self.policy = PreTrainedConfig.from_pretrained(
                policy_path,
                revision=pretrained_revision,
                cli_overrides=policy_overrides,
            )
            self.policy.pretrained_path = policy_path
        if self.policy is None:
            raise ValueError("--policy.path is required for rollout")

        # --- Task resolution ---
        # When any --dataset.* flag is passed, draccus creates a DatasetRecordConfig with single_task="".
        # If the user set the task via the top-level --task flag, propagate it so that all
        # downstream consumers (inference engine, dataset frame builders) see it.
        if self.dataset is not None and not self.dataset.single_task and self.task:
            logger.info("Propagating top-level task '%s' to dataset config", self.task)
            self.dataset.single_task = self.task
        elif self.dataset is not None and self.dataset.single_task and not self.task:
            logger.info("Propagating dataset single_task '%s' to top-level task", self.dataset.single_task)
            self.task = self.dataset.single_task

        # --- Device resolution ---
        # Resolve device from the policy config when not explicitly set so all
        # components (policy.to, preprocessor, inference engine) use the same
        # device string instead of inconsistent fallbacks.
        if self.device is None or not is_torch_device_available(self.device):
            resolved = self.policy.device
            if resolved:
                self.device = resolved
                logger.info("Resolved device from policy config: %s", self.device)
            else:
                self.device = auto_select_torch_device().type
                logger.info("No policy config to resolve device from; auto-selected device: %s", self.device)

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """Fields draccus resolves as pretrained-checkpoint paths (i.e. `--policy.path`)."""
        return ["policy"]
