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
from typing import ClassVar, Literal

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

    Use ``--strategy.type=<name>`` on the CLI to select a strategy.  The registry is
    open: a third-party package can register its own strategy and drive it from the
    same CLI — see ``docs/source/bring_your_own_rollout_strategies.mdx``.

    A strategy declares what the engine must arrange on its behalf through the
    capability ClassVars and hooks below, so that nothing outside the strategy needs
    to know its concrete type.  Anything expressed here works identically for
    built-in and third-party strategies.
    """

    # Whether the strategy honours the restartable-run() contract that
    # ``--interactive=true`` requires (see ``RolloutStrategy``).
    supports_interactive: ClassVar[bool] = False

    # How the strategy relates to a dataset:
    #   "none"     — records nothing; passing any ``--dataset.*`` flag is rejected.
    #   "optional" — records when ``--dataset.*`` is given (``ctx.data.dataset`` may be None).
    #   "required" — ``--dataset.repo_id`` is mandatory.
    # For "optional" and "required", ``build_rollout_context`` creates the dataset and
    # hands it over as ``ctx.data.dataset``.
    dataset_mode: ClassVar[Literal["none", "optional", "required"]] = "none"

    # Whether ``--teleop.type`` is mandatory (human-in-the-loop strategies).
    requires_teleop: ClassVar[bool] = False

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    def requires_streaming_encoding(self) -> bool:
        """Whether ``--dataset.streaming_encoding`` must be forced on.

        Return True when frames are written from inside the timed control loop:
        without streaming encoding the encode blocks the loop and the cadence
        collapses.  A method rather than a ClassVar because the answer can depend on
        the strategy's own fields — see :class:`DAggerStrategyConfig`.
        """
        return False

    def extra_dataset_features(self) -> dict[str, dict]:
        """Extra dataset columns this strategy records, merged into the dataset features.

        ``validate_frame`` rejects missing keys as well as extra ones, so every frame
        the strategy records must carry every key declared here.
        """
        return {}

    def resolve_defaults(self, dataset_cfg: DatasetRecordConfig | None) -> None:
        """Fill unset strategy fields from the dataset config, once, after validation.

        Called from ``RolloutConfig.__post_init__`` after the capability checks above
        have passed.  Raise ``ValueError`` for a field that cannot be resolved.
        """


@RolloutStrategyConfig.register_subclass("base")
@dataclass
class BaseStrategyConfig(RolloutStrategyConfig):
    """Autonomous rollout with no data recording."""

    supports_interactive: ClassVar[bool] = True


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
    """

    supports_interactive: ClassVar[bool] = True
    dataset_mode: ClassVar[Literal["none", "optional", "required"]] = "required"

    upload_every_n_episodes: int = 5
    # Target video file size in MB for episode rotation.  Episodes are
    # saved once the estimated video duration would exceed this limit.
    # Defaults to DEFAULT_VIDEO_FILE_SIZE_IN_MB when set to None.
    target_video_file_size_mb: int | None = None

    def requires_streaming_encoding(self) -> bool:
        """Always: frames are written from inside the control loop."""
        return True


@RolloutStrategyConfig.register_subclass("highlight")
@dataclass
class HighlightStrategyConfig(RolloutStrategyConfig):
    """Autonomous rollout with on-demand recording via ring buffer.

    A memory-bounded ring buffer continuously captures telemetry.  When
    the user presses the save key, the buffer contents are flushed to
    the dataset and live recording continues until the key is pressed
    again.
    """

    dataset_mode: ClassVar[Literal["none", "optional", "required"]] = "required"

    ring_buffer_seconds: float = 10.0
    ring_buffer_max_memory_mb: int = 1024
    save_key: str = "s"
    push_key: str = "h"

    def requires_streaming_encoding(self) -> bool:
        """Always: the ring buffer is flushed while the policy is still running."""
        return True


@dataclass
class DAggerKeyboardConfig:
    """Keyboard key bindings for DAgger controls.

    Keys are specified as single characters (e.g. ``"c"``, ``"h"``) or
    special key names (``"space"``).
    """

    pause_resume: str = "space"
    correction: str = "tab"
    upload: str = "enter"


@dataclass
class DAggerPedalConfig:
    """Foot pedal configuration for DAgger controls.

    Pedal codes are evdev key code strings (e.g. ``"KEY_A"``).
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
    """

    dataset_mode: ClassVar[Literal["none", "optional", "required"]] = "required"

    # This only applies if there are no teleop leaders specified.
    # When True (default), moves the robot back to the joint positions captured at startup.
    # Otherwise, leave the robot in its current position.
    reset_to_initial_position: bool = True

    # Whether to turn on or off the leader -> follower smooth handover behavior.
    # When False, fallback to follower -> leader handover.
    # Note that leader -> follower handover is only supported when the leader has `send_feedback` capability.
    smooth_leader_to_follower_handover: bool = True

    # Whether to turn on or off the smooth handover behavior at the start of the
    # reset phase: the leader is driven to the follower position (actuated
    # teleops, see `smooth_leader_to_follower_handover`), or the follower is
    # slid to the teleop pose (non-actuated teleops). Disable for clutch-style
    # teleoperators (e.g. VR controllers) that re-reference at the current robot
    # pose on engage: the handover is already continuous there, and the blocking
    # interpolation only delays the start of the reset phase.
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
    """

    # TODO(Steven): DAgger shouldn't require a dataset (user may want to just
    # rollout+intervene without recording), but for now we require it to simplify the
    # implementation.  Relaxing this to "optional" is all it takes on the config side.
    dataset_mode: ClassVar[Literal["none", "optional", "required"]] = "required"
    requires_teleop: ClassVar[bool] = True

    # Number of correction episodes to collect (corrections-only mode).
    # When None, falls back to ``--dataset.num_episodes``.
    num_episodes: int | None = None
    record_autonomous: bool = False
    upload_every_n_episodes: int = 5
    # Target video file size in MB for episode rotation (record_autonomous
    # mode only).  Defaults to DEFAULT_VIDEO_FILE_SIZE_IN_MB when None.
    target_video_file_size_mb: int | None = None
    # Whether to turn on or off the smooth handover behavior at phase transitions:
    # the leader is driven to the follower position on pause (teleops with
    # `send_feedback` capability), and the follower is slid to the teleop pose when
    # a correction starts (non-actuated teleops). Disable for clutch-style
    # teleoperators (e.g. VR controllers) that re-reference at the current robot
    # pose on engage: the handover is already continuous there, and the blocking
    # interpolation only delays the start of the correction.
    smooth_handover: bool = True
    input_device: str = "keyboard"
    keyboard: DAggerKeyboardConfig = field(default_factory=DAggerKeyboardConfig)
    pedal: DAggerPedalConfig = field(default_factory=DAggerPedalConfig)

    def __post_init__(self):
        if self.input_device not in ("keyboard", "pedal"):
            raise ValueError(f"DAgger input_device must be 'keyboard' or 'pedal', got '{self.input_device}'")

    def requires_streaming_encoding(self) -> bool:
        """Only when the autonomous phase is recorded too — corrections are saved between phases."""
        return self.record_autonomous

    def extra_dataset_features(self) -> dict[str, dict]:
        """Tag every frame with whether it came from a human correction."""
        return {"intervention": {"dtype": "bool", "shape": (1,), "names": None}}

    def resolve_defaults(self, dataset_cfg: DatasetRecordConfig | None) -> None:
        """Resolve ``num_episodes`` from the dataset config, and hint at streaming encoding."""
        if not self.record_autonomous and dataset_cfg is not None and not dataset_cfg.streaming_encoding:
            logger.info(
                "Streaming encoding is disabled for DAgger corrections-only mode. "
                "Consider enabling it for faster episode saving: "
                "--dataset.streaming_encoding=true --dataset.encoder_threads=2"
            )

        if self.num_episodes is not None:
            return
        if dataset_cfg is None:
            raise ValueError(
                "DAgger num_episodes must be set either via --strategy.num_episodes or --dataset.num_episodes"
            )
        self.num_episodes = dataset_cfg.num_episodes
        logger.info(
            "DAgger num_episodes not set — using --dataset.num_episodes=%d",
            self.num_episodes,
        )


# ---------------------------------------------------------------------------
# Top-level rollout config
# ---------------------------------------------------------------------------


@dataclass
class RolloutConfig:
    """Top-level configuration for the ``lerobot-rollout`` CLI.

    Combines hardware, policy, strategy, and runtime settings.  The
    ``__post_init__`` method performs fail-fast validation to reject
    invalid flag combinations early.
    """

    # Hardware
    robot: RobotConfig | None = None
    teleop: TeleoperatorConfig | None = None

    # Policy (loaded from --policy.path via __post_init__)
    policy: PreTrainedConfig | None = None

    # Strategy (polymorphic: --strategy.type=base|sentry|highlight|dagger|episodic, or
    # any name a third-party package registered on RolloutStrategyConfig)
    strategy: RolloutStrategyConfig = field(default_factory=BaseStrategyConfig)

    # Inference backend (polymorphic: --inference.type=sync|rtc)
    inference: InferenceEngineConfig = field(default_factory=SyncInferenceConfig)

    # Dataset (required or rejected according to the strategy's ``dataset_mode``)
    dataset: DatasetRecordConfig | None = None

    # Runtime
    fps: float = 30.0
    # Run time in seconds; 0 = infinite (24/7 mode).  In interactive mode this
    # bounds each /start segment, not the whole session.
    duration: float = 0.0
    # Control the rollout from stdin with chat-style commands (/start, /subtask,
    # /vqa, /autosteer, /reset, /stop) while hardware and policy stay warm.  The
    # robot does not move until /start, and logs below ERROR are muted for the
    # session's duration.
    interactive: bool = False
    # /autosteer: seconds of robot motion between two "what is the next subtask?"
    # queries, measured from the moment a subtask is applied.  Lower values
    # re-plan sooner but spend more of the loop generating text instead of acting.
    autosteer_interval_s: float = 10.0
    # Robot commands sent per policy action.  Values > 1 linearly interpolate
    # between consecutive policy actions for smoother motion: commands go to
    # the robot at ``fps × multiplier`` Hz while policy inference and dataset
    # recording stay at ``fps`` Hz.
    interpolation_multiplier: int = 1
    device: str | None = None
    task: str = ""
    display_data: bool = False
    # Visualization backend used when display_data is True: "rerun" or "foxglove".
    display_mode: str = "rerun"
    # For "rerun": IP of a remote server to send to. For "foxglove": interface to bind the WebSocket
    # server to (127.0.0.1 for local only, 0.0.0.0 for all interfaces).
    display_ip: str | None = None
    # For "rerun": port of the remote server. For "foxglove": port to bind the WebSocket server to.
    display_port: int | None = None
    # Whether to display compressed (JPEG) images instead of raw frames
    display_compressed_images: bool = False
    # Use vocal synthesis to read events
    play_sounds: bool = True
    resume: bool = False
    # Rename map for mapping robot/dataset observation keys to policy keys
    rename_map: dict[str, str] = field(default_factory=dict)

    # Hardware teardown
    # When True (default), smoothly interpolate the robot back to the joint
    # positions captured at startup before disconnecting.  Set to False to
    # leave the robot in its final achieved pose at shutdown.
    return_to_initial_position: bool = True

    # Torch compile
    use_torch_compile: bool = False
    torch_compile_backend: str = "inductor"
    torch_compile_mode: str = "default"
    compile_warmup_inferences: int = 2

    def __post_init__(self):
        """Validate config invariants and load the policy config from ``--policy.path``."""
        if self.interpolation_multiplier < 1:
            raise ValueError(f"interpolation_multiplier must be >= 1, got {self.interpolation_multiplier}")

        # --- Strategy-capability validation ---
        # Everything here reads the strategy's declarations rather than its concrete
        # type, so a third-party strategy is validated exactly like a built-in one.
        if self.strategy.requires_teleop and self.teleop is None:
            raise ValueError(f"{self.strategy.type} strategy requires --teleop.type to be set")

        if self.strategy.dataset_mode == "required" and (self.dataset is None or not self.dataset.repo_id):
            raise ValueError(f"{self.strategy.type} strategy requires --dataset.repo_id to be set")

        if self.strategy.dataset_mode == "none" and self.dataset is not None:
            recorders = ", ".join(
                sorted(
                    name
                    for name, choice_cls in RolloutStrategyConfig.get_known_choices().items()
                    if choice_cls.dataset_mode != "none"
                )
            )
            raise ValueError(
                f"{self.strategy.type} strategy does not record data. Use {recorders} for recording."
            )

        # Interactive mode calls strategy.run() once per segment, so only strategies
        # declaring ``supports_interactive`` may be driven by it.
        if self.interactive and not self.strategy.supports_interactive:
            supported = " or ".join(
                sorted(
                    name
                    for name, choice_cls in RolloutStrategyConfig.get_known_choices().items()
                    if choice_cls.supports_interactive
                )
            )
            raise ValueError(
                f"--interactive=true supports --strategy.type={supported} (got '{self.strategy.type}')."
            )

        if self.autosteer_interval_s < 0:
            raise ValueError(f"--autosteer_interval_s must be >= 0 (got {self.autosteer_interval_s}).")

        # A strategy that writes frames from inside the timed control loop cannot afford
        # a blocking encode: force streaming encoding on its behalf.
        if (
            self.dataset is not None
            and self.strategy.requires_streaming_encoding()
            and not self.dataset.streaming_encoding
        ):
            logger.warning("%s strategy forces streaming_encoding=True", self.strategy.type)
            self.dataset.streaming_encoding = True

        # Last: let the strategy fill any of its own fields left unset (e.g. DAgger's
        # num_episodes falling back to --dataset.num_episodes).
        self.strategy.resolve_defaults(self.dataset)

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
        return ["policy"]
