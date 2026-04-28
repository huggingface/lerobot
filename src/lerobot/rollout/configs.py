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
    """

    upload_every_n_episodes: int = 5
    # Target video file size in MB for episode rotation.  Episodes are
    # saved once the estimated video duration would exceed this limit.
    # Defaults to DEFAULT_VIDEO_FILE_SIZE_IN_MB when set to None.
    target_video_file_size_mb: int | None = None


@RolloutStrategyConfig.register_subclass("highlight")
@dataclass
class HighlightStrategyConfig(RolloutStrategyConfig):
    """Autonomous rollout with on-demand recording via ring buffer.

    A memory-bounded ring buffer continuously captures telemetry.  When
    the user presses the save key, the buffer contents are flushed to
    the dataset and live recording continues until the key is pressed
    again.
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

    # Number of correction episodes to collect (corrections-only mode).
    # When None, falls back to ``--dataset.num_episodes``.
    num_episodes: int | None = None
    record_autonomous: bool = False
    upload_every_n_episodes: int = 5
    # Target video file size in MB for episode rotation (record_autonomous
    # mode only).  Defaults to DEFAULT_VIDEO_FILE_SIZE_IN_MB when None.
    target_video_file_size_mb: int | None = None
    input_device: str = "keyboard"
    keyboard: DAggerKeyboardConfig = field(default_factory=DAggerKeyboardConfig)
    pedal: DAggerPedalConfig = field(default_factory=DAggerPedalConfig)

    def __post_init__(self):
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
    # Display data on a remote Rerun server
    display_ip: str | None = None
    # Port of the remote Rerun server
    display_port: int | None = None
    # Whether to display compressed images in Rerun
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
        # --- Strategy-specific validation ---
        if isinstance(self.strategy, DAggerStrategyConfig) and self.teleop is None:
            raise ValueError("DAgger strategy requires --teleop.type to be set")

        # TODO(Steven): DAgger shouldn't require a dataset (user may want to just rollout+intervene without recording), but for now we require it to simplify the implementation.
        needs_dataset = isinstance(
            self.strategy, (SentryStrategyConfig, HighlightStrategyConfig, DAggerStrategyConfig)
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
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
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
