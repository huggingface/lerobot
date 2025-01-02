import logging
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import draccus

from lerobot.common.policies.utils import get_pretrained_policy_path
from lerobot.common.robot_devices.robots.configs import RobotConfig
from lerobot.configs.policies import PretrainedConfig
from lerobot.configs.training import TrainPipelineConfig


@dataclass
class ControlConfig(draccus.ChoiceRegistry):
    pass


@ControlConfig.register_subclass("calibrate")
@dataclass
class CalibrateControlConfig(ControlConfig):
    # List of arms to calibrate (e.g. `--arms left_follower right_follower left_leader`)
    arms: list[str] | None = None


@ControlConfig.register_subclass("teleoperate")
@dataclass
class TeleoperateControlConfig(ControlConfig):
    # Limit the maximum frames per second. By default, no limit.
    fps: int | None = None
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_cameras: bool = False


@ControlConfig.register_subclass("record")
@dataclass
class RecordControlConfig(ControlConfig):
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # A short but accurate description of the task performed during the recording (e.g. "Pick the Lego block and drop it in the box on the right.")
    single_task: str
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    # Path to load a pretrained policy
    pretrained_policy_path: str | None = None
    # Config to override the one from the pretrained policy
    policy: PretrainedConfig | None = None
    # By default, use the value from policy checkpoint.
    device: str | None = None  # cuda | cpu | mps
    # Use Automatic Mixed Precision (AMP), expected to increase inference speed at the expend of float precision.
    # By default, use the value from policy checkpoint.
    use_amp: bool | None = None
    # Limit the frames per second. By default, uses the policy fps.
    fps: int | None = None
    # Number of seconds before starting data collection. It allows the robot devices to warmup and synchronize.
    warmup_time_s: int | float = 10
    # Number of seconds for data recording for each episode.
    episode_time_s: int | float = 60
    # Number of seconds for resetting the environment after each episode.
    reset_time_s: int | float = 60
    # Number of episodes to record.
    num_episodes: int = 50
    # Encode frames in the dataset into video
    video: bool = True
    # By default, run the computation of the data statistics at the end of data collection. Compute intensive and not required to just replay an episode.
    run_compute_stats: bool = True
    # Upload dataset to Hugging Face hub.
    push_to_hub: bool = True
    # Add tags to your dataset on the hub.
    tags: list[str] | None = None
    # Number of subprocesses handling the saving of frames as PNGs. Set to 0 to use threads only;
    # set to ≥1 to use subprocesses, each using threads to write images. The best number of processes
    # and threads depends on your system. We recommend 4 threads per camera with 0 processes.
    # If fps is unstable, adjust the thread count. If still unstable, try using 1 or more subprocesses.
    num_image_writer_processes: int = 0
    # Number of threads writing the frames as png images on disk, per camera.
    # Too many threads might cause unstable teleoperation fps due to main thread being blocked.
    # Not enough threads might cause low camera fps.
    num_image_writer_threads_per_camera: int = 4
    # Display all cameras on screen
    display_cameras: bool = True
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # Resume recording on an existing dataset.
    resume: bool = False
    # TODO(rcadene, aliberts): remove local_files_only when refactor with dataset as argument
    # Use local files only. By default, this script will try to fetch the dataset from the hub if it exists.
    local_files_only: bool = False

    def __post_init__(self):
        # TODO(aliberts, rcadene): move this logic out of the config
        from time import sleep

        if self.pretrained_policy_path is None:
            return

        sleep(1)
        self.resolve_policy_name_or_path()
        self.load_policy_config_from_path()
        self.load_fps_device_use_amp_from_path()

    def resolve_policy_name_or_path(self):
        self.pretrained_policy_path = get_pretrained_policy_path(self.pretrained_policy_path)

    def load_policy_config_from_path(self):
        # Load policy config from checkpoint
        cfg_path = self.pretrained_policy_path / "config.json"
        with open(cfg_path) as f:
            policy_cfg = draccus.load(PretrainedConfig, f)

        # Override policy config from command line
        if self.policy is not None:
            policy_cfg = replace(policy_cfg, **asdict(self.policy))

        self.policy = policy_cfg

    def load_fps_device_use_amp_from_path(self):
        # Load training config from checkpoint
        cfg_path = self.pretrained_policy_path / "config.yaml"
        with open(cfg_path) as f:
            train_cfg = draccus.load(TrainPipelineConfig, f)

        if self.fps is None:
            self.fps = train_cfg.env.fps
            logging.warning(f"No fps value provided, so using the one from policy checkpoint ({self.fps}).")

        if self.device is None:
            self.device = train_cfg.device
            logging.warning(
                f"No device value provided, so using the one from policy checkpoint ({self.device})."
            )

        if self.use_amp is None:
            self.use_amp = train_cfg.use_amp
            logging.warning(
                f"No use_amp value provided, so using the one from policy checkpoint ({self.use_amp})."
            )


@ControlConfig.register_subclass("replay")
@dataclass
class ReplayControlConfig(ControlConfig):
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # Index of the episode to replay.
    episode: int
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    # Limit the frames per second. By default, uses the dataset fps.
    fps: int | None = None
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # TODO(rcadene, aliberts): remove local_files_only when refactor with dataset as argument
    # Use local files only. By default, this script will try to fetch the dataset from the hub if it exists.
    local_files_only: bool = False


@dataclass
class ControlPipelineConfig:
    robot: RobotConfig
    control: ControlConfig
