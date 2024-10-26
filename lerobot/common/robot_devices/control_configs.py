from dataclasses import dataclass
from pathlib import Path

import draccus

from lerobot.common.robot_devices.robots.configs import RobotConfig
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig


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
    policy: PreTrainedConfig | None = None
    # TODO(rcadene, aliberts): By default, use device and use_amp values from policy checkpoint.
    device: str | None = None  # cuda | cpu | mps
    # `use_amp` determines whether to use Automatic Mixed Precision (AMP) for training and evaluation. With AMP,
    # automatic gradient scaling is used.
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
    # Upload on private repository on the Hugging Face hub.
    private: bool = False
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
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.policy is not None:
            if self.device is None:
                raise ValueError("Set one of the following device: cuda, cpu or mps")
            elif self.device == "cuda" and self.use_amp is None:
                raise ValueError("Set 'use_amp' to True or False.")


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
