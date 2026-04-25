import logging
import threading
from dataclasses import asdict, dataclass, field
from pprint import pformat

import draccus

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.rtc_inference.configs import AGGREGATE_FUNCTIONS, RobotClientConfig
from lerobot.rtc_inference.helpers import visualize_action_queue_size
from lerobot.rtc_inference.robot_client import RobotClient
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so_follower,
)
from lerobot.utils.import_utils import register_third_party_plugins


@dataclass
class RTCXVLAClientOnlyConfig:
    server_address: str = "192.168.1.107:4567"

    robot: RobotConfig | None = None
    task: str = field(default="", metadata={"help": "Task instruction"})

    policy_type: str = "xvla"
    pretrained_name_or_path: str = ""
    policy_device: str = "cuda"
    client_device: str = "cpu"
    rename_map: dict[str, str] = field(
        default_factory=lambda: {
            "observation.images.camera1": "observation.images.image",
            "observation.images.camera2": "observation.images.image2",
            "observation.images.camera3": "observation.images.image3",
            "observation.images.camera4": "observation.images.image4",
        }
    )

    # Chunking and aggregation behavior
    actions_per_chunk: int = 50
    chunk_size_threshold: float = 0.5
    aggregate_fn_name: str = "latest_only"

    # RTC parameters exposed to user
    rtc_execution_horizon: int = 10
    rtc_max_guidance_weight: float = 10.0
    rtc_prefix_attention_schedule: RTCAttentionSchedule = RTCAttentionSchedule.EXP
    rtc_debug: bool = False
    rtc_debug_maxlen: int = 100
    inference_delay_steps: int | None = None
    xvla_domain_id: int | None = None

    # Client loop/runtime parameters
    fps: int = 30
    obs_timestep_independent: bool = True
    image_compress_enable: bool = False
    image_compress_quality: int = 90
    interpolation_multiplier: int = 1
    debug_visualize_queue_size: bool = False

    def __post_init__(self):
        if self.robot is None:
            raise ValueError("robot configuration is required")
        if self.policy_type != "xvla":
            raise ValueError("policy_type must be 'xvla' for RTC XVLA orchestrator")
        if not self.pretrained_name_or_path:
            raise ValueError("pretrained_name_or_path is required")
        if not isinstance(self.rename_map, dict):
            raise ValueError("rename_map must be a dictionary")
        if self.actions_per_chunk <= 0:
            raise ValueError("actions_per_chunk must be > 0")
        if not (0 <= self.chunk_size_threshold <= 1):
            raise ValueError("chunk_size_threshold must be in [0, 1]")
        if self.aggregate_fn_name not in AGGREGATE_FUNCTIONS:
            raise ValueError(
                f"aggregate_fn_name must be one of {list(AGGREGATE_FUNCTIONS.keys())}"
            )
        if self.fps <= 0:
            raise ValueError("fps must be > 0")
        if not isinstance(self.obs_timestep_independent, bool):
            raise ValueError("obs_timestep_independent must be a boolean")
        if not isinstance(self.image_compress_enable, bool):
            raise ValueError("image_compress_enable must be a boolean")
        if not (1 <= self.image_compress_quality <= 100):
            raise ValueError("image_compress_quality must be in [1, 100]")
        if self.interpolation_multiplier <= 0:
            raise ValueError("interpolation_multiplier must be > 0")
        if self.inference_delay_steps is not None and self.inference_delay_steps < 0:
            raise ValueError("inference_delay_steps must be >= 0")


def _to_robot_client_config(cfg: RTCXVLAClientOnlyConfig) -> RobotClientConfig:
    return RobotClientConfig(
        policy_type=cfg.policy_type,
        pretrained_name_or_path=cfg.pretrained_name_or_path,
        robot=cfg.robot,
        actions_per_chunk=cfg.actions_per_chunk,
        task=cfg.task,
        rename_map=cfg.rename_map,
        server_address=cfg.server_address,
        policy_device=cfg.policy_device,
        client_device=cfg.client_device,
        chunk_size_threshold=cfg.chunk_size_threshold,
        fps=cfg.fps,
        obs_timestep_independent=cfg.obs_timestep_independent,
        image_compress_enable=cfg.image_compress_enable,
        image_compress_quality=cfg.image_compress_quality,
        interpolation_multiplier=cfg.interpolation_multiplier,
        aggregate_fn_name=cfg.aggregate_fn_name,
        debug_visualize_queue_size=cfg.debug_visualize_queue_size,
        rtc_enabled=True,
        rtc_execution_horizon=cfg.rtc_execution_horizon,
        rtc_max_guidance_weight=cfg.rtc_max_guidance_weight,
        rtc_prefix_attention_schedule=cfg.rtc_prefix_attention_schedule,
        rtc_debug=cfg.rtc_debug,
        rtc_debug_maxlen=cfg.rtc_debug_maxlen,
        inference_delay_steps=cfg.inference_delay_steps,
        xvla_domain_id=cfg.xvla_domain_id,
    )


def run_client_only(cfg: RTCXVLAClientOnlyConfig) -> None:
    client_cfg = _to_robot_client_config(cfg)
    client = RobotClient(client_cfg)

    if not client.start():
        client.logger.error("Failed to connect to policy server")
        return

    client.logger.info("Starting action receiver thread...")
    action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
    action_receiver_thread.start()

    try:
        client.control_loop(task=cfg.task)
    except KeyboardInterrupt:
        client.logger.info("KeyboardInterrupt received. Stopping client...")
    finally:
        client.stop()
        action_receiver_thread.join(timeout=2.0)

        if client_cfg.debug_visualize_queue_size:
            visualize_action_queue_size(client.action_queue_size)

        client.logger.info("RTC client stopped")


@draccus.wrap()
def main(cfg: RTCXVLAClientOnlyConfig) -> None:
    logging.info(pformat(asdict(cfg)))
    run_client_only(cfg)


if __name__ == "__main__":
    register_third_party_plugins()
    main()
