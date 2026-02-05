# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""
Simple script to control a robot from teleoperation.

Example:

```shell
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --display_data=true
```

Example teleoperation with bimanual so100:

```shell
lerobot-teleoperate \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/tty.usbmodem5A460822851 \
  --robot.right_arm_config.port=/dev/tty.usbmodem5A460814411 \
  --robot.id=bimanual_follower \
  --robot.left_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
  }' --robot.right_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
  }' \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/tty.usbmodem5A460852721 \
  --teleop.right_arm_config.port=/dev/tty.usbmodem5A460819811 \
  --teleop.id=bimanual_leader \
  --display_data=true
```

Example with DINOv3 vision feature visualization:

```shell
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem58760431541 \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
  --robot.id=black \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodem58760431551 \
  --teleop.id=blue \
  --display_data=true \
  --vision_visualizer=dinov2 \
  --dinov2_model=facebook/dinov3-vit-base-pretrain-lvd1689m \
  --visualize_attention=false
```

"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import rerun as rr

from lerobot.async_inference.bimanual_koch_utils import action_dict_to_tensor, get_bimanual_action_features
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_koch_follower,
    bi_so100_follower,
    bi_openarm_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_koch_leader,
    bi_so100_leader,
    bi_openarm_leader,
    bi_so_leader,
    gamepad,
    homunculus,
    keyboard,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    reachy2_teleoperator,
    so_leader,
    unitree_g1,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data, log_rerun_action_chunk
from lerobot.utils.vision_visualizers import VisionVisualizer, make_vision_visualizer

from lerobot.processor.processor_factory import make_robot_action_processor, make_teleop_action_processor

from lerobot.async_inference.bimanual_koch_utils import compute_current_ee

@dataclass
class TeleoperateConfig:
    # TODO: pepijn, steven: if more robots require multiple teleoperators (like lekiwi) its good to make this possibele in teleop.py and record.py with List[Teleoperator]
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # Limit the maximum frames per second.
    fps: int = 60
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Vision feature visualization ("none", "dinov2", or None to disable)
    vision_visualizer: str | None = None
    # DINOv2/v3 model to use if vision_visualizer="dinov2"
    # DINOv3 ViT: facebook/dinov3-vit-{small/base/large/huge}-pretrain-{lvd1689m/sat493m}
    # DINOv3 ConvNeXt: facebook/dinov3-convnext-{tiny/small/base/large}-pretrain-{lvd1689m/sat493m}
    # DINOv2: facebook/dinov2-{small/base/large/giant}
    dinov2_model: str = "facebook/dinov3-vit-base-pretrain-lvd1689m"
    # Whether to visualize attention maps (ViT only, slower)
    visualize_attention: bool = False
    # Display data on a remote Rerun server
    display_ip: str | None = None
    # Port of the remote Rerun server
    display_port: int | None = None
    # Whether to  display compressed images in Rerun
    display_compressed_images: bool = False


def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    display_data: bool = False,
    duration: float | None = None,
    vision_visualizer: VisionVisualizer | None = None,
    display_compressed_images: bool = False,
):
    """
    This function continuously reads actions from a teleoperation device, processes them through optional
    pipelines, sends them to a robot, and optionally displays the robot's state. The loop runs at a
    specified frequency until a set duration is reached or it is manually interrupted.

    Args:
        teleop: The teleoperator device instance providing control actions.
        robot: The robot instance being controlled.
        fps: The target frequency for the control loop in frames per second.
        display_data: If True, fetches robot observations and displays them in the console and Rerun.
        display_compressed_images: If True, compresses images before sending them to Rerun for display.
        duration: The maximum duration of the teleoperation loop in seconds. If None, the loop runs indefinitely.
        teleop_action_processor: An optional pipeline to process raw actions from the teleoperator.
        robot_action_processor: An optional pipeline to process actions before they are sent to the robot.
        robot_observation_processor: An optional pipeline to process raw observations from the robot.
        vision_visualizer: An optional vision feature visualizer (e.g., DINOv2).
    """

    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()
    for _ in range(10):
        print("\n")

    # Check if we're using FK/IK processors (koch-based systems)
    # These processors convert joint angles to/from end-effector poses
    uses_fk_ik = robot.robot_type in ["koch_follower", "bi_koch_follower"]

    if uses_fk_ik:
        action_features = get_bimanual_action_features(robot, teleop_action_processor)

    while True:
        loop_start = time.perf_counter()
        # Retrieve normalized action and raw values if available
        # Get robot observation
        # Not really needed for now other than for visualization
        # teleop_action_processor can take None as an observation
        # given that it is the identity processor as default
        try:
            raw_action, unnormalized_action = teleop.get_action_with_raw()
        except Exception:
            # Fallback to normalized-only in case a specific teleoperator does not support raw values
            raw_action = teleop.get_action()
            unnormalized_action = None
        # Retrieve normalized observation and raw values if available
        try:
            obs, raw_observation = robot.get_observation_with_raw()
        except Exception:
            obs = robot.get_observation()
            raw_observation = None
        if display_data:
            log_rerun_data(obs, raw_action)

        # Visualize vision features if enabled
        if vision_visualizer is not None:
            # Extract camera images from observation and visualize features
            for key, value in obs.items():
                if key.startswith("observation.image"):
                    # Extract camera name (e.g., "observation.image.top" -> "top")
                    camera_name = key.split(".")[-1] if len(key.split(".")) > 2 else "camera"
                    vision_visualizer(value, camera_name)

        # Process teleop action through pipeline
        teleop_action = teleop_action_processor((raw_action, obs))

        # FK/IK-specific visualization (compute and log current EE state)
        if uses_fk_ik:
            # Compute current end-effector pose from robot observation
            current_ee = compute_current_ee(obs, teleop_action_processor, action_features)
            log_rerun_action_chunk(current_ee.unsqueeze(0), name="current_ee_")

        robot_action_to_send = robot_action_processor((teleop_action, obs))

        # Send processed action to robot (robot_action_processor.to_output should return RobotAction)
        _ = robot.send_action(robot_action_to_send)

        output_lines: list[str] = []
        output_lines.append("-" * (display_len + 10))
        if display_data:
            # Process robot observation through pipeline
            obs_transition = robot_observation_processor(obs)

            log_rerun_data(
                observation=obs_transition,
                action=teleop_action,
                compress_images=display_compressed_images,
            )

            output_lines.append(f"{'TELEOP NAME':<{display_len}} | {'NORM':>7}")
            # Display the final robot action that was sent
            for motor, value in teleop_action.items():
                output_lines.append(f"{motor:<{display_len}} | {value * 100:>7.2f}")

            output_lines.append(f"{'ROBOT NAME':<{display_len}} | {'NORM':>7}")
            for motor, value in robot_action_to_send.items():
                output_lines.append(f"{motor:<{display_len}} | {value:>7.2f}")

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0.0))
        loop_s = time.perf_counter() - loop_start

        # Uncomment this to debug the raw encoder reading values
        # output_lines.append("-" * (display_len + 20))
        # # Actions block
        # output_lines.append("ACTIONS")
        # if unnormalized_action is not None:
        #     output_lines.append(f"{'NAME':<{display_len}} | {'NORM':>7} | {'RAW':>7}")
        #     for motor, value in raw_action.items():
        #         raw_val = unnormalized_action.get(motor) if isinstance(unnormalized_action, dict) else None
        #         raw_display = f"{int(raw_val):>7}" if isinstance(raw_val, (int, float)) else f"{'N/A':>7}"
        #         output_lines.append(f"{motor:<{display_len}} | {value:>7.2f} | {raw_display}")
        # else:
        #     output_lines.append(f"{'NAME':<{display_len}} | {'NORM':>7}")
        #     for motor, value in raw_action.items():
        #         output_lines.append(f"{motor:<{display_len}} | {value:>7.2f}")

        # # Observations block
        # output_lines.append("OBSERVATIONS")
        # if raw_observation is not None:
        #     output_lines.append(f"{'NAME':<{display_len}} | {'NORM':>7} | {'RAW':>7}")
        #     for key, value in obs.items():
        #         if not key.endswith(".pos"):
        #             continue
        #         raw_val = raw_observation.get(key) if isinstance(raw_observation, dict) else None
        #         raw_display = f"{int(raw_val):>7}" if isinstance(raw_val, (int, float)) else f"{'N/A':>7}"
        #         output_lines.append(f"{key:<{display_len}} | {value:>7.2f} | {raw_display}")
        # else:
        #     output_lines.append(f"{'NAME':<{display_len}} | {'NORM':>7}")
        #     for key, value in obs.items():
        #         if not key.endswith(".pos"):
        #             continue
        #         output_lines.append(f"{key:<{display_len}} | {value:>7.2f}")

        # # Timing line (keep a blank line before time for readability)
        output_lines.append("")
        output_lines.append(f"time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

        print("\n".join(output_lines))

        if duration is not None and time.perf_counter() - start >= duration:
            return

        # Move cursor up exactly the number of lines we printed
        move_cursor_up(len(output_lines))
        # move_cursor_up(10), # uncomment this to debug the raw encoder reading values


@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="teleoperation", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)
    teleop.connect()
    robot.connect()

    # # Create vision visualizer if requested
    vision_visualizer = None
    # if cfg.vision_visualizer is not None:
    #     logging.info(f"Creating vision visualizer: {cfg.vision_visualizer}")
    #     vision_visualizer = make_vision_visualizer(
    #         visualizer_type=cfg.vision_visualizer,
    #         model_name=cfg.dinov2_model,
    #         visualize_attention=cfg.visualize_attention,
    #         log_to_rerun=cfg.display_data,
    #     )

    _, _, robot_observation_processor = make_default_processors()
    try:
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=cfg.fps,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s,
            teleop_action_processor=make_teleop_action_processor(cfg.teleop, teleop, cfg.display_data),
            robot_action_processor=make_robot_action_processor(cfg.robot, robot, cfg.display_data),
            robot_observation_processor=robot_observation_processor,
            vision_visualizer=vision_visualizer,
            display_compressed_images=display_compressed_images,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()


def main():
    register_third_party_plugins()
    teleoperate()


if __name__ == "__main__":
    main()
