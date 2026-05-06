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

Requires: pip install 'lerobot[hardware]'

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

"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
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
    onerobotics_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_so_leader,
    gamepad,
    homunculus,
    keyboard,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    openarm_mini,
    reachy2_teleoperator,
    so_leader,
    unitree_g1,
    onerobotics_leader,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data, shutdown_rerun


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
    display_compressed_images: bool = False,
):
    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()

    # ============ CSV 记录设置 ============
    import csv
    import os

    csv_path = os.path.join(os.getcwd(), "teleop_data.csv")
    logging.info(f"========================================")
    logging.info(f"CSV will be saved to: {csv_path}")
    logging.info(f"========================================")

    csv_file = open(csv_path, "w", newline="")
    csv_writer = None
    frame_idx = 0
    # =====================================

    try:
        while True:
            loop_start = time.perf_counter()

            # 1. Get robot observation
            t_obs_start = time.perf_counter()
            obs = robot.get_observation()
            t_obs = (time.perf_counter() - t_obs_start) * 1000

            if robot.name == "unitree_g1":
                teleop.send_feedback(obs)

            # 2. Get teleop action
            t_act_start = time.perf_counter()
            raw_action = teleop.get_action()
            t_act = (time.perf_counter() - t_act_start) * 1000

            # Processing pipelines
            t_proc_start = time.perf_counter()
            teleop_action = teleop_action_processor((raw_action, obs))
            robot_action_to_send = robot_action_processor((teleop_action, obs))
            t_proc = (time.perf_counter() - t_proc_start) * 1000

            # 3. Send action out
            t_send_start = time.perf_counter()
            _ = robot.send_action(robot_action_to_send)
            t_send = (time.perf_counter() - t_send_start) * 1000

            # ============ 写CSV ============
            timestamp = time.perf_counter() - start

            # 提取从臂观测（只要数值类型）
            obs_joint_data = {}
            for key, val in obs.items():
                if isinstance(val, (int, float)):
                    obs_joint_data[f"obs_{key}"] = round(val, 6)

            # 提取主臂原始动作
            leader_data = {}
            for key, val in raw_action.items():
                if isinstance(val, (int, float)):
                    leader_data[f"leader_{key}"] = round(val, 6)

            # 提取处理后的动作
            action_data = {}
            for key, val in robot_action_to_send.items():
                if isinstance(val, (int, float)):
                    action_data[f"action_{key}"] = round(val, 6)

            row = {
                "frame": frame_idx,
                "timestamp": round(timestamp, 4),
                **leader_data,
                **action_data,
                **obs_joint_data,
            }

            # 第一帧：写表头
            if csv_writer is None:
                csv_writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
                csv_writer.writeheader()
                csv_file.flush()
                logging.info(f"CSV header written. Columns: {list(row.keys())}")

            csv_writer.writerow(row)

            # 每50帧flush一次，保证数据落盘
            if frame_idx % 50 == 0:
                csv_file.flush()
                if frame_idx % 500 == 0:
                    logging.info(f"CSV: {frame_idx} frames written to {csv_path}")

            frame_idx += 1
            # ================================

            if display_data:
                obs_transition = robot_observation_processor(obs)
                log_rerun_data(
                    observation=obs_transition,
                    action=teleop_action,
                    compress_images=display_compressed_images,
                )
                print("\n" + "-" * (display_len + 10))
                print(f"{'NAME':<{display_len}} | {'NORM':>7}")
                for motor, value in robot_action_to_send.items():
                    print(f"{motor:<{display_len}} | {value:>7.2f}")
                move_cursor_up(len(robot_action_to_send) + 3)

            dt_s = time.perf_counter() - loop_start
            precise_sleep(max(1 / fps - dt_s, 0.0))
            loop_s = time.perf_counter() - loop_start

            action_age_ms = getattr(teleop, "latest_action_age_ms", None)
            action_age_str = f"{action_age_ms:.1f}ms" if action_age_ms is not None else "N/A"

            logging.info(
                f"Hz: {1/loop_s:.1f} | Obs: {t_obs:.1f}ms | GetAct: {t_act:.1f}ms | "
                f"ActionAge: {action_age_str} | Proc: {t_proc:.1f}ms | SendAct: {t_send:.1f}ms"
            )

            if duration is not None and time.perf_counter() - start >= duration:
                break

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received, saving CSV...")
    finally:
        # ============ 确保CSV关闭 ============
        try:
            csv_file.flush()
            csv_file.close()
            logging.info(f"========================================")
            logging.info(f"CSV SAVED: {frame_idx} frames -> {csv_path}")
            logging.info(f"File size: {os.path.getsize(csv_path)} bytes")
            logging.info(f"========================================")
        except Exception as e:
            logging.error(f"Error closing CSV: {e}")
        # ======================================

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
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    teleop.connect()
    robot.connect()

    try:
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=cfg.fps,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            display_compressed_images=display_compressed_images,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            shutdown_rerun()
        teleop.disconnect()
        robot.disconnect()


def main():
    register_third_party_plugins()
    teleoperate()


if __name__ == "__main__":
    main()
