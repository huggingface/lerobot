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
python -m lerobot.teleoperate \
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
python -m lerobot.teleoperate \
  --robot.type=bi_so100_follower \
  --robot.left_arm_port=/dev/tty.usbmodem5A460851411 \
  --robot.right_arm_port=/dev/tty.usbmodem5A460812391 \
  --robot.id=bimanual_follower \
  --robot.cameras='{
    left: {"type": "opencv", "index_or_path": 0, "width": 1920, "height": 1080, "fps": 30},
    top: {"type": "opencv", "index_or_path": 1, "width": 1920, "height": 1080, "fps": 30},
    right: {"type": "opencv", "index_or_path": 2, "width": 1920, "height": 1080, "fps": 30}
  }' \
  --teleop.type=bi_so100_leader \
  --teleop.left_arm_port=/dev/tty.usbmodem5A460828611 \
  --teleop.right_arm_port=/dev/tty.usbmodem5A460826981 \
  --teleop.id=bimanual_leader \
  --display_data=true
```

"""

import logging
from pathlib import Path
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import draccus
import rerun as rr

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.kinect.configuration_kinect import KinectCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    bi_so101_follower,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_so100_leader,
    bi_so101_leader,
    gamepad,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
)
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data, register_depth_scale, register_depth_colorization
from lerobot.cameras.depth_defaults import resolve_depth_params


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
    # Visualization toggles
    viz_depth: bool = False
    # Optional log file path
    log_file: str | None = None
    # Depth viz overrides (apply to all streams unless config overrides)
    depth_colormap: str | None = None
    depth_min_m: float | None = None
    depth_max_m: float | None = None
    depth_display_colorized: bool = False
    # Performance logging control
    perf_logging: bool = False
    perf_level: str | None = None
    # Console output of action table
    console_output: bool = False


def teleop_loop(
    teleop: Teleoperator, robot: Robot, fps: int, display_data: bool = False, duration: float | None = None,
    viz_depth: bool = False, console_output: bool = False,
):
    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()
    # Perf: report control loop fps every 5s
    perf_last_log_t = time.perf_counter()
    perf_window_s = 5.0
    perf_iter = 0
    perf_fps_sum = 0.0
    perf_fps_sum_sq = 0.0
    perf_fps_min = float("inf")
    perf_fps_max = 0.0

    while True:
        loop_start = time.perf_counter()
        action = teleop.get_action()

        if display_data:
            observation = robot.get_observation()
            # When depth viz is on, drop RGB streams; when off, drop depth streams
            if viz_depth:
                observation = {k: v for k, v in observation.items() if k.endswith("_depth")}
            else:
                observation = {k: v for k, v in observation.items() if not k.endswith("_depth")}
            log_rerun_data(observation, action)

        robot.send_action(action)
        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)

        # Update FPS stats
        if dt_s > 0:
            fps_inst = 1.0 / dt_s
            perf_iter += 1
            perf_fps_sum += fps_inst
            perf_fps_sum_sq += fps_inst * fps_inst
            perf_fps_min = min(perf_fps_min, fps_inst)
            perf_fps_max = max(perf_fps_max, fps_inst)

        loop_s = time.perf_counter() - loop_start

        if console_output:
            print("\n" + "-" * (display_len + 10))
            print(f"{'NAME':<{display_len}} | {'NORM':>7}")
            for motor, value in action.items():
                print(f"{motor:<{display_len}} | {value:>7.2f}")
            print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

        if duration is not None and time.perf_counter() - start >= duration:
            return

        if console_output:
            move_cursor_up(len(action) + 5)

        # Periodic perf logging
        now_t = time.perf_counter()
        if (now_t - perf_last_log_t) >= perf_window_s and perf_iter > 0:
            try:
                perf_logger = logging.getLogger("performance")
                n = perf_iter
                fps_avg = perf_fps_sum / n
                mean_sq = perf_fps_sum_sq / n
                fps_var = max(0.0, mean_sq - (fps_avg ** 2))
                fps_std = fps_var ** 0.5
                perf_logger.info(
                    f"Teleop loop 5s stats — fps(avg={fps_avg:.1f}, std={fps_std:.1f}, min={perf_fps_min:.1f}, max={perf_fps_max:.1f})"
                )
            except Exception:
                pass
            # Reset window
            perf_last_log_t = now_t
            perf_iter = 0
            perf_fps_sum = 0.0
            perf_fps_sum_sq = 0.0
            perf_fps_min = float("inf")
            perf_fps_max = 0.0



@draccus.wrap()
def teleoperate(cfg: TeleoperateConfig):
    # Initialize logging; performance logger controlled by CLI flags
    perf_level = cfg.perf_level if cfg.perf_logging else None
    init_logging(
        log_file=Path(cfg.log_file) if cfg.log_file else None,
        perf_level=perf_level,
        console_enabled=False,
        only_perf_logging=True if cfg.perf_logging else False,
    )
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        _init_rerun(session_name="teleoperation")

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)

    teleop.connect()
    robot.connect()

    # Register per-stream depth scales and optional colorization for correct Rerun visualization
    try:
        for cam_key, cam in getattr(robot, "cameras", {}).items():
            # Identify sensor type from class name
            cls = cam.__class__.__name__.lower()
            sensor = "realsense" if "realsense" in cls else ("kinect" if "kinect" in cls else "unknown")

            device_scale = getattr(cam, "depth_scale", None)
            cam_cfg = getattr(cfg.robot, "cameras", {}).get(cam_key) if hasattr(cfg.robot, "cameras") else None
            params = resolve_depth_params(
                sensor=("realsense" if "realsense" in cls else ("kinect" if "kinect" in cls else "kinect")),
                cam_cfg=cam_cfg,
                cli_colormap=cfg.depth_colormap,
                cli_min=cfg.depth_min_m,
                cli_max=cfg.depth_max_m,
                device_scale=device_scale,
            )
            if params.meters_per_unit is not None:
                register_depth_scale(cam_key, params.meters_per_unit)
            # Always register colorization params; visualization uses them by default now
            register_depth_colorization(cam_key, params.colormap, params.min_m, params.max_m)
    except Exception:
        pass

    try:
        teleop_loop(teleop, robot, cfg.fps, display_data=cfg.display_data, duration=cfg.teleop_time_s, viz_depth=cfg.viz_depth, console_output=cfg.console_output)
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    teleoperate()
