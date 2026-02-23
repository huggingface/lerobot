"""Shared utilities for Human-in-the-Loop data collection scripts."""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.processor import (
    IdentityProcessorStep,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
)
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots import Robot
from lerobot.teleoperators import Teleoperator
from lerobot.utils.control_utils import is_headless
from lerobot.utils.robot_utils import precise_sleep

logger = logging.getLogger(__name__)


@dataclass
class HILDatasetConfig:
    repo_id: str
    single_task: str
    root: str | Path | None = None
    fps: int = 30
    episode_time_s: float = 120
    num_episodes: int = 50
    video: bool = True
    push_to_hub: bool = True
    private: bool = False
    tags: list[str] | None = None
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4
    video_encoding_batch_size: int = 1
    rename_map: dict[str, str] = field(default_factory=dict)


def teleop_has_motor_control(teleop: Teleoperator) -> bool:
    """Check if teleoperator has motor control capabilities."""
    return hasattr(teleop, "bus") and hasattr(teleop.bus, "disable_torque")


def teleop_disable_torque(teleop: Teleoperator) -> None:
    """Disable teleop torque if supported."""
    if teleop_has_motor_control(teleop):
        teleop.bus.disable_torque()


def teleop_enable_torque(teleop: Teleoperator) -> None:
    """Enable teleop torque if supported."""
    if teleop_has_motor_control(teleop):
        teleop.bus.enable_torque()


def teleop_smooth_move_to(teleop: Teleoperator, target_pos: dict, duration_s: float = 2.0, fps: int = 50):
    """Smoothly move teleop to target position if motor control is available."""
    if not teleop_has_motor_control(teleop):
        logger.warning("Teleop does not support motor control - cannot mirror robot position")
        return

    teleop_enable_torque(teleop)
    current = teleop.get_action()
    steps = int(duration_s * fps)

    for step in range(steps + 1):
        t = step / steps
        interp = {}
        for k in current:
            if k in target_pos:
                interp[k] = current[k] * (1 - t) + target_pos[k] * t
            else:
                interp[k] = current[k]
        teleop.bus.sync_write("Goal_Position", {k.replace(".pos", ""): v for k, v in interp.items()})
        time.sleep(1 / fps)


def init_keyboard_listener():
    """Initialize keyboard listener with HIL controls."""
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
        "policy_paused": False,
        "correction_active": False,
        "in_reset": False,
        "start_next_episode": False,
    }

    if is_headless():
        logger.warning("Headless environment - keyboard controls unavailable")
        return None, events

    from pynput import keyboard

    def on_press(key):
        try:
            if events["in_reset"]:
                if key in [keyboard.Key.space, keyboard.Key.right]:
                    print("\n[HIL] Starting next episode...")
                    events["start_next_episode"] = True
                elif hasattr(key, "char") and key.char == "c":
                    events["start_next_episode"] = True
                elif key == keyboard.Key.esc:
                    print("[HIL] ESC - Stop recording, pushing to hub...")
                    events["stop_recording"] = True
                    events["start_next_episode"] = True
            else:
                if key == keyboard.Key.space:
                    if not events["policy_paused"] and not events["correction_active"]:
                        print("\n[HIL] ⏸ PAUSED - Press 'c' to take control")
                        events["policy_paused"] = True
                elif hasattr(key, "char") and key.char == "c":
                    if events["policy_paused"] and not events["correction_active"]:
                        print("\n[HIL] ▶ Taking control...")
                        events["start_next_episode"] = True
                elif key == keyboard.Key.right:
                    print("[HIL] → End episode")
                    events["exit_early"] = True
                elif key == keyboard.Key.left:
                    print("[HIL] ← Re-record episode")
                    events["rerecord_episode"] = True
                    events["exit_early"] = True
                elif key == keyboard.Key.esc:
                    print("[HIL] ESC - Stop recording...")
                    events["stop_recording"] = True
                    events["exit_early"] = True
        except Exception as e:
            print(f"Key error: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener, events


def make_identity_processors():
    """Create identity processors for recording."""
    teleop_proc = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[IdentityProcessorStep()],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    obs_proc = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[IdentityProcessorStep()],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    return teleop_proc, obs_proc


def reset_loop(robot: Robot, teleop: Teleoperator, events: dict, fps: int):
    """Reset period where human repositions environment."""
    print("\n" + "=" * 60)
    print("  [HIL] RESET")
    print("=" * 60)

    events["in_reset"] = True
    events["start_next_episode"] = False

    obs = robot.get_observation()
    robot_pos = {k: v for k, v in obs.items() if k.endswith(".pos") and k in robot.observation_features}
    teleop_smooth_move_to(teleop, robot_pos, duration_s=2.0, fps=50)

    print("  Press any key to enable teleoperation")
    while not events["start_next_episode"] and not events["stop_recording"]:
        precise_sleep(0.05)

    if events["stop_recording"]:
        return

    events["start_next_episode"] = False
    teleop_disable_torque(teleop)
    print("  Teleop enabled - press any key to start episode")

    while not events["start_next_episode"] and not events["stop_recording"]:
        loop_start = time.perf_counter()
        action = teleop.get_action()
        robot.send_action(action)
        precise_sleep(1 / fps - (time.perf_counter() - loop_start))

    events["in_reset"] = False
    events["start_next_episode"] = False
    events["exit_early"] = False
    events["policy_paused"] = False
    events["correction_active"] = False


def print_controls(rtc: bool = False):
    """Print control instructions."""
    print("\n" + "=" * 60)
    print("  Human-in-the-Loop Data Collection" + (" (RTC)" if rtc else ""))
    print("=" * 60)
    print()
    print("  Controls:")
    print("    SPACE  - Pause policy")
    print("    c      - Take control")
    print("    →      - End episode")
    print("    ESC    - Stop and push to hub")
    print("=" * 60 + "\n")
