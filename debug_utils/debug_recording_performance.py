#!/usr/bin/env python3

import logging
import sys
import time
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.teleoperators.so100_leader.config_so100_leader import SO100LeaderConfig
from lerobot.teleoperators.so100_leader.so100_leader import SO100Leader

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def measure_component_times():
    """Measure the time taken by each component in the recording loop."""

    # Initialize cameras
    print("🔧 Initializing cameras...")
    camera_configs = {
        "gripper": OpenCVCameraConfig(
            index_or_path=0,
            fps=30,
            width=1280,
            height=720,
            color_mode=ColorMode.RGB,
        ),
        "top": OpenCVCameraConfig(
            index_or_path=1,
            fps=30,
            width=1280,
            height=720,
            color_mode=ColorMode.RGB,
        ),
    }

    cameras = {}
    for cam_key, config in camera_configs.items():
        try:
            cameras[cam_key] = OpenCVCamera(config)
            cameras[cam_key].connect()
            print(f"✅ {cam_key} camera connected")
        except Exception as e:
            print(f"❌ Failed to connect {cam_key} camera: {e}")
            return

    # Initialize robot
    print("🤖 Initializing SO100 follower...")
    try:
        robot_config = SO100FollowerConfig(port="/dev/tty.usbmodem58760434091", cameras=camera_configs)
        robot = SO100Follower(robot_config)
        robot.connect()
        print("✅ Robot connected")
    except Exception as e:
        print(f"❌ Failed to connect robot: {e}")
        return

    # Initialize teleoperator
    print("🎮 Initializing SO100 leader...")
    try:
        teleop_config = SO100LeaderConfig(port="/dev/tty.usbmodem58CD1771421")
        teleop = SO100Leader(teleop_config)
        teleop.connect()
        print("✅ Teleop connected")
    except Exception as e:
        print(f"❌ Failed to connect teleop: {e}")
        return

    print("\n📊 Starting performance measurement...")
    print("Recording 60 iterations (should be 2 seconds at 30 FPS)...")

    timing_data = {
        "robot_observation": [],
        "teleop_action": [],
        "robot_action": [],
        "total_loop": [],
        "camera_gripper": [],
        "camera_top": [],
        "robot_state": [],
    }

    target_fps = 30
    target_dt = 1.0 / target_fps

    try:
        for i in range(60):
            loop_start = time.perf_counter()

            # 1. Get robot observation (includes cameras)
            obs_start = time.perf_counter()
            observation = robot.get_observation()
            obs_time = time.perf_counter() - obs_start
            timing_data["robot_observation"].append(obs_time * 1000)  # Convert to ms

            # Break down observation time
            # Measure individual camera reads
            gripper_start = time.perf_counter()
            try:
                _ = cameras["gripper"].async_read(timeout_ms=100)
                gripper_time = time.perf_counter() - gripper_start
                timing_data["camera_gripper"].append(gripper_time * 1000)
            except Exception:
                timing_data["camera_gripper"].append(-1)  # Timeout/error

            top_start = time.perf_counter()
            try:
                _ = cameras["top"].async_read(timeout_ms=100)
                top_time = time.perf_counter() - top_start
                timing_data["camera_top"].append(top_time * 1000)
            except Exception:
                timing_data["camera_top"].append(-1)  # Timeout/error

            # Measure robot state read only
            state_start = time.perf_counter()
            try:
                _ = robot.bus.sync_read("Present_Position")
                state_time = time.perf_counter() - state_start
                timing_data["robot_state"].append(state_time * 1000)
            except Exception:
                timing_data["robot_state"].append(-1)

            # 2. Get teleop action
            action_start = time.perf_counter()
            action = teleop.get_action()
            action_time = time.perf_counter() - action_start
            timing_data["teleop_action"].append(action_time * 1000)

            # 3. Send action to robot
            send_start = time.perf_counter()
            robot.send_action(action)
            send_time = time.perf_counter() - send_start
            timing_data["robot_action"].append(send_time * 1000)

            # Total loop time
            total_time = time.perf_counter() - loop_start
            timing_data["total_loop"].append(total_time * 1000)

            # Show real-time progress
            if i % 10 == 0:
                current_fps = 1.0 / total_time if total_time > 0 else 0
                print(
                    f"  Iteration {i:2d}: {total_time * 1000:6.1f}ms total, "
                    f"{current_fps:5.1f} FPS, obs: {obs_time * 1000:5.1f}ms"
                )

            # Wait for target timing (if we're ahead)
            elapsed = time.perf_counter() - loop_start
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")
    except Exception as e:
        print(f"\n❌ Error during measurement: {e}")
    finally:
        # Cleanup
        try:
            robot.disconnect()
            teleop.disconnect()
            for cam in cameras.values():
                cam.disconnect()
        except:
            pass

    # Analyze results
    print("\n📈 PERFORMANCE ANALYSIS:")
    print("=" * 60)

    if timing_data["total_loop"]:
        for component, times in timing_data.items():
            if not times:
                continue

            valid_times = [t for t in times if t > 0]  # Filter out errors
            if not valid_times:
                print(f"{component:20s}: ALL FAILED")
                continue

            avg_ms = sum(valid_times) / len(valid_times)
            max_ms = max(valid_times)
            min_ms = min(valid_times)

            if component == "total_loop":
                actual_fps = 1000.0 / avg_ms if avg_ms > 0 else 0
                print(
                    f"{component:20s}: {avg_ms:6.1f}ms avg ({actual_fps:5.1f} FPS), "
                    f"{min_ms:5.1f}-{max_ms:5.1f}ms range"
                )
            else:
                print(f"{component:20s}: {avg_ms:6.1f}ms avg, {min_ms:5.1f}-{max_ms:5.1f}ms range")

    print("\n🔍 BOTTLENECK ANALYSIS:")
    if timing_data["total_loop"] and timing_data["robot_observation"]:
        avg_total = sum(t for t in timing_data["total_loop"] if t > 0) / len(
            [t for t in timing_data["total_loop"] if t > 0]
        )
        avg_obs = sum(t for t in timing_data["robot_observation"] if t > 0) / len(
            [t for t in timing_data["robot_observation"] if t > 0]
        )

        obs_percentage = (avg_obs / avg_total) * 100 if avg_total > 0 else 0
        print(f"• Observation takes {obs_percentage:.1f}% of total loop time")

        if avg_total > 33.3:  # More than 30 FPS
            print(f"• Target: 33.3ms (30 FPS), Actual: {avg_total:.1f}ms ({1000 / avg_total:.1f} FPS)")
            print(f"• You're {avg_total / 33.3:.1f}x slower than target!")

        # Camera analysis
        camera_times = []
        for cam_key in ["gripper", "top"]:
            cam_data = timing_data.get(f"camera_{cam_key}", [])
            if cam_data:
                valid_cam_times = [t for t in cam_data if t > 0]
                if valid_cam_times:
                    avg_cam = sum(valid_cam_times) / len(valid_cam_times)
                    camera_times.append((cam_key, avg_cam))

        if camera_times:
            print("• Camera breakdown:")
            for cam_name, cam_time in camera_times:
                cam_percentage = (cam_time / avg_total) * 100 if avg_total > 0 else 0
                print(f"  - {cam_name}: {cam_time:.1f}ms ({cam_percentage:.1f}%)")


if __name__ == "__main__":
    measure_component_times()
