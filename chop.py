#!/usr/bin/env python3
"""
Chop - A wrapper script for LeRobot commands with centralized configuration management.

This script simplifies running LeRobot commands by storing commonly used configurations
and providing a clean CLI interface.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import click

# Configuration directory
CONFIG_DIR = Path.home() / ".config" / "chop"
ROBOT_CONFIG_FILE = CONFIG_DIR / "robot_config.json"
TELEOP_CONFIG_FILE = CONFIG_DIR / "teleop_config.json"


def ensure_config_dir():
    """Create config directory if it doesn't exist."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_config(config_file: Path) -> dict[str, Any]:
    """Load configuration from JSON file."""
    if not config_file.exists():
        return {}
    with open(config_file) as f:
        return json.load(f)


def save_config(config_file: Path, config: dict[str, Any]):
    """Save configuration to JSON file."""
    ensure_config_dir()
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    click.echo(f"✓ Configuration saved to {config_file}")


def build_cli_args(prefix: str, config: dict[str, Any]) -> list[str]:
    """
    Build CLI arguments from a config dictionary.

    Args:
        prefix: The prefix for arguments (e.g., "robot", "teleop", "dataset")
        config: Dictionary of configuration values

    Returns:
        List of CLI argument strings
    """
    import shlex

    args = []
    for key, value in config.items():
        if isinstance(value, dict):
            # Handle nested dictionaries (like cameras)
            json_str = json.dumps(value)
            # Properly quote the JSON string for shell execution
            args.append(f"--{prefix}.{key}={shlex.quote(json_str)}")
        elif isinstance(value, bool):
            args.append(f"--{prefix}.{key}={str(value).lower()}")
        elif value is not None:
            args.append(f"--{prefix}.{key}={value}")
    return args


def check_dataset_exists(repo_id: str, root: str | None = None) -> bool:
    """
    Check if a dataset already exists at the expected location.

    Mimics the logic from LeRobotDataset to determine the dataset path.

    Args:
        repo_id: Dataset repository ID (e.g., "username/dataset_name")
        root: Optional root directory override

    Returns:
        True if dataset directory exists, False otherwise
    """
    # Get HF_LEROBOT_HOME from environment or use default
    from pathlib import Path

    hf_home = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    default_lerobot_home = Path(hf_home) / "lerobot"
    hf_lerobot_home = Path(os.environ.get("HF_LEROBOT_HOME", default_lerobot_home)).expanduser()

    # Calculate dataset path (same logic as LeRobotDataset)
    dataset_root = Path(root) if root is not None else hf_lerobot_home
    dataset_path = dataset_root / repo_id

    return dataset_path.exists()


@click.group(invoke_without_command=True)
@click.pass_context
@click.option("--set-robot", is_flag=True, help="Configure robot settings (so101_follower)")
@click.option("--set-teleop", is_flag=True, help="Configure teleop settings (so101_leader)")
@click.option("--show-config", is_flag=True, help="Show current configuration")
def cli(ctx, set_robot, set_teleop, show_config):
    """Chop - Centralized configuration manager for LeRobot commands."""

    if set_robot:
        configure_robot()
    elif set_teleop:
        configure_teleop()
    elif show_config:
        display_config()
    elif ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


def configure_robot():
    """Configure so101_follower robot settings."""
    click.echo("\n🤖 Configuring robot: so101_follower\n")

    config = {"type": "so101_follower"}

    # Port configuration
    config["port"] = click.prompt("Robot port", default="/dev/tty.usbmodem5A460814531")
    config["id"] = click.prompt("Robot ID", default="so101_follower")

    # Camera configuration (optional - can be set later via edit-config)
    click.echo("\nTo configure cameras, edit the config file with: chop edit-config robot")
    click.echo(f"Or set them manually in: {ROBOT_CONFIG_FILE}")

    save_config(ROBOT_CONFIG_FILE, config)


def configure_teleop():
    """Configure so101_leader teleoperator settings."""
    click.echo("\n🎮 Configuring teleoperator: so101_leader\n")

    config = {"type": "so101_leader"}

    # Port configuration
    config["port"] = click.prompt("Teleop port", default="/dev/tty.usbmodem5A460823651")
    config["id"] = click.prompt("Teleop ID", default="so101_leader")

    save_config(TELEOP_CONFIG_FILE, config)


def display_config():
    """Display current configuration."""
    click.echo("\n📋 Current Configuration\n")

    robot_config = load_config(ROBOT_CONFIG_FILE)
    teleop_config = load_config(TELEOP_CONFIG_FILE)

    if robot_config:
        click.echo("🤖 Robot Config:")
        click.echo(json.dumps(robot_config, indent=2))
        click.echo()
    else:
        click.echo("🤖 Robot Config: Not configured")
        click.echo()

    if teleop_config:
        click.echo("🎮 Teleop Config:")
        click.echo(json.dumps(teleop_config, indent=2))
        click.echo()
    else:
        click.echo("🎮 Teleop Config: Not configured")
        click.echo()

    if not any([robot_config, teleop_config]):
        click.echo("No configuration found. Use --set-robot or --set-teleop to configure.")


@cli.command()
@click.option("--dataset.repo_id", "dataset_repo_id", required=True, help="Dataset repository ID")
@click.option("--dataset.num_episodes", "dataset_num_episodes", type=int, help="Number of episodes")
@click.option("--dataset.single_task", "dataset_single_task", help="Single task description")
@click.option(
    "--dataset.episode_time_s", "dataset_episode_time_s", type=int, help="Episode duration in seconds"
)
@click.option("--dataset.reset_time_s", "dataset_reset_time_s", type=int, help="Reset duration in seconds")
@click.option("--policy.path", "policy_path", help="Path to pretrained policy for testing")
@click.option(
    "--policy.device", "policy_device", default="mps", help="Device for policy inference (default: mps)"
)
@click.option("--intervention_repo_id", help="Record intervention fragments to a separate dataset")
@click.option("--display_data", is_flag=True, help="Display data during recording")
@click.argument("extra_args", nargs=-1, type=click.UNPROCESSED)
def record(
    dataset_repo_id,
    dataset_num_episodes,
    dataset_single_task,
    dataset_episode_time_s,
    dataset_reset_time_s,
    policy_path,
    policy_device,
    intervention_repo_id,
    display_data,
    extra_args,
):
    """Record episodes with the robot. Automatically resumes if dataset exists.

    Pass additional args after -- (e.g., -- --dataset.push_to_hub=false)."""
    import shlex
    import tempfile

    robot_config = load_config(ROBOT_CONFIG_FILE)
    teleop_config = load_config(TELEOP_CONFIG_FILE)

    if not robot_config:
        click.echo("❌ Robot not configured. Run: chop --set-robot", err=True)
        sys.exit(1)

    if not teleop_config:
        click.echo("❌ Teleop not configured. Run: chop --set-teleop", err=True)
        sys.exit(1)

    # Enable intervention on teleop if intervention_repo_id is set
    if intervention_repo_id:
        teleop_config = {**teleop_config, "intervention_enabled": True}

    # Auto-detect resume based on dataset existence
    dataset_exists = check_dataset_exists(dataset_repo_id)

    if dataset_exists:
        click.echo(f"ℹ️  Found existing dataset, resuming: {dataset_repo_id}")
    else:
        click.echo(f"ℹ️  Starting new dataset: {dataset_repo_id}")

    # Build command
    cmd = ["lerobot-record"]

    # If cameras are present, use config file (Draccus can't parse nested ChoiceRegistry from CLI)
    has_cameras = "cameras" in robot_config and robot_config["cameras"]

    if has_cameras:
        # Create temp config file with all configs
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, prefix="chop_record_") as f:
            config = {"robot": robot_config, "teleop": teleop_config, "dataset": {"repo_id": dataset_repo_id}}

            if dataset_num_episodes is not None:
                config["dataset"]["num_episodes"] = dataset_num_episodes
            if dataset_single_task:
                config["dataset"]["single_task"] = dataset_single_task
            if dataset_episode_time_s is not None:
                config["dataset"]["episode_time_s"] = dataset_episode_time_s
            if dataset_reset_time_s is not None:
                config["dataset"]["reset_time_s"] = dataset_reset_time_s
            if dataset_exists:
                config["resume"] = True
            if display_data:
                config["display_data"] = True
            if intervention_repo_id:
                config["intervention_repo_id"] = intervention_repo_id

            json.dump(config, f, indent=2)
            temp_config_path = f.name

        try:
            cmd.append(f"--config_path={temp_config_path}")

            if policy_path:
                cmd.append(f"--policy.path={shlex.quote(policy_path)}")
                cmd.append(f"--policy.device={policy_device}")

            # Add any extra passthrough args
            if extra_args:
                cmd.extend(extra_args)

            # Execute command
            cmd_str = " \\\n  ".join(cmd)
            click.echo(f"\n🚀 Running command:\n{cmd_str}\n")
            subprocess.run(" ".join(cmd), shell=True, check=True)
        finally:
            Path(temp_config_path).unlink(missing_ok=True)
    else:
        # No cameras, use CLI args
        cmd.extend(build_cli_args("robot", robot_config))
        cmd.extend(build_cli_args("teleop", teleop_config))

        cmd.append(f"--dataset.repo_id={dataset_repo_id}")
        if dataset_num_episodes is not None:
            cmd.append(f"--dataset.num_episodes={dataset_num_episodes}")
        if dataset_single_task:
            cmd.append(f"--dataset.single_task={shlex.quote(dataset_single_task)}")
        if dataset_episode_time_s is not None:
            cmd.append(f"--dataset.episode_time_s={dataset_episode_time_s}")
        if dataset_reset_time_s is not None:
            cmd.append(f"--dataset.reset_time_s={dataset_reset_time_s}")

        if policy_path:
            cmd.append(f"--policy.path={shlex.quote(policy_path)}")
            cmd.append(f"--policy.device={policy_device}")

        if dataset_exists:
            cmd.append("--resume=true")

        if display_data:
            cmd.append("--display_data=true")

        if intervention_repo_id:
            cmd.append(f"--intervention_repo_id={intervention_repo_id}")

        # Add any extra passthrough args
        if extra_args:
            cmd.extend(extra_args)

        # Execute command
        cmd_str = " \\\n  ".join(cmd)
        click.echo(f"\n🚀 Running command:\n{cmd_str}\n")

        try:
            subprocess.run(" ".join(cmd), shell=True, check=True)
        except subprocess.CalledProcessError as e:
            click.echo(f"❌ Command failed with exit code {e.returncode}", err=True)
            sys.exit(e.returncode)


@cli.command()
@click.option("--display_data", is_flag=True, help="Display data during teleoperation")
@click.option("--fps", type=int, help="Frames per second")
@click.option("--teleop_time_s", type=float, help="Teleoperation duration in seconds")
def teleop(display_data, fps, teleop_time_s):
    """Teleoperate the robot."""
    import tempfile

    robot_config = load_config(ROBOT_CONFIG_FILE)
    teleop_config = load_config(TELEOP_CONFIG_FILE)

    if not robot_config:
        click.echo("❌ Robot not configured. Run: chop --set-robot", err=True)
        sys.exit(1)

    if not teleop_config:
        click.echo("❌ Teleop not configured. Run: chop --set-teleop", err=True)
        sys.exit(1)

    # Build command
    cmd = ["lerobot-teleoperate"]

    # If cameras are present, use config file (Draccus can't parse nested ChoiceRegistry from CLI)
    has_cameras = "cameras" in robot_config and robot_config["cameras"]

    if has_cameras:
        # Create temp config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, prefix="chop_teleop_") as f:
            config = {"robot": robot_config, "teleop": teleop_config}

            if display_data:
                config["display_data"] = True
            if fps is not None:
                config["fps"] = fps
            if teleop_time_s is not None:
                config["teleop_time_s"] = teleop_time_s

            json.dump(config, f, indent=2)
            temp_config_path = f.name

        try:
            cmd.append(f"--config_path={temp_config_path}")

            # Execute command
            cmd_str = " \\\n  ".join(cmd)
            click.echo(f"\n🚀 Running command:\n{cmd_str}\n")
            subprocess.run(" ".join(cmd), shell=True, check=True)
        finally:
            Path(temp_config_path).unlink(missing_ok=True)
    else:
        # No cameras, use CLI args
        cmd.extend(build_cli_args("robot", robot_config))
        cmd.extend(build_cli_args("teleop", teleop_config))

        if display_data:
            cmd.append("--display_data=true")

        if fps is not None:
            cmd.append(f"--fps={fps}")

        if teleop_time_s is not None:
            cmd.append(f"--teleop_time_s={teleop_time_s}")

        # Execute command
        cmd_str = " \\\n  ".join(cmd)
        click.echo(f"\n🚀 Running command:\n{cmd_str}\n")

        try:
            subprocess.run(" ".join(cmd), shell=True, check=True)
        except subprocess.CalledProcessError as e:
            click.echo(f"❌ Command failed with exit code {e.returncode}", err=True)
            sys.exit(e.returncode)


@cli.command()
@click.option("--dataset.repo_id", "dataset_repo_id", required=True, help="Dataset repository ID")
@click.option(
    "--dataset.episode", "dataset_episode", type=int, required=True, help="Episode number to replay"
)
def replay(dataset_repo_id, dataset_episode):
    """Replay an episode on the robot (cameras not needed for replay)."""

    robot_config = load_config(ROBOT_CONFIG_FILE)

    if not robot_config:
        click.echo("❌ Robot not configured. Run: chop --set-robot", err=True)
        sys.exit(1)

    # Build command (exclude cameras - they're not needed for replay)
    cmd = ["lerobot-replay"]

    # Build robot args excluding cameras
    # Cameras are optional for replay and can't be passed via CLI due to Draccus ChoiceRegistry limitations
    for key, value in robot_config.items():
        if key == "cameras":
            continue  # Skip cameras
        elif isinstance(value, bool):
            cmd.append(f"--robot.{key}={str(value).lower()}")
        elif value is not None:
            cmd.append(f"--robot.{key}={value}")

    cmd.append(f"--dataset.repo_id={dataset_repo_id}")
    cmd.append(f"--dataset.episode={dataset_episode}")

    # Execute command
    cmd_str = " \\\n  ".join(cmd)
    click.echo(f"\n🚀 Running command:\n{cmd_str}\n")

    try:
        subprocess.run(" ".join(cmd), shell=True, check=True)
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Command failed with exit code {e.returncode}", err=True)
        sys.exit(e.returncode)


@cli.command(name="identify-ports")
def identify_ports():
    """Identify robot ports and automatically update configuration files for SO101."""
    import time

    try:
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus
    except ImportError:
        click.echo(
            "❌ Could not import lerobot modules. Make sure you're in the lerobot environment.", err=True
        )
        sys.exit(1)

    def test_port(port: str):
        """Test if an SO101 robot is connected to the given port by moving its base motor."""
        # SO101 has 6 motors (no forearm_roll like SO107)
        motors = {
            "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
            "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
            "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
            "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
            "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
            "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
        }

        try:
            click.echo(f"\nTesting {port}...")
            bus = FeetechMotorsBus(port=port, motors=motors)
            bus.connect()

            current_pos = bus.read("Present_Position", "shoulder_pan", normalize=False)
            click.echo("Moving base motor... WATCH YOUR ROBOT!")
            movement_ticks = 200

            bus.write("Goal_Position", "shoulder_pan", current_pos + movement_ticks, normalize=False)
            time.sleep(0.8)
            bus.write("Goal_Position", "shoulder_pan", current_pos - movement_ticks, normalize=False)
            time.sleep(0.8)
            bus.write("Goal_Position", "shoulder_pan", current_pos, normalize=False)
            time.sleep(0.3)

            bus.disconnect()
            return True

        except Exception as e:
            click.echo(f"  ✗ Failed: {e}")
            return False

    click.echo("=" * 60)
    click.echo("LeRobot SO101 Port Identifier")
    click.echo("=" * 60)
    click.echo("\nThis script will scan for USB serial ports.")
    click.echo("Watch which arm moves, then identify it:")
    click.echo("  L = Leader (teleop arm)")
    click.echo("  F = Follower (robot arm)")
    click.echo("=" * 60)

    # Scan for available ports (macOS style)
    import glob

    available_ports = sorted(glob.glob("/dev/tty.usbmodem*"))
    if not available_ports:
        # Try Linux-style ports
        available_ports = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))

    if not available_ports:
        click.echo("\n❌ No USB serial ports found!")
        click.echo("   Make sure your arms are connected via USB.")
        sys.exit(1)

    click.echo(f"\nFound {len(available_ports)} port(s): {', '.join(available_ports)}")

    port_mapping = {}

    for port in available_ports:
        if test_port(port):
            while True:
                response = (
                    click.prompt("\nWhich arm moved? [L/F] (or 'skip')", type=str, default="")
                    .strip()
                    .upper()
                )
                if response in ["L", "F"]:
                    port_mapping[response] = port
                    click.echo(f"  ✓ Recorded: {response} -> {port}")
                    break
                elif response == "SKIP":
                    click.echo("  Skipped")
                    break
                else:
                    click.echo("  Invalid input. Please enter L (Leader), F (Follower), or skip")

    # Display results
    click.echo("\n" + "=" * 60)
    click.echo("PORT CONFIGURATION")
    click.echo("=" * 60)

    if len(port_mapping) == 2:
        click.echo("\nDetected configuration:")
        click.echo(f"  Robot (Follower):  {port_mapping['F']}")
        click.echo(f"  Teleop (Leader):   {port_mapping['L']}")

        # Update configurations
        robot_config = {"type": "so101_follower", "port": port_mapping["F"], "id": "so101_follower"}
        teleop_config = {"type": "so101_leader", "port": port_mapping["L"], "id": "so101_leader"}

        # Save configs
        save_config(ROBOT_CONFIG_FILE, robot_config)
        save_config(TELEOP_CONFIG_FILE, teleop_config)

        click.echo("\n✅ Configuration files updated successfully!")
        click.echo(f"   Robot config: {ROBOT_CONFIG_FILE}")
        click.echo(f"   Teleop config: {TELEOP_CONFIG_FILE}")
    else:
        click.echo("\n⚠️  Warning: Not all arms identified. Configuration not updated.")
        click.echo("   Identified arms:")
        for arm_type, port in sorted(port_mapping.items()):
            arm_name = {"L": "Leader (Teleop)", "F": "Follower (Robot)"}
            click.echo(f"     {arm_name.get(arm_type, arm_type)}: {port}")
        click.echo("\n   Please run the command again to identify all arms.")

    click.echo("=" * 60 + "\n")


@cli.command(name="find-cameras")
def find_cameras():
    """Detect cameras and interactively assign them to roles with live video feed."""
    import cv2

    try:
        from lerobot.cameras.configs import ColorMode
        from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
        from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
    except ImportError:
        click.echo(
            "❌ Could not import lerobot modules. Make sure you're in the lerobot environment.", err=True
        )
        sys.exit(1)

    click.echo("=" * 60)
    click.echo("LeRobot Camera Detector & Configurator")
    click.echo("=" * 60)
    click.echo("\nDetecting cameras...")

    # Detect all cameras
    all_cameras = []

    # Find OpenCV cameras
    try:
        opencv_cameras = OpenCVCamera.find_cameras()
        all_cameras.extend(opencv_cameras)
        click.echo(f"Found {len(opencv_cameras)} OpenCV camera(s)")
    except Exception as e:
        click.echo(f"⚠️  Error detecting OpenCV cameras: {e}")

    # Find RealSense cameras
    try:
        realsense_cameras = RealSenseCamera.find_cameras()
        all_cameras.extend(realsense_cameras)
        click.echo(f"Found {len(realsense_cameras)} RealSense camera(s)")
    except Exception as e:
        click.echo(f"⚠️  Error detecting RealSense cameras: {e}")

    if not all_cameras:
        click.echo("\n❌ No cameras detected!")
        sys.exit(1)

    click.echo(f"\nTotal cameras detected: {len(all_cameras)}")
    click.echo("=" * 60)

    camera_mapping = {}
    # Default roles - users can add custom ones
    available_roles = ["top", "left_wrist", "right_wrist"]

    for i, cam_info in enumerate(all_cameras):
        cam_type = cam_info.get("type")
        cam_id = cam_info.get("id")
        cam_name = cam_info.get("name", f"{cam_type} {cam_id}")

        click.echo(f"\n{'=' * 60}")
        click.echo(f"Camera #{i + 1}: {cam_name}")
        click.echo(f"  Type: {cam_type}")
        click.echo(f"  ID: {cam_id}")

        # Create camera instance and show video feed
        camera = None
        try:
            click.echo("  Connecting to camera...")

            if cam_type == "OpenCV":
                config = OpenCVCameraConfig(index_or_path=cam_id, color_mode=ColorMode.RGB)
                camera = OpenCVCamera(config)
            elif cam_type == "RealSense":
                config = RealSenseCameraConfig(serial_number_or_name=cam_id, color_mode=ColorMode.RGB)
                camera = RealSenseCamera(config)
            else:
                click.echo(f"  ⚠️  Unknown camera type: {cam_type}, skipping...")
                continue

            camera.connect(warmup=True)

            # Show live video feed
            window_name = f"Camera #{i + 1}: {cam_name} - Press any key to close"
            click.echo("  ✓ Showing live video feed. Press any key in the video window to close.")

            while True:
                image_array = camera.read()
                # Convert RGB to BGR for OpenCV display
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                # Add text overlay
                cv2.putText(
                    image_bgr,
                    f"Camera #{i + 1}: {cam_name}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    image_bgr,
                    "Press any key to close",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    1,
                )
                cv2.imshow(window_name, image_bgr)

                # Check for key press (wait 30ms)
                if cv2.waitKey(30) != -1:
                    break

            cv2.destroyWindow(window_name)
            camera.disconnect()

        except Exception as e:
            click.echo(f"  ✗ Failed to connect to camera: {e}")
            click.echo("  Skipping this camera...")
            if camera:
                try:
                    camera.disconnect()
                except Exception:
                    pass
            cv2.destroyAllWindows()
            continue

        # Ask user to assign role
        click.echo(f"\n  Available roles: {', '.join(available_roles)}")
        click.echo("  (You can also enter a new custom role name)")
        while True:
            response = (
                click.prompt("  Assign this camera to a role (or 'skip' to skip)", type=str, default="")
                .strip()
                .lower()
            )

            if response == "skip" or response == "":
                click.echo("  Skipped")
                break

            # Check if role is already used in this session
            if response in camera_mapping:
                click.echo(f"  ⚠️  Role '{response}' already assigned. Choose a different name.")
                continue

            # If it's a new custom role, confirm with the user
            if response not in available_roles:
                if click.confirm(f"  Add '{response}' as a new camera role?", default=True):
                    available_roles.append(response)
                else:
                    click.echo(f"  Choose from: {', '.join(available_roles)} or enter a new name")
                    continue

            # Extract configuration based on camera type
            if cam_type == "OpenCV":
                # Extract index from path if it's /dev/videoX
                if isinstance(cam_id, str) and cam_id.startswith("/dev/video"):
                    index = int(cam_id.replace("/dev/video", ""))
                else:
                    index = cam_id

                cam_config = {
                    "type": "opencv",
                    "index_or_path": index,
                    "width": cam_info.get("default_stream_profile", {}).get("width", 640),
                    "height": cam_info.get("default_stream_profile", {}).get("height", 480),
                    "fps": int(cam_info.get("default_stream_profile", {}).get("fps", 30)),
                }
            elif cam_type == "RealSense":
                use_depth = click.confirm("  Enable depth for this camera?", default=True)
                cam_config = {
                    "type": "intelrealsense",
                    "serial_number_or_name": cam_id,
                    "width": cam_info.get("default_stream_profile", {}).get("width", 640),
                    "height": cam_info.get("default_stream_profile", {}).get("height", 480),
                    "fps": cam_info.get("default_stream_profile", {}).get("fps", 30),
                    "use_depth": use_depth,
                }

            camera_mapping[response] = cam_config
            # Remove from available roles if it was a predefined one
            if response in available_roles:
                available_roles.remove(response)
            click.echo(f"  ✓ Assigned to '{response}'")
            break

    # Display results
    click.echo(f"\n{'=' * 60}")
    click.echo("CAMERA CONFIGURATION")
    click.echo("=" * 60)

    if camera_mapping:
        click.echo("\nAssigned cameras:")
        for role, config in camera_mapping.items():
            click.echo(
                f"  {role}: {config['type']} ({config.get('serial_number_or_name') or config.get('index_or_path')})"
            )

        # Update robot configuration
        robot_config = load_config(ROBOT_CONFIG_FILE)

        if not robot_config:
            robot_config = {"type": "so101_follower"}

        robot_config["cameras"] = camera_mapping

        save_config(ROBOT_CONFIG_FILE, robot_config)

        click.echo("\n✅ Robot configuration updated with camera settings!")
        click.echo(f"   Config file: {ROBOT_CONFIG_FILE}")
    else:
        click.echo("\n⚠️  No cameras assigned. Configuration not updated.")

    cv2.destroyAllWindows()
    click.echo("=" * 60 + "\n")


@cli.command(name="edit-config")
@click.argument("config_type", type=click.Choice(["robot", "teleop"]))
def edit_config(config_type):
    """Edit configuration files directly."""

    config_map = {
        "robot": ROBOT_CONFIG_FILE,
        "teleop": TELEOP_CONFIG_FILE,
    }

    config_file = config_map[config_type]

    if not config_file.exists():
        click.echo(f"Configuration file does not exist: {config_file}")
        click.echo("Creating empty configuration file...")
        save_config(config_file, {})

    # Open in default editor
    editor = os.environ.get("EDITOR", "vim")
    try:
        subprocess.run([editor, str(config_file)], check=True)
        click.echo(f"✓ Configuration updated: {config_file}")
    except subprocess.CalledProcessError:
        click.echo("❌ Failed to edit configuration", err=True)
        sys.exit(1)


@cli.command()
@click.option("--dataset.repo_id", "dataset_repo_id", help="Dataset repository ID (not used when resuming)")
@click.option("--policy.type", "policy_type", help="Policy type (e.g., act, pi05, smolvla)")
@click.option("--policy.repo_id", "policy_repo_id", help="Policy repository ID for saving")
@click.option(
    "--output_dir", help="Override output directory (default: outputs/{policy_type}_{dataset_name})"
)
@click.option("--batch_size", type=int, help="Training batch size")
@click.option("--steps", type=int, help="Number of training steps")
@click.option("--save_freq", type=int, help="Checkpoint save frequency")
@click.option("--policy.device", "policy_device", default="cuda", help="Device for training (default: cuda)")
@click.option(
    "--wandb/--no-wandb", "wandb_enable", default=True, help="Enable/disable wandb logging (default: enabled)"
)
@click.option(
    "--policy.pretrained_path", "policy_pretrained_path", help="Path to pretrained policy checkpoint"
)
@click.argument("extra_args", nargs=-1, type=click.UNPROCESSED)
def train(
    dataset_repo_id,
    policy_type,
    policy_repo_id,
    output_dir,
    batch_size,
    steps,
    save_freq,
    policy_device,
    wandb_enable,
    policy_pretrained_path,
    extra_args,
):
    """Train a policy on a dataset. Pass through additional args like --policy.n_obs_steps, --peft.method_type, etc."""

    # Determine output directory first
    if output_dir:
        output_path = Path(output_dir).expanduser()
    elif policy_type and dataset_repo_id:
        dataset_name = dataset_repo_id.split("/")[-1]
        output_dir = f"outputs/{policy_type}_{dataset_name}"
        output_path = Path(output_dir).expanduser()
    else:
        # Check if we can infer from output_dir in extra_args
        output_path = None
        for arg in extra_args:
            if arg.startswith("--output_dir="):
                output_dir = arg.split("=", 1)[1]
                output_path = Path(output_dir).expanduser()
                break

        if not output_path:
            click.echo(
                "❌ Either --output_dir or both --policy.type and --dataset.repo_id are required", err=True
            )
            sys.exit(1)

    # Check if output directory exists and has checkpoints
    resume_mode = False
    config_path = None

    if output_path.exists():
        checkpoint_dir = output_path / "checkpoints" / "last" / "pretrained_model"
        train_config = checkpoint_dir / "train_config.json"

        if train_config.exists():
            click.echo(f"\n⚠️  Found existing training directory with checkpoint: {output_dir}")
            click.echo("What would you like to do?")
            click.echo("  [r] Resume training from last checkpoint")
            click.echo("  [d] Delete and start fresh")

            while True:
                choice = click.prompt("\nYour choice", type=str, default="").strip().lower()
                if choice == "r":
                    resume_mode = True
                    config_path = train_config
                    click.echo("✓ Resuming training from checkpoint")
                    click.echo(f"   Config: {config_path}")
                    break
                elif choice == "d":
                    import shutil

                    click.echo(f"Deleting directory: {output_path}")
                    shutil.rmtree(output_path)
                    click.echo("✓ Directory deleted, starting fresh")
                    break
                else:
                    click.echo("Invalid choice. Please enter 'r' to resume or 'd' to delete.")
        else:
            click.echo(f"ℹ️  Found existing output directory (no checkpoint config): {output_dir}")
            click.echo("   Continuing (will fail if incompatible)")

    # Build command
    cmd = ["lerobot-train"]

    if resume_mode:
        # Resume mode: only use config_path, resume flag, and steps
        # All other parameters are loaded from the saved config
        click.echo("\n⚠️  Resume mode: Only --steps can be overridden. All other parameters ignored.")

        cmd.append(f"--config_path={config_path}")
        cmd.append("--resume=true")

        # Only allow overriding steps when resuming
        if steps is not None:
            cmd.append(f"--steps={steps}")

        # Ignore all other options and extra args in resume mode
        if extra_args:
            click.echo(f"⚠️  Ignoring extra args in resume mode: {' '.join(extra_args)}")

    else:
        # Fresh training: validate required params and build full command
        if not dataset_repo_id:
            click.echo("❌ --dataset.repo_id is required for fresh training", err=True)
            sys.exit(1)
        if not policy_type:
            click.echo("❌ --policy.type is required for fresh training", err=True)
            sys.exit(1)
        if not policy_repo_id:
            click.echo("❌ --policy.repo_id is required for fresh training", err=True)
            sys.exit(1)

        # Add core options
        cmd.append(f"--dataset.repo_id={dataset_repo_id}")
        cmd.append(f"--policy.type={policy_type}")
        cmd.append(f"--policy.repo_id={policy_repo_id}")
        cmd.append(f"--output_dir={output_dir}")
        cmd.append(f"--policy.device={policy_device}")

        if batch_size is not None:
            cmd.append(f"--batch_size={batch_size}")

        if steps is not None:
            cmd.append(f"--steps={steps}")

        if save_freq is not None:
            cmd.append(f"--save_freq={save_freq}")

        # Wandb
        cmd.append(f"--wandb.enable={str(wandb_enable).lower()}")

        # Pretrained path
        if policy_pretrained_path:
            cmd.append(f"--policy.pretrained_path={policy_pretrained_path}")

        # Add extra args
        if extra_args:
            cmd.extend(extra_args)

    # Execute command
    cmd_str = " \\\n  ".join(cmd)
    click.echo(f"\n🚀 Running command:\n{cmd_str}\n")

    try:
        subprocess.run(" ".join(cmd), shell=True, check=True)
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Command failed with exit code {e.returncode}", err=True)
        sys.exit(e.returncode)


if __name__ == "__main__":
    cli()
