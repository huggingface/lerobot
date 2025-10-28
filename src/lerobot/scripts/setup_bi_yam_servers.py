#!/usr/bin/env python3
"""
Helper script to launch the bimanual Yam arm servers for use with LeRobot.

This script starts four server processes:
- Two follower arm servers (ports 1234 and 1235)
- Two leader arm servers (ports 5001 and 5002)

The follower servers will be controlled by LeRobot's bi_yam_follower robot.
The leader servers expose the teaching handle positions to LeRobot's bi_yam_leader teleoperator.

Expected CAN interfaces:
- can_follower_r: Right follower arm
- can_follower_l: Left follower arm  
- can_leader_r: Right leader arm (with teaching handle)
- can_leader_l: Left leader arm (with teaching handle)

Usage:
    python -m lerobot.scripts.setup_bi_yam_servers

Requirements:
    - LeRobot installed with yam support: pip install -e ".[yam]"
    - i2rt library (installed automatically with the above command)
    - CAN interfaces configured and available
    - Proper permissions to access CAN devices
"""

import os
import signal
import subprocess
import sys
import time


def check_can_interface(interface):
    """Check if a CAN interface exists and is available."""
    try:
        result = subprocess.run(
            ["ip", "link", "show", interface], capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            return False

        if "state UP" in result.stdout or "state UNKNOWN" in result.stdout:
            return True
        else:
            print(f"Warning: CAN interface {interface} exists but is not UP")
            return False

    except Exception as e:
        print(f"Error checking CAN interface {interface}: {e}")
        return False


def check_all_can_interfaces():
    """Check if all required CAN interfaces exist."""
    required_interfaces = ["can_follower_r", "can_leader_r", "can_follower_l", "can_leader_l"]

    missing_interfaces = []

    for interface in required_interfaces:
        if not check_can_interface(interface):
            missing_interfaces.append(interface)

    if missing_interfaces:
        raise RuntimeError(f"Missing or unavailable CAN interfaces: {', '.join(missing_interfaces)}")

    print("✓ All CAN interfaces are available")
    return True


def find_i2rt_script():
    """Find the i2rt minimum_gello.py script from the installed package."""
    try:
        import i2rt

        i2rt_path = os.path.dirname(i2rt.__file__)
        script_path = os.path.join(os.path.dirname(i2rt_path), "scripts", "minimum_gello.py")
        if os.path.exists(script_path):
            return script_path
    except ImportError:
        raise RuntimeError(
            "Could not import i2rt. Please install it with: pip install -e '.[yam]'"
        )

    raise RuntimeError(
        "Could not find i2rt minimum_gello.py script. "
        "The i2rt installation may be incomplete."
    )


def launch_server_process(can_channel, gripper, mode, server_port):
    """Launch a single server process for a Yam arm."""
    try:
        script_path = find_i2rt_script()
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    cmd = [
        sys.executable,
        script_path,
        "--can_channel",
        can_channel,
        "--gripper",
        gripper,
        "--mode",
        mode,
        "--server_port",
        str(server_port),
    ]

    print(f"Starting: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(cmd)
        return process
    except Exception as e:
        print(f"Error starting process for {can_channel}: {e}")
        return None


def main():
    """Main function to launch all server processes."""
    processes = []

    try:
        # Check CAN interfaces
        print("Checking CAN interfaces...")
        check_all_can_interfaces()

        # Define the server processes to launch
        server_configs = [
            # Right follower arm
            {
                "can_channel": "can_follower_r",
                "gripper": "linear_4310",
                "mode": "follower",
                "server_port": 1234,
            },
            # Left follower arm
            {
                "can_channel": "can_follower_l",
                "gripper": "linear_4310",
                "mode": "follower",
                "server_port": 1235,
            },
            # Right leader arm (teaching handle)
            {
                "can_channel": "can_leader_r",
                "gripper": "yam_teaching_handle",
                "mode": "follower",  # Note: We use follower mode to expose as a read-only server
                "server_port": 5001,
            },
            # Left leader arm (teaching handle)
            {
                "can_channel": "can_leader_l",
                "gripper": "yam_teaching_handle",
                "mode": "follower",  # Note: We use follower mode to expose as a read-only server
                "server_port": 5002,
            },
        ]

        # Launch all processes
        print("\nLaunching server processes...")
        for config in server_configs:
            process = launch_server_process(**config)
            if process:
                processes.append(process)
                print(f"✓ Started process {process.pid} for {config['can_channel']} on port {config['server_port']}")
            else:
                raise RuntimeError(f"Failed to start process for {config['can_channel']}")

        print(f"\n✓ Successfully launched {len(processes)} server processes")
        print("\nServer setup:")
        print("  - Right follower arm: localhost:1234")
        print("  - Left follower arm:  localhost:1235")
        print("  - Right leader arm:   localhost:5001")
        print("  - Left leader arm:    localhost:5002")
        print("\nYou can now use lerobot-record with:")
        print("  --robot.type=bi_yam_follower")
        print("  --teleop.type=bi_yam_leader")
        print("\nPress Ctrl+C to stop all server processes")

        # Wait for processes and handle termination
        try:
            while True:
                # Check if any process has died
                for i, process in enumerate(processes):
                    if process.poll() is not None:
                        print(f"\nProcess {process.pid} has terminated")
                        processes.pop(i)
                        break

                if not processes:
                    print("All processes have terminated")
                    break

                time.sleep(1)

        except KeyboardInterrupt:
            print("\n\nReceived Ctrl+C, terminating all server processes...")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    finally:
        # Clean up: terminate all running processes
        for process in processes:
            try:
                print(f"Terminating process {process.pid}...")
                process.terminate()

                # Wait up to 5 seconds for graceful termination
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"Force killing process {process.pid}...")
                    process.kill()
                    process.wait()

            except Exception as e:
                print(f"Error terminating process {process.pid}: {e}")

        print("All server processes terminated")


if __name__ == "__main__":
    main()

