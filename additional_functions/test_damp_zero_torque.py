#!/usr/bin/env python3
"""
Test script to switch between damp and zero torque mode every 2 seconds.
This script demonstrates switching between the safest robot modes.
"""

import time
import sys
import signal

# Add the unitree_sdk2_python path to sys.path
sys.path.append('/Users/nepyope/Documents/unitree/unitree_IL_lerobot/unitree_sdk2_python')

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient


class DampZeroTorqueTester:
    def __init__(self):
        # Initialize locomotion client directly (like the working examples)
        self.loco_client = LocoClient()
        self.loco_client.SetTimeout(10.0)  # Increased timeout to allow robot to connect
        self.loco_client.Init()
        self.running = True

        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully."""
        print("\nReceived interrupt signal. Shutting down...")
        self.running = False

    def switch_to_damp(self):
        """Switch robot to damping mode."""
        try:
            print("[Command] Switching to DAMP mode (motors off)...")
            result = self.loco_client.Damp()
            print(f"[Result] Damp command result: {result}")
            return True
        except Exception as e:
            print(f"[Error] Failed to switch to damp mode: {e}")
            return False

    def switch_to_zero_torque(self):
        """Switch robot to zero torque mode."""
        try:
            print("[Command] Switching to ZERO TORQUE mode (motors on, no torque)...")
            result = self.loco_client.ZeroTorque()
            print(f"[Result] ZeroTorque command result: {result}")
            return True
        except Exception as e:
            print(f"[Error] Failed to switch to zero torque mode: {e}")
            return False

    def run_test(self):
        """Run exactly 2 cycles: damp → zero_torque → damp → zero_torque."""
        print("=" * 60)
        print("G1 Robot Damp/Zero Torque Mode Switcher")
        print("=" * 60)
        print("Running exactly 2 cycles:")
        print("   Cycle 1: DAMP → ZERO TORQUE")
        print("   Cycle 2: ZERO TORQUE → DAMP → ZERO TORQUE")
        print("   Final mode: ZERO TORQUE")
        print("\nWARNING: Ensure robot is in a safe position!")
        print("Press Ctrl+C to stop safely")
        print("=" * 60)

        # Wait for user confirmation
        input("Press Enter to start the test...")

        start_time = time.time()

        # Initial setup - start in damp mode
        print(f"\nStarting test...")
        if not self.switch_to_damp():
            print("Failed to initialize damp mode. Aborting.")
            return

        # Cycle 1: DAMP → ZERO TORQUE
        print(f"\nCycle 1: DAMP → ZERO TORQUE")
        print("Waiting 2 seconds...")
        time.sleep(2)

        if not self.running:
            return

        if not self.switch_to_zero_torque():
            print("Failed to switch to zero torque in cycle 1.")
            return

        # Cycle 2: ZERO TORQUE → DAMP → ZERO TORQUE
    
        print(f"\nCycle 2: ZERO TORQUE → DAMP → ZERO TORQUE")
        print("Waiting 2 seconds...")
        time.sleep(2)

        if not self.running:
            return

        if not self.switch_to_damp():
            print("Failed to switch to damp in cycle 2.")
            return

        print("Waiting 2 seconds...")
        time.sleep(2)

        if not self.running:
            return

        if not self.switch_to_zero_torque():
            print("Failed to switch to zero torque in cycle 2.")
            return

        # Test completed - robot ends in zero torque mode
        print(f"\nTest completed after 2 cycles")
        print("Final mode: ZERO TORQUE")

        print("\nTest finished!")
        print(f"Total cycles: 2")
        print(f"Total time: {time.time() - start_time:.1f} seconds")


def main():
    print("G1 Robot Damp/Zero Torque Mode Switcher")
    print("This script will run exactly 2 cycles ending in zero torque mode.")

    # Get network interface from command line or use default
    if len(sys.argv) > 1:
        network_interface = sys.argv[1]
        print(f"Using network interface: {network_interface}")
    else:
        network_interface = "en7"  # Default based on your setup
        print(f"Using default network interface: {network_interface}")

    print("\nInitializing connection...")

    tester = None
    try:
        # Initialize the channel factory
        ChannelFactoryInitialize(0, network_interface)
        print("Channel factory initialized")
        
        # Wait for DDS to discover robot (CRITICAL!)
        print("Waiting for DDS discovery (3 seconds)...")
        time.sleep(3)
        print("Discovery complete")

        # Create tester
        tester = DampZeroTorqueTester()
        print("LocoClient created and ready")

        # Run the test
        tester.run_test()
        del tester #calls close()

    except Exception as e:
        print(f"Error during initialization: {e}")
        print("\nTroubleshooting:")
        print("   - Make sure you're connected to the robot's network")
        print("   - Try: sudo ifconfig en7 192.168.123.100 netmask 255.255.255.0")
        print("   - Verify robot is powered on and accessible")
        print("   - Run check_robot_connection.py for detailed diagnostics")


if __name__ == "__main__":
    main()