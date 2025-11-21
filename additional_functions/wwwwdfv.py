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
        # Initialize motion switcher client first
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()
        
        # Check current robot mode
        print("Checking robot mode...")
        status, result = self.msc.CheckMode()
        if status == 0:
            print(f"Current robot mode: {result.get('name', 'unknown')}")
            if result.get('name'):
                print("Releasing current mode...")
                self.msc.ReleaseMode()
                time.sleep(2.0)
        else:
            print(f"Could not check robot mode. Status: {status}")
        
        # Ensure robot is in 'ai' mode for locomotion commands
        print("Setting robot to 'ai' mode...")
        try:
            self.msc.SelectMode("ai")
            time.sleep(1.0)
            print("Robot set to 'ai' mode successfully")
        except Exception as e:
            print(f"Warning: Could not set robot to 'ai' mode: {e}")
            print("Trying to continue anyway...")
        
        # Initialize locomotion client
        self.loco_client = LocoClient()
        self.loco_client.SetTimeout(10.0)
        self.loco_client.Init()
        self.running = True
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully."""
        print("\n🛑 Received interrupt signal. Shutting down safely...")
        self.running = False
        # Release robot control
        try:
            self.msc.ReleaseMode()
            print("🔓 Robot control released")
        except:
            pass
    
    def switch_to_damp(self):
        """Switch robot to damping mode."""
        try:
            print("🔒 Switching to DAMP mode (motors off)...")
            self.loco_client.Damp()
            return True
        except Exception as e:
            print(f"❌ Error switching to damp mode: {e}")
            return False
    
    def switch_to_zero_torque(self):
        """Switch robot to zero torque mode."""
        try:
            print("⚡ Switching to ZERO TORQUE mode (motors on, no torque)...")
            self.loco_client.ZeroTorque()
            return True
        except Exception as e:
            print(f"❌ Error switching to zero torque mode: {e}")
            return False
    
    def run_test(self):
        """Run exactly 2 cycles: damp → zero_torque → damp → zero_torque."""
        print("=" * 60)
        print("🤖 G1 Robot Damp/Zero Torque Mode Switcher")
        print("=" * 60)
        print("🔄 Running exactly 2 cycles:")
        print("   Cycle 1: DAMP → ZERO TORQUE")
        print("   Cycle 2: ZERO TORQUE → DAMP → ZERO TORQUE")
        print("   Final mode: ZERO TORQUE")
        print("\n⚠️  WARNING: Ensure robot is in a safe position!")
        print("🛑 Press Ctrl+C to stop safely")
        print("=" * 60)
        
        # Wait for user confirmation
        input("Press Enter to start the test...")
        
        start_time = time.time()
        
        # Initial setup - start in damp mode
        print(f"\n🚀 Starting test...")
        if not self.switch_to_damp():
            print("❌ Failed to initialize damp mode. Aborting.")
            return
        
        # Cycle 1: DAMP → ZERO TORQUE
        print(f"\n📊 Cycle 1: DAMP → ZERO TORQUE")
        print("⏳ Waiting 2 seconds...")
        time.sleep(2)
        
        if not self.running:
            return
            
        if not self.switch_to_zero_torque():
            print("❌ Failed to switch to zero torque in cycle 1.")
            return
        
        # Cycle 2: ZERO TORQUE → DAMP → ZERO TORQUE
        print(f"\n📊 Cycle 2: ZERO TORQUE → DAMP → ZERO TORQUE")
        print("⏳ Waiting 2 seconds...")
        time.sleep(2)
        
        if not self.running:
            return
            
        if not self.switch_to_damp():
            print("❌ Failed to switch to damp in cycle 2.")
            return
            
        print("⏳ Waiting 2 seconds...")
        time.sleep(2)
        
        if not self.running:
            return
            
        if not self.switch_to_zero_torque():
            print("❌ Failed to switch to zero torque in cycle 2.")
            return
        
        # Test completed - robot ends in zero torque mode
        print(f"\n🏁 Test completed after 2 cycles")
        print("⚡ Final mode: ZERO TORQUE")
        
        # Release robot control to allow other programs to connect
        print("🔓 Releasing robot control...")
        try:
            self.msc.ReleaseMode()
            time.sleep(1.0)
            # Release again to ensure complete cleanup
            self.msc.ReleaseMode()
            print("✅ Robot control released successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not release robot control: {e}")
        
        # Final check - ensure robot is in a free state
        print("🔍 Final mode check...")
        try:
            status, result = self.msc.CheckMode()
            if status == 0:
                print(f"Final robot mode: {result.get('name', 'unknown')}")
                if result.get('name'):
                    print("🔓 Performing final release...")
                    self.msc.ReleaseMode()
            else:
                print(f"Could not check final mode. Status: {status}")
        except Exception as e:
            print(f"Warning: Could not check final mode: {e}")
        
        print("\n✅ Test finished!")
        print(f"📈 Total cycles: 2")
        print(f"⏱️  Total time: {time.time() - start_time:.1f} seconds")


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
    
    try:
        # Initialize the channel factory
        ChannelFactoryInitialize(0, network_interface)
        print("✅ Channel factory initialized")
        
        # Create tester
        tester = DampZeroTorqueTester()
        print("✅ LocoClient created")
        
        # Run the test
        tester.run_test()
        
    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        print("\n🔧 Troubleshooting:")
        print("   - Make sure you're connected to the robot's network")
        print("   - Try: sudo ifconfig en7 192.168.123.100 netmask 255.255.255.0")
        print("   - Verify robot is powered on and accessible")
        print("   - Check if robot is in 'ai' mode")


if __name__ == "__main__":
    main()
