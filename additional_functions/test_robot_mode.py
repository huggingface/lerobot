#!/usr/bin/env python3
"""
Test script to check G1 robot's current mode and print it to screen.
This script uses MotionSwitcherClient to check the robot's control mode.
"""

import time
import sys
import json

# Add the unitree_sdk2_python path to sys.path
sys.path.append('/Users/nepyope/Documents/unitree/unitree_IL_lerobot/unitree_sdk2_python')

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient


class RobotModeChecker:
    def __init__(self):
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()
    
    def check_mode(self):
        """Check the robot's current mode and return the result."""
        try:
            status, result = self.msc.CheckMode()
            return status, result
        except Exception as e:
            print(f"Error checking mode: {e}")
            return -1, None
    
    def print_mode_info(self):
        """Print detailed mode information to screen."""
        print("=" * 50)
        print("G1 Robot Mode Checker")
        print("=" * 50)
        
        print("\nChecking robot mode...")
        status, result = self.check_mode()
        
        if status == 0 and result is not None:
            print("✅ Successfully connected to robot!")
            print(f"📊 Status Code: {status}")
            print(f"📋 Mode Result: {json.dumps(result, indent=2)}")
            
            # Extract mode name if available
            if isinstance(result, dict) and 'name' in result:
                mode_name = result['name']
                if mode_name:
                    print(f"🤖 Current Mode: {mode_name}")
                    
                    # Provide mode interpretation
                    mode_interpretations = {
                        'ai': 'AI/Autonomous Mode - Ready for external commands',
                        'normal': 'Normal Mode - Basic operation mode',
                        'advanced': 'Advanced Mode - Advanced control mode',
                        'ai-w': 'AI-Wheeled Mode - For wheeled robots',
                        '': 'Safe Mode - No active control mode'
                    }
                    
                    interpretation = mode_interpretations.get(mode_name, 'Unknown Mode')
                    print(f"💡 Interpretation: {interpretation}")
                else:
                    print("🤖 Current Mode: Safe Mode (No active mode)")
                    print("💡 Interpretation: Robot is in safe state, ready to accept commands")
            else:
                print("🤖 Current Mode: Unknown (check result structure)")
                
        elif status == -1:
            print("❌ Failed to connect to robot")
            print("🔧 Troubleshooting:")
            print("   - Check network connection")
            print("   - Verify robot is powered on")
            print("   - Ensure correct network interface (try en7)")
            print("   - Check if robot IP is 192.168.123.164")
        else:
            print(f"❌ Error checking mode. Status: {status}")
            if result:
                print(f"📋 Result: {result}")
        
        print("\n" + "=" * 50)


def main():
    print("G1 Robot Mode Checker")
    print("This script will check the robot's current control mode.")
    
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
        
        # Create mode checker
        checker = RobotModeChecker()
        print("✅ MotionSwitcherClient created")
        
        # Check and print mode
        checker.print_mode_info()
        
        # Optionally check mode multiple times
        print("\n🔄 Checking mode again in 2 seconds...")
        time.sleep(2)
        checker.print_mode_info()
        
    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        print("\n🔧 Troubleshooting:")
        print("   - Make sure you're connected to the robot's network")
        print("   - Try: sudo ifconfig en7 192.168.123.100 netmask 255.255.255.0")
        print("   - Verify robot is powered on and accessible")


if __name__ == "__main__":
    main()
