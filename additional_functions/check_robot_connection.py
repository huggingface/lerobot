#!/usr/bin/env python3
"""
Check if robot is reachable and responding to DDS messages.
This helps diagnose the [ClientStub] send request error.
"""

import time
import sys

sys.path.append('/Users/nepyope/Documents/unitree/unitree_IL_lerobot/unitree_sdk2_python')

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

def check_connection(network_interface="en7"):
    """Check connection to robot step by step."""
    print("=" * 60)
    print("G1 Robot Connection Diagnostic")
    print("=" * 60)
    print(f"Network interface: {network_interface}")
    print()
    
    # Step 1: Initialize DDS
    print("[1/5] Initializing DDS ChannelFactory...")
    try:
        ChannelFactoryInitialize(0, network_interface)
        print("      SUCCESS: ChannelFactory initialized")
    except Exception as e:
        print(f"      FAILED: {e}")
        return False
    
    # Step 2: Wait for discovery
    print("[2/5] Waiting for DDS discovery (5 seconds)...")
    print("      (This allows DDS to find the robot on the network)")
    time.sleep(5)
    print("      SUCCESS: Discovery period complete")
    
    # Step 3: Create LocoClient
    print("[3/5] Creating LocoClient...")
    try:
        loco_client = LocoClient()
        print("      SUCCESS: LocoClient created")
    except Exception as e:
        print(f"      FAILED: {e}")
        return False
    
    # Step 4: Set timeout and initialize
    print("[4/5] Initializing LocoClient (timeout=10s)...")
    try:
        loco_client.SetTimeout(10.0)
        loco_client.Init()
        print("      SUCCESS: LocoClient initialized")
    except Exception as e:
        print(f"      FAILED: {e}")
        return False
    
    # Step 5: Test command
    print("[5/5] Sending test command (ZeroTorque)...")
    print("      This will show if robot is listening...")
    try:
        result = loco_client.ZeroTorque()
        print(f"      SUCCESS: Command sent, result: {result}")
        print()
        print("=" * 60)
        print("CONNECTION TEST PASSED!")
        print("=" * 60)
        print("Your robot is connected and responding.")
        print("The test_damp_zero_torque.py script should work now.")
        return True
    except Exception as e:
        print(f"      FAILED: {e}")
        print()
        print("=" * 60)
        print("CONNECTION TEST FAILED")
        print("=" * 60)
        print()
        print("Troubleshooting:")
        print("  1. Check robot is powered on")
        print("  2. Check network connection:")
        print(f"     sudo ifconfig {network_interface} 192.168.123.100 netmask 255.255.255.0")
        print("  3. Ping robot: ping 192.168.123.164")
        print("  4. Check robot mode (should be 'ai' mode)")
        print("  5. Check if another program is controlling the robot")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        interface = sys.argv[1]
    else:
        interface = "en7"
    
    check_connection(interface)
