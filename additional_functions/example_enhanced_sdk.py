#!/usr/bin/env python3
"""
Simple example using the Enhanced LocoClient with cleanup.

This demonstrates the enhanced SDK with proper resource management.
"""

import sys
import time

sys.path.append('/Users/nepyope/Documents/unitree/unitree_IL_lerobot/unitree_sdk2_python')

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client_enhanced import LocoClient


def main():
    network_interface = "en7" if len(sys.argv) < 2 else sys.argv[1]
    
    print("=" * 70)
    print("G1 Enhanced SDK Example: Damp (1s) → Zero Torque (1s)")
    print("=" * 70)
    print(f"Network: {network_interface}\n")
    
    try:
        # Initialize DDS
        print("[1/4] Initializing DDS...")
        ChannelFactoryInitialize(0, network_interface)
        time.sleep(3)
        print("      ✓ DDS ready\n")
        
        # Use context manager for automatic cleanup
        print("[2/4] Connecting to robot...")
        with LocoClient(auto_damp_on_close=True) as client:
            client.SetTimeout(5.0)
            print("      ✓ Connected\n")
            
            # Commands
            print("[3/4] Executing commands...")
            
            print("      → DAMP mode...")
            client.Damp()
            time.sleep(1)
            print("      ✓ Damp complete")
            
            print("      → ZERO TORQUE mode...")
            client.ZeroTorque()
            time.sleep(1)
            print("      ✓ Zero torque complete\n")
            
            print("[4/4] Disconnecting...")
            # Context manager will auto-cleanup and damp here
        
        print("      ✓ Disconnected safely\n")
        print("=" * 70)
        print("SUCCESS: All operations completed")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
