#!/usr/bin/env python3
"""
Test script to transition through robot modes: zero torque → damp → start → balance stand.
This script demonstrates a sequence of robot mode transitions.
"""

import time
import sys
import signal

# Add the unitree_sdk2_python path to sys.path
sys.path.append('/Users/nepyope/Documents/unitree/unitree_IL_lerobot/unitree_sdk2_python')

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient


class ModeSequenceTester:
    def __init__(self):
        # Initialize motion switcher client first
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()
        
        # Initialize locomotion client
        self.loco_client = LocoClient()
        self.loco_client.SetTimeout(10.0)
        self.loco_client.Init()
        self.running = True
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully."""
        print("\nReceived interrupt signal. Shutting down...")
        self.running = False
        # Just exit immediately
        exit(0)
    
    def reset_robot_state(self):
        """Reset robot to a clean state before starting sequence."""
        try:
            print("Resetting robot to clean state...")
            
            # Check current mode first
            try:
                status, result = self.msc.CheckMode()
                if status == 0:
                    print(f"📊 Current mode: {result.get('name', 'unknown')}")
                    # Release current mode if any
                    if result.get('name'):
                        print(" Releasing current mode...")
                        self.msc.ReleaseMode()
                        time.sleep(2.0)
            except Exception as e:
                print(f" Could not check/release mode: {e}")
            
            # Force reset through locomotion
            print(" Setting to damp mode...")
            self.loco_client.Damp()
            time.sleep(2.0)
            
            print(" Setting to zero torque...")
            self.loco_client.ZeroTorque()
            time.sleep(2.0)
            
            print(" Back to damp mode...")
            self.loco_client.Damp()
            time.sleep(2.0)
            
            print(" Robot state reset complete")
        except Exception as e:
            print(f" Warning: Could not fully reset robot state: {e}")
            # Try minimal reset
            try:
                print(" Attempting minimal reset...")
                self.loco_client.Damp()
                time.sleep(1.0)
                print(" Minimal reset complete")
            except Exception as e2:
                print(f" Even minimal reset failed: {e2}")
    
    def cleanup_and_reset(self):
        """Clean up and reset robot after sequence."""
        try:
            print(" Cleaning up and resetting robot...")
            
            # Release any held modes (but don't force damp mode)
            try:
                print(" Releasing any held modes...")
                self.msc.ReleaseMode()
                time.sleep(1.0)
            except Exception as e:
                print(f" Could not release mode: {e}")
            
            print(" Cleanup complete - robot ready for next commands")
        except Exception as e:
            print(f" Warning: Could not clean up robot state: {e}")
            print(" Cleanup attempted - robot may still be ready")
    
    def switch_to_zero_torque(self):
        """Switch robot to zero torque mode."""
        try:
            print(" Switching to ZERO TORQUE mode (FSM ID: 0)...")
            self.loco_client.ZeroTorque()
            return True
        except Exception as e:
            print(f" Error switching to zero torque mode: {e}")
            return False
    
    def switch_to_damp(self):
        """Switch robot to damping mode."""
        try:
            print(" Switching to DAMP mode (FSM ID: 1)...")
            self.loco_client.Damp()
            return True
        except Exception as e:
            print(f" Error switching to damp mode: {e}")
            return False
    
    def switch_to_start(self):
        """Switch robot to start mode."""
        try:
            print(" Switching to START mode (FSM ID: 200)...")
            self.loco_client.Start()
            return True
        except Exception as e:
            print(f" Error switching to start mode: {e}")
            return False
    
    def switch_to_high_stand(self):
        """Switch robot to high stand mode."""
        try:
            print(" Switching to HIGH STAND mode (max height)...")
            self.loco_client.HighStand()
            return True
        except Exception as e:
            print(f" Error switching to high stand mode: {e}")
            return False
    
    
    def switch_to_low_stand(self):
        """Switch robot to low stand mode."""
        try:
            print(" Switching to LOW STAND mode...")
            self.loco_client.LowStand()
            return True
        except Exception as e:
            print(f" Error switching to low stand mode: {e}")
            return False
    
    def switch_to_squat2standup(self):
        """Switch robot to squat2standup mode."""
        try:
            print(" Switching to SQUAT2STANDUP mode (FSM ID: 706)...")
            self.loco_client.Squat2StandUp()
            return True
        except Exception as e:
            print(f" Error switching to squat2standup mode: {e}")
            return False
    
    def switch_to_lie2standup(self):
        """Switch robot to lie2standup mode."""
        try:
            print(" Switching to LIE2STANDUP mode (FSM ID: 702)...")
            self.loco_client.Lie2StandUp()
            return True
        except Exception as e:
            print(f" Error switching to lie2standup mode: {e}")
            return False
    
    def switch_to_wave_hand(self, turn_flag=False):
        """Switch robot to wave hand mode."""
        try:
            print(f" Switching to WAVE HAND mode (turn_flag: {turn_flag})...")
            self.loco_client.WaveHand(turn_flag)
            return True
        except Exception as e:
            print(f" Error switching to wave hand mode: {e}")
            return False
    
    def switch_to_shake_hand(self, stage=-1):
        """Switch robot to shake hand mode."""
        try:
            print(f" Switching to SHAKE HAND mode (stage: {stage})...")
            self.loco_client.ShakeHand(stage)
            return True
        except Exception as e:
            print(f" Error switching to shake hand mode: {e}")
            return False
    
    def interactive_control(self):
        """Interactive keyboard control for robot modes."""
        print("=" * 70)
        print("G1 Robot Interactive Control")
        print("=" * 70)
        print("Commands:")
        print("   0 = Zero Torque (FSM ID: 0)")
        print("   1 = Damp (FSM ID: 1)")
        print("   2 = Start (FSM ID: 200)")
        print("   3 = High Stand")
        print("   4 = Low Stand")
        print("   5 = Squat2StandUp")
        print("   6 = Lie2StandUp")
        print("   7 = Wave Hand")
        print("   8 = Shake Hand")
        print("   q = Quit")
        print("\nWARNING: Ensure robot is in a safe position!")
        print("=" * 70)
        
        while self.running:
            try:
                command = input("\nEnter command (0-8 or q): ").strip()
                
                if command == 'q':
                    print("Quitting...")
                    break
                elif command == '0':
                    self.switch_to_zero_torque()
                elif command == '1':
                    self.switch_to_damp()
                elif command == '2':
                    self.switch_to_start()
                elif command == '3':
                    self.switch_to_high_stand()
                elif command == '4':
                    self.switch_to_low_stand()
                elif command == '5':
                    self.switch_to_squat2standup()
                elif command == '6':
                    self.switch_to_lie2standup()
                elif command == '7':
                    self.switch_to_wave_hand()
                elif command == '8':
                    self.switch_to_shake_hand()
                else:
                    print("Invalid command. Try 0-8 or q.")
                    
            except KeyboardInterrupt:
                print("\nReceived interrupt signal. Shutting down...")
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("Interactive control finished.")


def main():
    print("G1 Robot Interactive Control")
    print("This script provides keyboard control for robot modes")
    
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
        print(" Channel factory initialized")
        
        # Create tester
        tester = ModeSequenceTester()
        print(" LocoClient created")
        
        # Run interactive control
        tester.interactive_control()
        
    except Exception as e:
        print(f" Error during initialization: {e}")
        print("\n🔧 Troubleshooting:")
        print("   - Make sure you're connected to the robot's network")
        print("   - Try: sudo ifconfig en7 192.168.123.100 netmask 255.255.255.0")
        print("   - Verify robot is powered on and accessible")
        print("   - Check if robot is in 'ai' mode")


if __name__ == "__main__":
    main()
