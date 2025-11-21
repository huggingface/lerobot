#!/usr/bin/env python3
"""
Test script for G1 robot LED control using the new audio_control function.

Usage:
    python test_led_control.py
"""

import time
import logging
from lerobot.robots.unitree_g1.unitree_g1 import UnitreeG1
from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    print("="*60)
    print("G1 Robot LED Control Test")
    print("="*60)
    
    # Create robot config
    config = UnitreeG1Config(
        cameras={},  # No cameras needed for LED test
        motion_mode=False,
        simulation_mode=False,
    )
    
    # Initialize robot
    print("\n[1/4] Initializing robot...")
    robot = UnitreeG1(config)
    
    try:
        # Test LED: Red
        print("\n[2/4] Setting LED to RED (255, 0, 0)...")
        robot.audio_control((255, 0, 0))
        time.sleep(2)
        
        # Test LED: Green
        print("\n[3/4] Setting LED to GREEN (0, 255, 0)...")
        robot.audio_control((0, 255, 0))
        time.sleep(2)
        
        # Test LED: Blue
        print("\n[4/4] Setting LED to BLUE (0, 0, 255)...")
        robot.audio_control((0, 0, 255))
        time.sleep(2)
        
        # Test TTS
        print("\n[Bonus 1] Testing text-to-speech...")
        robot.audio_control("LED test complete!", volume=80)
        time.sleep(3)
        
        # Test WAV playback
        print("\n[Bonus 2] Testing WAV file playback...")
        wav_file = "out.wav"
        import os
        if os.path.exists(wav_file):
            robot.audio_control(wav_file, volume=80)
            print(f"Playing {wav_file}... (waiting for playback to complete)")
            # Wait a bit for audio to play (adjust based on your file length)
            time.sleep(5)
        else:
            print(f"⚠ Warning: {wav_file} not found, skipping WAV playback test")
            print(f"  To test WAV playback, place a 16kHz mono 16-bit WAV file named 'out.wav' in this directory")
        
        print("\n" + "="*60)
        print("✓ LED control test completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Error during test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\nDisconnecting robot...")
        robot.disconnect()
        print("Done!")


if __name__ == "__main__":
    main()

