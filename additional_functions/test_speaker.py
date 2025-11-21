#!/usr/bin/env python3
"""
Simple test script to use the G1 robot's speaker.
Demonstrates TTS (text-to-speech) and volume control.

Usage:
    python test_speaker.py en7
    (replace 'en7' with your network interface)
"""

import sys
import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <network_interface>")
        print("Example: python3 test_speaker.py en7")
        sys.exit(1)

    # Initialize DDS communication
    network_interface = sys.argv[1]
    print(f"Initializing communication on {network_interface}...")
    ChannelFactoryInitialize(0, network_interface)

    # Create audio client
    audio_client = AudioClient()
    audio_client.SetTimeout(10.0)
    audio_client.Init()
    print("Audio client initialized!")

    # Get current volume
    code, volume_data = audio_client.GetVolume()
    if code == 0:
        print(f"Current volume: {volume_data}")
    
    # Set volume to 80%
    print("Setting volume to 80%...")
    audio_client.SetVolume(100)
    time.sleep(0.5)

    # # Test English TTS
    # print("Speaking (English)...")
    audio_client.LedControl(255, 0, 0)
    audio_client.TtsMaker("Hello!", 0)
    exit()
    # time.sleep(5)

    # # Test Chinese TTS
    # print("Speaking (Chinese)...")
    # audio_client.TtsMaker("大家好！我是宇树科技人形机器人。", 0)
    # time.sleep(5)

    # # Test LED control (bonus!)
    # print("Testing LED strip...")
    # audio_client.TtsMaker("Now testing LED lights.", 0)
    # time.sleep(2)
    
    print("LED: Red")
      # Red
    time.sleep(1)
    
    print("LED: Green")
    #audio_client.LedControl(0, 255, 0)  # Green
    time.sleep(1)
    
    print("LED: Blue")
    #audio_client.LedControl(0, 0, 255)  # Blue
    time.sleep(1)


    print("Done!")


if __name__ == "__main__":
    main()

