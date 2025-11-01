#!/usr/bin/env python3
"""
Quick diagnostic script to check serial port status.
"""

import serial
import serial.tools.list_ports

# List all available ports
print("Available serial ports:")
ports = serial.tools.list_ports.comports()
for port in ports:
    print(f"  {port.device}: {port.description}")

# Try to open the problematic port
port_name = "/dev/tty.usbmodem58FD0163901"
print(f"\nAttempting to open {port_name}...")
try:
    ser = serial.Serial(port_name, baudrate=1000000, timeout=1)
    print(f"✓ Successfully opened {port_name}")
    print(f"  Port details: {ser}")
    ser.close()
    print("✓ Port closed successfully")
except serial.SerialException as e:
    print(f"✗ Error: {e}")
    print("\nThis port is busy. Possible solutions:")
    print("1. Unplug and replug the USB device")
    print("2. Check for hung processes with: ps aux | grep python")
    print("3. Restart your computer (last resort)")
