#!/usr/bin/env python3
"""
PS4 Controller HID Test Script

This script uses hidapi to read and display PS4 controller output.
It shows both raw data and interpreted values for debugging and testing.

Requirements:
    pip install hidapi

Usage:
    python temp/test_hid.py
"""

import hid
import time
import sys
from typing import Dict, List, Optional


class PS4ControllerTester:
    """Test PS4 controller using HID."""
    
    def __init__(self):
        self.device = None
        self.device_info = None
        self.running = True
        
        # PS4 Controller vendor and product IDs
        self.PS4_VENDOR_ID = 0x054C  # Sony
        # self.PS4_PRODUCT_ID = 0x09CC  # DualShock 4
        self.PS4_PRODUCT_ID = 0x05C4  # DualShock 4
        
        # Current state
        self.buttons = {}
        self.analogs = {}
        self.dpad = 0
        
    def find_ps4_controller(self) -> Optional[Dict]:
        """Find PS4 controller device."""
        print("Searching for PS4 controller...")
        
        devices = hid.enumerate()
        for device in devices:
            # Check for PS4 controller by vendor/product ID
            if (device['vendor_id'] == self.PS4_VENDOR_ID and 
                device['product_id'] == self.PS4_PRODUCT_ID):
                print(f"Found PS4 controller: {device['product_string']}")
                return device
            
            # Also check by name for other PS4 controllers
            device_name = device.get('product_string', '').lower()
            if any(name in device_name for name in ['ps4', 'dualshock', 'playstation']):
                print(f"Found potential PS4 controller: {device['product_string']}")
                print(f"  Vendor ID: {device['vendor_id']:04X}")
                print(f"  Product ID: {device['product_id']:04X}")
                return device
        
        print("No PS4 controller found!")
        print("\nAvailable HID devices:")
        for device in devices:
            if device.get('product_string'):
                print(f"  {device['product_string']} (VID: {device['vendor_id']:04X}, PID: {device['product_id']:04X})")
        
        return None
    
    def connect(self) -> bool:
        """Connect to PS4 controller."""
        self.device_info = self.find_ps4_controller()
        if not self.device_info:
            return False
            
        try:
            print(f"Connecting to controller at path: {self.device_info['path']}")
            self.device = hid.device()
            self.device.open_path(self.device_info['path'])
            self.device.set_nonblocking(1)
            
            manufacturer = self.device.get_manufacturer_string()
            product = self.device.get_product_string()
            print(f"Connected to {manufacturer} {product}")
            return True
            
        except OSError as e:
            print(f"Error opening controller: {e}")
            print("You may need to run with sudo/admin privileges on some systems")
            return False
    
    def disconnect(self):
        """Disconnect from controller."""
        if self.device:
            self.device.close()
            self.device = None
            print("Disconnected from controller")
    
    def normalize_analog(self, value: int, min_val: int = 0, max_val: int = 255) -> float:
        """Normalize analog value to -1.0 to 1.0 range."""
        center = (max_val + min_val) / 2
        range_val = (max_val - min_val) / 2
        normalized = (value - center) / range_val
        return max(-1.0, min(1.0, normalized))
    
    def parse_dpad(self, dpad_value: int) -> str:
        """Parse D-pad value to direction string."""
        directions = {
            0: "N", 1: "NE", 2: "E", 3: "SE",
            4: "S", 5: "SW", 6: "W", 7: "NW", 8: "None"
        }
        return directions.get(dpad_value, f"Unknown({dpad_value})")
    
    def parse_buttons(self, button_data: List[int]) -> Dict[str, bool]:
        """Parse button data from raw bytes."""
        buttons = {}
        
        if len(button_data) >= 8:
            # Parse buttons from first few bytes
            # This mapping may need adjustment for your specific controller
            byte0 = button_data[0]
            byte1 = button_data[1] 
            byte2 = button_data[2]
            
            # D-pad is often in the first few bits
            self.dpad = byte0 & 0x0F
            
            # Parse individual buttons
            buttons = {
                'square': bool(byte0 & 0x10),
                'cross': bool(byte0 & 0x20),
                'circle': bool(byte0 & 0x40),
                'triangle': bool(byte0 & 0x80),
                'l1': bool(byte1 & 0x01),
                'r1': bool(byte1 & 0x02),
                'l2': bool(byte1 & 0x04),
                'r2': bool(byte1 & 0x08),
                'share': bool(byte1 & 0x10),
                'options': bool(byte1 & 0x20),
                'l3': bool(byte1 & 0x40),
                'r3': bool(byte1 & 0x80),
                'ps': bool(byte2 & 0x01),
                'touchpad': bool(byte2 & 0x02),
            }
        
        return buttons
    
    def read_data(self) -> Optional[List[int]]:
        """Read raw data from controller."""
        if not self.device:
            return None
            
        try:
            # PS4 controller typically sends 64-byte packets
            data = self.device.read(64, timeout_ms=100)
            return data if data else None
        except OSError as e:
            print(f"Error reading from controller: {e}")
            return None
    
    def parse_data(self, data: List[int]) -> Dict:
        """Parse raw controller data."""
        if not data or len(data) < 8:
            return {}
        
        # Parse analog sticks (common PS4 mapping)
        analogs = {
            'left_x': data[1] if len(data) > 1 else 128,
            'left_y': data[2] if len(data) > 2 else 128,
            'right_x': data[3] if len(data) > 3 else 128,
            'right_y': data[4] if len(data) > 4 else 128,
            'l2': data[5] if len(data) > 5 else 0,
            'r2': data[6] if len(data) > 6 else 0,
        }
        
        # Parse buttons
        buttons = self.parse_buttons(data)
        
        return {
            'analogs': analogs,
            'buttons': buttons,
            'dpad': self.dpad,
            'raw_data': data[:16]  # Show first 16 bytes for debugging
        }
    
    def print_status(self, data: Dict):
        """Print formatted controller status."""
        if not data:
            return
            
        print("\n" + "="*60)
        print("PS4 CONTROLLER TEST OUTPUT")
        print("="*60)
        
        # Print analog stick values
        print("\nANALOG STICKS:")
        analogs = data.get('analogs', {})
        for stick, value in analogs.items():
            normalized = self.normalize_analog(value)
            print(f"  {stick:8}: Raw={value:3d} | Normalized={normalized:6.3f}")
        
        # Print button states
        print("\nBUTTONS:")
        buttons = data.get('buttons', {})
        button_states = []
        for button, pressed in buttons.items():
            status = "PRESSED" if pressed else "      "
            button_states.append(f"{button:10}: {status}")
        
        # Print in columns
        for i in range(0, len(button_states), 2):
            if i + 1 < len(button_states):
                print(f"  {button_states[i]} | {button_states[i+1]}")
            else:
                print(f"  {button_states[i]}")
        
        # Print D-pad
        dpad_direction = self.parse_dpad(data.get('dpad', 8))
        print(f"\nD-PAD: {dpad_direction}")
        
        # Print raw data for debugging
        print("\nRAW DATA (first 16 bytes):")
        raw_data = data.get('raw_data', [])
        hex_data = ' '.join([f"{b:02X}" for b in raw_data])
        print(f"  {hex_data}")
        
        print("="*60)
    
    def run(self):
        """Main loop to continuously read and display controller data."""
        if not self.connect():
            return
        
        print("\nPress Ctrl+C to exit")
        print("Move controller sticks and press buttons to see their values...")
        
        try:
            while self.running:
                data = self.read_data()
                if data:
                    parsed_data = self.parse_data(data)
                    self.print_status(parsed_data)
                else:
                    print("No data received from controller")
                
                time.sleep(0.1)  # 10 FPS update rate
                
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            self.disconnect()


def main():
    """Main function."""
    print("PS4 Controller HID Test")
    print("=" * 30)
    
    try:
        tester = PS4ControllerTester()
        tester.run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
