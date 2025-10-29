#!/usr/bin/env python3
"""
Debug script for Damiao motor CAN communication on macOS.
This replaces candump for macOS SLCAN debugging.
"""

import can
import time
import sys

def test_can_communication(port="/dev/cu.usbmodem2101"):
    """Test basic CAN communication with a Damiao motor."""
    
    print("=" * 60)
    print("Damiao Motor CAN Communication Debug Tool")
    print("=" * 60)
    print(f"\nPort: {port}")
    print()
    
    try:
        # Connect to SLCAN
        print("Step 1: Connecting to SLCAN...")
        bus = can.interface.Bus(
            channel=port,
            interface='slcan',
            bitrate=1000000
        )
        print("✓ Connected to SLCAN")
        
        # Test 1: Send enable command and listen for ANY response
        print("\n" + "=" * 60)
        print("Test 1: Enable Motor (ID 0x01)")
        print("=" * 60)
        print("Sending enable command to 0x01...")
        
        enable_msg = can.Message(
            arbitration_id=0x01,
            data=[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC],
            is_extended_id=False
        )
        bus.send(enable_msg)
        print("✓ Enable command sent")
        
        print("\nListening for responses (2 seconds)...")
        print("Expected: Response from 0x11 (master ID)")
        print()
        
        responses = []
        start_time = time.time()
        timeout = 2.0
        
        while time.time() - start_time < timeout:
            msg = bus.recv(timeout=0.1)
            if msg:
                responses.append(msg)
                print(f"  → Response from 0x{msg.arbitration_id:02X}: {msg.data.hex()}")
        
        if not responses:
            print("✗ NO RESPONSES RECEIVED")
            print("\nPossible issues:")
            print("  1. Motor not powered (check 24V supply)")
            print("  2. CAN wiring incorrect (CANH, CANL, GND)")
            print("  3. Motor master ID not set to 0x11")
            print("  4. SLCAN adapter not working properly")
            print("  5. Wrong CAN port specified")
        else:
            print(f"\n✓ Received {len(responses)} response(s)")
            
            # Check if we got response from expected ID
            recv_ids = [msg.arbitration_id for msg in responses]
            if 0x11 in recv_ids:
                print("✓ Motor 0x11 is responding!")
            else:
                print(f"⚠ Responses from unexpected IDs: {[hex(id) for id in recv_ids]}")
        
        # Test 2: Send refresh command
        print("\n" + "=" * 60)
        print("Test 2: Refresh Motor State (ID 0x01)")
        print("=" * 60)
        print("Sending refresh command...")
        
        refresh_msg = can.Message(
            arbitration_id=0x7FF,  # Parameter ID
            data=[0x01, 0x00, 0xCC, 0x00, 0x00, 0x00, 0x00, 0x00],
            is_extended_id=False
        )
        bus.send(refresh_msg)
        print("✓ Refresh command sent")
        
        print("\nListening for responses (2 seconds)...")
        responses = []
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            msg = bus.recv(timeout=0.1)
            if msg:
                responses.append(msg)
                print(f"  → Response from 0x{msg.arbitration_id:02X}: {msg.data.hex()}")
        
        if not responses:
            print("✗ NO RESPONSES RECEIVED")
        else:
            print(f"\n✓ Received {len(responses)} response(s)")
        
        # Test 3: Listen for any spontaneous traffic
        print("\n" + "=" * 60)
        print("Test 3: Listen for Any CAN Traffic")
        print("=" * 60)
        print("Listening for 5 seconds...")
        print("(This will catch any background CAN traffic)")
        print()
        
        start_time = time.time()
        traffic_count = 0
        
        while time.time() - start_time < 5.0:
            msg = bus.recv(timeout=0.1)
            if msg:
                traffic_count += 1
                print(f"  [{time.time() - start_time:.2f}s] ID=0x{msg.arbitration_id:03X}: {msg.data.hex()}")
        
        if traffic_count == 0:
            print("✗ No CAN traffic detected at all")
            print("\nThis suggests:")
            print("  - SLCAN adapter may not be working")
            print("  - No devices on the CAN bus are active")
            print("  - Wrong port specified")
        else:
            print(f"\n✓ Detected {traffic_count} CAN messages")
        
        # Cleanup
        print("\n" + "=" * 60)
        print("Cleanup")
        print("=" * 60)
        print("Sending disable command...")
        disable_msg = can.Message(
            arbitration_id=0x01,
            data=[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD],
            is_extended_id=False
        )
        bus.send(disable_msg)
        time.sleep(0.1)
        
        bus.shutdown()
        print("✓ Disconnected from CAN bus")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("Debug Complete")
    print("=" * 60)
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        port = sys.argv[1]
    else:
        port = "/dev/cu.usbmodem2101"
    
    test_can_communication(port)

