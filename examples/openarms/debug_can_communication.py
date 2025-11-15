#!/usr/bin/env python3
"""
Comprehensive debug script for OpenArms CAN FD communication.
Tests all 4 CAN interfaces with CAN FD support.
"""

import can
import time
import sys
import subprocess

def check_can_interface(port):
    """Check if CAN interface is UP and configured."""
    try:
        result = subprocess.run(['ip', 'link', 'show', port], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            return False, "Interface not found", None
        
        output = result.stdout
        if 'UP' not in output:
            return False, "Interface is DOWN", None
        
        # Check if CAN FD is enabled
        is_fd = 'fd on' in output.lower() or 'canfd' in output.lower()
        
        return True, "Interface is UP", is_fd
    except FileNotFoundError:
        return None, "Cannot check (ip command not found)", None


def test_motor_on_interface(bus, motor_id, timeout=2.0, use_fd=False):
    """
    Test a single motor and return all responses.
    
    Returns:
        list of (arbitration_id, data) tuples for all responses received
    """
    # Send enable command
    enable_msg = can.Message(
        arbitration_id=motor_id,
        data=[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC],
        is_extended_id=False,
        is_fd=use_fd
    )
    
    try:
        bus.send(enable_msg)
    except Exception as e:
        return None, f"Send error: {e}"
    
    # Listen for responses
    responses = []
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        msg = bus.recv(timeout=0.1)
        if msg:
            responses.append((msg.arbitration_id, msg.data, msg.is_fd if hasattr(msg, 'is_fd') else False))
    
    # Send disable command
    disable_msg = can.Message(
        arbitration_id=motor_id,
        data=[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD],
        is_extended_id=False,
        is_fd=use_fd
    )
    try:
        bus.send(disable_msg)
    except:
        pass
    
    return responses, None


def test_interface(port, interface_type="socketcan", use_can_fd=True):
    """Test all 8 motors on a single CAN interface."""
    
    results = {
        'interface': port,
        'status': None,
        'is_fd': use_can_fd,
        'motors': {}
    }
    
    # Check interface status
    status_ok, status_msg, interface_has_fd = check_can_interface(port)
    
    if interface_has_fd is not None:
        results['interface_fd_enabled'] = interface_has_fd
        if use_can_fd and not interface_has_fd:
            status_msg += " (CAN FD NOT enabled on interface!)"
        elif interface_has_fd:
            status_msg += " (CAN FD enabled)"
    
    results['status'] = status_msg
    
    if status_ok is False:
        return results
    
    # Try to connect
    try:
        if use_can_fd:
            print(f"  Connecting to {port} with CAN FD (1 Mbps / 5 Mbps)...")
            bus = can.interface.Bus(
                channel=port,
                interface=interface_type,
                bitrate=1000000,
                data_bitrate=5000000,
                fd=True
            )
        else:
            print(f"  Connecting to {port} with CAN 2.0 (1 Mbps)...")
            bus = can.interface.Bus(
                channel=port,
                interface=interface_type,
                bitrate=1000000
            )
    except Exception as e:
        results['status'] = f"Connection failed: {e}"
        return results
    
    try:
        # Clear any pending messages
        while bus.recv(timeout=0.01):
            pass
        
        # Test each motor (0x01 to 0x08)
        for motor_id in range(0x01, 0x09):
            responses, error = test_motor_on_interface(bus, motor_id, timeout=1.0, use_fd=use_can_fd)
            
            if error:
                results['motors'][motor_id] = {'error': error}
            elif responses:
                results['motors'][motor_id] = {
                    'found': True,
                    'responses': responses
                }
            else:
                results['motors'][motor_id] = {
                    'found': False,
                    'responses': []
                }
            
            time.sleep(0.05)  # Small delay between motors
        
    finally:
        bus.shutdown()
    
    return results


def print_results(all_results):
    """Print formatted results for all interfaces."""
    
    print("SUMMARY - Motors Found on Each Interface")
    
    motor_names = {
        0x01: "joint_1 (Shoulder pan)",
        0x02: "joint_2 (Shoulder lift)",
        0x03: "joint_3 (Shoulder rotation)",
        0x04: "joint_4 (Elbow flex)",
        0x05: "joint_5 (Wrist roll)",
        0x06: "joint_6 (Wrist pitch)",
        0x07: "joint_7 (Wrist rotation)",
        0x08: "gripper",
    }
    
    total_found = 0
    
    for result in all_results:
        interface = result['interface']
        status = result['status']
        
        print(f"{interface}: {status}")
        if result.get('is_fd'):
            print(f"  Mode: CAN FD")
        else:
            print(f"  Mode: CAN 2.0")
        
        if 'Connection failed' in status or 'DOWN' in status:
            print(f"  ⚠ Cannot test {interface}")
            continue
        
        motors_found = 0
        
        for motor_id in range(0x01, 0x09):
            motor_data = result['motors'].get(motor_id, {})
            motor_name = motor_names.get(motor_id, "Unknown")
            
            if motor_data.get('error'):
                print(f"  Motor 0x{motor_id:02X} ({motor_name}): ✗ {motor_data['error']}")
            elif motor_data.get('found'):
                motors_found += 1
                total_found += 1
                responses = motor_data['responses']
                print(f"  Motor 0x{motor_id:02X} ({motor_name}): ✓ FOUND")
                
                for resp_id, data, is_fd in responses:
                    data_hex = data.hex()
                    fd_flag = " [FD]" if is_fd else " [2.0]"
                    print(f"    → Response from 0x{resp_id:02X}{fd_flag}: {data_hex}")
            else:
                print(f"  Motor 0x{motor_id:02X} ({motor_name}): ✗ No response")
        
        print(f"\n  Summary: {motors_found}/8 motors found on {interface}")
    
    # Overall summary
    print("OVERALL SUMMARY")
    print(f"Total motors found across all interfaces: {total_found}")
    
    # Analyze configuration
    print("DIAGNOSIS")
    
    for result in all_results:
        interface = result['interface']
        motors_found = sum(1 for m in result['motors'].values() if m.get('found'))
        
        if motors_found == 0:
            print(f"\n⚠ {interface}: NO MOTORS FOUND")
            print("  Possible issues:")
            print("    1. CAN FD mode mismatch (interface vs motor configuration)")
            print("    2. Missing 120Ω termination resistors at BOTH cable ends")
            print("    3. Motor timeout parameter set incorrectly (should NOT be 0)")
            print("    4. CANH/CANL wiring issue")
            print("    5. Cable too long (>40m for CAN FD at 5Mbps)")
            
            # Check FD mismatch
            if result.get('is_fd') and not result.get('interface_fd_enabled'):
                print("    ⚠️ CRITICAL: Trying CAN FD but interface NOT configured for FD!")
                print(f"       Fix: sudo ip link set {interface} type can bitrate 1000000 dbitrate 5000000 fd on")
                
        elif motors_found < 8:
            print(f"\n⚠ {interface}: Only {motors_found}/8 motors responding")
            print("  Check power and connections for missing motors")
        else:
            print(f"\n✓ {interface}: All 8 motors responding correctly!")
    
    # Check for unexpected response IDs
    print("RESPONSE ID ANALYSIS")
    
    for result in all_results:
        interface = result['interface']
        unexpected = []
        
        for motor_id, motor_data in result['motors'].items():
            if motor_data.get('found'):
                expected_id = motor_id + 0x10
                actual_ids = [resp[0] for resp in motor_data['responses']]
                
                if expected_id not in actual_ids:
                    unexpected.append((motor_id, actual_ids))
        
        if unexpected:
            print(f"\n⚠ {interface}: Unexpected response IDs detected")
            for motor_id, actual_ids in unexpected:
                expected_id = motor_id + 0x10
                print(f"  Motor 0x{motor_id:02X}: Expected 0x{expected_id:02X}, "
                      f"got {[f'0x{id:02X}' for id in actual_ids]}")
            print("  → Motor Master IDs need reconfiguration")
        else:
            motors_found = sum(1 for m in result['motors'].values() if m.get('found'))
            if motors_found > 0:
                print(f"\n✓ {interface}: All responding motors use correct IDs")


def test_communication_speed(interface, motor_id, num_iterations=100):
    """
    Test communication speed with a motor.
    
    Returns:
        tuple: (hz, avg_latency_ms) or (None, None) if test failed
    """
    try:
        # Connect to interface
        bus = can.interface.Bus(
            channel=interface,
            interface="socketcan",
            bitrate=1000000,
            data_bitrate=5000000,
            fd=True
        )
        
        # Send refresh commands and measure round-trip time
        latencies = []
        successful = 0
        
        for _ in range(num_iterations):
            start = time.perf_counter()
            
            # Send enable command (lightweight operation)
            enable_msg = can.Message(
                arbitration_id=motor_id,
                data=[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC],
                is_extended_id=False,
                is_fd=True
            )
            bus.send(enable_msg)
            
            # Wait for response
            msg = bus.recv(timeout=0.1)
            
            if msg:
                latency = (time.perf_counter() - start) * 1000  # Convert to ms
                latencies.append(latency)
                successful += 1
        
        bus.shutdown()
        
        if successful > 0:
            avg_latency = sum(latencies) / len(latencies)
            hz = 1000.0 / avg_latency if avg_latency > 0 else 0
            return hz, avg_latency
        
        return None, None
        
    except Exception as e:
        print(f"    Speed test error: {e}")
        return None, None


def main():
    """Main function to test all CAN interfaces with CAN FD."""
    
    print("\nThis will test all 4 CAN interfaces (can0-can3) with CAN FD")
    print("Testing motors 0x01-0x08 on each interface")
    print()
    print("Make sure:")
    print("  ✓ Motors are powered (24V)")
    print("  ✓ CAN interfaces configured with FD mode:")
    print("    ./examples/openarms/setup_can.sh")
    print("  ✓ Motor 'timeout' parameter NOT set to 0 (use Damiao tools)")
    print("  ✓ CAN wiring includes 120Ω termination at BOTH ends")
    print()
    
    input("Press ENTER to start testing...")
    
    # Test all 4 interfaces with CAN FD
    all_results = []
    
    for i in range(4):
        interface = f"can{i}"
        print(f"Testing {interface}...")
        
        result = test_interface(interface, use_can_fd=True)
        all_results.append(result)
        
        # Quick status
        if 'Connection failed' in result['status'] or 'DOWN' in result['status']:
            print(f"  ⚠ {interface}: {result['status']}")
        else:
            motors_found = sum(1 for m in result['motors'].values() if m.get('found'))
            print(f"  {interface}: {motors_found}/8 motors found")
        
        time.sleep(0.2)
    
    # Print detailed results
    print_results(all_results)
    
    print("Testing Complete!")
    
    all_found = sum(sum(1 for m in r['motors'].values() if m.get('found')) for r in all_results)
    
    if all_found == 0:
        print("\n⚠️ CRITICAL: No motors found on any interface!")
        print("\nTop issues to check:")
        print("  1. Motor 'timeout' parameter (use Damiao tools to set > 0)")
        print("  2. CAN FD not enabled (run ./examples/openarms/setup_can.sh)")
        print("  3. Missing termination resistors")
        print("\nTry:")
        print("  a) Check motor parameters with Damiao Debugging Tools")
        print("  b) Verify CAN FD is enabled: ip -d link show can0 | grep fd")
        print("  c) Run setup script: ./examples/openarms/setup_can.sh")
    else:
        # Run speed test on interfaces with motors
        print("COMMUNICATION SPEED TEST")
        print("\nTesting maximum communication frequency...")
        
        for result in all_results:
            interface = result['interface']
            
            # Find first responding motor
            responding_motor = None
            for motor_id, motor_data in result['motors'].items():
                if motor_data.get('found'):
                    responding_motor = motor_id
                    break
            
            if responding_motor:
                print(f"\n{interface}: Testing with motor 0x{responding_motor:02X}...")
                hz, latency = test_communication_speed(interface, responding_motor, num_iterations=100)
                
                if hz:
                    print(f"  ✓ Max frequency: {hz:.1f} Hz")
                    print(f"  ✓ Avg latency: {latency:.2f} ms")
                    print(f"  ✓ Commands per second: ~{int(hz)}")
                else:
                    print(f"  ✗ Speed test failed")
            else:
                print(f"\n{interface}: No motors found, skipping speed test")

        print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTesting interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

