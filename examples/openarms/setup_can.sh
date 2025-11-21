#!/bin/bash
# Setup all OpenArms CAN interfaces with CAN FD

set -e

echo "=========================================="
echo "OpenArms CAN FD Interface Setup"
echo "=========================================="
echo ""
echo "Mode: CAN FD"
echo "  - Nominal bitrate: 1 Mbps"
echo "  - Data bitrate: 5 Mbps"
echo ""
echo "Configuring interfaces can0, can1, can2, can3..."
echo ""

# Configure each CAN interface with CAN FD
for i in 0 1 2 3; do
    interface="can$i"
    
    # Check if interface exists
    if ! ip link show "$interface" &> /dev/null; then
        echo "⚠ $interface: Not found, skipping"
        continue
    fi
    
    # Bring down interface
    sudo ip link set "$interface" down 2>/dev/null
    
    # Configure CAN FD mode
    sudo ip link set "$interface" type can \
        bitrate 1000000 \
        dbitrate 5000000 \
        fd on
    
    # Bring up interface
    sudo ip link set "$interface" up
    
    # Verify configuration
    if ip link show "$interface" | grep -q "UP"; then
        echo "✓ $interface: Configured and UP"
    else
        echo "✗ $interface: Failed to bring UP"
    fi
done

echo ""
echo "=========================================="
echo "Verification"
echo "=========================================="
echo ""

# Show detailed status for each interface
for i in 0 1 2 3; do
    interface="can$i"
    if ip link show "$interface" &> /dev/null; then
        echo "$interface:"
        # Show key parameters
        ip -d link show "$interface" | grep -E "can|state|bitrate|dbitrate" | head -3
        echo ""
    fi
done

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "All interfaces configured for CAN FD mode"
echo ""
echo "Next steps:"
echo "  1. Test motors: python debug_can_communication.py"
echo "  2. Run teleoperation: python examples/openarms/teleop.py"
echo ""