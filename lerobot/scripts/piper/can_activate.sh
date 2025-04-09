#!/bin/bash

# The default CAN name can be set by the user via command-line parameters.
DEFAULT_CAN_NAME="${1:-can0}"

# The default bitrate for a single CAN module can be set by the user via command-line parameters.
DEFAULT_BITRATE="${2:-1000000}"

# USB hardware address (optional parameter)
USB_ADDRESS="${3}"
echo "-------------------START-----------------------"
# Check if ethtool is installed.
if ! dpkg -l | grep -q "ethtool"; then
    echo "\e[31mError: ethtool not detected in the system.\e[0m"
    echo "Please use the following command to install ethtool:"
    echo "sudo apt update && sudo apt install ethtool"
    exit 1
fi

# Check if can-utils is installed.
if ! dpkg -l | grep -q "can-utils"; then
    echo "\e[31mError: can-utils not detected in the system.\e[0m"
    echo "Please use the following command to install ethtool:"
    echo "sudo apt update && sudo apt install can-utils"
    exit 1
fi

echo "Both ethtool and can-utils are installed."

# Retrieve the number of CAN modules in the current system.
CURRENT_CAN_COUNT=$(ip link show type can | grep -c "link/can")

# Verify if the number of CAN modules in the current system matches the expected value.
if [ "$CURRENT_CAN_COUNT" -ne "1" ]; then
    if [ -z "$USB_ADDRESS" ]; then
        # Iterate through all CAN interfaces.
        for iface in $(ip -br link show type can | awk '{print $1}'); do
            # Use ethtool to retrieve bus-info.
            BUS_INFO=$(sudo ethtool -i "$iface" | grep "bus-info" | awk '{print $2}')
            
            if [ -z "$BUS_INFO" ];then
                echo "Error: Unable to retrieve bus-info for interface $iface."
                continue
            fi
            
            echo "Interface $iface is inserted into USB port $BUS_INFO"
        done
        echo -e " \e[31m Error: The number of CAN modules detected by the system ($CURRENT_CAN_COUNT) does not match the expected number (1). \e[0m"
        echo -e " \e[31m Please add the USB hardware address parameter, such as: \e[0m"
        echo -e " bash can_activate.sh can0 1000000 1-2:1.0"
        echo "-------------------ERROR-----------------------"
        exit 1
    fi
fi

# Load the gs_usb module.
# sudo modprobe gs_usb
# if [ $? -ne 0 ]; then
#     echo "Error: Unable to load the gs_usb module."
#     exit 1
# fi

if [ -n "$USB_ADDRESS" ]; then
    echo "Detected USB hardware address parameter: $USB_ADDRESS"
    
    # Use ethtool to find the CAN interface corresponding to the USB hardware address.
    INTERFACE_NAME=""
    for iface in $(ip -br link show type can | awk '{print $1}'); do
        BUS_INFO=$(sudo ethtool -i "$iface" | grep "bus-info" | awk '{print $2}')
        if [ "$BUS_INFO" == "$USB_ADDRESS" ]; then
            INTERFACE_NAME="$iface"
            break
        fi
    done
    
    if [ -z "$INTERFACE_NAME" ]; then
        echo "Error: Unable to find CAN interface corresponding to USB hardware address $USB_ADDRESS."
        exit 1
    else
        echo "Found the interface corresponding to USB hardware address $USB_ADDRESS: $INTERFACE_NAME."
    fi
else
    # Retrieve the unique CAN interface.
    INTERFACE_NAME=$(ip -br link show type can | awk '{print $1}')
    
    # Check if the interface name has been retrieved.
    if [ -z "$INTERFACE_NAME" ]; then
        echo "Error: Unable to detect CAN interface."
        exit 1
    fi
    BUS_INFO=$(sudo ethtool -i "$INTERFACE_NAME" | grep "bus-info" | awk '{print $2}')
    echo "Expected to configure a single CAN module, detected interface $INTERFACE_NAME with corresponding USB address $BUS_INFO."
fi

# Check if the current interface is already activated.
IS_LINK_UP=$(ip link show "$INTERFACE_NAME" | grep -q "UP" && echo "yes" || echo "no")

# Retrieve the bitrate of the current interface.
CURRENT_BITRATE=$(ip -details link show "$INTERFACE_NAME" | grep -oP 'bitrate \K\d+')

if [ "$IS_LINK_UP" == "yes" ] && [ "$CURRENT_BITRATE" -eq "$DEFAULT_BITRATE" ]; then
    echo "Interface $INTERFACE_NAME is already activated with a bitrate of $DEFAULT_BITRATE."
    
    # Check if the interface name matches the default name.
    if [ "$INTERFACE_NAME" != "$DEFAULT_CAN_NAME" ]; then
        echo "Rename interface $INTERFACE_NAME to $DEFAULT_CAN_NAME."
        sudo ip link set "$INTERFACE_NAME" down
        sudo ip link set "$INTERFACE_NAME" name "$DEFAULT_CAN_NAME"
        sudo ip link set "$DEFAULT_CAN_NAME" up
        echo "The interface has been renamed to $DEFAULT_CAN_NAME and reactivated."
    else
        echo "The interface name is already $DEFAULT_CAN_NAME."
    fi
else
    # If the interface is not activated or the bitrate is different, configure it.
    if [ "$IS_LINK_UP" == "yes" ]; then
        echo "Interface $INTERFACE_NAME is already activated, but the bitrate is $CURRENT_BITRATE, which does not match the set value of $DEFAULT_BITRATE."
    else
        echo "Interface $INTERFACE_NAME is not activated or bitrate is not set."
    fi
    
    # Set the interface bitrate and activate it.
    sudo ip link set "$INTERFACE_NAME" down
    sudo ip link set "$INTERFACE_NAME" type can bitrate $DEFAULT_BITRATE
    sudo ip link set "$INTERFACE_NAME" up
    echo "Interface $INTERFACE_NAME has been reset to bitrate $DEFAULT_BITRATE and activated."
    
    # Rename the interface to the default name.
    if [ "$INTERFACE_NAME" != "$DEFAULT_CAN_NAME" ]; then
        echo "Rename interface $INTERFACE_NAME to $DEFAULT_CAN_NAME."
        sudo ip link set "$INTERFACE_NAME" down
        sudo ip link set "$INTERFACE_NAME" name "$DEFAULT_CAN_NAME"
        sudo ip link set "$DEFAULT_CAN_NAME" up
        echo "The interface has been renamed to $DEFAULT_CAN_NAME and reactivated."
    fi
fi

echo "-------------------OVER------------------------"
