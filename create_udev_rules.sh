#!/bin/bash

# Create a new udev rules file
echo "# Udev rules for USB servo devices" | sudo tee /etc/udev/rules.d/99-usb-servo.rules

# Loop through the devices
for device in /dev/ttyACM{0..1}; do
    # Get device information
    SERIAL=$(udevadm info -a -n $device | grep '{serial}' | head -n1 | cut -d'"' -f2)
    VENDOR=$(udevadm info -a -n $device | grep '{idVendor}' | head -n1 | cut -d'"' -f2)
    PRODUCT=$(udevadm info -a -n $device | grep '{idProduct}' | head -n1 | cut -d'"' -f2)
    
    # Create a rule for this device
    echo "SUBSYSTEM==\"tty\", ATTRS{idVendor}==\"$VENDOR\", ATTRS{idProduct}==\"$PRODUCT\", ATTRS{serial}==\"$SERIAL\", SYMLINK+=\"servo_$SERIAL\"" | sudo tee -a /etc/udev/rules.d/99-usb-servo.rules
done

# Reload udev rules
sudo udevadm control --reload-rules
sudo udevadm trigger