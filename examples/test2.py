#!/usr/bin/env python
#
# *********     Ping Example      *********
#
#
# Available SCServo model on this example : All models using Protocol SCS
# This example is tested with a SCServo(STS/SMS/SCS), and an URT
# Be sure that SCServo(STS/SMS/SCS) properties are already set as %% ID : 1 / Baudnum : 6 (Baudrate : 1000000)
#

import os

if os.name == "nt":
    import msvcrt

    def getch():
        return msvcrt.getch().decode()
else:
    import sys
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    def getch():
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


from scservo_sdk import *  # Uses SCServo SDK library

# Default setting
SCS_ID = 1  # SCServo ID : 1
BAUDRATE = 1000000  # SCServo default baudrate : 1000000
DEVICENAME = "/dev/tty.usbserial-2130"  # Check which port is being used on your controller
# ex) Windows: "COM1"   Linux: "/dev/ttyUSB0" Mac: "/dev/tty.usbserial-*"

protocol_end = 1  # SCServo bit end(STS/SMS=0, SCS=1)

# Initialize PortHandler instance
# Set the port path
# Get methods and members of PortHandlerLinux or PortHandlerWindows
portHandler = PortHandler(DEVICENAME)

# Initialize PacketHandler instance
# Get methods and members of Protocol
packetHandler = PacketHandler(protocol_end)

# Open port
if portHandler.openPort():
    print("Succeeded to open the port")
else:
    print("Failed to open the port")
    print("Press any key to terminate...")
    getch()
    quit()


# Set port baudrate
if portHandler.setBaudRate(BAUDRATE):
    print("Succeeded to change the baudrate")
else:
    print("Failed to change the baudrate")
    print("Press any key to terminate...")
    getch()
    quit()

# Try to ping the SCServo
# Get SCServo model number
scs_model_number, scs_comm_result, scs_error = packetHandler.ping(portHandler, SCS_ID)
if scs_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(scs_comm_result))
elif scs_error != 0:
    print("%s" % packetHandler.getRxPacketError(scs_error))
else:
    print("[ID:%03d] ping Succeeded. SCServo model number : %d" % (SCS_ID, scs_model_number))


ADDR_SCS_PRESENT_POSITION = 56
scs_present_position, scs_comm_result, scs_error = packetHandler.read2ByteTxRx(
    portHandler, SCS_ID, ADDR_SCS_PRESENT_POSITION
)
if scs_comm_result != COMM_SUCCESS:
    print(packetHandler.getTxRxResult(scs_comm_result))
elif scs_error != 0:
    print(packetHandler.getRxPacketError(scs_error))

breakpoint()
scs_present_position = SCS_LOWORD(scs_present_position)
# scs_present_speed = SCS_HIWORD(scs_present_position_speed)
# print("[ID:%03d] PresPos:%03d PresSpd:%03d" % (SCS_ID, scs_present_position, SCS_TOHOST(scs_present_speed, 15)))
print("[ID:%03d] PresPos:%03d" % (SCS_ID, scs_present_position))

groupSyncRead = GroupSyncRead(portHandler, packetHandler, ADDR_SCS_PRESENT_POSITION, 2)

scs_addparam_result = groupSyncRead.addParam(SCS_ID)
if scs_addparam_result != True:
    print("[ID:%03d] groupSyncRead addparam failed" % SCS_ID)
    quit()

# Syncread present position
scs_comm_result = groupSyncRead.txRxPacket()
if scs_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(scs_comm_result))

# Check if groupsyncread data of SCServo#1 is available
scs_getdata_result = groupSyncRead.isAvailable(SCS_ID, ADDR_SCS_PRESENT_POSITION, 2)
if scs_getdata_result == True:
    # Get SCServo#1 present position value
    scs_present_position = groupSyncRead.getData(SCS_ID, ADDR_SCS_PRESENT_POSITION, 2)
else:
    scs_present_position = 0
    print("[ID:%03d] groupSyncRead getdata failed" % SCS_ID)

# # Check if groupsyncread data of SCServo#2 is available
# scs_getdata_result = groupSyncRead.isAvailable(SCS2_ID, ADDR_SCS_PRESENT_POSITION, 2)
# if scs_getdata_result == True:
#     # Get SCServo#2 present position value
#     scs2_present_position_speed = groupSyncRead.getData(SCS2_ID, ADDR_SCS_PRESENT_POSITION, 2)
# else:
#     print("[ID:%03d] groupSyncRead getdata failed" % SCS2_ID)

scs_present_position = SCS_LOWORD(scs_present_position)
print("[ID:%03d] PresPos:%03d" % (SCS_ID, scs_present_position))


# Close port
portHandler.closePort()
