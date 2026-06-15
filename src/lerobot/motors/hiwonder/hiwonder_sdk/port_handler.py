# Copyright 2024 Hiwonder. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import time

import serial

DEFAULT_BAUDRATE = 1_000_000
LATENCY_TIMER = 50


class PortHandler:
    def __init__(self, port_name):
        self.is_open = False
        self.baudrate = DEFAULT_BAUDRATE
        self.packet_start_time = 0.0
        self.packet_timeout = 0.0
        self.tx_time_per_byte = 0.0
        self.is_using = False
        self.port_name = port_name
        self.ser = None

    def openPort(self):  # noqa: N802
        return self.setBaudRate(self.baudrate)

    def closePort(self):  # noqa: N802
        self.ser.close()
        self.is_open = False

    def clearPort(self):  # noqa: N802
        self.ser.flush()

    def setPortName(self, port_name):  # noqa: N802
        self.port_name = port_name

    def getPortName(self):  # noqa: N802
        return self.port_name

    def setBaudRate(self, baudrate):  # noqa: N802
        baud = self.getCFlagBaud(baudrate)
        if baud <= 0:
            return False
        self.baudrate = baudrate
        return self.setupPort(baud)

    def getBaudRate(self):  # noqa: N802
        return self.baudrate

    def getBytesAvailable(self):  # noqa: N802
        return self.ser.in_waiting

    def readPort(self, length):  # noqa: N802
        if sys.version_info > (3, 0):
            return self.ser.read(length)
        else:
            return [ord(ch) for ch in self.ser.read(length)]

    def writePort(self, packet):  # noqa: N802
        return self.ser.write(packet)

    def setPacketTimeout(self, packet_length):  # noqa: N802
        self.packet_start_time = self.getCurrentTime()
        self.packet_timeout = (
            (self.tx_time_per_byte * packet_length) + (self.tx_time_per_byte * 3.0) + LATENCY_TIMER
        )

    def setPacketTimeoutMillis(self, msec):  # noqa: N802
        self.packet_start_time = self.getCurrentTime()
        self.packet_timeout = msec

    def isPacketTimeout(self):  # noqa: N802
        if self.getTimeSinceStart() > self.packet_timeout:
            self.packet_timeout = 0
            return True
        return False

    def getCurrentTime(self):  # noqa: N802
        return round(time.time() * 1_000_000_000) / 1_000_000.0

    def getTimeSinceStart(self):  # noqa: N802
        time_since = self.getCurrentTime() - self.packet_start_time
        if time_since < 0.0:
            self.packet_start_time = self.getCurrentTime()
        return time_since

    def setupPort(self, cflag_baud):  # noqa: N802
        if self.is_open:
            self.closePort()
        self.ser = serial.Serial(
            port=self.port_name,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            timeout=0,
        )
        self.is_open = True
        self.ser.reset_input_buffer()
        self.tx_time_per_byte = (1000.0 / self.baudrate) * 10.0
        return True

    def getCFlagBaud(self, baudrate):  # noqa: N802
        if baudrate in [4800, 9600, 14400, 19200, 38400, 57600, 76800, 115200, 128000, 250000, 500000, 1000000]:
            return baudrate
        return -1
