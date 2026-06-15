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

TXPACKET_MAX_LEN = 250
RXPACKET_MAX_LEN = 250

# Packet field indices
PKT_HEADER0 = 0
PKT_HEADER1 = 1
PKT_ID = 2
PKT_LENGTH = 3
PKT_INSTRUCTION = 4
PKT_ERROR = 4
PKT_PARAMETER0 = 5

# Error bits
ERRBIT_VOLTAGE = 1
ERRBIT_SENSOR = 2
ERRBIT_OVERHEAT = 4
ERRBIT_CURRENT = 8
ERRBIT_ANGLE = 16
ERRBIT_OVERLOAD = 32

BROADCAST_ID = 254
MAX_ID = 252

# Instructions
INST_PING = 1
INST_READ = 2
INST_WRITE = 3
INST_REG_WRITE = 4
INST_ACTION = 5
INST_RESET = 6
INST_SYNC_READ = 130
INST_SYNC_WRITE = 131

# Communication results
COMM_SUCCESS = 0
COMM_PORT_BUSY = -1
COMM_TX_FAIL = -2
COMM_RX_FAIL = -3
COMM_TX_ERROR = -4
COMM_RX_WAITING = -5
COMM_RX_TIMEOUT = -6
COMM_RX_CORRUPT = -7
COMM_NOT_AVAILABLE = -9


class PacketHandler:
    def __init__(self, port_handler, endianness=0):
        self.port_handler = port_handler
        # 0: little-endian (default for Hiwonder STS-compatible servos)
        # 1: big-endian
        self.endianness = endianness

    def getEndian(self):  # noqa: N802
        return self.endianness

    def setEndian(self, e):  # noqa: N802
        self.endianness = e

    def getTxRxResult(self, result):  # noqa: N802
        messages = {
            COMM_SUCCESS: "[TxRxResult] Communication success!",
            COMM_PORT_BUSY: "[TxRxResult] Port is in use!",
            COMM_TX_FAIL: "[TxRxResult] Failed transmit instruction packet!",
            COMM_RX_FAIL: "[TxRxResult] Failed get status packet from device!",
            COMM_TX_ERROR: "[TxRxResult] Incorrect instruction packet!",
            COMM_RX_WAITING: "[TxRxResult] Now receiving status packet!",
            COMM_RX_TIMEOUT: "[TxRxResult] There is no status packet!",
            COMM_RX_CORRUPT: "[TxRxResult] Incorrect status packet!",
            COMM_NOT_AVAILABLE: "[TxRxResult] Protocol does not support this function!",
        }
        return messages.get(result, "")

    def getRxPacketError(self, error):  # noqa: N802
        if error & ERRBIT_VOLTAGE:
            return "[ServoStatus] Input voltage error!"
        if error & ERRBIT_SENSOR:
            return "[ServoStatus] Sensor error!"
        if error & ERRBIT_OVERHEAT:
            return "[ServoStatus] Overheat error!"
        if error & ERRBIT_CURRENT:
            return "[ServoStatus] Current error!"
        if error & ERRBIT_ANGLE:
            return "[ServoStatus] Angle error!"
        if error & ERRBIT_OVERLOAD:
            return "[ServoStatus] Overload error!"
        return ""

    def toHost(self, a, b):  # noqa: N802
        return -(a & ~(1 << b)) if (a & (1 << b)) else a

    def toServo(self, a, b):  # noqa: N802
        return (-a | (1 << b)) if a < 0 else a

    def makeWord16(self, a, b):  # noqa: N802
        if self.endianness == 0:
            return (a & 0xFF) | ((b & 0xFF) << 8)
        return (b & 0xFF) | ((a & 0xFF) << 8)

    def makeWord32(self, a, b):  # noqa: N802
        return (a & 0xFFFF) | ((b & 0xFFFF) << 16)

    def getLowWord32(self, l):  # noqa: N802, E741
        return l & 0xFFFF

    def get_Highword32(self, h):  # noqa: N802
        return (h >> 16) & 0xFFFF

    def getLowByte(self, w):  # noqa: N802
        return w & 0xFF if self.endianness == 0 else (w >> 8) & 0xFF

    def getHighByte(self, w):  # noqa: N802
        return (w >> 8) & 0xFF if self.endianness == 0 else w & 0xFF

    # --- SCS_LOBYTE / SCS_HIBYTE aliases (used by lerobot internals) ---
    def SCS_LOBYTE(self, w):  # noqa: N802
        return w & 0xFF

    def SCS_HIBYTE(self, w):  # noqa: N802
        return (w >> 8) & 0xFF

    def SCS_LOWORD(self, l):  # noqa: N802
        return l & 0xFFFF

    def SCS_HIWORD(self, l):  # noqa: N802
        return (l >> 16) & 0xFFFF

    def txPacket(self, txpacket):  # noqa: N802
        total_packet_length = txpacket[PKT_LENGTH] + 4

        if self.port_handler.is_using:
            return COMM_PORT_BUSY
        self.port_handler.is_using = True

        if total_packet_length > TXPACKET_MAX_LEN:
            self.port_handler.is_using = False
            return COMM_TX_ERROR

        txpacket[PKT_HEADER0] = 0xFF
        txpacket[PKT_HEADER1] = 0xFF

        checksum = 0
        for index in range(2, total_packet_length - 1):
            checksum += txpacket[index]
        txpacket[total_packet_length - 1] = ~checksum & 0xFF

        self.port_handler.clearPort()
        written_packet_length = self.port_handler.writePort(txpacket)
        if total_packet_length != written_packet_length:
            self.port_handler.is_using = False
            return COMM_TX_FAIL

        return COMM_SUCCESS

    def rxPacket(self):  # noqa: N802
        rxpacket = []
        result = COMM_TX_FAIL
        checksum = 0
        rx_length = 0
        wait_length = 6

        while True:
            rxpacket.extend(self.port_handler.readPort(wait_length - rx_length))
            rx_length = len(rxpacket)
            if rx_length >= wait_length:
                index = 0
                for index in range(0, rx_length - 1):
                    if rxpacket[index] == 0xFF and rxpacket[index + 1] == 0xFF:
                        break

                if index == 0:
                    if (
                        rxpacket[PKT_ID] > 0xFD
                        or rxpacket[PKT_LENGTH] > RXPACKET_MAX_LEN
                        or rxpacket[PKT_ERROR] > 0x7F
                    ):
                        del rxpacket[0]
                        rx_length -= 1
                        continue

                    if wait_length != rxpacket[PKT_LENGTH] + PKT_LENGTH + 1:
                        wait_length = rxpacket[PKT_LENGTH] + PKT_LENGTH + 1
                        continue

                    if rx_length < wait_length:
                        if self.port_handler.isPacketTimeout():
                            result = COMM_RX_TIMEOUT if rx_length == 0 else COMM_RX_CORRUPT
                            break
                        continue

                    checksum = 0
                    for i in range(2, wait_length - 1):
                        checksum += rxpacket[i]
                    checksum = ~checksum & 0xFF

                    result = COMM_SUCCESS if rxpacket[wait_length - 1] == checksum else COMM_RX_CORRUPT
                    break
                else:
                    del rxpacket[0:index]
                    rx_length -= index
            else:
                if self.port_handler.isPacketTimeout():
                    result = COMM_RX_TIMEOUT if rx_length == 0 else COMM_RX_CORRUPT
                    break

        self.port_handler.is_using = False
        return rxpacket, result

    def txRxPacket(self, txpacket):  # noqa: N802
        rxpacket = None
        error = 0

        result = self.txPacket(txpacket)
        if result != COMM_SUCCESS:
            return rxpacket, result, error

        if txpacket[PKT_ID] == BROADCAST_ID:
            self.port_handler.is_using = False
            return rxpacket, result, error

        if txpacket[PKT_INSTRUCTION] == INST_READ:
            self.port_handler.setPacketTimeout(txpacket[PKT_PARAMETER0 + 1] + 6)
        else:
            self.port_handler.setPacketTimeout(6)

        while True:
            rxpacket, result = self.rxPacket()
            if result != COMM_SUCCESS or txpacket[PKT_ID] == rxpacket[PKT_ID]:
                break

        if result == COMM_SUCCESS and txpacket[PKT_ID] == rxpacket[PKT_ID]:
            error = rxpacket[PKT_ERROR]

        return rxpacket, result, error

    def ping(self, id_):
        txpacket = [0] * 6
        if id_ > BROADCAST_ID:
            return COMM_NOT_AVAILABLE, 0
        txpacket[PKT_ID] = id_
        txpacket[PKT_LENGTH] = 2
        txpacket[PKT_INSTRUCTION] = INST_PING
        return self.txRxPacket(txpacket)

    def action(self, id_):
        txpacket = [0] * 6
        txpacket[PKT_ID] = id_
        txpacket[PKT_LENGTH] = 2
        txpacket[PKT_INSTRUCTION] = INST_ACTION
        _, result, _ = self.txRxPacket(txpacket)
        return result

    def readData(self, id_, address, length):  # noqa: N802
        txpacket = [0] * 8
        data = []
        if id_ > BROADCAST_ID:
            return data, COMM_NOT_AVAILABLE, 0
        txpacket[PKT_ID] = id_
        txpacket[PKT_LENGTH] = 4
        txpacket[PKT_INSTRUCTION] = INST_READ
        txpacket[PKT_PARAMETER0] = address
        txpacket[PKT_PARAMETER0 + 1] = length
        rxpacket, result, error = self.txRxPacket(txpacket)
        if result == COMM_SUCCESS:
            error = rxpacket[PKT_ERROR]
            data.extend(rxpacket[PKT_PARAMETER0 : PKT_PARAMETER0 + length])
        return data, result, error

    def read1ByteData(self, id_, address):  # noqa: N802
        data, result, error = self.readData(id_, address, 1)
        return data[0] if result == COMM_SUCCESS else 0, result, error

    def read2ByteData(self, id_, address):  # noqa: N802
        data, result, error = self.readData(id_, address, 2)
        return self.makeWord16(data[0], data[1]) if result == COMM_SUCCESS else 0, result, error

    def read4ByteData(self, id_, address):  # noqa: N802
        data, result, error = self.readData(id_, address, 4)
        return (
            self.makeWord32(self.makeWord16(data[0], data[1]), self.makeWord16(data[2], data[3]))
            if result == COMM_SUCCESS
            else 0
        ), result, error

    def writeDataOnly(self, id_, address, length, data):  # noqa: N802
        txpacket = [0] * (length + 7)
        txpacket[PKT_ID] = id_
        txpacket[PKT_LENGTH] = length + 3
        txpacket[PKT_INSTRUCTION] = INST_WRITE
        txpacket[PKT_PARAMETER0] = address
        txpacket[PKT_PARAMETER0 + 1 : PKT_PARAMETER0 + 1 + length] = data[0:length]
        result = self.txPacket(txpacket)
        self.port_handler.is_using = False
        return result

    def writeReadData(self, id_, address, length, data):  # noqa: N802
        txpacket = [0] * (length + 7)
        txpacket[PKT_ID] = id_
        txpacket[PKT_LENGTH] = length + 3
        txpacket[PKT_INSTRUCTION] = INST_WRITE
        txpacket[PKT_PARAMETER0] = address
        txpacket[PKT_PARAMETER0 + 1 : PKT_PARAMETER0 + 1 + length] = data[0:length]
        _, result, error = self.txRxPacket(txpacket)
        return result, error

    def syncReadTx(self, start_address, data_length, param, param_length):  # noqa: N802
        txpacket = [0] * (param_length + 8)
        txpacket[PKT_ID] = BROADCAST_ID
        txpacket[PKT_LENGTH] = param_length + 4
        txpacket[PKT_INSTRUCTION] = INST_SYNC_READ
        txpacket[PKT_PARAMETER0] = start_address
        txpacket[PKT_PARAMETER0 + 1] = data_length
        txpacket[PKT_PARAMETER0 + 2 : PKT_PARAMETER0 + 2 + param_length] = param[0:param_length]
        return self.txPacket(txpacket)

    def syncReadRx(self, data_length, param_length):  # noqa: N802
        wait_length = (6 + data_length) * param_length
        self.port_handler.setPacketTimeout(wait_length)
        rxpacket = []
        rx_length = 0
        while True:
            rxpacket.extend(self.port_handler.readPort(wait_length - rx_length))
            rx_length = len(rxpacket)
            if rx_length >= wait_length:
                result = COMM_SUCCESS
                break
            if self.port_handler.isPacketTimeout():
                result = COMM_RX_TIMEOUT if rx_length == 0 else COMM_RX_CORRUPT
                break
        self.port_handler.is_using = False
        return result, rxpacket

    def syncWriteTxOnly(self, start_address, data_length, param, param_length):  # noqa: N802
        txpacket = [0] * (param_length + 8)
        txpacket[PKT_ID] = BROADCAST_ID
        txpacket[PKT_LENGTH] = param_length + 4
        txpacket[PKT_INSTRUCTION] = INST_SYNC_WRITE
        txpacket[PKT_PARAMETER0] = start_address
        txpacket[PKT_PARAMETER0 + 1] = data_length
        txpacket[PKT_PARAMETER0 + 2 : PKT_PARAMETER0 + 2 + param_length] = param[0:param_length]
        _, result, _ = self.txRxPacket(txpacket)
        return result

    def reset(self, id_):
        txpacket = [0] * 6
        if id_ > BROADCAST_ID:
            return COMM_NOT_AVAILABLE, 0
        txpacket[PKT_ID] = id_
        txpacket[PKT_LENGTH] = 2
        txpacket[PKT_INSTRUCTION] = INST_RESET
        _, result, error = self.txRxPacket(txpacket)
        return result, error
