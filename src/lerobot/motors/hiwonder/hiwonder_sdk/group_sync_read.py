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

from .packet_handler import COMM_NOT_AVAILABLE, COMM_RX_CORRUPT, COMM_RX_TIMEOUT, COMM_SUCCESS


class GroupSyncRead:
    def __init__(self, packet_handler, start_address, data_length):
        self.packet_handler = packet_handler
        self.start_address = start_address
        self.data_length = data_length
        self.last_result = False
        self.is_param_changed = False
        self.param = []
        self.data_dict = {}

    def makeParam(self):  # noqa: N802
        if not self.data_dict:
            return
        self.param = list(self.data_dict.keys())

    def addParam(self, id_):  # noqa: N802
        if id_ in self.data_dict:
            return False
        self.data_dict[id_] = []
        self.is_param_changed = True
        return True

    def removeParam(self, id_):  # noqa: N802
        if id_ not in self.data_dict:
            return
        del self.data_dict[id_]
        self.is_param_changed = True

    def clearParam(self):  # noqa: N802
        self.data_dict.clear()

    def txPacket(self):  # noqa: N802
        if not self.data_dict:
            return COMM_NOT_AVAILABLE
        if self.is_param_changed or not self.param:
            self.makeParam()
        return self.packet_handler.syncReadTx(
            self.start_address, self.data_length, self.param, len(self.data_dict)
        )

    def rxPacket(self):  # noqa: N802
        self.last_result = True
        if not self.data_dict:
            return COMM_NOT_AVAILABLE

        result, rxpacket = self.packet_handler.syncReadRx(self.data_length, len(self.data_dict))

        if len(rxpacket) >= (self.data_length + 6):
            for id_ in self.data_dict:
                self.data_dict[id_], result = self._readRx(rxpacket, id_, self.data_length)
                if result != COMM_SUCCESS:
                    self.last_result = False
        else:
            self.last_result = False

        return result

    def txRxPacket(self):  # noqa: N802
        result = self.txPacket()
        if result != COMM_SUCCESS:
            return result
        return self.rxPacket()

    def _readRx(self, rxpacket, id_, data_length):
        data = []
        rx_length = len(rxpacket)
        rx_index = 0

        while (rx_index + 6 + data_length) <= rx_length:
            headpacket = [0x00, 0x00, 0x00]
            while rx_index < rx_length:
                headpacket[2] = headpacket[1]
                headpacket[1] = headpacket[0]
                headpacket[0] = rxpacket[rx_index]
                rx_index += 1
                if headpacket[2] == 0xFF and headpacket[1] == 0xFF and headpacket[0] == id_:
                    break

            if (rx_index + 3 + data_length) > rx_length:
                break
            if rxpacket[rx_index] != (data_length + 2):
                rx_index += 1
                continue

            rx_index += 1
            error = rxpacket[rx_index]
            rx_index += 1
            cal_sum = id_ + (data_length + 2) + error
            data = [error]
            data.extend(rxpacket[rx_index : rx_index + data_length])
            for i in range(data_length):
                cal_sum += rxpacket[rx_index]
                rx_index += 1
            cal_sum = ~cal_sum & 0xFF

            if cal_sum != rxpacket[rx_index]:
                return None, COMM_RX_CORRUPT
            return data, COMM_SUCCESS

        return None, COMM_RX_CORRUPT

    def isAvailable(self, id_, address, data_length):  # noqa: N802
        if id_ not in self.data_dict:
            return False, 0
        if address < self.start_address or self.start_address + self.data_length - data_length < address:
            return False, 0
        if not self.data_dict[id_] or len(self.data_dict[id_]) < (data_length + 1):
            return False, 0
        return True, self.data_dict[id_][0]

    def getData(self, id_, address, data_length):  # noqa: N802
        offset = address - self.start_address + 1
        if data_length == 1:
            return self.data_dict[id_][offset]
        elif data_length == 2:
            return self.packet_handler.makeWord16(
                self.data_dict[id_][offset], self.data_dict[id_][offset + 1]
            )
        elif data_length == 4:
            return self.packet_handler.makeWord32(
                self.packet_handler.makeWord16(
                    self.data_dict[id_][offset], self.data_dict[id_][offset + 1]
                ),
                self.packet_handler.makeWord16(
                    self.data_dict[id_][offset + 2], self.data_dict[id_][offset + 3]
                ),
            )
        return 0
