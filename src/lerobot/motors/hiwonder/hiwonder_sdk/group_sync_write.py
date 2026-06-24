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

from .packet_handler import COMM_NOT_AVAILABLE


class GroupSyncWrite:
    def __init__(self, packet_handler, start_address, data_length):
        self.packet_handler = packet_handler
        self.start_address = start_address
        self.data_length = data_length
        self.is_param_changed = False
        self.param = []
        self.data_dict = {}

    def makeParam(self):  # noqa: N802
        if not self.data_dict:
            return
        self.param = []
        for id_ in self.data_dict:
            if not self.data_dict[id_]:
                return
            self.param.append(id_)
            self.param.extend(self.data_dict[id_])

    def addParam(self, id_, data):  # noqa: N802
        if id_ in self.data_dict:
            return False
        if len(data) > self.data_length:
            return False
        self.data_dict[id_] = data
        self.is_param_changed = True
        return True

    def removeParam(self, id_):  # noqa: N802
        if id_ not in self.data_dict:
            return
        del self.data_dict[id_]
        self.is_param_changed = True

    def changeParam(self, id_, data):  # noqa: N802
        if id_ not in self.data_dict:
            return False
        if len(data) > self.data_length:
            return False
        self.data_dict[id_] = data
        self.is_param_changed = True
        return True

    def clearParam(self):  # noqa: N802
        self.data_dict.clear()

    def txPacket(self):  # noqa: N802
        if not self.data_dict:
            return COMM_NOT_AVAILABLE
        if self.is_param_changed or not self.param:
            self.makeParam()
        return self.packet_handler.syncWriteTxOnly(
            self.start_address,
            self.data_length,
            self.param,
            len(self.data_dict) * (1 + self.data_length),
        )
