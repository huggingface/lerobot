#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import abc
from collections.abc import Callable

import serial
from mock_serial import MockSerial

from lerobot.motors.feetech.feetech import patch_setPacketTimeout
from lerobot.motors.hiwonder import hiwonder_sdk as hw

from .mock_serial_patch import WaitableStub


def _split_into_byte_chunks(value: int, length: int) -> list[int]:
    if length == 1:
        return [value]
    elif length == 2:
        return [value & 0xFF, (value >> 8) & 0xFF]
    elif length == 4:
        lo = value & 0xFFFF
        hi = (value >> 16) & 0xFFFF
        return [lo & 0xFF, (lo >> 8) & 0xFF, hi & 0xFF, (hi >> 8) & 0xFF]
    raise ValueError(f"Unsupported data length: {length}")


class MockHiwonderPacket(abc.ABC):
    @classmethod
    def build(cls, hw_id: int, params: list[int], length: int, *args, **kwargs) -> bytes:
        packet = cls._build(hw_id, params, length, *args, **kwargs)
        packet = cls._add_checksum(packet)
        return bytes(packet)

    @classmethod
    @abc.abstractmethod
    def _build(cls, hw_id: int, params: list[int], length: int, *args, **kwargs) -> list[int]:
        pass

    @staticmethod
    def _add_checksum(packet: list[int]) -> list[int]:
        checksum = 0
        for id_ in range(2, len(packet) - 1):
            checksum += packet[id_]
        packet[-1] = ~checksum & 0xFF
        return packet


class MockInstructionPacket(MockHiwonderPacket):
    """
    Helper class to build valid Hiwonder Instruction Packets.

    Instruction Packet structure:

    | Header    | Packet ID | Length | Instruction | Params            | Checksum |
    | --------- | --------- | ------ | ----------- | ----------------- | -------- |
    | 0xFF 0xFF | ID        | Len    | Instr       | Param 1 … Param N | Sum      |
    """

    @classmethod
    def _build(cls, hw_id: int, params: list[int], length: int, instruction: int) -> list[int]:
        return [
            0xFF, 0xFF,   # header
            hw_id,        # servo id
            length,       # length
            instruction,  # instruction type
            *params,      # data bytes
            0x00,         # placeholder for checksum
        ]  # fmt: skip

    @classmethod
    def ping(cls, hw_id: int) -> bytes:
        return cls.build(hw_id=hw_id, params=[], length=2, instruction=hw.INST_PING)

    @classmethod
    def read(cls, hw_id: int, start_address: int, data_length: int) -> bytes:
        params = [start_address, data_length]
        return cls.build(hw_id=hw_id, params=params, length=4, instruction=hw.INST_READ)

    @classmethod
    def write(cls, hw_id: int, value: int, start_address: int, data_length: int) -> bytes:
        data = _split_into_byte_chunks(value, data_length)
        params = [start_address, *data]
        length = data_length + 3
        return cls.build(hw_id=hw_id, params=params, length=length, instruction=hw.INST_WRITE)

    @classmethod
    def sync_read(cls, hw_ids: list[int], start_address: int, data_length: int) -> bytes:
        params = [start_address, data_length, *hw_ids]
        length = len(hw_ids) + 4
        return cls.build(hw_id=hw.BROADCAST_ID, params=params, length=length, instruction=hw.INST_SYNC_READ)

    @classmethod
    def sync_write(cls, ids_values: dict[int, int], start_address: int, data_length: int) -> bytes:
        data = []
        for id_, value in ids_values.items():
            data += [id_, *_split_into_byte_chunks(value, data_length)]
        params = [start_address, data_length, *data]
        length = len(ids_values) * (1 + data_length) + 4
        return cls.build(hw_id=hw.BROADCAST_ID, params=params, length=length, instruction=hw.INST_SYNC_WRITE)


class MockStatusPacket(MockHiwonderPacket):
    """
    Helper class to build valid Hiwonder Status Packets.

    Status Packet structure:

    | Header    | Packet ID | Length | Error | Params            | Checksum |
    | --------- | --------- | ------ | ----- | ----------------- | -------- |
    | 0xFF 0xFF | ID        | Len    | Err   | Param 1 … Param N | Sum      |
    """

    @classmethod
    def _build(cls, hw_id: int, params: list[int], length: int, error: int = 0) -> list[int]:
        return [
            0xFF, 0xFF,  # header
            hw_id,       # servo id
            length,      # length
            error,       # status
            *params,     # data bytes
            0x00,        # placeholder for checksum
        ]  # fmt: skip

    @classmethod
    def ping(cls, hw_id: int, error: int = 0) -> bytes:
        return cls.build(hw_id, params=[], length=2, error=error)

    @classmethod
    def read(cls, hw_id: int, value: int, param_length: int, error: int = 0) -> bytes:
        params = _split_into_byte_chunks(value, param_length)
        length = param_length + 2
        return cls.build(hw_id, params=params, length=length, error=error)


class MockPortHandler(hw.PortHandler):
    """
    Overrides setupPort to allow running tests without a real serial port (e.g. on macOS).
    """

    def setupPort(self, cflag_baud):  # noqa: N802
        if self.is_open:
            self.closePort()
        self.ser = serial.Serial(
            port=self.port_name,
            bytesize=serial.EIGHTBITS,
            timeout=0,
        )
        self.is_open = True
        self.ser.reset_input_buffer()
        self.tx_time_per_byte = (1000.0 / self.baudrate) * 10.0
        return True

    def setPacketTimeout(self, packet_length):  # noqa: N802
        return patch_setPacketTimeout(self, packet_length)


class MockMotors(MockSerial):
    """
    Simulates physical Hiwonder servos by responding with valid status packets.
    """

    def __init__(self):
        super().__init__()

    @property
    def stubs(self) -> dict[str, WaitableStub]:
        return super().stubs

    def stub(self, *, name=None, **kwargs):
        new_stub = WaitableStub(**kwargs)
        self._MockSerial__stubs[name or new_stub.receive_bytes] = new_stub
        return new_stub

    def build_broadcast_ping_stub(self, ids: list[int], num_invalid_try: int = 0) -> str:
        ping_request = MockInstructionPacket.ping(hw.BROADCAST_ID)
        return_packets = b"".join(MockStatusPacket.ping(id_) for id_ in ids)
        stub_name = "Ping_" + "_".join(str(id_) for id_ in ids)
        self.stub(name=stub_name, receive_bytes=ping_request, send_fn=self._build_send_fn(return_packets, num_invalid_try))
        return stub_name

    def build_ping_stub(self, hw_id: int, num_invalid_try: int = 0, error: int = 0) -> str:
        ping_request = MockInstructionPacket.ping(hw_id)
        return_packet = MockStatusPacket.ping(hw_id, error)
        stub_name = f"Ping_{hw_id}_{error}"
        self.stub(name=stub_name, receive_bytes=ping_request, send_fn=self._build_send_fn(return_packet, num_invalid_try))
        return stub_name

    def build_read_stub(self, address: int, length: int, hw_id: int, value: int, reply: bool = True, error: int = 0, num_invalid_try: int = 0) -> str:
        read_request = MockInstructionPacket.read(hw_id, address, length)
        return_packet = MockStatusPacket.read(hw_id, value, length, error) if reply else b""
        stub_name = f"Read_{address}_{length}_{hw_id}_{value}_{error}"
        self.stub(name=stub_name, receive_bytes=read_request, send_fn=self._build_send_fn(return_packet, num_invalid_try))
        return stub_name

    def build_write_stub(self, address: int, length: int, hw_id: int, value: int, reply: bool = True, error: int = 0, num_invalid_try: int = 0) -> str:
        write_request = MockInstructionPacket.write(hw_id, value, address, length)
        return_packet = MockStatusPacket.build(hw_id, params=[], length=2, error=error) if reply else b""
        stub_name = f"Write_{address}_{length}_{hw_id}"
        self.stub(name=stub_name, receive_bytes=write_request, send_fn=self._build_send_fn(return_packet, num_invalid_try))
        return stub_name

    def build_sync_read_stub(self, address: int, length: int, ids_values: dict[int, int], reply: bool = True, num_invalid_try: int = 0) -> str:
        sync_read_request = MockInstructionPacket.sync_read(list(ids_values), address, length)
        return_packets = b"".join(MockStatusPacket.read(id_, pos, length) for id_, pos in ids_values.items()) if reply else b""
        stub_name = f"Sync_Read_{address}_{length}_" + "_".join(str(id_) for id_ in ids_values)
        self.stub(name=stub_name, receive_bytes=sync_read_request, send_fn=self._build_send_fn(return_packets, num_invalid_try))
        return stub_name

    def build_sequential_sync_read_stub(
        self, address: int, length: int, ids_values: dict[int, list[int]] | None = None
    ) -> str:
        sequence_length = len(next(iter(ids_values.values())))
        assert all(len(positions) == sequence_length for positions in ids_values.values())
        sync_read_request = MockInstructionPacket.sync_read(list(ids_values), address, length)
        sequential_packets = []
        for count in range(sequence_length):
            return_packets = b"".join(
                MockStatusPacket.read(id_, positions[count], length) for id_, positions in ids_values.items()
            )
            sequential_packets.append(return_packets)
        sync_read_response = self._build_sequential_send_fn(sequential_packets)
        stub_name = f"Seq_Sync_Read_{address}_{length}_" + "_".join(str(id_) for id_ in ids_values)
        self.stub(name=stub_name, receive_bytes=sync_read_request, send_fn=sync_read_response)
        return stub_name

    def build_sync_write_stub(self, address: int, length: int, ids_values: dict[int, int], num_invalid_try: int = 0) -> str:
        sync_write_request = MockInstructionPacket.sync_write(ids_values, address, length)
        stub_name = f"Sync_Write_{address}_{length}_" + "_".join(str(id_) for id_ in ids_values)
        self.stub(name=stub_name, receive_bytes=sync_write_request, send_fn=self._build_send_fn(b"", num_invalid_try))
        return stub_name

    @staticmethod
    def _build_send_fn(packet: bytes, num_invalid_try: int = 0) -> Callable[[int], bytes]:
        def send_fn(_call_count: int) -> bytes:
            if num_invalid_try >= _call_count:
                return b""
            return packet
        return send_fn

    @staticmethod
    def _build_sequential_send_fn(packets: list[bytes]) -> Callable[[int], bytes]:
        def send_fn(_call_count: int) -> bytes:
            return packets[_call_count - 1]
        return send_fn
