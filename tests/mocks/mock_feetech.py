import abc
from typing import Callable

import scservo_sdk as scs
import serial
from mock_serial import MockSerial

from lerobot.common.motors.feetech import STS_SMS_SERIES_CONTROL_TABLE
from lerobot.common.motors.feetech.feetech import _split_into_byte_chunks, patch_setPacketTimeout

from .mock_serial_patch import WaitableStub

# https://files.waveshare.com/upload/2/27/Communication_Protocol_User_Manual-EN%28191218-0923%29.pdf
INSTRUCTION_TYPES = {
    "Read": scs.INST_PING,              # Read data from the Device
    "Ping": scs.INST_READ,              # Checks whether the Packet has arrived at a device with the same ID as the specified packet ID
    "Write": scs.INST_WRITE,            # Write data to the Device
    "Reg_Write": scs.INST_REG_WRITE,    # Register the Instruction Packet in standby status; Packet can later be executed using the Action command
    "Action": scs.INST_ACTION,          # Executes a Packet that was registered beforehand using Reg Write
    "Factory_Reset": 0x06,              # Resets the Control Table to its initial factory default settings
    "Sync_Write": scs.INST_SYNC_WRITE,  # Write data to multiple devices with the same Address with the same length at once
    "Sync_Read": scs.INST_SYNC_READ,    # Read data from multiple devices with the same Address with the same length at once
}  # fmt: skip

ERROR_TYPE = {
    "Success": 0x00,
    "Voltage": scs.ERRBIT_VOLTAGE,
    "Angle": scs.ERRBIT_ANGLE,
    "Overheat": scs.ERRBIT_OVERHEAT,
    "Overele": scs.ERRBIT_OVERELE,
    "Overload": scs.ERRBIT_OVERLOAD,
}


class MockFeetechPacket(abc.ABC):
    @classmethod
    def build(cls, scs_id: int, params: list[int], length: int, *args, **kwargs) -> bytes:
        packet = cls._build(scs_id, params, length, *args, **kwargs)
        packet = cls._add_checksum(packet)
        return bytes(packet)

    @abc.abstractclassmethod
    def _build(cls, scs_id: int, params: list[int], length: int, *args, **kwargs) -> list[int]:
        pass

    @staticmethod
    def _add_checksum(packet: list[int]) -> list[int]:
        checksum = 0
        for id_ in range(2, len(packet) - 1):  # except header & checksum
            checksum += packet[id_]

        packet[-1] = ~checksum & 0xFF

        return packet


class MockInstructionPacket(MockFeetechPacket):
    """
    Helper class to build valid Feetech Instruction Packets.

    Instruction Packet structure
    (from https://files.waveshare.com/upload/2/27/Communication_Protocol_User_Manual-EN%28191218-0923%29.pdf)

    | Header    | Packet ID | Length | Instruction | Params            | Checksum |
    | --------- | --------- | ------ | ----------- | ----------------- | -------- |
    | 0xFF 0xFF | ID        | Len    | Instr       | Param 1 … Param N | Sum      |

    """

    @classmethod
    def _build(cls, scs_id: int, params: list[int], length: int, instruct_type: str) -> list[int]:
        instruct_value = INSTRUCTION_TYPES[instruct_type]
        return [
            0xFF, 0xFF,      # header
            scs_id,          # servo id
            length,          # length
            instruct_value,  # instruction type
            *params,         # data bytes
            0x00,            # placeholder for checksum
        ]  # fmt: skip

    @classmethod
    def ping(
        cls,
        scs_id: int,
    ) -> bytes:
        """
        Builds a "Ping" broadcast instruction.

        No parameters required.
        """
        return cls.build(scs_id=scs_id, params=[], length=2, instruct_type="Ping")

    @classmethod
    def read(
        cls,
        scs_id: int,
        start_address: int,
        data_length: int,
    ) -> bytes:
        """
        Builds a "Read" instruction.

        The parameters for Read are:
            param[0]   = start_address
            param[1]   = data_length

        And 'length' = 4, where:
            +1 is for instruction byte,
            +1 is for the address byte,
            +1 is for the length bytes,
            +1 is for the checksum at the end.
        """
        params = [start_address, data_length]
        length = 4
        return cls.build(scs_id=scs_id, params=params, length=length, instruct_type="Read")

    @classmethod
    def write(
        cls,
        scs_id: int,
        value: int,
        start_address: int,
        data_length: int,
    ) -> bytes:
        """
        Builds a "Write" instruction.

        The parameters for Write are:
            param[0]   = start_address L
            param[1]   = start_address H
            param[2]   = 1st Byte
            param[3]   = 2nd Byte
            ...
            param[1+X] = X-th Byte

        And 'length' = data_length + 3, where:
            +1 is for instruction byte,
            +1 is for the length bytes,
            +1 is for the checksum at the end.
        """
        data = _split_into_byte_chunks(value, data_length)
        params = [start_address, *data]
        length = data_length + 3
        return cls.build(scs_id=scs_id, params=params, length=length, instruct_type="Write")

    @classmethod
    def sync_read(
        cls,
        scs_ids: list[int],
        start_address: int,
        data_length: int,
    ) -> bytes:
        """
        Builds a "Sync_Read" broadcast instruction.

        The parameters for Sync Read are:
            param[0]   = start_address
            param[1]   = data_length
            param[2+]  = motor IDs to read from

        And 'length' = (number_of_params + 4), where:
            +1 is for instruction byte,
            +1 is for the address byte,
            +1 is for the length bytes,
            +1 is for the checksum at the end.
        """
        params = [start_address, data_length, *scs_ids]
        length = len(scs_ids) + 4
        return cls.build(scs_id=scs.BROADCAST_ID, params=params, length=length, instruct_type="Sync_Read")

    @classmethod
    def sync_write(
        cls,
        ids_values: dict[int],
        start_address: int,
        data_length: int,
    ) -> bytes:
        """
        Builds a "Sync_Write" broadcast instruction.

        The parameters for Sync_Write are:
            param[0]   = start_address
            param[1]   = data_length
            param[2]   = [1st motor] ID
            param[2+1] = [1st motor] 1st Byte
            param[2+2] = [1st motor] 2nd Byte
            ...
            param[5+X] = [1st motor] X-th Byte
            param[6]   = [2nd motor] ID
            param[6+1] = [2nd motor] 1st Byte
            param[6+2] = [2nd motor] 2nd Byte
            ...
            param[6+X] = [2nd motor] X-th Byte

        And 'length' = ((number_of_params * 1 + data_length) + 4), where:
            +1 is for instruction byte,
            +1 is for the address byte,
            +1 is for the length bytes,
            +1 is for the checksum at the end.
        """
        data = []
        for id_, value in ids_values.items():
            split_value = _split_into_byte_chunks(value, data_length)
            data += [id_, *split_value]
        params = [start_address, data_length, *data]
        length = len(ids_values) * (1 + data_length) + 4
        return cls.build(scs_id=scs.BROADCAST_ID, params=params, length=length, instruct_type="Sync_Write")


class MockStatusPacket(MockFeetechPacket):
    """
    Helper class to build valid Feetech Status Packets.

    Status Packet structure
    (from https://files.waveshare.com/upload/2/27/Communication_Protocol_User_Manual-EN%28191218-0923%29.pdf)

    | Header    | Packet ID | Length | Error | Params            | Checksum |
    | --------- | --------- | ------ | ----- | ----------------- | -------- |
    | 0xFF 0xFF | ID        | Len    | Err   | Param 1 … Param N | Sum      |

    """

    @classmethod
    def _build(cls, scs_id: int, params: list[int], length: int, error: str = "Success") -> list[int]:
        err_byte = ERROR_TYPE[error]
        return [
            0xFF, 0xFF,  # header
            scs_id,      # servo id
            length,      # length
            err_byte,    # status
            *params,     # data bytes
            0x00,        # placeholder for checksum
        ]  # fmt: skip

    @classmethod
    def ping(cls, scs_id: int, error: str = "Success") -> bytes:
        """Builds a 'Ping' status packet.

        Args:
            scs_id (int): ID of the servo responding.
            error (str, optional): Error to be returned. Defaults to "Success".

        Returns:
            bytes: The raw 'Ping' status packet ready to be sent through serial.
        """
        return cls.build(scs_id, params=[], length=2, error=error)

    @classmethod
    def read(cls, scs_id: int, value: int, param_length: int) -> bytes:
        """Builds a 'Read' status packet.

        Args:
            scs_id (int): ID of the servo responding.
            value (int): Desired value to be returned in the packet.
            param_length (int): The address length as reported in the control table.

        Returns:
            bytes: The raw 'Sync Read' status packet ready to be sent through serial.
        """
        params = _split_into_byte_chunks(value, param_length)
        length = param_length + 2
        return cls.build(scs_id, params=params, length=length)


class MockPortHandler(scs.PortHandler):
    """
    This class overwrite the 'setupPort' method of the Feetech PortHandler because it can specify
    baudrates that are not supported with a serial port on MacOS.
    """

    def setupPort(self, cflag_baud):  # noqa: N802
        if self.is_open:
            self.closePort()

        self.ser = serial.Serial(
            port=self.port_name,
            # baudrate=self.baudrate,  <- This will fail on MacOS
            # parity = serial.PARITY_ODD,
            # stopbits = serial.STOPBITS_TWO,
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
    This class will simulate physical motors by responding with valid status packets upon receiving some
    instruction packets. It is meant to test MotorsBus classes.
    """

    ctrl_table = STS_SMS_SERIES_CONTROL_TABLE

    def __init__(self):
        super().__init__()

    @property
    def stubs(self) -> dict[str, WaitableStub]:
        return super().stubs

    def stub(self, *, name=None, **kwargs):
        new_stub = WaitableStub(**kwargs)
        self._MockSerial__stubs[name or new_stub.receive_bytes] = new_stub
        return new_stub

    def build_broadcast_ping_stub(self, ids: list[int] | None = None, num_invalid_try: int = 0) -> str:
        ping_request = MockInstructionPacket.ping(scs.BROADCAST_ID)
        return_packets = b"".join(MockStatusPacket.ping(id_) for id_ in ids)
        ping_response = self._build_send_fn(return_packets, num_invalid_try)
        stub_name = "Ping_" + "_".join([str(id_) for id_ in ids])
        self.stub(
            name=stub_name,
            receive_bytes=ping_request,
            send_fn=ping_response,
        )
        return stub_name

    def build_ping_stub(self, scs_id: int, num_invalid_try: int = 0) -> str:
        ping_request = MockInstructionPacket.ping(scs_id)
        return_packet = MockStatusPacket.ping(scs_id)
        ping_response = self._build_send_fn(return_packet, num_invalid_try)
        stub_name = f"Ping_{scs_id}"
        self.stub(
            name=stub_name,
            receive_bytes=ping_request,
            send_fn=ping_response,
        )
        return stub_name

    def build_read_stub(
        self, data_name: str, scs_id: int, value: int | None = None, num_invalid_try: int = 0
    ) -> str:
        address, length = self.ctrl_table[data_name]
        read_request = MockInstructionPacket.read(scs_id, address, length)
        return_packet = MockStatusPacket.read(scs_id, value, length)
        read_response = self._build_send_fn(return_packet, num_invalid_try)
        stub_name = f"Read_{data_name}_{scs_id}"
        self.stub(
            name=stub_name,
            receive_bytes=read_request,
            send_fn=read_response,
        )
        return stub_name

    def build_sync_read_stub(
        self, data_name: str, ids_values: dict[int, int] | None = None, num_invalid_try: int = 0
    ) -> str:
        address, length = self.ctrl_table[data_name]
        sync_read_request = MockInstructionPacket.sync_read(list(ids_values), address, length)
        return_packets = b"".join(MockStatusPacket.read(id_, pos, length) for id_, pos in ids_values.items())

        sync_read_response = self._build_send_fn(return_packets, num_invalid_try)
        stub_name = f"Sync_Read_{data_name}_" + "_".join([str(id_) for id_ in ids_values])
        self.stub(
            name=stub_name,
            receive_bytes=sync_read_request,
            send_fn=sync_read_response,
        )
        return stub_name

    def build_sequential_sync_read_stub(
        self, data_name: str, ids_values: dict[int, list[int]] | None = None
    ) -> str:
        sequence_length = len(next(iter(ids_values.values())))
        assert all(len(positions) == sequence_length for positions in ids_values.values())
        address, length = self.ctrl_table[data_name]
        sync_read_request = MockInstructionPacket.sync_read(list(ids_values), address, length)
        sequential_packets = []
        for count in range(sequence_length):
            return_packets = b"".join(
                MockStatusPacket.read(id_, positions[count], length) for id_, positions in ids_values.items()
            )
            sequential_packets.append(return_packets)

        sync_read_response = self._build_sequential_send_fn(sequential_packets)
        stub_name = f"Seq_Sync_Read_{data_name}_" + "_".join([str(id_) for id_ in ids_values])
        self.stub(
            name=stub_name,
            receive_bytes=sync_read_request,
            send_fn=sync_read_response,
        )
        return stub_name

    def build_sync_write_stub(
        self, data_name: str, ids_values: dict[int, int] | None = None, num_invalid_try: int = 0
    ) -> str:
        address, length = self.ctrl_table[data_name]
        sync_read_request = MockInstructionPacket.sync_write(ids_values, address, length)
        stub_name = f"Sync_Write_{data_name}_" + "_".join([str(id_) for id_ in ids_values])
        self.stub(
            name=stub_name,
            receive_bytes=sync_read_request,
            send_fn=self._build_send_fn(b"", num_invalid_try),
        )
        return stub_name

    def build_write_stub(
        self, data_name: str, scs_id: int, value: int, error: str = "Success", num_invalid_try: int = 0
    ) -> str:
        address, length = self.ctrl_table[data_name]
        sync_read_request = MockInstructionPacket.write(scs_id, value, address, length)
        return_packet = MockStatusPacket.build(scs_id, params=[], length=2, error=error)
        stub_name = f"Write_{data_name}_{scs_id}"
        self.stub(
            name=stub_name,
            receive_bytes=sync_read_request,
            send_fn=self._build_send_fn(return_packet, num_invalid_try),
        )
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
