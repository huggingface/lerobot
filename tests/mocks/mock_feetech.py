import abc
import random
from typing import Callable

import scservo_sdk as scs
import serial
from mock_serial import MockSerial

from lerobot.common.motors.feetech import SCS_SERIES_CONTROL_TABLE, FeetechMotorsBus

from .mock_serial_patch import WaitableStub

# https://files.waveshare.com/upload/2/27/Communication_Protocol_User_Manual-EN%28191218-0923%29.pdf
INSTRUCTION_TYPES = {
    "Ping": 0x01,	        # Checks whether the Packet has arrived at a device with the same ID as the specified packet ID
    "Read": 0x02,	        # Read data from the Device
    "Write": 0x03,	        # Write data to the Device
    "Reg_Write": 0x04,	    # Register the Instruction Packet in standby status; Packet can later be executed using the Action command
    "Action": 0x05,	        # Executes a Packet that was registered beforehand using Reg Write
    "Factory_Reset": 0x06,  # Resets the Control Table to its initial factory default settings
    "Sync_Read": 0x82,	           # Read data from multiple devices with the same Address with the same length at once
    "Sync_Write": 0x83,	    # Write data to multiple devices with the same Address with the same length at once
}  # fmt: skip

ERROR_TYPE = {
    "Success": 0x00,
    "Voltage": 0x01,
    "Angle": 0x02,
    "Overheat": 0x04,
    "Overele": 0x08,
    "Overload": 0x20,
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
        for idx in range(2, len(packet) - 1):  # except header & checksum
            checksum += packet[idx]

        packet[-1] = scs.SCS_LOBYTE(~checksum)

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
        params, length = [], 2
        return cls.build(scs_id=scs_id, params=params, length=length, instruct_type="Ping")

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
        for idx, value in ids_values.items():
            split_value = FeetechMotorsBus._split_int_to_bytes(value, data_length)
            data += [idx, *split_value]
        params = [start_address, data_length, *data]
        length = len(ids_values) * (1 + data_length) + 4
        return cls.build(scs_id=scs.BROADCAST_ID, params=params, length=length, instruct_type="Sync_Write")

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
        data = FeetechMotorsBus._split_int_to_bytes(value, data_length)
        params = [start_address, *data]
        length = data_length + 3
        return cls.build(scs_id=scs_id, params=params, length=length, instruct_type="Write")


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
    def ping(cls, scs_id: int, model_nb: int = 1190, firm_ver: int = 50) -> bytes:
        """Builds a 'Ping' status packet.

        Args:
            scs_id (int): ID of the servo responding.
            model_nb (int, optional): Desired 'model number' to be returned in the packet. Defaults to 1190
                which corresponds to a XL330-M077-T.
            firm_ver (int, optional): Desired 'firmware version' to be returned in the packet.
                Defaults to 50.

        Returns:
            bytes: The raw 'Ping' status packet ready to be sent through serial.
        """
        # raise NotImplementedError
        params = [scs.SCS_LOBYTE(model_nb), scs.SCS_HIBYTE(model_nb), firm_ver]
        length = 2
        return cls.build(scs_id, params=params, length=length)

    @classmethod
    def present_position(cls, scs_id: int, pos: int | None = None, min_max_range: tuple = (0, 4095)) -> bytes:
        """Builds a 'Present_Position' status packet.

        Args:
            scs_id (int): List of the servos ids.
            pos (int | None, optional): Desired 'Present_Position' to be returned in the packet. If None, it
                will use a random value in the min_max_range. Defaults to None.
            min_max_range (tuple, optional): Min/max range to generate the position values used for when 'pos'
                is None. Note that the bounds are included in the range. Defaults to (0, 4095).

        Returns:
            bytes: The raw 'Present_Position' status packet ready to be sent through serial.
        """
        pos = random.randint(*min_max_range) if pos is None else pos
        params = [scs.SCS_LOBYTE(pos), scs.SCS_HIBYTE(pos)]
        length = 4
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


class MockMotors(MockSerial):
    """
    This class will simulate physical motors by responding with valid status packets upon receiving some
    instruction packets. It is meant to test MotorsBus classes.
    """

    ctrl_table = SCS_SERIES_CONTROL_TABLE

    def __init__(self):
        super().__init__()

    @property
    def stubs(self) -> dict[str, WaitableStub]:
        return super().stubs

    def stub(self, *, name=None, **kwargs):
        new_stub = WaitableStub(**kwargs)
        self._MockSerial__stubs[name or new_stub.receive_bytes] = new_stub
        return new_stub

    def build_broadcast_ping_stub(
        self, ids_models_firmwares: dict[int, list[int]] | None = None, num_invalid_try: int = 0
    ) -> str:
        ping_request = MockInstructionPacket.ping(scs.BROADCAST_ID)
        return_packets = b"".join(
            MockStatusPacket.ping(idx, model, firm_ver)
            for idx, (model, firm_ver) in ids_models_firmwares.items()
        )
        ping_response = self._build_send_fn(return_packets, num_invalid_try)

        stub_name = "Ping_" + "_".join([str(idx) for idx in ids_models_firmwares])
        self.stub(
            name=stub_name,
            receive_bytes=ping_request,
            send_fn=ping_response,
        )
        return stub_name

    def build_ping_stub(
        self, scs_id: int, model_nb: int, firm_ver: int = 50, num_invalid_try: int = 0
    ) -> str:
        ping_request = MockInstructionPacket.ping(scs_id)
        return_packet = MockStatusPacket.ping(scs_id, model_nb, firm_ver)
        ping_response = self._build_send_fn(return_packet, num_invalid_try)
        stub_name = f"Ping_{scs_id}"
        self.stub(
            name=stub_name,
            receive_bytes=ping_request,
            send_fn=ping_response,
        )
        return stub_name

    def build_sync_read_stub(
        self, data_name: str, ids_values: dict[int, int] | None = None, num_invalid_try: int = 0
    ) -> str:
        """
        'data_name' supported:
            - Present_Position
        """
        if data_name != "Present_Position":
            raise NotImplementedError

        address, length = self.ctrl_table[data_name]
        sync_read_request = MockInstructionPacket.sync_read(list(ids_values), address, length)
        return_packets = b"".join(
            MockStatusPacket.present_position(idx, pos) for idx, pos in ids_values.items()
        )
        sync_read_response = self._build_send_fn(return_packets, num_invalid_try)
        stub_name = f"Sync_Read_{data_name}_" + "_".join([str(idx) for idx in ids_values])
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
        stub_name = f"Sync_Write_{data_name}_" + "_".join([str(idx) for idx in ids_values])
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
