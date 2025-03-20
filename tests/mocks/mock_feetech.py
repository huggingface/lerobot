import abc
import random
from typing import Callable

import scservo_sdk as scs
import serial
from mock_serial import MockSerial

from lerobot.common.motors.feetech.feetech import SCS_SERIES_CONTROL_TABLE

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
    def sync_read(
        cls,
        scs_ids: list[int],
        start_address: int,
        data_length: int,
    ) -> bytes:
        """
        Builds a "Sync_Read" broadcast instruction.

        The parameters for Sync Read (Protocol 2.0) are:
            param[0]   = start_address
            param[1]   = data_length
            param[2+]  = motor IDs to read from

        And 'length' = (number_of_params + 7), where:
            +1 is for instruction byte,
            +1 is for the address byte,
            +1 is for the length bytes,
            +1 is for the checksum at the end.
        """
        params = [start_address, data_length, *scs_ids]
        length = len(scs_ids) + 4
        return cls.build(scs_id=scs.BROADCAST_ID, params=params, length=length, instruct_type="Sync_Read")


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
    def present_position(cls, scs_id: int, pos: int | None = None, min_max_range: tuple = (0, 4095)) -> bytes:
        """Builds a 'Present_Position' status packet.

        Args:
            scs_id (int): List of the servos ids
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

    'data_name' supported:
        - Present_Position
    """

    ctrl_table = SCS_SERIES_CONTROL_TABLE

    def __init__(self, scs_ids: list[int]):
        super().__init__()
        self._ids = scs_ids
        self.open()

    def build_single_motor_stubs(
        self, data_name: str, return_value: int | None = None, num_invalid_try: int | None = None
    ) -> None:
        address, length = self.ctrl_table[data_name]
        for idx in self._ids:
            if data_name == "Present_Position":
                sync_read_request_single = MockInstructionPacket.sync_read([idx], address, length)
                sync_read_response_single = self._build_present_pos_send_fn(
                    [idx], [return_value], num_invalid_try
                )
            else:
                raise NotImplementedError  # TODO(aliberts): add ping?

            self.stub(
                name=f"SyncRead_{data_name}_{idx}",
                receive_bytes=sync_read_request_single,
                send_fn=sync_read_response_single,
            )

    def build_all_motors_stub(
        self, data_name: str, return_values: list[int] | None = None, num_invalid_try: int | None = None
    ) -> None:
        address, length = self.ctrl_table[data_name]
        if data_name == "Present_Position":
            sync_read_request_all = MockInstructionPacket.sync_read(self._ids, address, length)
            sync_read_response_all = self._build_present_pos_send_fn(
                self._ids, return_values, num_invalid_try
            )
        else:
            raise NotImplementedError  # TODO(aliberts): add ping?

        self.stub(
            name=f"SyncRead_{data_name}_all",
            receive_bytes=sync_read_request_all,
            send_fn=sync_read_response_all,
        )

    def _build_present_pos_send_fn(
        self, scs_ids: list[int], return_pos: list[int] | None = None, num_invalid_try: int | None = None
    ) -> Callable[[int], bytes]:
        return_pos = [None for _ in scs_ids] if return_pos is None else return_pos
        assert len(return_pos) == len(scs_ids)

        def send_fn(_call_count: int) -> bytes:
            if num_invalid_try is not None and num_invalid_try >= _call_count:
                return b""

            packets = b"".join(
                MockStatusPacket.present_position(idx, pos)
                for idx, pos in zip(scs_ids, return_pos, strict=True)
            )
            return packets

        return send_fn
