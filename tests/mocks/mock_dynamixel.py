import abc
from typing import Callable

import dynamixel_sdk as dxl
import serial
from mock_serial.mock_serial import MockSerial

from lerobot.common.motors.dynamixel.dynamixel import _split_into_byte_chunks

from .mock_serial_patch import WaitableStub

# https://emanual.robotis.com/docs/en/dxl/crc/
DXL_CRC_TABLE = [
    0x0000, 0x8005, 0x800F, 0x000A, 0x801B, 0x001E, 0x0014, 0x8011,
    0x8033, 0x0036, 0x003C, 0x8039, 0x0028, 0x802D, 0x8027, 0x0022,
    0x8063, 0x0066, 0x006C, 0x8069, 0x0078, 0x807D, 0x8077, 0x0072,
    0x0050, 0x8055, 0x805F, 0x005A, 0x804B, 0x004E, 0x0044, 0x8041,
    0x80C3, 0x00C6, 0x00CC, 0x80C9, 0x00D8, 0x80DD, 0x80D7, 0x00D2,
    0x00F0, 0x80F5, 0x80FF, 0x00FA, 0x80EB, 0x00EE, 0x00E4, 0x80E1,
    0x00A0, 0x80A5, 0x80AF, 0x00AA, 0x80BB, 0x00BE, 0x00B4, 0x80B1,
    0x8093, 0x0096, 0x009C, 0x8099, 0x0088, 0x808D, 0x8087, 0x0082,
    0x8183, 0x0186, 0x018C, 0x8189, 0x0198, 0x819D, 0x8197, 0x0192,
    0x01B0, 0x81B5, 0x81BF, 0x01BA, 0x81AB, 0x01AE, 0x01A4, 0x81A1,
    0x01E0, 0x81E5, 0x81EF, 0x01EA, 0x81FB, 0x01FE, 0x01F4, 0x81F1,
    0x81D3, 0x01D6, 0x01DC, 0x81D9, 0x01C8, 0x81CD, 0x81C7, 0x01C2,
    0x0140, 0x8145, 0x814F, 0x014A, 0x815B, 0x015E, 0x0154, 0x8151,
    0x8173, 0x0176, 0x017C, 0x8179, 0x0168, 0x816D, 0x8167, 0x0162,
    0x8123, 0x0126, 0x012C, 0x8129, 0x0138, 0x813D, 0x8137, 0x0132,
    0x0110, 0x8115, 0x811F, 0x011A, 0x810B, 0x010E, 0x0104, 0x8101,
    0x8303, 0x0306, 0x030C, 0x8309, 0x0318, 0x831D, 0x8317, 0x0312,
    0x0330, 0x8335, 0x833F, 0x033A, 0x832B, 0x032E, 0x0324, 0x8321,
    0x0360, 0x8365, 0x836F, 0x036A, 0x837B, 0x037E, 0x0374, 0x8371,
    0x8353, 0x0356, 0x035C, 0x8359, 0x0348, 0x834D, 0x8347, 0x0342,
    0x03C0, 0x83C5, 0x83CF, 0x03CA, 0x83DB, 0x03DE, 0x03D4, 0x83D1,
    0x83F3, 0x03F6, 0x03FC, 0x83F9, 0x03E8, 0x83ED, 0x83E7, 0x03E2,
    0x83A3, 0x03A6, 0x03AC, 0x83A9, 0x03B8, 0x83BD, 0x83B7, 0x03B2,
    0x0390, 0x8395, 0x839F, 0x039A, 0x838B, 0x038E, 0x0384, 0x8381,
    0x0280, 0x8285, 0x828F, 0x028A, 0x829B, 0x029E, 0x0294, 0x8291,
    0x82B3, 0x02B6, 0x02BC, 0x82B9, 0x02A8, 0x82AD, 0x82A7, 0x02A2,
    0x82E3, 0x02E6, 0x02EC, 0x82E9, 0x02F8, 0x82FD, 0x82F7, 0x02F2,
    0x02D0, 0x82D5, 0x82DF, 0x02DA, 0x82CB, 0x02CE, 0x02C4, 0x82C1,
    0x8243, 0x0246, 0x024C, 0x8249, 0x0258, 0x825D, 0x8257, 0x0252,
    0x0270, 0x8275, 0x827F, 0x027A, 0x826B, 0x026E, 0x0264, 0x8261,
    0x0220, 0x8225, 0x822F, 0x022A, 0x823B, 0x023E, 0x0234, 0x8231,
    0x8213, 0x0216, 0x021C, 0x8219, 0x0208, 0x820D, 0x8207, 0x0202
]  # fmt: skip


class MockDynamixelPacketv2(abc.ABC):
    @classmethod
    def build(cls, dxl_id: int, params: list[int], length: int, *args, **kwargs) -> bytes:
        packet = cls._build(dxl_id, params, length, *args, **kwargs)
        packet = cls._add_stuffing(packet)
        packet = cls._add_crc(packet)
        return bytes(packet)

    @abc.abstractclassmethod
    def _build(cls, dxl_id: int, params: list[int], length: int, *args, **kwargs) -> list[int]:
        pass

    @staticmethod
    def _add_stuffing(packet: list[int]) -> list[int]:
        """
        Byte stuffing is a method of adding additional data to generated instruction packets to ensure that
        the packets are processed successfully. When the byte pattern "0xFF 0xFF 0xFD" appears in a packet,
        byte stuffing adds 0xFD to the end of the pattern to convert it to “0xFF 0xFF 0xFD 0xFD” to ensure
        that it is not interpreted as the header at the start of another packet.

        Source: https://emanual.robotis.com/docs/en/dxl/protocol2/#transmission-process

        Args:
            packet (list[int]): The raw packet without stuffing.

        Returns:
            list[int]: The packet stuffed if it contained a "0xFF 0xFF 0xFD" byte sequence in its data bytes.
        """
        packet_length_in = dxl.DXL_MAKEWORD(packet[dxl.PKT_LENGTH_L], packet[dxl.PKT_LENGTH_H])
        packet_length_out = packet_length_in

        temp = [0] * dxl.TXPACKET_MAX_LEN

        # FF FF FD XX ID LEN_L LEN_H
        temp[dxl.PKT_HEADER0 : dxl.PKT_HEADER0 + dxl.PKT_LENGTH_H + 1] = packet[
            dxl.PKT_HEADER0 : dxl.PKT_HEADER0 + dxl.PKT_LENGTH_H + 1
        ]

        index = dxl.PKT_INSTRUCTION

        for i in range(0, packet_length_in - 2):  # except CRC
            temp[index] = packet[i + dxl.PKT_INSTRUCTION]
            index = index + 1
            if (
                packet[i + dxl.PKT_INSTRUCTION] == 0xFD
                and packet[i + dxl.PKT_INSTRUCTION - 1] == 0xFF
                and packet[i + dxl.PKT_INSTRUCTION - 2] == 0xFF
            ):
                # FF FF FD
                temp[index] = 0xFD
                index = index + 1
                packet_length_out = packet_length_out + 1

        temp[index] = packet[dxl.PKT_INSTRUCTION + packet_length_in - 2]
        temp[index + 1] = packet[dxl.PKT_INSTRUCTION + packet_length_in - 1]
        index = index + 2

        if packet_length_in != packet_length_out:
            packet = [0] * index

        packet[0:index] = temp[0:index]

        packet[dxl.PKT_LENGTH_L] = dxl.DXL_LOBYTE(packet_length_out)
        packet[dxl.PKT_LENGTH_H] = dxl.DXL_HIBYTE(packet_length_out)

        return packet

    @staticmethod
    def _add_crc(packet: list[int]) -> list[int]:
        """Computes and add CRC to the packet.

        https://emanual.robotis.com/docs/en/dxl/crc/
        https://en.wikipedia.org/wiki/Cyclic_redundancy_check

        Args:
            packet (list[int]): The raw packet without CRC (but with placeholders for it).

        Returns:
            list[int]: The raw packet with a valid CRC.
        """
        crc = 0
        for j in range(len(packet) - 2):
            i = ((crc >> 8) ^ packet[j]) & 0xFF
            crc = ((crc << 8) ^ DXL_CRC_TABLE[i]) & 0xFFFF

        packet[-2] = dxl.DXL_LOBYTE(crc)
        packet[-1] = dxl.DXL_HIBYTE(crc)

        return packet


class MockInstructionPacket(MockDynamixelPacketv2):
    """
    Helper class to build valid Dynamixel Protocol 2.0 Instruction Packets.

    Protocol 2.0 Instruction Packet structure
    https://emanual.robotis.com/docs/en/dxl/protocol2/#instruction-packet

    | Header              | Packet ID | Length      | Instruction | Params            | CRC         |
    | ------------------- | --------- | ----------- | ----------- | ----------------- | ----------- |
    | 0xFF 0xFF 0xFD 0x00 | ID        | Len_L Len_H | Instr       | Param 1 … Param N | CRC_L CRC_H |

    """

    @classmethod
    def _build(cls, dxl_id: int, params: list[int], length: int, instruction: int) -> list[int]:
        length = len(params) + 3
        return [
            0xFF, 0xFF, 0xFD, 0x00,  # header
            dxl_id,                  # servo id
            dxl.DXL_LOBYTE(length),  # length_l
            dxl.DXL_HIBYTE(length),  # length_h
            instruction,             # instruction type
            *params,                 # data bytes
            0x00, 0x00               # placeholder for CRC
        ]  # fmt: skip

    @classmethod
    def ping(
        cls,
        dxl_id: int,
    ) -> bytes:
        """
        Builds a "Ping" broadcast instruction.
        https://emanual.robotis.com/docs/en/dxl/protocol2/#ping-0x01

        No parameters required.
        """
        return cls.build(dxl_id=dxl_id, params=[], length=3, instruction=dxl.INST_PING)

    @classmethod
    def read(
        cls,
        dxl_id: int,
        start_address: int,
        data_length: int,
    ) -> bytes:
        """
        Builds a "Read" instruction.
        https://emanual.robotis.com/docs/en/dxl/protocol2/#read-0x02

        The parameters for Read (Protocol 2.0) are:
            param[0]   = start_address L
            param[1]   = start_address H
            param[2]   = data_length L
            param[3]   = data_length H

        And 'length' = data_length + 5, where:
            +1 is for instruction byte,
            +2 is for the length bytes,
            +2 is for the CRC at the end.
        """
        params = [
            dxl.DXL_LOBYTE(start_address),
            dxl.DXL_HIBYTE(start_address),
            dxl.DXL_LOBYTE(data_length),
            dxl.DXL_HIBYTE(data_length),
        ]
        length = len(params) + 3
        # length = data_length + 5
        return cls.build(dxl_id=dxl_id, params=params, length=length, instruction=dxl.INST_READ)

    @classmethod
    def write(
        cls,
        dxl_id: int,
        value: int,
        start_address: int,
        data_length: int,
    ) -> bytes:
        """
        Builds a "Write" instruction.
        https://emanual.robotis.com/docs/en/dxl/protocol2/#write-0x03

        The parameters for Write (Protocol 2.0) are:
            param[0]   = start_address L
            param[1]   = start_address H
            param[2]   = 1st Byte
            param[3]   = 2nd Byte
            ...
            param[1+X] = X-th Byte

        And 'length' = data_length + 5, where:
            +1 is for instruction byte,
            +2 is for the length bytes,
            +2 is for the CRC at the end.
        """
        data = _split_into_byte_chunks(value, data_length)
        params = [
            dxl.DXL_LOBYTE(start_address),
            dxl.DXL_HIBYTE(start_address),
            *data,
        ]
        length = data_length + 5
        return cls.build(dxl_id=dxl_id, params=params, length=length, instruction=dxl.INST_WRITE)

    @classmethod
    def sync_read(
        cls,
        dxl_ids: list[int],
        start_address: int,
        data_length: int,
    ) -> bytes:
        """
        Builds a "Sync_Read" broadcast instruction.
        https://emanual.robotis.com/docs/en/dxl/protocol2/#sync-read-0x82

        The parameters for Sync_Read (Protocol 2.0) are:
            param[0]   = start_address L
            param[1]   = start_address H
            param[2]   = data_length L
            param[3]   = data_length H
            param[4+]  = motor IDs to read from

        And 'length' = (number_of_params + 7), where:
            +1 is for instruction byte,
            +2 is for the address bytes,
            +2 is for the length bytes,
            +2 is for the CRC at the end.
        """
        params = [
            dxl.DXL_LOBYTE(start_address),
            dxl.DXL_HIBYTE(start_address),
            dxl.DXL_LOBYTE(data_length),
            dxl.DXL_HIBYTE(data_length),
            *dxl_ids,
        ]
        length = len(dxl_ids) + 7
        return cls.build(
            dxl_id=dxl.BROADCAST_ID, params=params, length=length, instruction=dxl.INST_SYNC_READ
        )

    @classmethod
    def sync_write(
        cls,
        ids_values: dict[int, int],
        start_address: int,
        data_length: int,
    ) -> bytes:
        """
        Builds a "Sync_Write" broadcast instruction.
        https://emanual.robotis.com/docs/en/dxl/protocol2/#sync-write-0x83

        The parameters for Sync_Write (Protocol 2.0) are:
            param[0]   = start_address L
            param[1]   = start_address H
            param[2]   = data_length L
            param[3]   = data_length H
            param[5]   = [1st motor] ID
            param[5+1] = [1st motor] 1st Byte
            param[5+2] = [1st motor] 2nd Byte
            ...
            param[5+X] = [1st motor] X-th Byte
            param[6]   = [2nd motor] ID
            param[6+1] = [2nd motor] 1st Byte
            param[6+2] = [2nd motor] 2nd Byte
            ...
            param[6+X] = [2nd motor] X-th Byte

        And 'length' = ((number_of_params * 1 + data_length) + 7), where:
            +1 is for instruction byte,
            +2 is for the address bytes,
            +2 is for the length bytes,
            +2 is for the CRC at the end.
        """
        data = []
        for id_, value in ids_values.items():
            split_value = _split_into_byte_chunks(value, data_length)
            data += [id_, *split_value]
        params = [
            dxl.DXL_LOBYTE(start_address),
            dxl.DXL_HIBYTE(start_address),
            dxl.DXL_LOBYTE(data_length),
            dxl.DXL_HIBYTE(data_length),
            *data,
        ]
        length = len(ids_values) * (1 + data_length) + 7
        return cls.build(
            dxl_id=dxl.BROADCAST_ID, params=params, length=length, instruction=dxl.INST_SYNC_WRITE
        )


class MockStatusPacket(MockDynamixelPacketv2):
    """
    Helper class to build valid Dynamixel Protocol 2.0 Status Packets.

    Protocol 2.0 Status Packet structure
    https://emanual.robotis.com/docs/en/dxl/protocol2/#status-packet

    | Header              | Packet ID | Length      | Instruction | Error | Params            | CRC         |
    | ------------------- | --------- | ----------- | ----------- | ----- | ----------------- | ----------- |
    | 0xFF 0xFF 0xFD 0x00 | ID        | Len_L Len_H | 0x55        | Err   | Param 1 … Param N | CRC_L CRC_H |
    """

    @classmethod
    def _build(cls, dxl_id: int, params: list[int], length: int, error: int = 0) -> list[int]:
        return [
            0xFF, 0xFF, 0xFD, 0x00,  # header
            dxl_id,                  # servo id
            dxl.DXL_LOBYTE(length),  # length_l
            dxl.DXL_HIBYTE(length),  # length_h
            0x55,                    # instruction = 'status'
            error,                   # error
            *params,                 # data bytes
            0x00, 0x00               # placeholder for CRC
        ]  # fmt: skip

    @classmethod
    def ping(cls, dxl_id: int, model_nb: int = 1190, firm_ver: int = 50, error: int = 0) -> bytes:
        """
        Builds a 'Ping' status packet.
        https://emanual.robotis.com/docs/en/dxl/protocol2/#ping-0x01

        Args:
            dxl_id (int): ID of the servo responding.
            model_nb (int, optional): Desired 'model number' to be returned in the packet. Defaults to 1190
                which corresponds to a XL330-M077-T.
            firm_ver (int, optional): Desired 'firmware version' to be returned in the packet.
                Defaults to 50.

        Returns:
            bytes: The raw 'Ping' status packet ready to be sent through serial.
        """
        params = [dxl.DXL_LOBYTE(model_nb), dxl.DXL_HIBYTE(model_nb), firm_ver]
        length = 7
        return cls.build(dxl_id, params=params, length=length, error=error)

    @classmethod
    def read(cls, dxl_id: int, value: int, param_length: int, error: int = 0) -> bytes:
        """
        Builds a 'Read' status packet (also works for 'Sync Read')
        https://emanual.robotis.com/docs/en/dxl/protocol2/#read-0x02
        https://emanual.robotis.com/docs/en/dxl/protocol2/#sync-read-0x82

        Args:
            dxl_id (int): ID of the servo responding.
            value (int): Desired value to be returned in the packet.
            param_length (int): The address length as reported in the control table.

        Returns:
            bytes: The raw 'Present_Position' status packet ready to be sent through serial.
        """
        params = _split_into_byte_chunks(value, param_length)
        length = param_length + 4
        return cls.build(dxl_id, params=params, length=length, error=error)


class MockPortHandler(dxl.PortHandler):
    """
    This class overwrite the 'setupPort' method of the Dynamixel PortHandler because it can specify
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
        self, ids_models: dict[int, list[int]] | None = None, num_invalid_try: int = 0
    ) -> str:
        ping_request = MockInstructionPacket.ping(dxl.BROADCAST_ID)
        return_packets = b"".join(MockStatusPacket.ping(id_, model) for id_, model in ids_models.items())
        ping_response = self._build_send_fn(return_packets, num_invalid_try)

        stub_name = "Ping_" + "_".join([str(id_) for id_ in ids_models])
        self.stub(
            name=stub_name,
            receive_bytes=ping_request,
            send_fn=ping_response,
        )
        return stub_name

    def build_ping_stub(
        self, dxl_id: int, model_nb: int, firm_ver: int = 50, num_invalid_try: int = 0, error: int = 0
    ) -> str:
        ping_request = MockInstructionPacket.ping(dxl_id)
        return_packet = MockStatusPacket.ping(dxl_id, model_nb, firm_ver, error)
        ping_response = self._build_send_fn(return_packet, num_invalid_try)
        stub_name = f"Ping_{dxl_id}"
        self.stub(
            name=stub_name,
            receive_bytes=ping_request,
            send_fn=ping_response,
        )
        return stub_name

    def build_read_stub(
        self,
        address: int,
        length: int,
        dxl_id: int,
        value: int,
        reply: bool = True,
        error: int = 0,
        num_invalid_try: int = 0,
    ) -> str:
        read_request = MockInstructionPacket.read(dxl_id, address, length)
        return_packet = MockStatusPacket.read(dxl_id, value, length, error) if reply else b""
        read_response = self._build_send_fn(return_packet, num_invalid_try)
        stub_name = f"Read_{address}_{length}_{dxl_id}_{value}_{error}"
        self.stub(
            name=stub_name,
            receive_bytes=read_request,
            send_fn=read_response,
        )
        return stub_name

    def build_write_stub(
        self,
        address: int,
        length: int,
        dxl_id: int,
        value: int,
        reply: bool = True,
        error: int = 0,
        num_invalid_try: int = 0,
    ) -> str:
        sync_read_request = MockInstructionPacket.write(dxl_id, value, address, length)
        return_packet = MockStatusPacket.build(dxl_id, params=[], length=4, error=error) if reply else b""
        stub_name = f"Write_{address}_{length}_{dxl_id}"
        self.stub(
            name=stub_name,
            receive_bytes=sync_read_request,
            send_fn=self._build_send_fn(return_packet, num_invalid_try),
        )
        return stub_name

    def build_sync_read_stub(
        self,
        address: int,
        length: int,
        ids_values: dict[int, int],
        reply: bool = True,
        num_invalid_try: int = 0,
    ) -> str:
        sync_read_request = MockInstructionPacket.sync_read(list(ids_values), address, length)
        return_packets = (
            b"".join(MockStatusPacket.read(id_, pos, length) for id_, pos in ids_values.items())
            if reply
            else b""
        )
        sync_read_response = self._build_send_fn(return_packets, num_invalid_try)
        stub_name = f"Sync_Read_{address}_{length}_" + "_".join([str(id_) for id_ in ids_values])
        self.stub(
            name=stub_name,
            receive_bytes=sync_read_request,
            send_fn=sync_read_response,
        )
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
        stub_name = f"Seq_Sync_Read_{address}_{length}_" + "_".join([str(id_) for id_ in ids_values])
        self.stub(
            name=stub_name,
            receive_bytes=sync_read_request,
            send_fn=sync_read_response,
        )
        return stub_name

    def build_sync_write_stub(
        self, address: int, length: int, ids_values: dict[int, int], num_invalid_try: int = 0
    ) -> str:
        sync_read_request = MockInstructionPacket.sync_write(ids_values, address, length)
        stub_name = f"Sync_Write_{address}_{length}_" + "_".join([str(id_) for id_ in ids_values])
        self.stub(
            name=stub_name,
            receive_bytes=sync_read_request,
            send_fn=self._build_send_fn(b"", num_invalid_try),
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
