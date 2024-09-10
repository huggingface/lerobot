from dynamixel_sdk import COMM_SUCCESS


class PortHandler:
    def __init__(self, port):
        self.port = port

    def openPort(self):  # noqa: N802
        return True

    def closePort(self):  # noqa: N802
        pass

    def setPacketTimeoutMillis(self, timeout_ms):  # noqa: N802
        del timeout_ms  # unused


class PacketHandler:
    def __init__(self, protocol_version):
        del protocol_version  # unused


class GroupSyncRead:
    def __init__(self, port_handler, packet_handler, address, bytes):
        pass

    def addParam(self, motor_index):  # noqa: N802
        pass

    def txRxPacket(self):  # noqa: N802
        return COMM_SUCCESS

    def getData(self, index, address, bytes):  # noqa: N802
        return value  # noqa: F821


class GroupSyncWrite:
    def __init__(self, port_handler, packet_handler, address, bytes):
        pass

    def addParam(self, index, data):  # noqa: N802
        pass

    def txPacket(self):  # noqa: N802
        return COMM_SUCCESS

    def changeParam(self, index, data):  # noqa: N802
        pass
