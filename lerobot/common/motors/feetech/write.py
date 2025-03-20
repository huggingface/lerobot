import time
from pprint import pprint

import scservo_sdk as scs

from lerobot.common.motors.feetech.feetech import SCS_SERIES_CONTROL_TABLE, FeetechMotorsBus


def read(port_handler, packet_handler, data_name):
    data_list, comm = packet_handler.ping(port_handler)
    # data_list, comm = packet_handler.broadcastPing(port_handler)
    pprint(data_list)
    address, length = SCS_SERIES_CONTROL_TABLE[data_name]
    reader = scs.GroupSyncRead(port_handler, packet_handler, address, length)
    for idx in data_list:
        reader.addParam(idx)

    reader.txRxPacket()

    values = {}
    for idx in data_list:
        values[idx] = reader.getData(idx, 9, 1)

    pprint(values)


def write(port_handler, packet_handler, data_name, value):
    data_list, comm = packet_handler.ping(port_handler)
    # data_list, comm = packet_handler.broadcastPing(port_handler)
    pprint(data_list)
    address, length = SCS_SERIES_CONTROL_TABLE[data_name]
    writer = scs.GroupSyncWrite(port_handler, packet_handler, address, length)
    data = FeetechMotorsBus.split_int_bytes(value, length)
    for idx in data_list:
        writer.addParam(idx, data)

    writer.txPacket()


def loop(port_handler, packet_handler):
    data_list, comm = packet_handler.ping(port_handler)
    # data_list, comm = packet_handler.broadcastPing(port_handler)
    # data_list = [1]
    address, length = SCS_SERIES_CONTROL_TABLE["Present_Position"]
    reader = scs.GroupSyncRead(port_handler, packet_handler, address, length)

    for idx in data_list:
        reader.addParam(idx)

    try:
        while True:
            start = time.perf_counter()
            reader.txRxPacket()
            values = {}
            for idx in data_list:
                values[idx] = reader.getData(idx, address, length)
            read_s = time.perf_counter() - start
            # print("--------------------------------------")
            print(f"\nread_s: {read_s * 1e3:.2f}ms ({1 / read_s:.0f} Hz)")
            pprint(values)
    except KeyboardInterrupt:
        pass
    finally:
        port_handler.closePort()


def main():
    port_handler = scs.PortHandler("/dev/tty.usbmodem58760430541")
    packet_handler = scs.PacketHandler(0)
    port_handler.openPort()

    model_number, comm, _ = packet_handler.ping(port_handler, scs.BROADCAST_ID)
    print(model_number)

    # read(port_handler, packet_handler, "Return_Delay_Time")
    # write(port_handler, packet_handler, "Return_Delay_Time", 2)
    # read(port_handler, packet_handler, "Return_Delay_Time")

    # loop(port_handler, packet_handler)

    if port_handler.is_open:
        port_handler.closePort()


if __name__ == "__main__":
    main()
