#!/usr/bin/env python3
import contextlib
import pickle
import threading
import time

import zmq
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as hg_LowCmd, LowState_ as hg_LowState
from unitree_sdk2py.utils.crc import CRC

# DDS topic names follow Unitree SDK naming conventions
# ruff: noqa: N816
kTopicLowCommand_Debug = "rt/lowcmd"  # action to robot
kTopicLowState = "rt/lowstate"  # observation from robot

LOWCMD_PORT = 6000
LOWSTATE_PORT = 6001


def state_forward_loop(
    lowstate_sub, lowstate_sock, state_period: float
):  # read observation from DDS and send to server
    last_state_time = 0.0

    while True:
        # read from DDS
        msg = lowstate_sub.Read()
        if msg is None:
            continue

        now = time.time()
        # optional downsampling (if robot dds rate > state_period)
        if now - last_state_time >= state_period:
            payload = pickle.dumps((kTopicLowState, msg), protocol=pickle.HIGHEST_PROTOCOL)
            # if no subscribers / tx buffer full, just drop
            with contextlib.suppress(zmq.Again):
                lowstate_sock.send(payload, zmq.NOBLOCK)
            last_state_time = now


def cmd_forward_loop(lowcmd_sock, lowcmd_pub_debug, crc: CRC):  # send action to robot
    while True:
        payload = lowcmd_sock.recv()
        topic, cmd = pickle.loads(payload)

        # recompute crc just in case
        cmd.crc = crc.Crc(cmd)

        if topic == kTopicLowCommand_Debug:
            lowcmd_pub_debug.Write(cmd)
        else:
            pass


def main():
    # initialize DDS
    ChannelFactoryInitialize(0)

    # stop all active publishers on the robot
    msc = MotionSwitcherClient()
    msc.SetTimeout(5.0)
    msc.Init()

    status, result = msc.CheckMode()
    while result is not None and "name" in result and result["name"]:
        msc.ReleaseMode()
        status, result = msc.CheckMode()
        time.sleep(1.0)

    crc = CRC()

    # initialize DDS publisher
    lowcmd_pub_debug = ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
    lowcmd_pub_debug.Init()

    # initialize DDS subscriber
    lowstate_sub = ChannelSubscriber(kTopicLowState, hg_LowState)
    lowstate_sub.Init()

    # initialize ZMQ
    ctx = zmq.Context.instance()

    # send action to robot
    lowcmd_sock = ctx.socket(zmq.PULL)
    lowcmd_sock.bind(f"tcp://0.0.0.0:{LOWCMD_PORT}")

    # send observation to server
    lowstate_sock = ctx.socket(zmq.PUB)
    lowstate_sock.bind(f"tcp://0.0.0.0:{LOWSTATE_PORT}")

    state_period = 0.002  # ~500 hz

    # start observation forwarding thread
    t_state = threading.Thread(
        target=state_forward_loop,
        args=(lowstate_sub, lowstate_sock, state_period),
        daemon=True,
    )
    t_state.start()

    # start action forwarding thread
    t_cmd = threading.Thread(
        target=cmd_forward_loop,
        args=(lowcmd_sock, lowcmd_pub_debug, crc),
        daemon=True,
    )
    t_cmd.start()

    print("bridge running (lowstate -> zmq, lowcmd -> dds)")
    # keep main thread alive so daemon threads don’t exit
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("shutting down bridge...")


if __name__ == "__main__":
    main()
