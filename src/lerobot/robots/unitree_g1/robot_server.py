#!/usr/bin/env python3
import time
import pickle
import threading

import zmq

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as hg_LowCmd, LowState_ as hg_LowState
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

kTopicLowCommand_Debug = "rt/lowcmd"
kTopicLowCommand_Motion = "rt/arm_sdk"
kTopicLowState = "rt/lowstate"

LOWCMD_PORT = 6000      # laptop -> robot
LOWSTATE_PORT = 6001    # robot -> laptop


def state_forward_loop(lowstate_sub, lowstate_sock, state_period: float):
    """
    read lowstate from dds and push to laptop at ~state_period.
    runs in its own thread.
    """
    last_state_time = 0.0

    while True:
        # read from dds (blocking)
        msg = lowstate_sub.Read()
        if msg is None:
            continue

        now = time.time()
        # optional downsampling (if robot dds rate > state_period)
        if now - last_state_time >= state_period:
            payload = pickle.dumps((kTopicLowState, msg), protocol=pickle.HIGHEST_PROTOCOL)
            try:
                lowstate_sock.send(payload, zmq.NOBLOCK)
            except zmq.Again:
                # if no subscribers / tx buffer full, just drop
                pass
            last_state_time = now


def cmd_forward_loop(lowcmd_sock, lowcmd_pub_debug, lowcmd_pub_motion, crc: CRC):
    """
    read lowcmd from laptop (zmq) and push to dds.
    runs in its own thread.
    """
    while True:
        # blocking wait for commands from laptop
        payload = lowcmd_sock.recv()
        topic, cmd = pickle.loads(payload)  # cmd is hg_LowCmd

        # recompute crc just in case
        cmd.crc = crc.Crc(cmd)

        if topic == kTopicLowCommand_Debug:
            lowcmd_pub_debug.Write(cmd)
        elif topic == kTopicLowCommand_Motion:
            lowcmd_pub_motion.Write(cmd)
        else:
            # ignore unknown topics
            pass


def main():
    # dds init
    ChannelFactoryInitialize(0)

    # acquire motion mode on the robot
    msc = MotionSwitcherClient()
    msc.SetTimeout(5.0)
    msc.Init()

    status, result = msc.CheckMode()
    while result is not None and "name" in result and result["name"]:
        msc.ReleaseMode()
        status, result = msc.CheckMode()
        time.sleep(1.0)

    crc = CRC()

    # dds publishers / subscriber
    lowcmd_pub_debug = ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
    lowcmd_pub_motion = ChannelPublisher(kTopicLowCommand_Motion, hg_LowCmd)
    lowcmd_pub_debug.Init()
    lowcmd_pub_motion.Init()

    lowstate_sub = ChannelSubscriber(kTopicLowState, hg_LowState)
    lowstate_sub.Init()

    # zmq setup
    ctx = zmq.Context.instance()

    # commands from laptop
    lowcmd_sock = ctx.socket(zmq.PULL)
    lowcmd_sock.bind(f"tcp://0.0.0.0:{LOWCMD_PORT}")

    # state to laptop
    lowstate_sock = ctx.socket(zmq.PUB)
    lowstate_sock.bind(f"tcp://0.0.0.0:{LOWSTATE_PORT}")

    state_period = 0.002  # ~500 hz

    # start threads
    t_state = threading.Thread(
        target=state_forward_loop,
        args=(lowstate_sub, lowstate_sock, state_period),
        daemon=True,
    )
    t_cmd = threading.Thread(
        target=cmd_forward_loop,
        args=(lowcmd_sock, lowcmd_pub_debug, lowcmd_pub_motion, crc),
        daemon=True,
    )

    t_state.start()
    t_cmd.start()

    print("bridge running (lowstate -> zmq, lowcmd -> dds)")

    # keep main thread alive so daemon threads don’t exit
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("shutting down bridge...")
        # sockets/context will be cleaned up on process exit


if __name__ == "__main__":
    main()
