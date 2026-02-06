#!/usr/bin/env python3

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
DDS-to-ZMQ bridge server for Unitree G1 robot.

This server runs on the robot and forwards:
- Robot state (LowState) from DDS to ZMQ (for remote clients)
- Robot commands (LowCmd) from ZMQ to DDS (from remote clients)

Uses JSON for secure serialization instead of pickle.
"""

import base64
import contextlib
import json
import threading
import time
from typing import Any

import zmq
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as hg_LowCmd, LowState_ as hg_LowState
from unitree_sdk2py.utils.crc import CRC

# DDS topic names follow Unitree SDK naming conventions
# ruff: noqa: N816
kTopicLowCommand_Debug = "rt/lowcmd"  # action to robot
kTopicLowState = "rt/lowstate"  # observation from robot

LOWCMD_PORT = 6000
LOWSTATE_PORT = 6001
NUM_MOTORS = 35


def lowstate_to_dict(msg: hg_LowState) -> dict[str, Any]:
    """Convert LowState SDK message to a JSON-serializable dictionary."""
    motor_states = []
    for i in range(NUM_MOTORS):
        temp = msg.motor_state[i].temperature
        avg_temp = float(sum(temp) / len(temp)) if isinstance(temp, list) else float(temp)
        motor_states.append(
            {
                "q": float(msg.motor_state[i].q),
                "dq": float(msg.motor_state[i].dq),
                "tau_est": float(msg.motor_state[i].tau_est),
                "temperature": avg_temp,
            }
        )

    return {
        "motor_state": motor_states,
        "imu_state": {
            "quaternion": [float(x) for x in msg.imu_state.quaternion],
            "gyroscope": [float(x) for x in msg.imu_state.gyroscope],
            "accelerometer": [float(x) for x in msg.imu_state.accelerometer],
            "rpy": [float(x) for x in msg.imu_state.rpy],
            "temperature": float(msg.imu_state.temperature),
        },
        # Encode bytes as base64 for JSON compatibility
        "wireless_remote": base64.b64encode(bytes(msg.wireless_remote)).decode("ascii"),
        "mode_machine": int(msg.mode_machine),
    }


def dict_to_lowcmd(data: dict[str, Any]) -> hg_LowCmd:
    """Convert dictionary back to LowCmd SDK message."""
    cmd = unitree_hg_msg_dds__LowCmd_()
    cmd.mode_pr = data.get("mode_pr", 0)
    cmd.mode_machine = data.get("mode_machine", 0)

    for i, motor_data in enumerate(data.get("motor_cmd", [])):
        cmd.motor_cmd[i].mode = motor_data.get("mode", 0)
        cmd.motor_cmd[i].q = motor_data.get("q", 0.0)
        cmd.motor_cmd[i].dq = motor_data.get("dq", 0.0)
        cmd.motor_cmd[i].kp = motor_data.get("kp", 0.0)
        cmd.motor_cmd[i].kd = motor_data.get("kd", 0.0)
        cmd.motor_cmd[i].tau = motor_data.get("tau", 0.0)

    return cmd


def state_forward_loop(
    lowstate_sub: ChannelSubscriber,
    lowstate_sock: zmq.Socket,
    state_period: float,
    shutdown_event: threading.Event,
) -> None:
    """Read observation from DDS and forward to ZMQ clients."""
    last_state_time = 0.0

    while not shutdown_event.is_set():
        # read from DDS
        msg = lowstate_sub.Read()
        if msg is None:
            continue

        now = time.time()
        # optional downsampling (if robot dds rate > state_period)
        if now - last_state_time >= state_period:
            # Convert to dict and serialize with JSON
            state_dict = lowstate_to_dict(msg)
            payload = json.dumps({"topic": kTopicLowState, "data": state_dict}).encode("utf-8")
            # if no subscribers / tx buffer full, just drop
            with contextlib.suppress(zmq.Again):
                lowstate_sock.send(payload, zmq.NOBLOCK)
            last_state_time = now


def cmd_forward_loop(
    lowcmd_sock: zmq.Socket,
    lowcmd_pub_debug: ChannelPublisher,
    crc: CRC,
) -> None:
    """Receive commands from ZMQ and forward to DDS."""
    while True:
        try:
            payload = lowcmd_sock.recv()
        except zmq.ContextTerminated:
            break
        msg_dict = json.loads(payload.decode("utf-8"))

        topic = msg_dict.get("topic", "")
        cmd_data = msg_dict.get("data", {})

        # Reconstruct LowCmd object from dict
        cmd = dict_to_lowcmd(cmd_data)

        # recompute crc
        cmd.crc = crc.Crc(cmd)

        if topic == kTopicLowCommand_Debug:
            lowcmd_pub_debug.Write(cmd)


def main() -> None:
    """Main entry point for the robot server bridge."""
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

    # receive commands from remote client
    lowcmd_sock = ctx.socket(zmq.PULL)
    lowcmd_sock.bind(f"tcp://0.0.0.0:{LOWCMD_PORT}")

    # publish state to remote clients
    lowstate_sock = ctx.socket(zmq.PUB)
    lowstate_sock.bind(f"tcp://0.0.0.0:{LOWSTATE_PORT}")

    state_period = 0.002  # ~500 hz
    shutdown_event = threading.Event()

    # start observation forwarding in background thread
    t_state = threading.Thread(
        target=state_forward_loop,
        args=(lowstate_sub, lowstate_sock, state_period, shutdown_event),
    )
    t_state.start()

    print("bridge running (lowstate -> zmq, lowcmd -> dds)")

    # run command forwarding in main thread
    try:
        cmd_forward_loop(lowcmd_sock, lowcmd_pub_debug, crc)
    except KeyboardInterrupt:
        print("shutting down bridge...")
    finally:
        shutdown_event.set()
        ctx.term()  # terminates blocking zmq.recv() calls
        t_state.join(timeout=2.0)


if __name__ == "__main__":
    main()
