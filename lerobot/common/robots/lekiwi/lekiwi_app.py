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

import json
import logging
import time

import numpy as np
import zmq

from lerobot.common.constants import OBS_STATE
from lerobot.common.robots.lekiwi.configuration_lekiwi import LeKiwiRobotConfig
from lerobot.common.robots.lekiwi.lekiwi_robot import LeKiwiRobot

# Network Configuration
PORT_ZMQ_CMD: int = 5555
PORT_ZMQ_OBSERVATIONS: int = 5556


class RemoteAgent:
    def __init__(self):
        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_cmd_socket.bind(f"tcp://*:{PORT_ZMQ_CMD}")

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_observation_socket.bind(f"tcp://*:{PORT_ZMQ_OBSERVATIONS}")

    def disconnect(self):
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()


def main():
    logging.info("Configuring LeKiwiRobot")
    robot_config = LeKiwiRobotConfig()
    robot = LeKiwiRobot(robot_config)

    logging.info("Connecting LeKiwiRobot")
    robot.connect()

    logging.info("Starting RemoteAgent")
    remote_agent = RemoteAgent()

    last_cmd_time = time.time()
    logging.info("Waiting for commands...")
    try:
        # Business logic
        start = time.perf_counter()
        duration = 0
        while duration < 100:
            loop_start_time = time.time()
            try:
                msg = remote_agent.zmq_cmd_socket.recv_string(zmq.NOBLOCK)
                data = np.array(json.loads(msg))
                _action_sent = robot.send_action(data)
                last_cmd_time = time.time()
            except zmq.Again:
                logging.warning("No command available")
            except Exception as e:
                logging.error("Message fetching failed: %s", e)

            # TODO(Steven): Check this value
            # Watchdog: stop the robot if no command is received for over 0.5 seconds.
            now = time.time()
            if now - last_cmd_time > 0.5:
                robot.stop_base()

            last_observation = robot.get_observation()
            last_observation[OBS_STATE] = last_observation[OBS_STATE].tolist()
            remote_agent.zmq_observation_socket.send_string(json.dumps(last_observation))

            # Ensure a short sleep to avoid overloading the CPU.
            elapsed = time.time() - loop_start_time

            # TODO(Steven): Check this value
            time.sleep(
                max(0.033 - elapsed, 0)
            )  # If robot jitters increase the sleep and monitor cpu load with `top` in cmd
            duration = time.perf_counter() - start

    except KeyboardInterrupt:
        print("Shutting down LeKiwi server.")
    finally:
        robot.disconnect()
        remote_agent.disconnect()

    logging.info("Finished LeKiwiRobot cleanly")


if __name__ == "__main__":
    main()
