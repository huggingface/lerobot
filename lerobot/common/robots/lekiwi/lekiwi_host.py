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

import base64
import json
import logging
import time

import cv2
import zmq

from lerobot.common.constants import OBS_IMAGES

from .config_lekiwi import LeKiwiConfig
from .lekiwi import LeKiwi


class HostAgent:
    def __init__(self, port_zmq_cmd, port_zmq_observations):
        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_cmd_socket.bind(f"tcp://*:{port_zmq_cmd}")

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_observation_socket.bind(f"tcp://*:{port_zmq_observations}")

    def disconnect(self):
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()


def main():
    logging.info("Configuring LeKiwi")
    robot_config = LeKiwiConfig()
    robot = LeKiwi(robot_config)

    logging.info("Connecting LeKiwi")
    robot.connect()

    logging.info("Starting HostAgent")
    remote_agent = HostAgent(robot_config.port_zmq_cmd, robot_config.port_zmq_observations)

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
                data = dict(json.loads(msg))
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

            # Encode ndarrays to base64 strings
            for cam_key, _ in robot.cameras.items():
                ret, buffer = cv2.imencode(
                    ".jpg", last_observation[OBS_IMAGES][cam_key], [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                )
                if ret:
                    last_observation[OBS_IMAGES][cam_key] = base64.b64encode(buffer).decode("utf-8")
                else:
                    last_observation[OBS_IMAGES][cam_key] = ""

            # Send the observation to the remote agent
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

    logging.info("Finished LeKiwi cleanly")


if __name__ == "__main__":
    main()
