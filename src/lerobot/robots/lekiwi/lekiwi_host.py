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
from dataclasses import dataclass, field

import cv2
import draccus
import zmq

from lerobot.utils.lifecycle import Cleanup

from .config_lekiwi import LeKiwiConfig, LeKiwiHostConfig
from .lekiwi import LeKiwi


@dataclass
class LeKiwiServerConfig:
    """Configuration for the LeKiwi host script."""

    robot: LeKiwiConfig = field(default_factory=LeKiwiConfig)
    host: LeKiwiHostConfig = field(default_factory=LeKiwiHostConfig)


class LeKiwiHost:
    def __init__(self, config: LeKiwiHostConfig):
        self.zmq_context = None
        self.zmq_cmd_socket = None
        self.zmq_observation_socket = None
        try:
            self.zmq_context = zmq.Context()
            self.zmq_cmd_socket = self.zmq_context.socket(zmq.PULL)
            self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)
            self.zmq_cmd_socket.bind(f"tcp://*:{config.port_zmq_cmd}")

            self.zmq_observation_socket = self.zmq_context.socket(zmq.PUSH)
            # CONFLATE does not support multipart messages; a 2-deep send queue keeps
            # near-latest-only semantics and sheds stale observations during stalls.
            self.zmq_observation_socket.setsockopt(zmq.SNDHWM, 2)
            self.zmq_observation_socket.bind(f"tcp://*:{config.port_zmq_observations}")
        except Exception as connect_error:
            connect_error.add_note("while creating the LeKiwi host sockets")
            try:
                self.disconnect()
            except Exception as disconnect_error:
                raise ExceptionGroup(
                    "Failed to create the LeKiwi host and to fully roll back",
                    [connect_error, disconnect_error],
                ) from None
            raise

        self.connection_time_s = config.connection_time_s
        self.watchdog_timeout_ms = config.watchdog_timeout_ms
        self.max_loop_freq_hz = config.max_loop_freq_hz

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.disconnect()

    def disconnect(self) -> None:
        with Cleanup("the LeKiwi host") as cleanup:
            for name in ("zmq_observation_socket", "zmq_cmd_socket"):
                socket = getattr(self, name, None)
                if socket is None:
                    continue
                with cleanup.step(f"the {name}"):
                    socket.close()
                    setattr(self, name, None)
            if self.zmq_context is not None:
                with cleanup.step("the ZMQ context"):
                    self.zmq_context.term()
                    self.zmq_context = None


@draccus.wrap()
def main(cfg: LeKiwiServerConfig):
    logging.info("Configuring LeKiwi")
    robot = LeKiwi(cfg.robot)

    try:
        logging.info("Connecting LeKiwi and starting HostAgent")
        with robot, LeKiwiHost(cfg.host) as host:
            last_cmd_time = time.time()
            watchdog_active = False
            logging.info("Waiting for commands...")

            # Business logic
            start = time.perf_counter()
            duration = 0
            while duration < host.connection_time_s:
                loop_start_time = time.time()
                try:
                    msg = host.zmq_cmd_socket.recv_string(zmq.NOBLOCK)
                    data = dict(json.loads(msg))
                    _action_sent = robot.send_action(data)
                    last_cmd_time = time.time()
                    watchdog_active = False
                except zmq.Again:
                    if not watchdog_active:
                        logging.warning("No command available")
                except Exception as e:
                    logging.error("Message fetching failed: %s", e)

                now = time.time()
                if (now - last_cmd_time > host.watchdog_timeout_ms / 1000) and not watchdog_active:
                    logging.warning(
                        f"Command not received for more than {host.watchdog_timeout_ms} milliseconds. Stopping the base."
                    )
                    watchdog_active = True
                    robot.stop_base()

                last_observation = robot.get_observation()

                # Send one multipart message: a JSON header frame (state + camera
                # order) followed by one raw JPEG frame per camera. Raw JPEG avoids
                # the 33% base64 inflation of embedding binary data in JSON.
                cam_keys = list(robot.cameras.keys())
                jpeg_frames = []
                for cam_key in cam_keys:
                    ret, jpeg = cv2.imencode(
                        ".jpg", last_observation.pop(cam_key), [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                    )
                    jpeg_frames.append(jpeg if ret else b"")
                header = {"_cams": cam_keys, **last_observation}

                # Send the observation to the remote agent
                try:
                    host.zmq_observation_socket.send_multipart(
                        [json.dumps(header).encode()] + jpeg_frames, flags=zmq.NOBLOCK
                    )
                except zmq.Again:
                    logging.info("Dropping observation, no client connected")

                # Ensure a short sleep to avoid overloading the CPU.
                elapsed = time.time() - loop_start_time

                time.sleep(max(1 / host.max_loop_freq_hz - elapsed, 0))
                duration = time.perf_counter() - start
            print("Cycle time reached.")

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")
    finally:
        print("Shutting down Lekiwi Host.")

    logging.info("Finished LeKiwi cleanly")


if __name__ == "__main__":
    main()
