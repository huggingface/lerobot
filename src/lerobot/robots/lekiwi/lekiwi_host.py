#!/usr/bin/env python

import argparse
import base64
import json
import logging
import time

import cv2
import zmq

from .config_lekiwi import LeKiwiConfig, LeKiwiHostConfig
from .lekiwi import LeKiwi


class LeKiwiHost:
    def __init__(self, config: LeKiwiHostConfig):
        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_cmd_socket.bind(f"tcp://*:{config.port_zmq_cmd}")

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_observation_socket.bind(f"tcp://*:{config.port_zmq_observations}")

        self.connection_time_s = config.connection_time_s
        self.watchdog_timeout_ms = config.watchdog_timeout_ms
        self.max_loop_freq_hz = config.max_loop_freq_hz

    def disconnect(self):
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()


def parse_args():
    p = argparse.ArgumentParser()
    # 这三个别名都能用：--id / --robot.id / --robot_id
    p.add_argument(
        "--id",
        "--robot.id",
        "--robot_id",
        dest="robot_id",
        type=str,
        help="Robot ID (used for calibration file lookup)",
    )
    p.add_argument(
        "--port",
        "--robot.port",
        dest="robot_port",
        type=str,
        help="Serial port for Feetech bus (e.g. /dev/so101_leader)",
    )
    p.add_argument("--cmd_port", type=int, default=5556, help="ZMQ command bind port")
    p.add_argument(
        "--state_port", "--obs_port", type=int, default=5555, help="ZMQ observation/state bind port"
    )
    p.add_argument("--watchdog_timeout_ms", type=int, default=500)
    p.add_argument("--max_loop_hz", type=float, default=100.0)
    p.add_argument(
        "--connection_time_s", type=float, default=3600.0, help="Run time limit; <=0 means run forever"
    )

    # ★ 新增：两类日志的节流间隔（秒）
    p.add_argument(
        "--no_cmd_log_interval_s", type=float, default=1.0, help="Rate-limit 'No command available' logs"
    )
    p.add_argument(
        "--drop_obs_log_interval_s", type=float, default=1.0, help="Rate-limit 'Dropping observation' logs"
    )
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args = parse_args()

    logging.info("Configuring LeKiwi")
    try:
        robot_config = LeKiwiConfig(
            id=args.robot_id,
            port=args.robot_port,
        )
    except TypeError:
        robot_config = LeKiwiConfig()
        if args.robot_id:
            robot_config.id = args.robot_id
        if args.robot_port:
            robot_config.port = args.robot_port

    robot = LeKiwi(robot_config)

    logging.info("Connecting LeKiwi")
    robot.connect()

    logging.info("Starting HostAgent")
    host_config = LeKiwiHostConfig(
        # 如果希望 CLI 覆盖 dataclass 默认，这里也可以把端口/时长等接过去
        connection_time_s=args.connection_time_s,
        watchdog_timeout_ms=args.watchdog_timeout_ms,
        max_loop_freq_hz=args.max_loop_hz,
        # 注意：LeKiwiHostConfig 的字段名和你的 CLI 名称可能不同步，保持一致即可
        # 若需要也可传端口：port_zmq_cmd=args.cmd_port, port_zmq_observations=args.state_port,
    )
    host = LeKiwiHost(host_config)

    last_cmd_time = time.time()
    watchdog_active = False

    # ★ 新增：首条命令标志 + 两类日志的“上次打印时间戳”
    seen_first_cmd = False
    last_no_cmd_warn_ts = 0.0
    last_drop_obs_warn_ts = 0.0

    logging.info("Waiting for commands...")
    try:
        start = time.perf_counter()
        duration = 0
        while duration < host.connection_time_s or host.connection_time_s <= 0:
            loop_start_time = time.time()
            try:
                msg = host.zmq_cmd_socket.recv_string(zmq.NOBLOCK)
                data = dict(json.loads(msg))
                _action_sent = robot.send_action(data)
                last_cmd_time = time.time()
                watchdog_active = False

                if not seen_first_cmd:
                    logging.info("First command received from client; link is active.")
                    seen_first_cmd = True

            except zmq.Again:
                nowt = time.time()
                # ★ 节流“无命令”日志
                if not watchdog_active and (nowt - last_no_cmd_warn_ts) > args.no_cmd_log_interval_s:
                    logging.warning("No command available")
                    last_no_cmd_warn_ts = nowt
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

            # Encode ndarrays to base64 strings
            for cam_key, _ in robot.cameras.items():
                ret, buffer = cv2.imencode(
                    ".jpg", last_observation[cam_key], [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                )
                if ret:
                    last_observation[cam_key] = base64.b64encode(buffer).decode("utf-8")
                else:
                    last_observation[cam_key] = ""

            # Send the observation to the remote agent
            try:
                host.zmq_observation_socket.send_string(json.dumps(last_observation), flags=zmq.NOBLOCK)
            except zmq.Again:
                # ★ 节流“丢弃观测”日志
                nowt = time.time()
                if (nowt - last_drop_obs_warn_ts) > args.drop_obs_log_interval_s:
                    logging.info("Dropping observation, no client connected")
                    last_drop_obs_warn_ts = nowt

            # Ensure a short sleep to avoid overloading the CPU.
            elapsed = time.time() - loop_start_time
            time.sleep(max(1 / host.max_loop_freq_hz - elapsed, 0))

            duration = time.perf_counter() - start

        print("Cycle time reached.")

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")
    finally:
        print("Shutting down Lekiwi Host.")
        robot.disconnect()
        host.disconnect()

    logging.info("Finished LeKiwi cleanly")


if __name__ == "__main__":
    main()
