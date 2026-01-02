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
Example command:
```shell
python src/lerobot/async_inference/robot_client.py \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --task="dummy" \
    --server_address=127.0.0.1:8080 \
    --policy_type=act \
    --pretrained_name_or_path=user/model \
    --policy_device=mps \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```
"""

# ruff: noqa: E402, I001

import os as _os
import sys as _sys
import time as _time

_IMPORT_TIMING_ENABLED = _os.getenv("LEROBOT_IMPORT_TIMING", "0") == "1"
_IMPORT_T0 = _time.perf_counter() if _IMPORT_TIMING_ENABLED else 0.0

import logging
import pickle  # nosec
import threading
import time
from contextlib import suppress
from collections.abc import Callable
from dataclasses import asdict
from pprint import pformat
from queue import Empty, Queue
from typing import Any

import cv2  # type: ignore
import numpy as np
import grpc

from lerobot.robots.utils import make_robot_from_config
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks

from .configs import RobotClientConfig
from .constants import SUPPORTED_ROBOTS
from .helpers import (
    Action,
    FPSTracker,
    Observation,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
    visualize_action_queue_size,
)

if _IMPORT_TIMING_ENABLED:
    _sys.stderr.write(
        f"[import-timing] {__name__} imports: {(_time.perf_counter() - _IMPORT_T0) * 1000.0:.2f}ms\n"
    )


class RobotClient:
    prefix = "robot_client"
    logger = get_logger(prefix)

    @staticmethod
    def _ms(seconds: float) -> float:
        return seconds * 1000.0

    def __init__(self, config: RobotClientConfig):
        """Initialize RobotClient with unified configuration.

        Args:
            config: RobotClientConfig containing all configuration parameters
        """
        # Store configuration
        self.config = config
        self.robot = make_robot_from_config(config.robot)
        self.robot.connect()

        lerobot_features = map_robot_keys_to_lerobot_features(self.robot)

        # Use environment variable if server_address is not provided in config
        self.server_address = config.server_address

        self.policy_config = RemotePolicyConfig(
            config.policy_type,
            config.pretrained_name_or_path,
            lerobot_features,
            config.actions_per_chunk,
            config.policy_device,
        )
        self.channel = grpc.insecure_channel(
            self.server_address, grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.logger.info(f"Initializing client to connect to server at {self.server_address}")

        self.shutdown_event = threading.Event()

        # Initialize client side variables
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = -1

        self._chunk_size_threshold = config.chunk_size_threshold

        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()  # Protect queue operations
        self.action_queue_size = []
        self.start_barrier = threading.Barrier(2)  # 2 threads: action receiver, control loop

        # Observation sending is offloaded to a background thread so the control loop doesn't block on gRPC.
        # Keep only the latest observation to avoid unbounded backlog when networking is slow.
        self._obs_send_queue: Queue[TimedObservation] = Queue(maxsize=1)
        self._obs_sender_thread: threading.Thread | None = None
        self._obs_sender_stop_event = threading.Event()
        self._last_obs_enqueued_t = 0.0

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        self.logger.info("Robot connected and ready")

        # Use an event for thread-safe coordination
        self.must_go = threading.Event()
        self.must_go.set()  # Initially set - observations qualify for direct processing

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def start(self):
        """Start the robot client and connect to the policy server"""
        try:
            # client-server handshake
            t_total_start = time.perf_counter()

            t_ready_start = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            t_ready_done = time.perf_counter()
            self.logger.debug(
                "Connected to policy server (Ready RPC) in %.2fms",
                self._ms(t_ready_done - t_ready_start),
            )

            # send policy instructions
            t_policy_ser_start = time.perf_counter()
            policy_config_bytes = pickle.dumps(self.policy_config)
            t_policy_ser_done = time.perf_counter()
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            self.logger.info("Sending policy instructions to policy server")
            self.logger.debug(
                f"Policy type: {self.policy_config.policy_type} | "
                f"Pretrained name or path: {self.policy_config.pretrained_name_or_path} | "
                f"Device: {self.policy_config.device}"
            )
            self.logger.debug(
                "Policy config serialized in %.2fms | bytes: %s",
                self._ms(t_policy_ser_done - t_policy_ser_start),
                len(policy_config_bytes),
            )

            t_policy_rpc_start = time.perf_counter()
            self.stub.SendPolicyInstructions(policy_setup)
            t_policy_rpc_done = time.perf_counter()
            self.logger.debug(
                "SendPolicyInstructions RPC completed in %.2fms",
                self._ms(t_policy_rpc_done - t_policy_rpc_start),
            )

            self.shutdown_event.clear()
            self._start_observation_sender_thread()
            self.logger.info(
                "Client init complete | Ready: %.2fms | Policy serialize: %.2fms | Policy RPC: %.2fms | Total: %.2fms",
                self._ms(t_ready_done - t_ready_start),
                self._ms(t_policy_ser_done - t_policy_ser_start),
                self._ms(t_policy_rpc_done - t_policy_rpc_start),
                self._ms(time.perf_counter() - t_total_start),
            )

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    def stop(self):
        """Stop the robot client"""
        self.shutdown_event.set()
        self._stop_observation_sender_thread()

        self.robot.disconnect()
        self.logger.debug("Robot disconnected")

        self.channel.close()
        self.logger.debug("Client stopped, channel closed")

    def _start_observation_sender_thread(self) -> None:
        if self._obs_sender_thread is not None and self._obs_sender_thread.is_alive():
            return

        self._obs_sender_stop_event.clear()
        self._obs_sender_thread = threading.Thread(
            target=self._observation_sender_loop,
            name="robot_client_observation_sender",
            daemon=True,
        )
        self._obs_sender_thread.start()

    def _stop_observation_sender_thread(self) -> None:
        self._obs_sender_stop_event.set()
        # Unblock queue.get() promptly.
        with suppress(Exception):
            self._obs_send_queue.put_nowait(
                TimedObservation(timestamp=0.0, timestep=-1, observation={}, must_go=False)
            )

        if self._obs_sender_thread is not None and self._obs_sender_thread.is_alive():
            self._obs_sender_thread.join(timeout=2.0)
        self._obs_sender_thread = None

    def _observation_sender_loop(self) -> None:
        while not self._obs_sender_stop_event.is_set() and self.running:
            try:
                obs = self._obs_send_queue.get(timeout=0.1)
            except Empty:
                continue

            # Sentinel used to unblock on shutdown.
            if obs.get_timestep() < 0:
                continue

            try:
                _ = self.send_observation(obs)
            except Exception as e:
                self.logger.error(f"Error in observation sender thread: {e}")

    def send_observation(
        self,
        obs: TimedObservation,
    ) -> bool:
        """Send observation to the policy server.
        Returns True if the observation was sent successfully, False otherwise."""
        if not self.running:
            raise RuntimeError("Client not running. Run RobotClient.start() before sending observations.")

        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")

        t_encode_start = time.perf_counter()
        encoded_observation, encode_stats = _encode_images_for_transport(
            obs.get_observation(), jpeg_quality=60
        )
        # NOTE: Mutate in-place to avoid extra copies of large dicts; we only touch image entries.
        obs.observation = encoded_observation
        t_encode_done = time.perf_counter()
        if encode_stats["images_encoded"] > 0:
            self.logger.debug(
                "Encoded %s images for transport in %.2fms | raw_bytes=%s -> encoded_bytes=%s",
                encode_stats["images_encoded"],
                self._ms(t_encode_done - t_encode_start),
                encode_stats["raw_bytes_total"],
                encode_stats["encoded_bytes_total"],
            )

        t_total_start = time.perf_counter()

        t_ser_start = time.perf_counter()
        observation_bytes = pickle.dumps(obs)
        t_ser_done = time.perf_counter()
        self.logger.debug(
            "Observation #%s serialization time: %.2fms | bytes: %s | must_go: %s",
            obs.get_timestep(),
            self._ms(t_ser_done - t_ser_start),
            len(observation_bytes),
            obs.must_go,
        )

        try:
            t_iter_start = time.perf_counter()
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            t_iter_done = time.perf_counter()

            t_rpc_start = time.perf_counter()
            _ = self.stub.SendObservations(observation_iterator)
            t_rpc_done = time.perf_counter()
            obs_timestep = obs.get_timestep()
            self.logger.debug(
                "Sent observation #%s | iterator prep: %.2fms | SendObservations RPC: %.2fms | total: %.2fms",
                obs_timestep,
                self._ms(t_iter_done - t_iter_start),
                self._ms(t_rpc_done - t_rpc_start),
                self._ms(time.perf_counter() - t_total_start),
            )

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Error sending observation #{obs.get_timestep()}: {e}")
            return False

    def _inspect_action_queue(self):
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
            timestamps = sorted([action.get_timestep() for action in self.action_queue.queue])
        self.logger.debug(f"Queue size: {queue_size}, Queue contents: {timestamps}")
        return queue_size, timestamps

    def _aggregate_action_queues(
        self,
        incoming_actions: list[TimedAction],
        aggregate_fn: Callable[[Any, Any], Any] | None = None,
    ):
        """Finds the same timestep actions in the queue and aggregates them using the aggregate_fn"""
        if aggregate_fn is None:
            # default aggregate function: take the latest action
            def aggregate_fn(x1, x2):
                return x2

        future_action_queue = Queue()
        with self.action_queue_lock:
            internal_queue = self.action_queue.queue

        current_action_queue = {action.get_timestep(): action.get_action() for action in internal_queue}

        discontinuity = _action_discontinuity_stats(current_action_queue, incoming_actions)
        if discontinuity["overlaps"] > 0:
            self.logger.debug(
                "Incoming action discontinuity on overlap | overlaps=%s | mean_abs=%.6f | max_abs=%.6f | mean_l2=%.6f | max_l2=%.6f",
                discontinuity["overlaps"],
                discontinuity["mean_abs"],
                discontinuity["max_abs"],
                discontinuity["mean_l2"],
                discontinuity["max_l2"],
            )

        # Avoid changing near-future actions right before execution: it can introduce visible jitter.
        # We only allow overlap aggregation for actions sufficiently far in the future.
        protect_steps = 5

        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action

            # New action is older than the latest action in the queue, skip it
            if new_action.get_timestep() <= latest_action:
                continue

            # If the new action's timestep is not in the current action queue, add it directly
            elif new_action.get_timestep() not in current_action_queue:
                future_action_queue.put(new_action)
                continue

            # If the new action's timestep is in the current action queue, aggregate it
            # TODO: There is probably a way to do this with broadcasting of the two action tensors
            overlap_ts = new_action.get_timestep()
            if overlap_ts <= latest_action + protect_steps:
                # Keep the previously queued action for this timestep.
                # (We still accept the rest of the incoming chunk.)
                continue

            future_action_queue.put(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=overlap_ts,
                    action=aggregate_fn(
                        current_action_queue[overlap_ts],
                        new_action.get_action(),
                    ),
                )
            )

        with self.action_queue_lock:
            self.action_queue = future_action_queue

    def receive_actions(self, verbose: bool = False):
        """Receive actions from the policy server"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")

        while self.running:
            try:
                # Use StreamActions to get a stream of actions from the server
                t_rpc_start = time.perf_counter()
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                t_rpc_done = time.perf_counter()
                if len(actions_chunk.data) == 0:
                    self.logger.debug(
                        "GetActions returned Empty | RPC: %.2fms",
                        self._ms(t_rpc_done - t_rpc_start),
                    )
                    continue  # received `Empty` from server, wait for next call

                receive_time = time.time()

                # Deserialize bytes back into list[TimedAction]
                deserialize_start = time.perf_counter()
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                deserialize_time = time.perf_counter() - deserialize_start

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                # Calculate network latency if we have matching observations
                if len(timed_actions) > 0 and verbose:
                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.debug(f"Current latest action: {latest_action}")

                    # Get queue state before changes
                    old_size, old_timesteps = self._inspect_action_queue()
                    if not old_timesteps:
                        old_timesteps = [latest_action]  # queue was empty

                    # Log incoming actions
                    incoming_timesteps = [a.get_timestep() for a in timed_actions]

                    first_action_timestep = timed_actions[0].get_timestep()
                    server_to_client_latency = (receive_time - timed_actions[0].get_timestamp()) * 1000

                    self.logger.info(
                        f"Received action chunk for step #{first_action_timestep} | "
                        f"Latest action: #{latest_action} | "
                        f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Network latency (server->client): {server_to_client_latency:.2f}ms | "
                        f"Deserialization time: {deserialize_time * 1000:.2f}ms"
                    )

                self.logger.debug(
                    "GetActions RPC: %.2fms | bytes: %s | actions: %s | deserialize: %.2fms",
                    self._ms(t_rpc_done - t_rpc_start),
                    len(actions_chunk.data),
                    len(timed_actions),
                    self._ms(deserialize_time),
                )

                # Update action queue
                start_time = time.perf_counter()
                self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
                queue_update_time = time.perf_counter() - start_time

                self.must_go.set()  # after receiving actions, next empty queue triggers must-go processing!

                if verbose:
                    # Get queue state after changes
                    new_size, new_timesteps = self._inspect_action_queue()

                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.info(
                        f"Latest action: {latest_action} | "
                        f"Old action steps: {old_timesteps[0]}:{old_timesteps[-1]} | "
                        f"Incoming action steps: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Updated action steps: {new_timesteps[0]}:{new_timesteps[-1]}"
                    )
                    self.logger.debug(
                        f"Queue update complete ({queue_update_time:.6f}s) | "
                        f"Before: {old_size} items | "
                        f"After: {new_size} items | "
                    )
                else:
                    self.logger.debug(
                        "Action queue update time: %.2fms",
                        self._ms(queue_update_time),
                    )

            except grpc.RpcError as e:
                self.logger.error(f"Error receiving actions: {e}")

    def actions_available(self):
        """Check if there are actions available in the queue"""
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _action_tensor_to_action_dict(self, action_tensor: Any) -> dict[str, float]:
        action = {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}
        return action

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any]:
        """Reading and performing actions in local queue"""

        # Lock only for queue operations
        get_start = time.perf_counter()
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            # Get action from queue
            timed_action = self.action_queue.get_nowait()
        get_end = time.perf_counter() - get_start

        t_send_start = time.perf_counter()
        _performed_action = self.robot.send_action(
            self._action_tensor_to_action_dict(timed_action.get_action())
        )
        t_send_done = time.perf_counter()
        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()

        if verbose:
            with self.action_queue_lock:
                current_queue_size = self.action_queue.qsize()

            self.logger.debug(
                f"Ts={timed_action.get_timestamp()} | "
                f"Action #{timed_action.get_timestep()} performed | "
                f"Queue size: {current_queue_size}"
            )

            self.logger.debug(
                f"Popping action from queue to perform took {get_end:.6f}s | Queue size: {current_queue_size}"
            )
            self.logger.debug(
                "Robot send_action time: %.2fms",
                self._ms(t_send_done - t_send_start),
            )
        else:
            self.logger.debug(
                "Action #%s timings | pop: %.2fms | send_action: %.2fms",
                timed_action.get_timestep(),
                self._ms(get_end),
                self._ms(t_send_done - t_send_start),
            )

        return _performed_action

    def _ready_to_send_observation(self):
        """Flags when the client is ready to send an observation"""
        with self.action_queue_lock:
            return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold

    def control_loop_observation(self, task: str, verbose: bool = False) -> RawObservation:
        self.logger.info("ready to capture observation")
        try:
            # If an observation is already queued for sending, don't capture another one yet.
            # This prevents thrashing when the action queue is low (and smooths control).
            if self._obs_send_queue.qsize() > 0:
                return {}

            # Get serialized observation bytes from the function
            start_time = time.perf_counter()

            t_capture_start = time.perf_counter()
            raw_observation: RawObservation = self.robot.get_observation()
            t_capture_done = time.perf_counter()
            raw_observation["task"] = task

            with self.latest_action_lock:
                latest_action = self.latest_action

            t_pack_start = time.perf_counter()
            observation = TimedObservation(
                timestamp=time.time(),  # need time.time() to compare timestamps across client and server
                observation=raw_observation,
                timestep=max(latest_action, 0),
            )
            t_pack_done = time.perf_counter()

            obs_capture_time = time.perf_counter() - start_time

            # If there are no actions left in the queue, the observation must go through processing!
            with self.action_queue_lock:
                observation.must_go = self.must_go.is_set() and self.action_queue.empty()
                current_queue_size = self.action_queue.qsize()

            t_send_obs_start = time.perf_counter()
            enqueued = True
            if self._obs_send_queue.full():
                with suppress(Exception):
                    _ = self._obs_send_queue.get_nowait()
            try:
                self._obs_send_queue.put_nowait(observation)
                self._last_obs_enqueued_t = time.perf_counter()
            except Exception:
                enqueued = False
            t_send_obs_done = time.perf_counter()

            self.logger.debug(f"QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go})")
            if observation.must_go:
                # must-go event will be set again after receiving actions
                self.must_go.clear()

            self.logger.debug(
                "Observation #%s timings | capture: %.2fms | pack: %.2fms | enqueue: %.2fms | total: %.2fms | enqueued: %s",
                observation.get_timestep(),
                self._ms(t_capture_done - t_capture_start),
                self._ms(t_pack_done - t_pack_start),
                self._ms(t_send_obs_done - t_send_obs_start),
                self._ms(obs_capture_time),
                enqueued,
            )

            if verbose:
                # Calculate comprehensive FPS metrics
                fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())

                self.logger.info(
                    f"Obs #{observation.get_timestep()} | "
                    f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                    f"Target: {fps_metrics['target_fps']:.2f}"
                )

                self.logger.debug(
                    f"Ts={observation.get_timestamp():.6f} | Capturing observation took {obs_capture_time:.6f}s"
                )

            return raw_observation

        except Exception as e:
            self.logger.error(f"Error in observation sender: {e}")

    def control_loop(self, task: str, verbose: bool = False) -> tuple[Observation, Action]:
        """Combined function for executing actions and streaming observations"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting")

        _performed_action = None
        _captured_observation = None

        while self.running:
            control_loop_start = time.perf_counter()
            """Control loop: (1) Performing actions, when available"""
            if self.actions_available():
                _performed_action = self.control_loop_action(verbose)

            """Control loop: (2) Streaming observations to the remote policy server"""
            if self._ready_to_send_observation():
                _captured_observation = self.control_loop_observation(task, verbose)

            self.logger.debug(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
            # Dynamically adjust sleep time to maintain the desired control frequency
            elapsed = time.perf_counter() - control_loop_start
            sleep_s = max(0.0, self.config.environment_dt - elapsed)
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                self.logger.debug(
                    "Control loop overran dt | elapsed: %.2fms | target: %.2fms",
                    self._ms(elapsed),
                    self._ms(self.config.environment_dt),
                )

        return _captured_observation, _performed_action


def async_client(cfg: RobotClientConfig):
    logging.info(pformat(asdict(cfg)))

    if cfg.robot.type not in SUPPORTED_ROBOTS:
        raise ValueError(f"Robot {cfg.robot.type} not yet supported!")

    client = RobotClient(cfg)

    if client.start():
        client.logger.info("Starting action receiver thread...")

        # Create and start action receiver thread
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)

        # Start action receiver thread
        action_receiver_thread.start()

        try:
            # The main thread runs the control loop
            client.control_loop(task=cfg.task)

        finally:
            client.stop()
            action_receiver_thread.join()
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("Client stopped")


if __name__ == "__main__":
    import draccus

    draccus.wrap()(async_client)()  # run the client


def _is_uint8_hwc3_image(x: Any) -> bool:
    if not isinstance(x, np.ndarray):
        return False
    if x.dtype != np.uint8:
        return False
    if x.ndim != 3:
        return False
    h, w, c = x.shape
    if h <= 0 or w <= 0:
        return False
    return c == 3


def _encode_images_for_transport(
    observation: Any,
    jpeg_quality: int,
) -> tuple[Any, dict[str, int]]:
    """Recursively JPEG-encode uint8 HWC3 images inside an observation structure.

    Encoded images are replaced with a marker dict so the server can decode them back
    into `np.ndarray` before policy preprocessing.
    """
    stats = {"images_encoded": 0, "raw_bytes_total": 0, "encoded_bytes_total": 0}

    def _encode_any(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: _encode_any(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_encode_any(v) for v in x]
        if isinstance(x, tuple):
            return tuple(_encode_any(v) for v in x)

        if not _is_uint8_hwc3_image(x):
            return x

        # Treat input as RGB (LeRobot convention); OpenCV expects BGR for encoding.
        bgr = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(
            ".jpg",
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
        )
        if not ok:
            raise RuntimeError("OpenCV failed to JPEG-encode image for transport")

        payload = bytes(buf)
        stats["images_encoded"] += 1
        stats["raw_bytes_total"] += int(x.nbytes)
        stats["encoded_bytes_total"] += len(payload)
        return {"__lerobot_image_encoding__": "jpeg", "quality": int(jpeg_quality), "data": payload}

    return _encode_any(observation), stats


def _action_discontinuity_stats(
    current_action_queue: dict[int, Any],
    incoming_actions: list[TimedAction],
) -> dict[str, float]:
    """Compute a rough discontinuity metric between existing queued actions and newly received ones.

    Only compares timesteps that overlap (same timestep already present in queue).
    """
    overlaps = 0
    sum_abs = 0.0
    max_abs = 0.0
    sum_l2 = 0.0
    max_l2 = 0.0

    for a in incoming_actions:
        ts = a.get_timestep()
        if ts not in current_action_queue:
            continue
        old = current_action_queue[ts]
        new = a.get_action()

        try:
            old_arr = np.asarray(old, dtype=np.float32).reshape(-1)
            new_arr = np.asarray(new, dtype=np.float32).reshape(-1)
        except Exception:
            continue
        if old_arr.shape != new_arr.shape or old_arr.size == 0:
            continue

        diff = new_arr - old_arr
        abs_diff = np.abs(diff)
        overlaps += 1
        sum_abs += float(abs_diff.mean())
        max_abs = max(max_abs, float(abs_diff.max()))
        l2 = float(np.linalg.norm(diff))
        sum_l2 += l2
        max_l2 = max(max_l2, l2)

    if overlaps == 0:
        return {"overlaps": 0.0, "mean_abs": 0.0, "max_abs": 0.0, "mean_l2": 0.0, "max_l2": 0.0}

    return {
        "overlaps": float(overlaps),
        "mean_abs": sum_abs / overlaps,
        "max_abs": max_abs,
        "mean_l2": sum_l2 / overlaps,
        "max_l2": max_l2,
    }
