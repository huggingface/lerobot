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
    --client_device=cpu \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```
"""

import logging
import pickle  # nosec
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from pprint import pformat
from queue import Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so_follower,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.visualization_utils import (
    init_visualization,
    log_visualization_data,
    shutdown_visualization,
)

from .configs import RobotClientConfig
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


class RobotClient:
    prefix = "robot_client"
    logger = get_logger(prefix)

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

        # --- Async-inference diagnostics (streamed to the live viewer under the `diagnostics.*` namespace) ---
        # Metadata of the chunk whose action we are currently executing, refreshed on each dequeue.
        self._executing_chunk_id: int | None = None
        self._executing_obs_timestamp: float | None = None
        self._executing_server_recv_timestamp: float | None = None
        self._executing_inference_start: float | None = None
        self._executing_inference_end: float | None = None
        # Queue length observed at the moment we last dequeued an action to execute.
        self._last_dequeue_queue_size: int = 0
        # Latest robot joint state (in action-feature order), used to measure the first-action-vs-state
        # jump when a new chunk arrives. Written by the control loop, read by the action-receiver thread.
        self._latest_state_lock = threading.Lock()
        self._latest_state_vec: torch.Tensor | None = None
        # One-shot diagnostic events produced off the control-loop thread (chunk_received, in the
        # receiver thread) and drained by the control loop so all viewer logging stays single-threaded.
        self._diag_events_lock = threading.Lock()
        self._pending_diag_events: list[dict[str, float]] = []
        # chunk_requested marker stashed by control_loop_observation (same thread) for the viewer step.
        self._pending_chunk_requested: dict[str, float] | None = None

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)
        # Throttle for always-on FPS logging (log at most once per interval, in seconds)
        self._fps_log_interval_s = 1.0
        self._last_fps_log_t = 0.0

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
            start_time = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            end_time = time.perf_counter()
            self.logger.debug(f"Connected to policy server in {end_time - start_time:.4f}s")

            # send policy instructions
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            self.logger.info("Sending policy instructions to policy server")
            self.logger.debug(
                f"Policy type: {self.policy_config.policy_type} | "
                f"Pretrained name or path: {self.policy_config.pretrained_name_or_path} | "
                f"Device: {self.policy_config.device}"
            )

            self.stub.SendPolicyInstructions(policy_setup)

            self.shutdown_event.clear()

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    def stop(self):
        """Stop the robot client"""
        self.shutdown_event.set()

        self.robot.disconnect()
        self.logger.debug("Robot disconnected")

        self.channel.close()
        self.logger.debug("Client stopped, channel closed")

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

        start_time = time.perf_counter()
        observation_bytes = pickle.dumps(obs)
        serialize_time = time.perf_counter() - start_time
        self.logger.debug(f"Observation serialization time: {serialize_time:.6f}s")

        try:
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            _ = self.stub.SendObservations(observation_iterator)
            obs_timestep = obs.get_timestep()
            self.logger.debug(f"Sent observation #{obs_timestep} | ")

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
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
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
            # Attribute the aggregated action to the newest contributing chunk (carry its diagnostics
            # metadata), since the blend is dominated by / freshest from the incoming chunk.
            future_action_queue.put(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=new_action.get_timestep(),
                    action=aggregate_fn(
                        current_action_queue[new_action.get_timestep()], new_action.get_action()
                    ),
                    chunk_id=new_action.get_chunk_id(),
                    obs_timestamp=new_action.get_obs_timestamp(),
                    server_recv_timestamp=new_action.server_recv_timestamp,
                    inference_start_timestamp=new_action.inference_start_timestamp,
                    inference_end_timestamp=new_action.inference_end_timestamp,
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
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue  # received `Empty` from server, wait for next call

                receive_time = time.time()

                # Deserialize bytes back into list[TimedAction]
                deserialize_start = time.perf_counter()
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                deserialize_time = time.perf_counter() - deserialize_start

                # Log device type of received actions
                if len(timed_actions) > 0:
                    received_device = timed_actions[0].get_action().device.type
                    self.logger.debug(f"Received actions on device: {received_device}")

                # Move actions to client_device (e.g., for downstream planners that need GPU)
                client_device = self.config.client_device
                if client_device != "cpu":
                    for timed_action in timed_actions:
                        if timed_action.get_action().device.type != client_device:
                            timed_action.action = timed_action.get_action().to(client_device)
                    self.logger.debug(f"Converted actions to device: {client_device}")
                else:
                    self.logger.debug(f"Actions kept on device: {client_device}")

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                # Diagnostic event: a new chunk arrived. Record its id, the end-to-end latency, and how
                # far the first action is from the current robot state (the jump you'd see at a chunk
                # boundary). Stashed for the control loop to log so viewer writes stay single-threaded.
                if len(timed_actions) > 0:
                    first_action = timed_actions[0].get_action()
                    chunk_id = timed_actions[0].get_chunk_id()
                    obs_ts = timed_actions[0].get_obs_timestamp()

                    with self._latest_state_lock:
                        state_vec = self._latest_state_vec

                    event: dict[str, float] = {}
                    if chunk_id is not None:
                        event["chunk_received_chunk_id"] = float(chunk_id)
                    if obs_ts is not None:
                        event["chunk_received_latency_ms"] = (receive_time - obs_ts) * 1000.0
                    if state_vec is not None and first_action is not None:
                        fa = first_action.detach().to("cpu").reshape(-1).float()
                        if fa.shape == state_vec.shape:
                            event["chunk_received_first_action_delta"] = float(
                                torch.linalg.norm(fa - state_vec)
                            )

                    if event:
                        with self._diag_events_lock:
                            self._pending_diag_events.append(event)

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

            except grpc.RpcError as e:
                self.logger.error(f"Error receiving actions: {e}")

    def actions_available(self):
        """Check if there are actions available in the queue"""
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        action = {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}
        return action

    def _state_vector_from_raw_observation(self, raw_observation: RawObservation) -> torch.Tensor | None:
        """Build a joint-state tensor ordered like ``robot.action_features`` (the same order used to
        turn action tensors into action dicts), or ``None`` if any action feature is missing/non-scalar.

        Diagnostics-only: lets the receiver thread measure how far a new chunk's first action is from
        the current robot state (the jump visible at a chunk boundary)."""
        try:
            values = [raw_observation[key] for key in self.robot.action_features]
            return torch.tensor([float(v) for v in values])
        except (KeyError, TypeError, ValueError):
            return None

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any]:
        """Reading and performing actions in local queue"""

        # Lock only for queue operations
        get_start = time.perf_counter()
        with self.action_queue_lock:
            queue_size_at_dequeue = self.action_queue.qsize()
            self.action_queue_size.append(queue_size_at_dequeue)
            # Get action from queue
            timed_action = self.action_queue.get_nowait()
        get_end = time.perf_counter() - get_start

        # Record which chunk this action came from (and its timing) so the control loop can stream
        # per-tick diagnostics: queue size at dequeue, chunk id, staleness and server-side latencies.
        self._last_dequeue_queue_size = queue_size_at_dequeue
        self._executing_chunk_id = timed_action.get_chunk_id()
        self._executing_obs_timestamp = timed_action.get_obs_timestamp()
        self._executing_server_recv_timestamp = timed_action.server_recv_timestamp
        self._executing_inference_start = timed_action.inference_start_timestamp
        self._executing_inference_end = timed_action.inference_end_timestamp

        _performed_action = self.robot.send_action(
            self._action_tensor_to_action_dict(timed_action.get_action())
        )
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

        return _performed_action

    def _ready_to_send_observation(self):
        """Flags when the client is ready to send an observation"""
        with self.action_queue_lock:
            return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold

    def control_loop_observation(self, task: str, verbose: bool = False) -> RawObservation:
        try:
            # Get serialized observation bytes from the function
            start_time = time.perf_counter()

            raw_observation: RawObservation = self.robot.get_observation()
            raw_observation["task"] = task

            # Cache the current joint state (action-feature order) for the chunk-jump diagnostic.
            state_vec = self._state_vector_from_raw_observation(raw_observation)
            if state_vec is not None:
                with self._latest_state_lock:
                    self._latest_state_vec = state_vec

            with self.latest_action_lock:
                latest_action = self.latest_action

            observation = TimedObservation(
                timestamp=time.time(),  # need time.time() to compare timestamps across client and server
                observation=raw_observation,
                timestep=max(latest_action, 0),
            )

            obs_capture_time = time.perf_counter() - start_time

            # If there are no actions left in the queue, the observation must go through processing!
            with self.action_queue_lock:
                observation.must_go = self.must_go.is_set() and self.action_queue.empty()
                current_queue_size = self.action_queue.qsize()

            _ = self.send_observation(observation)

            # Sending an observation is effectively a request for the server to produce a new chunk.
            # Record it as a one-shot marker (value = source observation timestep) for the viewer.
            self._pending_chunk_requested = {
                "chunk_requested_obs_timestep": float(observation.get_timestep()),
            }

            self.logger.debug(f"QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go})")
            if observation.must_go:
                # must-go event will be set again after receiving actions
                self.must_go.clear()

            # Always track FPS so it can be monitored without --verbose. Logged at INFO, throttled to
            # `self._fps_log_interval_s` so it doesn't spam the console at the control frequency.
            fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())
            now = time.perf_counter()
            if now - self._last_fps_log_t >= self._fps_log_interval_s:
                self._last_fps_log_t = now
                self.logger.info(
                    f"Obs #{observation.get_timestep()} | "
                    f"FPS now: {fps_metrics['instant_fps']:.2f} | "
                    f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                    f"Target: {fps_metrics['target_fps']:.2f}"
                )

            if verbose:
                self.logger.debug(
                    f"Ts={observation.get_timestamp():.6f} | Capturing observation took {obs_capture_time:.6f}s"
                )

            return raw_observation

        except Exception as e:
            self.logger.error(f"Error in observation sender: {e}")

    def _collect_diagnostics(self, had_action: bool) -> dict[str, float]:
        """Assemble the per-tick diagnostics scalars streamed to the live viewer.

        Combines steady per-tick signals (queue size, which chunk is executing, staleness, server
        latencies) with one-shot events produced this tick on the control-loop thread
        (``chunk_requested``) and drained from the action-receiver thread (``chunk_received``). All
        viewer logging happens on the control-loop thread, so draining here keeps it single-threaded.

        Args:
            had_action: Whether an action was dequeued and executed this tick.
        """
        diagnostics: dict[str, float] = {}

        # Queue-starvation marker: 1 while the queue is empty and we have no fresh action to execute
        # (the client holds/repeats the last commanded action), 0 otherwise.
        diagnostics["queue_starved"] = 0.0 if had_action else 1.0

        if had_action:
            diagnostics["queue_size"] = float(self._last_dequeue_queue_size)
            if self._executing_chunk_id is not None:
                diagnostics["action_source_chunk_id"] = float(self._executing_chunk_id)
            if self._executing_inference_start is not None and self._executing_inference_end is not None:
                diagnostics["server_inference_latency_ms"] = (
                    self._executing_inference_end - self._executing_inference_start
                ) * 1000.0
            if (
                self._executing_inference_start is not None
                and self._executing_server_recv_timestamp is not None
            ):
                diagnostics["server_queue_wait_ms"] = (
                    self._executing_inference_start - self._executing_server_recv_timestamp
                ) * 1000.0

        # Staleness of what we're executing right now: wall-clock now minus the timestamp of the
        # observation that generated the currently-executing action.
        if self._executing_obs_timestamp is not None:
            diagnostics["time_since_chunk_requested_ms"] = (
                time.time() - self._executing_obs_timestamp
            ) * 1000.0

        # Drain the chunk_requested marker (produced on this thread in control_loop_observation).
        if self._pending_chunk_requested is not None:
            diagnostics.update(self._pending_chunk_requested)
            self._pending_chunk_requested = None

        # Drain one-shot events produced by the action-receiver thread (chunk_received).
        with self._diag_events_lock:
            events = self._pending_diag_events
            self._pending_diag_events = []
        for event in events:
            diagnostics.update(event)

        return diagnostics

    def control_loop(self, task: str, verbose: bool = False) -> tuple[Observation, Action]:
        """Combined function for executing actions and streaming observations"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting")

        _performed_action = None
        _captured_observation = None

        while self.running:
            control_loop_start = time.perf_counter()

            # Track only data produced this iteration so we don't re-log stale values to the viewer.
            _fresh_action = None
            _fresh_observation = None

            """Control loop: (1) Performing actions, when available"""
            had_action = self.actions_available()
            if had_action:
                _performed_action = _fresh_action = self.control_loop_action(verbose)

            """Control loop: (2) Streaming observations to the remote policy server"""
            if self._ready_to_send_observation():
                _captured_observation = _fresh_observation = self.control_loop_observation(task, verbose)

            """Control loop: (3) Streaming observations/actions/diagnostics to the live viewer, when enabled"""
            if self.config.display_data:
                diagnostics = self._collect_diagnostics(had_action=had_action)
                # Drop the non-numeric "task" instruction; the viewer only handles scalars/images.
                obs_to_log = (
                    {k: v for k, v in _fresh_observation.items() if k != "task"}
                    if _fresh_observation is not None
                    else None
                )
                if obs_to_log is not None or _fresh_action is not None or diagnostics:
                    log_visualization_data(
                        self.config.display_mode,
                        observation=obs_to_log,
                        action=_fresh_action,
                        compress_images=self.config.display_compressed_images,
                        diagnostics=diagnostics or None,
                    )

            self.logger.debug(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
            # Dynamically adjust sleep time to maintain the desired control frequency
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

        return _captured_observation, _performed_action


@draccus.wrap()
def async_client(cfg: RobotClientConfig):
    logging.info(pformat(asdict(cfg)))

    # TODO: Assert if checking robot support is still needed with the plugin system
    # if cfg.robot.type not in SUPPORTED_ROBOTS:
    #     raise ValueError(f"Robot {cfg.robot.type} not yet supported!")

    client = RobotClient(cfg)

    if cfg.display_data:
        init_visualization(
            cfg.display_mode,
            session_name="async_inference",
            ip=cfg.display_ip,
            port=cfg.display_port,
        )

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
            if cfg.display_data:
                shutdown_visualization(cfg.display_mode)
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("Client stopped")


if __name__ == "__main__":
    register_third_party_plugins()
    async_client()  # run the client
