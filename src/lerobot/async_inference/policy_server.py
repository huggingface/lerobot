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
Example:
```shell
python -m lerobot.async_inference.policy_server \
     --host=127.0.0.1 \
     --port=8080 \
     --fps=30 \
     --inference_latency=0.033 \
     --obs_queue_timeout=1
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
from concurrent import futures
from dataclasses import asdict
from pprint import pformat
from queue import Empty, Queue
from typing import Any

import cv2  # type: ignore
import numpy as np
import draccus
import grpc
import torch

from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import receive_bytes_in_chunks

from .configs import PolicyServerConfig
from .constants import SUPPORTED_POLICIES
from .helpers import (
    FPSTracker,
    Observation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    observations_similar,
    raw_observation_to_observation,
)

if _IMPORT_TIMING_ENABLED:
    _sys.stderr.write(
        f"[import-timing] {__name__} imports: {(_time.perf_counter() - _IMPORT_T0) * 1000.0:.2f}ms\n"
    )


def _infer_model_action_horizon(policy_config: Any) -> tuple[str, int] | None:
    """Infer the maximum action horizon from a loaded policy config."""
    if policy_config is None:
        return None

    for field_name in ("chunk_size", "n_action_steps", "horizon"):
        value = getattr(policy_config, field_name, None)
        if isinstance(value, int) and value > 0:
            return field_name, value

    return None


class PolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    prefix = "policy_server"
    logger = get_logger(prefix)

    def __init__(self, config: PolicyServerConfig):
        self.config = config
        self.shutdown_event = threading.Event()

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=config.fps)

        self.observation_queue = Queue(maxsize=1)

        self._predicted_timesteps_lock = threading.Lock()
        self._predicted_timesteps = set()

        self.last_processed_obs = None

        # Attributes will be set by SendPolicyInstructions
        self.device = None
        self.policy_type = None
        self.lerobot_features = None
        self.actions_per_chunk = None
        self.policy = None
        self.preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None
        self.postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None

    @staticmethod
    def _ms(seconds: float) -> float:
        return seconds * 1000.0

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    @property
    def policy_image_features(self):
        return self.policy.config.image_features

    def _reset_server(self) -> None:
        """Flushes server state when new client connects."""
        # only running inference on the latest observation received by the server
        self.shutdown_event.set()
        self.observation_queue = Queue(maxsize=1)

        with self._predicted_timesteps_lock:
            self._predicted_timesteps = set()

    def Ready(self, request, context):  # noqa: N802
        client_id = context.peer()
        self.logger.info(f"Client {client_id} connected and ready")
        self._reset_server()
        self.shutdown_event.clear()

        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Receive policy instructions from the robot client"""

        t_total_start = time.perf_counter()

        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        client_id = context.peer()

        t0 = time.perf_counter()
        policy_specs = pickle.loads(request.data)  # nosec
        t_deserialize = time.perf_counter() - t0

        if not isinstance(policy_specs, RemotePolicyConfig):
            raise TypeError(f"Policy specs must be a RemotePolicyConfig. Got {type(policy_specs)}")

        if policy_specs.policy_type not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Policy type {policy_specs.policy_type} not supported. "
                f"Supported policies: {SUPPORTED_POLICIES}"
            )

        self.logger.info(
            f"Receiving policy instructions from {client_id} | "
            f"Policy type: {policy_specs.policy_type} | "
            f"Pretrained name or path: {policy_specs.pretrained_name_or_path} | "
            f"Actions per chunk: {policy_specs.actions_per_chunk} | "
            f"Device: {policy_specs.device}"
        )
        self.logger.debug(
            "Policy instructions payload deserialized in %.2fms", self._ms(t_deserialize)
        )

        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type  # act, pi0, etc.
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk

        policy_class = get_policy_class(self.policy_type)

        t_load_start = time.perf_counter()
        self.policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)
        t_loaded = time.perf_counter()

        t_to_start = time.perf_counter()
        self.policy.to(self.device)  # includes parameter/device moves
        t_to_done = time.perf_counter()

        inferred_horizon = _infer_model_action_horizon(getattr(self.policy, "config", None))
        if inferred_horizon is not None:
            horizon_field, model_horizon = inferred_horizon
            if self.actions_per_chunk > model_horizon:
                raise ValueError(
                    "Requested actions_per_chunk "
                    f"({self.actions_per_chunk}) exceeds model-supported horizon "
                    f"({model_horizon}, from policy config field '{horizon_field}') "
                    f"for checkpoint '{policy_specs.pretrained_name_or_path}'. "
                    f"Set actions_per_chunk <= {model_horizon}."
                )

        # Load preprocessor and postprocessor, overriding device to match requested device
        device_override = {"device": self.device}
        t_pp_start = time.perf_counter()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=policy_specs.pretrained_name_or_path,
            preprocessor_overrides={
                "device_processor": device_override,
                "rename_observations_processor": {"rename_map": policy_specs.rename_map},
            },
            postprocessor_overrides={"device_processor": device_override},
        )
        t_pp_done = time.perf_counter()

        self.logger.info(
            "Policy init timing | from_pretrained: %.2fms | to(%s): %.2fms | pre/post: %.2fms | total: %.2fms",
            self._ms(t_loaded - t_load_start),
            self.device,
            self._ms(t_to_done - t_to_start),
            self._ms(t_pp_done - t_pp_start),
            self._ms(time.perf_counter() - t_total_start),
        )

        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        """Receive observations from the robot client"""
        client_id = context.peer()
        self.logger.debug(f"Receiving observations from {client_id}")

        t_total_start = time.perf_counter()
        receive_time = time.time()  # comparing timestamps so need time.time()

        t_recv_start = time.perf_counter()
        received_bytes = receive_bytes_in_chunks(
            request_iterator, None, self.shutdown_event, self.logger
        )  # blocking call while looping over request_iterator
        t_recv_done = time.perf_counter()

        t_deser_start = time.perf_counter()
        timed_observation = pickle.loads(received_bytes)  # nosec
        t_deser_done = time.perf_counter()

        t_decode_start = time.perf_counter()
        decoded_observation, decode_stats = _decode_images_from_transport(timed_observation.get_observation())
        timed_observation.observation = decoded_observation
        t_decode_done = time.perf_counter()
        if decode_stats["images_decoded"] > 0:
            self.logger.debug(
                "Decoded %s images from transport in %.2fms | encoded_bytes=%s -> raw_bytes=%s",
                decode_stats["images_decoded"],
                self._ms(t_decode_done - t_decode_start),
                decode_stats["encoded_bytes_total"],
                decode_stats["raw_bytes_total"],
            )

        self.logger.debug(f"Received observation #{timed_observation.get_timestep()}")

        obs_timestep = timed_observation.get_timestep()
        obs_timestamp = timed_observation.get_timestamp()

        # Calculate FPS metrics
        fps_metrics = self.fps_tracker.calculate_fps_metrics(obs_timestamp)

        self.logger.debug(
            f"Received observation #{obs_timestep} | "
            f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "  # fps at which observations are received from client
            f"Target: {fps_metrics['target_fps']:.2f} | "
            f"One-way latency: {(receive_time - obs_timestamp) * 1000:.2f}ms"
        )

        self.logger.debug(
            f"Server timestamp: {receive_time:.6f} | "
            f"Client timestamp: {obs_timestamp:.6f} | "
            f"Chunk-receive time: {self._ms(t_recv_done - t_recv_start):.2f}ms | "
            f"Deserialize time: {self._ms(t_deser_done - t_deser_start):.2f}ms | "
            f"Payload bytes: {len(received_bytes)}"
        )

        t_enqueue_start = time.perf_counter()
        enqueued = self._enqueue_observation(timed_observation)  # wrapping a RawObservation
        t_enqueue_done = time.perf_counter()

        if not enqueued:
            self.logger.debug(f"Observation #{obs_timestep} has been filtered out")
        else:
            self.logger.debug(
                "Observation #%s enqueued | enqueue time: %.2fms | queue size: %s",
                obs_timestep,
                self._ms(t_enqueue_done - t_enqueue_start),
                self.observation_queue.qsize(),
            )

        self.logger.debug(
            "SendObservations total time: %.2fms",
            self._ms(time.perf_counter() - t_total_start),
        )

        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        """Returns actions to the robot client. Actions are sent as a single
        chunk, containing multiple actions."""
        client_id = context.peer()
        self.logger.debug(f"Client {client_id} connected for action streaming")

        # Generate action based on the most recent observation and its timestep
        try:
            t_total_start = time.perf_counter()

            t_wait_start = time.perf_counter()
            obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)
            t_wait_done = time.perf_counter()

            self.logger.info(
                f"Running inference for observation #{obs.get_timestep()} (must_go: {obs.must_go})"
            )

            self.logger.debug(
                "GetActions waited %.2fms for observation | queue size after get: %s",
                self._ms(t_wait_done - t_wait_start),
                self.observation_queue.qsize(),
            )

            with self._predicted_timesteps_lock:
                self._predicted_timesteps.add(obs.get_timestep())

            start_time = time.perf_counter()
            action_chunk = self._predict_action_chunk(obs)
            inference_time = time.perf_counter() - start_time

            start_time = time.perf_counter()
            actions_bytes = pickle.dumps(action_chunk)  # nosec
            serialize_time = time.perf_counter() - start_time

            # Create and return the action chunk
            actions = services_pb2.Actions(data=actions_bytes)

            self.logger.info(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Total time: {(inference_time + serialize_time) * 1000:.2f}ms"
            )

            self.logger.debug(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Inference time: {self._ms(inference_time):.2f}ms | "
                f"Serialize time: {self._ms(serialize_time):.2f}ms | "
                f"Pickle bytes: {len(actions_bytes)}"
            )

            # sleep controls inference latency (wall-clock budget for the entire GetActions call)
            elapsed = time.perf_counter() - t_total_start
            target = self.config.inference_latency
            sleep_s = max(0.0, target - max(0.0, elapsed))
            if sleep_s > 0:
                t_sleep_start = time.perf_counter()
                time.sleep(sleep_s)
                t_sleep_done = time.perf_counter()
                self.logger.debug(
                    "GetActions throttling sleep: %.2fms (target %.2fms, elapsed %.2fms)",
                    self._ms(t_sleep_done - t_sleep_start),
                    self._ms(target),
                    self._ms(elapsed),
                )
            else:
                self.logger.debug(
                    "GetActions no sleep (target %.2fms, elapsed %.2fms)",
                    self._ms(target),
                    self._ms(elapsed),
                )

            return actions

        except Empty:  # no observation added to queue in obs_queue_timeout
            return services_pb2.Empty()

        except Exception as e:
            self.logger.error(f"Error in StreamActions: {e}")

            return services_pb2.Empty()

    def _obs_sanity_checks(self, obs: TimedObservation, previous_obs: TimedObservation) -> bool:
        """Check if the observation is valid to be processed by the policy"""
        with self._predicted_timesteps_lock:
            predicted_timesteps = self._predicted_timesteps

        if obs.get_timestep() in predicted_timesteps:
            self.logger.debug(f"Skipping observation #{obs.get_timestep()} - Timestep predicted already!")
            return False

        elif observations_similar(obs, previous_obs, lerobot_features=self.lerobot_features):
            self.logger.debug(
                f"Skipping observation #{obs.get_timestep()} - Observation too similar to last obs predicted!"
            )
            return False

        else:
            return True

    def _enqueue_observation(self, obs: TimedObservation) -> bool:
        """Enqueue an observation if it must go through processing, otherwise skip it.
        Observations not in queue are never run through the policy network"""

        if (
            obs.must_go
            or self.last_processed_obs is None
            or self._obs_sanity_checks(obs, self.last_processed_obs)
        ):
            last_obs = self.last_processed_obs.get_timestep() if self.last_processed_obs else "None"
            self.logger.debug(
                f"Enqueuing observation. Must go: {obs.must_go} | Last processed obs: {last_obs}"
            )

            # If queue is full, get the old observation to make room
            if self.observation_queue.full():
                # pops from queue
                _ = self.observation_queue.get_nowait()
                self.logger.debug("Observation queue was full, removed oldest observation")

            # Now put the new observation (never blocks as queue is non-full here)
            self.observation_queue.put(obs)
            return True

        return False

    def _time_action_chunk(self, t_0: float, action_chunk: list[torch.Tensor], i_0: int) -> list[TimedAction]:
        """Turn a chunk of actions into a list of TimedAction instances,
        with the first action corresponding to t_0 and the rest corresponding to
        t_0 + i*environment_dt for i in range(len(action_chunk))
        """
        return [
            # Convert to numpy so the robot client does not need torch installed just to unpickle actions.
            TimedAction(
                timestamp=t_0 + i * self.config.environment_dt,
                timestep=i_0 + i,
                action=action.detach().cpu().numpy(),
            )
            for i, action in enumerate(action_chunk)
        ]

    def _get_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """Get an action chunk from the policy. The chunk contains only"""
        t0 = time.perf_counter()
        chunk = self.policy.predict_action_chunk(observation)
        t1 = time.perf_counter()
        self.logger.debug("Policy predict_action_chunk time: %.2fms", self._ms(t1 - t0))
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)  # adding batch dimension, now shape is (B, chunk_size, action_dim)

        return chunk[:, : self.actions_per_chunk, :]

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        """Predict an action chunk based on an observation.

        Pipeline:
        1. Convert raw observation to LeRobot format
        2. Apply preprocessor (tokenization, normalization, batching, device placement)
        3. Run policy inference to get action chunk
        4. Apply postprocessor (unnormalization, device movement)
        5. Convert to TimedAction list
        """
        """1. Prepare observation"""
        start_prepare = time.perf_counter()
        observation: Observation = raw_observation_to_observation(
            observation_t.get_observation(),
            self.lerobot_features,
            self.policy_image_features,
        )
        prepare_time = time.perf_counter() - start_prepare

        """2. Apply preprocessor"""
        start_preprocess = time.perf_counter()
        observation = self.preprocessor(observation)
        self.last_processed_obs: TimedObservation = observation_t
        preprocessing_time = time.perf_counter() - start_preprocess

        """3. Get action chunk"""
        start_inference = time.perf_counter()
        action_tensor = self._get_action_chunk(observation)
        inference_time = time.perf_counter() - start_inference
        self.logger.debug(
            "Model timings | prepare: %.2fms | preprocess: %.2fms | inference: %.2fms | action shape: %s",
            self._ms(prepare_time),
            self._ms(preprocessing_time),
            self._ms(inference_time),
            tuple(action_tensor.shape),
        )

        """4. Apply postprocessor"""
        # Apply postprocessor (handles unnormalization and device movement)
        # Postprocessor expects (B, action_dim) per action, but we have (B, chunk_size, action_dim)
        # So we process each action in the chunk individually
        start_postprocess = time.perf_counter()
        _, chunk_size, _ = action_tensor.shape

        # Process each action in the chunk
        processed_actions = []
        for i in range(chunk_size):
            # Extract action at timestep i: (B, action_dim)
            single_action = action_tensor[:, i, :]
            t_action_post_start = time.perf_counter()
            processed_action = self.postprocessor(single_action)
            t_action_post_done = time.perf_counter()
            self.logger.debug(
                "Postprocess action[%s/%s] time: %.2fms",
                i + 1,
                chunk_size,
                self._ms(t_action_post_done - t_action_post_start),
            )
            processed_actions.append(processed_action)

        # Stack back to (B, chunk_size, action_dim), then remove batch dim
        action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)
        self.logger.debug(f"Postprocessed action shape: {action_tensor.shape}")

        """5. Convert to TimedAction list"""
        t_time_chunk_start = time.perf_counter()
        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep()
        )
        t_time_chunk_done = time.perf_counter()
        postprocess_stops = time.perf_counter()
        postprocessing_time = postprocess_stops - start_postprocess

        self.logger.debug(
            f"Observation {observation_t.get_timestep()} | "
            f"Prepare time: {1000 * prepare_time:.2f}ms | "
            f"Preprocessing time: {1000 * preprocessing_time:.2f}ms | "
            f"Inference time: {1000 * inference_time:.2f}ms | "
            f"Postprocessing time: {1000 * postprocessing_time:.2f}ms | "
            f"Timing chunk time: {1000 * (t_time_chunk_done - t_time_chunk_start):.2f}ms | "
            f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )

        return action_chunk

    def stop(self):
        """Stop the server"""
        self._reset_server()
        self.logger.info("Server stopping...")


@draccus.wrap()
def serve(cfg: PolicyServerConfig):
    """Start the PolicyServer with the given configuration.

    Args:
        config: PolicyServerConfig instance. If None, uses default configuration.
    """
    logging.info(pformat(asdict(cfg)))

    # Create the server instance first
    policy_server = PolicyServer(cfg)

    # Setup and start gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    policy_server.logger.info(f"PolicyServer started on {cfg.host}:{cfg.port}")
    server.start()

    server.wait_for_termination()

    policy_server.logger.info("Server terminated")


if __name__ == "__main__":
    serve()


def _decode_images_from_transport(observation: Any) -> tuple[Any, dict[str, int]]:
    """Recursively decode JPEG-marked images back into uint8 HWC3 RGB numpy arrays."""
    stats = {"images_decoded": 0, "raw_bytes_total": 0, "encoded_bytes_total": 0}

    def _maybe_decode_payload(x: Any) -> Any:
        if isinstance(x, dict) and x.get("__lerobot_image_encoding__") == "jpeg":
            data = x.get("data")
            if not isinstance(data, (bytes, bytearray)):
                raise TypeError("JPEG payload missing bytes 'data'")

            buf = np.frombuffer(data, dtype=np.uint8)
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError("OpenCV failed to decode JPEG payload")

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            stats["images_decoded"] += 1
            stats["encoded_bytes_total"] += len(data)
            stats["raw_bytes_total"] += int(rgb.nbytes)
            return rgb

        if isinstance(x, dict):
            return {k: _maybe_decode_payload(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_maybe_decode_payload(v) for v in x]
        if isinstance(x, tuple):
            return tuple(_maybe_decode_payload(v) for v in x)
        return x

    return _maybe_decode_payload(observation), stats
