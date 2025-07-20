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
python src/lerobot/scripts/server/policy_server.py \
     --host=127.0.0.1 \
     --port=8080
```
"""

import logging
import pickle
import threading
import time
from concurrent import futures
from dataclasses import asdict
from pprint import pformat

import draccus
import grpc
import torch

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import AsyncStats
from lerobot.policies.factory import get_policy_class
from lerobot.scripts.server.configs import PolicyServerConfig
from lerobot.scripts.server.constants import SUPPORTED_POLICIES
from lerobot.scripts.server.helpers import (
    Observation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    raw_observation_to_observation,
)
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import receive_bytes_in_chunks


class PolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    prefix = "policy_server"
    logger = get_logger(prefix)

    def __init__(self, config: PolicyServerConfig):
        self.config = config
        self.shutdown_event = threading.Event()

        # Attributes will be set by SendPolicyInstructions
        self.policy = None

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

    def Ready(self, request, context):  # noqa: N802
        client_id = context.peer()
        self.logger.info(f"Client {client_id} connected and ready")
        self._reset_server()
        self.shutdown_event.clear()

        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Receive policy instructions from the robot client"""

        start = time.perf_counter()

        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        client_id = context.peer()

        policy_setup: RemotePolicyConfig = pickle.loads(request.data)  # nosec

        self.logger.info(
            f"Receiving policy instructions from {client_id}"
            f" | Server args: {policy_setup.server_args}"
            f" | Actions per chunk: {policy_setup.actions_per_chunk}"
        )

        policy_path = parser.get_path_arg("policy", args=policy_setup.server_args)
        cli_overrides = parser.get_cli_overrides("policy", args=policy_setup.server_args)
        policy_config = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)

        if policy_config.type not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Policy type '{policy_config.type}' not supported. Supported policies: {SUPPORTED_POLICIES}"
            )

        policy_class = get_policy_class(policy_config.type)
        self.policy = policy_class.from_pretrained(policy_path, config=policy_config)

        self.lerobot_features = policy_setup.lerobot_features
        self.actions_per_chunk = policy_setup.actions_per_chunk
        self.last_processed_obs = None

        end = time.perf_counter()

        self.logger.info(f"Policy loaded on {self.policy.config.device} in {end - start:.4f} seconds")

        return services_pb2.Empty()

    def GetActions(self, request_iterator, context):  # noqa: N802
        """Returns actions to the robot client. Actions are sent as a single
        chunk, containing multiple actions."""
        client_id = context.peer()
        try:
            self.logger.debug(f"Client {client_id} connected for action streaming")

            receive_start = time.perf_counter()
            received_bytes = receive_bytes_in_chunks(
                request_iterator, None, self.shutdown_event, PolicyServer.prefix
            )
            receive_end = time.perf_counter()
            unpack_start = receive_end
            obs = pickle.loads(received_bytes)  # nosec
            unpack_end = time.perf_counter()
            predict_start = unpack_end
            action_chunk = self._predict_action_chunk(obs)
            predict_end = time.perf_counter()
            pack_start = predict_end
            actions_bytes = pickle.dumps(action_chunk)  # nosec
            pack_end = time.perf_counter()
            actions = services_pb2.Actions(data=actions_bytes)
            send_start = pack_end
            send_end = time.perf_counter()
            total_time = send_end - receive_start

            self.logger.info(
                f"Observation {obs.get_timestep()}"
                f" | Receive: {receive_end - receive_start:.3f}s"
                f" | Unpack: {unpack_end - unpack_start:.3f}s"
                f" | Predict: {predict_end - predict_start:.3f}s"
                f" | Pack: {pack_end - pack_start:.3f}s"
                f" | Send: {send_end - send_start:.3f}s"
                f" | Total: {total_time:.3f}s"
            )

            return actions
        except Exception as e:
            self.logger.error(f"Error processing observation from client {client_id}: {str(e)}")
            import traceback

            self.logger.error(traceback.format_exc())
            raise

    def _time_action_chunk(self, i_0: int, action_chunk: list[torch.Tensor]) -> list[TimedAction]:
        """Turn a chunk of actions into a list of TimedAction instances,
        with the first action corresponding to t_0 and the rest corresponding to
        t_0 + i*environment_dt for i in range(len(action_chunk))
        """
        return [TimedAction(timestep=i_0 + i, action=action) for i, action in enumerate(action_chunk)]

    def _prepare_observation(self, observation_t: TimedObservation) -> Observation:
        """
        Prepare observation, ready for policy inference.
        E.g.: To keep observation sampling rate high (and network packet tiny) we send int8 [0,255] images from the
        client and then convert them to float32 [0,1] images here, before running inference.
        """
        # RawObservation from robot.get_observation() - wrong keys, wrong dtype, wrong image shape
        observation: Observation = raw_observation_to_observation(
            observation_t.get_observation(),
            self.lerobot_features,
            self.policy_image_features,
            self.policy.config.device,
        )
        # processed Observation - right keys, right dtype, right image shape

        return observation

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        """Predict an action chunk based on an observation"""
        step_start = time.perf_counter()

        """1. Prepare observation"""
        preprocessing_start = time.perf_counter()
        observation = self._prepare_observation(observation_t)
        inference_latency_steps = observation_t.get_inference_latency_steps()

        if self.last_processed_obs is None:
            async_stats = AsyncStats(
                steps_since_last_chunk_start=0,
                inference_latency_steps=inference_latency_steps,
            )
        else:
            async_stats = AsyncStats(
                steps_since_last_chunk_start=observation_t.get_timestep()
                - self.last_processed_obs.get_timestep(),
                inference_latency_steps=inference_latency_steps,
            )

        self.last_processed_obs: TimedObservation = observation_t
        preprocessing_end = time.perf_counter()

        """2. Get action tensor"""
        inference_start = preprocessing_end
        action_tensor = self.policy.predict_action_chunk(observation, async_stats=async_stats)
        if action_tensor.ndim != 3:
            action_tensor = action_tensor.unsqueeze(
                0
            )  # adding batch dimension, now shape is (B, chunk_size, action_dim)
        if action_tensor.shape[1] != self.actions_per_chunk:
            raise ValueError(
                f"Expected action tensor to have {self.actions_per_chunk} actions, got {action_tensor.shape[1]}. {action_tensor.shape=}"
            )

        inference_end = time.perf_counter()

        """3. Post-inference processing"""
        postprocessing_start = inference_end
        # Move to CPU before serializing
        action_tensor = action_tensor.cpu().squeeze(0)  # remove the first dimension

        action_chunk = self._time_action_chunk(observation_t.get_timestep(), list(action_tensor))
        postprocessing_end = time.perf_counter()
        total_time = postprocessing_end - step_start

        self.logger.info(
            f"Observation {observation_t.get_timestep()}"
            f" | Preprocessing: {(preprocessing_end - preprocessing_start):.3f}s"
            f" | Inference: {(inference_end - inference_start):.3f}s"
            f" | Postprocessing: {(postprocessing_end - postprocessing_start):.3f}s"
            f" | Total: {(total_time):.3f}s"
            f" | {async_stats=}"
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
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1), compression=grpc.Compression.Deflate)
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    policy_server.logger.info(f"PolicyServer started on {cfg.host}:{cfg.port}")
    server.start()

    server.wait_for_termination()

    policy_server.logger.info("Server terminated")


if __name__ == "__main__":
    serve()
