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

import io
import logging
import os
import pickle  # nosec
import threading
import time
from concurrent import futures
from dataclasses import asdict
from pprint import pformat
from queue import Empty, Queue
from typing import Any
from urllib.parse import quote

import draccus
import grpc
import torch

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency in some server envs
    np = None

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, Response
    import uvicorn
except ImportError:  # pragma: no cover - policy server can still run without the API
    FastAPI = None
    HTTPException = None
    JSONResponse = None
    Response = None
    uvicorn = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional fallback; OpenCV may still be available
    Image = None

try:
    import cv2
except ImportError:  # pragma: no cover - optional fallback; Pillow may still be available
    cv2 = None

from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import receive_bytes_in_chunks
from lerobot.types import PolicyAction

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


def _is_image_like(value: Any) -> bool:
    """Cheap check for camera-like arrays/tensors in raw observations."""
    shape = None
    if isinstance(value, torch.Tensor):
        shape = tuple(value.shape)
    elif np is not None and isinstance(value, np.ndarray):
        shape = value.shape
    elif isinstance(value, (bytes, bytearray)):
        return True

    if shape is None:
        return False

    while len(shape) == 4 and shape[0] == 1:
        shape = shape[1:]

    if len(shape) == 2:
        return shape[0] >= 16 and shape[1] >= 16

    if len(shape) == 3:
        return (
            shape[0] >= 16
            and shape[1] >= 16
            and shape[2] in (1, 3, 4)
        ) or (
            shape[1] >= 16
            and shape[2] >= 16
            and shape[0] in (1, 3, 4)
        )

    return False


def _to_uint8_hwc_image(value: Any):
    """Return an image-like value as a uint8 HWC/gray numpy array, or None."""
    if np is None:
        return None

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    if not isinstance(value, np.ndarray):
        return None

    image = value
    while image.ndim == 4 and image.shape[0] == 1:
        image = image[0]

    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)  # CHW -> HWC

    if image.ndim not in (2, 3):
        return None

    if image.ndim == 3 and image.shape[-1] not in (1, 3, 4):
        return None

    # Avoid mistaking small vectors/matrices for camera images.
    if image.shape[0] < 16 or image.shape[1] < 16:
        return None

    if image.dtype != np.uint8:
        image = image.astype(np.float32, copy=False)
        finite = image[np.isfinite(image)]
        if finite.size == 0:
            image = np.zeros_like(image, dtype=np.uint8)
        else:
            min_value = float(finite.min())
            max_value = float(finite.max())
            if 0.0 <= min_value and max_value <= 1.0:
                image = image * 255.0
            elif min_value < 0.0 or max_value > 255.0:
                denom = max(max_value - min_value, 1e-6)
                image = (image - min_value) * (255.0 / denom)
            image = np.clip(image, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(image)


def _encode_image(value: Any) -> tuple[bytes, str, list[int]] | None:
    """Encode a raw observation image value as bytes for the FastAPI endpoint."""
    if isinstance(value, (bytes, bytearray)):
        return bytes(value), "application/octet-stream", [len(value)]

    image = _to_uint8_hwc_image(value)
    if image is None:
        return None

    if cv2 is not None:
        image_for_cv2 = image
        if image.ndim == 3 and image.shape[-1] == 3:
            image_for_cv2 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif image.ndim == 3 and image.shape[-1] == 4:
            image_for_cv2 = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
        ok, encoded = cv2.imencode(".jpg", image_for_cv2)
        if ok:
            return encoded.tobytes(), "image/jpeg", list(image.shape)

    if Image is not None:
        pil_image = Image.fromarray(image)
        if pil_image.mode not in ("RGB", "L"):
            pil_image = pil_image.convert("RGB")
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        return buffer.getvalue(), "image/jpeg", list(image.shape)

    return None


def _summarize_value(value: Any) -> Any:
    """Keep /observation JSON small while still exposing useful raw_obs metadata."""
    if isinstance(value, torch.Tensor):
        return {
            "type": "torch.Tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if np is not None and isinstance(value, np.ndarray):
        return {
            "type": "np.ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        if len(value) <= 16 and all(isinstance(x, (str, int, float, bool, type(None))) for x in value):
            return list(value)
        return {"type": type(value).__name__, "length": len(value)}
    if isinstance(value, dict):
        return {str(k): _summarize_value(v) for k, v in value.items()}
    return {"type": type(value).__name__}


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

        # Latest raw observation snapshot for the FastAPI publisher.
        self.latest_observation_lock = threading.Lock()
        self.latest_observation: dict[str, Any] | None = None
        self.latest_images: dict[str, dict[str, Any]] = {}
        self.latest_client_timestamp: float | None = None
        self.latest_server_timestamp: float | None = None
        self.latest_timestep: int | None = None

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

        with self.latest_observation_lock:
            self.latest_observation = None
            self.latest_images = {}
            self.latest_client_timestamp = None
            self.latest_server_timestamp = None
            self.latest_timestep = None

    def Ready(self, request, context):  # noqa: N802
        client_id = context.peer()
        self.logger.info(f"Client {client_id} connected and ready")
        self._reset_server()
        self.shutdown_event.clear()

        return services_pb2.Empty()

    def _update_latest_observation_snapshot(
        self,
        timestep: int,
        client_timestamp: float,
        server_timestamp: float,
        raw_observation: dict[str, Any],
    ) -> None:
        """Publish the newest raw observation for FastAPI readers.

        This keeps the gRPC hot path simple: store the latest timestamp and raw_obs;
        image encoding happens only when an HTTP client requests an image.
        """
        image_keys = []
        if isinstance(raw_observation, dict):
            for key, value in raw_observation.items():
                if _is_image_like(value):
                    image_keys.append(str(key))

        with self.latest_observation_lock:
            self.latest_observation = dict(raw_observation)
            self.latest_images = {key: {} for key in image_keys}
            self.latest_client_timestamp = client_timestamp
            self.latest_server_timestamp = server_timestamp
            self.latest_timestep = timestep

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Receive policy instructions from the robot client"""

        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        client_id = context.peer()

        policy_specs = pickle.loads(request.data)  # nosec

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

        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type  # act, pi0, etc.
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk

        policy_class = get_policy_class(self.policy_type)

        start = time.perf_counter()
        self.policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)
        self.policy.to(self.device)

        # Load preprocessor and postprocessor, overriding device to match requested device
        device_override = {"device": self.device}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=policy_specs.pretrained_name_or_path,
            preprocessor_overrides={
                "device_processor": device_override,
                "rename_observations_processor": {"rename_map": policy_specs.rename_map},
            },
            postprocessor_overrides={"device_processor": device_override},
        )

        end = time.perf_counter()

        self.logger.info(f"Time taken to put policy on {self.device}: {end - start:.4f} seconds")

        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        """Receive observations from the robot client"""
        client_id = context.peer()
        self.logger.debug(f"Receiving observations from {client_id}")

        receive_time = time.time()  # comparing timestamps so need time.time()
        start_deserialize = time.perf_counter()
        received_bytes = receive_bytes_in_chunks(
            request_iterator, None, self.shutdown_event, self.logger
        )  # blocking call while looping over request_iterator
        timed_observation = pickle.loads(received_bytes)  # nosec
        deserialize_time = time.perf_counter() - start_deserialize

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
            f"Deserialization time: {deserialize_time:.6f}s"
        )

        self._update_latest_observation_snapshot(
            timestep=obs_timestep,
            client_timestamp=obs_timestamp,
            server_timestamp=receive_time,
            raw_observation=timed_observation.get_observation(),
        )

        if not self._enqueue_observation(
            timed_observation  # wrapping a RawObservation
        ):
            self.logger.debug(f"Observation #{obs_timestep} has been filtered out")

        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        """Returns actions to the robot client. Actions are sent as a single
        chunk, containing multiple actions."""
        client_id = context.peer()
        self.logger.debug(f"Client {client_id} connected for action streaming")

        # Generate action based on the most recent observation and its timestep
        try:
            getactions_starts = time.perf_counter()
            obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)
            self.logger.info(
                f"Running inference for observation #{obs.get_timestep()} (must_go: {obs.must_go})"
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
                f"Inference time: {inference_time:.2f}s |"
                f"Serialize time: {serialize_time:.2f}s |"
                f"Total time: {inference_time + serialize_time:.2f}s"
            )

            time.sleep(
                max(0, self.config.inference_latency - max(0, time.perf_counter() - getactions_starts))
            )  # sleep controls inference latency

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
            TimedAction(timestamp=t_0 + i * self.config.environment_dt, timestep=i_0 + i, action=action)
            for i, action in enumerate(action_chunk)
        ]

    def _get_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """Get an action chunk from the policy. The chunk contains only"""
        chunk = self.policy.predict_action_chunk(observation)
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
        self.logger.info(
            f"Preprocessing and inference took {inference_time:.4f}s, action shape: {action_tensor.shape}"
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
            processed_action = self.postprocessor(single_action)
            processed_actions.append(processed_action)

        # Stack back to (B, chunk_size, action_dim), then remove batch dim
        action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)
        self.logger.debug(f"Postprocessed action shape: {action_tensor.shape}")

        action_tensor = action_tensor.detach().cpu()

        """5. Convert to TimedAction list"""
        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep()
        )
        postprocess_stops = time.perf_counter()
        postprocessing_time = postprocess_stops - start_postprocess

        self.logger.info(
            f"Observation {observation_t.get_timestep()} | "
            f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )

        self.logger.debug(
            f"Observation {observation_t.get_timestep()} | "
            f"Prepare time: {1000 * prepare_time:.2f}ms | "
            f"Preprocessing time: {1000 * preprocessing_time:.2f}ms | "
            f"Inference time: {1000 * inference_time:.2f}ms | "
            f"Postprocessing time: {1000 * postprocessing_time:.2f}ms | "
            f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )

        return action_chunk

    def get_latest_snapshot(self) -> dict[str, Any]:
        with self.latest_observation_lock:
            raw_observation = self.latest_observation
            return {
                "timestep": self.latest_timestep,
                "client_timestamp": self.latest_client_timestamp,
                "server_timestamp": self.latest_server_timestamp,
                "observation": dict(raw_observation) if raw_observation is not None else None,
                "image_keys": list(self.latest_images.keys()),
            }

    def stop(self):
        """Stop the server"""
        self._reset_server()
        self.logger.info("Server stopping...")


def create_observation_api(policy_server: PolicyServer):
    if FastAPI is None:
        return None

    app = FastAPI(title="LeRobot Policy Server Observation API")

    @app.get("/")
    def root():
        return {
            "status": "/status",
            "observation": "/observation",
            "images": "/images",
            "first_image": "/image",
        }

    @app.get("/status")
    def status():
        snapshot = policy_server.get_latest_snapshot()
        return {
            "has_observation": snapshot["observation"] is not None,
            "timestep": snapshot["timestep"],
            "client_timestamp": snapshot["client_timestamp"],
            "server_timestamp": snapshot["server_timestamp"],
            "image_keys": snapshot["image_keys"],
            "observation_keys": list(snapshot["observation"].keys()) if snapshot["observation"] else [],
        }

    @app.get("/observation")
    def observation():
        snapshot = policy_server.get_latest_snapshot()
        if snapshot["observation"] is None:
            raise HTTPException(status_code=404, detail="No observation has been received yet")
        return JSONResponse(
            {
                "timestep": snapshot["timestep"],
                "client_timestamp": snapshot["client_timestamp"],
                "server_timestamp": snapshot["server_timestamp"],
                "image_keys": snapshot["image_keys"],
                "observation": {
                    key: _summarize_value(value) for key, value in snapshot["observation"].items()
                },
            }
        )

    @app.get("/images")
    def images():
        snapshot = policy_server.get_latest_snapshot()
        return {
            "timestep": snapshot["timestep"],
            "client_timestamp": snapshot["client_timestamp"],
            "server_timestamp": snapshot["server_timestamp"],
            "image_keys": snapshot["image_keys"],
            "urls": {key: f"/images/{quote(key, safe='')}" for key in snapshot["image_keys"]},
        }

    def _get_image_response(key: str | None = None):
        snapshot = policy_server.get_latest_snapshot()
        observation = snapshot["observation"]
        image_keys = snapshot["image_keys"]
        if observation is None:
            raise HTTPException(status_code=404, detail="No observation has been received yet")
        if not image_keys:
            raise HTTPException(status_code=404, detail="No image-like values found in the latest raw observation")

        selected_key = key or image_keys[0]
        if selected_key not in observation:
            raise HTTPException(status_code=404, detail=f"No image found for key '{selected_key}'")

        encoded = _encode_image(observation[selected_key])
        if encoded is None:
            raise HTTPException(status_code=415, detail=f"Value for key '{selected_key}' could not be encoded as an image")

        data, media_type, shape = encoded
        return Response(
            content=data,
            media_type=media_type,
            headers={
                "X-LeRobot-Image-Key": selected_key,
                "X-LeRobot-Image-Shape": str(shape),
                "X-LeRobot-Timestep": str(snapshot["timestep"]),
                "X-LeRobot-Client-Timestamp": str(snapshot["client_timestamp"]),
                "X-LeRobot-Server-Timestamp": str(snapshot["server_timestamp"]),
            },
        )

    @app.get("/image")
    def first_image():
        return _get_image_response()

    @app.get("/images/{key}")
    def image_by_key(key: str):
        return _get_image_response(key)

    return app


def start_observation_api(policy_server: PolicyServer, cfg: PolicyServerConfig) -> None:
    if FastAPI is None or uvicorn is None:
        policy_server.logger.warning("FastAPI/uvicorn is not installed; observation API is disabled")
        return

    http_port = getattr(cfg, "http_port", 8001)
    if http_port is None:
        policy_server.logger.info("Observation API disabled because http_port is None")
        return

    http_host = os.environ.get("LEROBOT_OBS_API_HOST", str(getattr(cfg, "http_host", "0.0.0.0")))
    ssl_certfile = os.environ.get("LEROBOT_OBS_API_SSL_CERTFILE")
    ssl_keyfile = os.environ.get("LEROBOT_OBS_API_SSL_KEYFILE")
    scheme = "https" if ssl_certfile and ssl_keyfile else "http"

    app = create_observation_api(policy_server)
    config = uvicorn.Config(
        app,
        host=http_host,
        port=int(http_port),
        log_level="warning",
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
    )
    api_server = uvicorn.Server(config)
    api_thread = threading.Thread(target=api_server.run, name="observation_api", daemon=True)
    api_thread.start()
    policy_server.logger.info(f"Observation API started at {scheme}://{http_host}:{http_port}")


@draccus.wrap()
def serve(cfg: PolicyServerConfig):
    """Start the PolicyServer with the given configuration.

    Args:
        config: PolicyServerConfig instance. If None, uses default configuration.
    """
    logging.info(pformat(asdict(cfg)))

    # Create the server instance first
    policy_server = PolicyServer(cfg)
    start_observation_api(policy_server, cfg)

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
