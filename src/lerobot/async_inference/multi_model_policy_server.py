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
Multi-Model Policy Server that launches multiple policy servers with pre-loaded models on different ports.

Example:
```shell
python src/lerobot/scripts/server/multi_model_policy_server.py \
     --host=127.0.0.1 \
     --base_port=8080 \
     --policy_paths="outputs/train/smolvla_pick_knife/checkpoints/last/pretrained_model,outputs/train/smolvla_place_left/checkpoints/last/pretrained_model,outputs/train/act_pick/checkpoints/last/pretrained_model,outputs/train/pi0_place/checkpoints/last/pretrained_model" \
     --robot_port=/dev/ttyACM1 \
     --robot_id=juanito_follower_arm \
     --robot_cameras="{ up: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
     --fps=30 \
     --inference_latency=0.033 \
     --obs_queue_timeout=1
```

This will launch 4 servers on ports 8080, 8081, 8082, 8083 with the respective pre-loaded models.
"""

import logging
import multiprocessing
import signal
import sys
import threading
import time
from concurrent import futures
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import List

import draccus
import grpc
import torch

from lerobot.policies.factory import get_policy_class
from lerobot.scripts.server.configs import PolicyServerConfig
from lerobot.scripts.server.helpers import get_logger
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)


@dataclass
class MultiModelPolicyServerConfig:
    # Host to bind the servers to
    host: str = "127.0.0.1"
    # Base port number (servers will use base_port, base_port+1, base_port+2, etc.)
    base_port: int = 8080
    # Policy model paths separated by commas (e.g., "path1,path2,path3")
    policy_paths: str = ""
    # Default robot configuration shared across all servers
    robot_port: str = "/dev/ttyACM1"
    robot_id: str = "follower_arm"
    robot_cameras: str = '{ up: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}'
    # Server configuration
    fps: int = 30
    inference_latency: float = 0.033
    obs_queue_timeout: float = 1.0
    # Device to load models on
    device: str = "cuda"

    def __post_init__(self):
        # Convert comma-separated string to list
        if isinstance(self.policy_paths, str):
            if self.policy_paths:
                self.policy_paths = [path.strip() for path in self.policy_paths.split(",")]
            else:
                self.policy_paths = []
        
        # Validate that policy paths exist
        for i, policy_path in enumerate(self.policy_paths):
            if not Path(policy_path).exists():
                logging.warning(f"Policy path {i+1} does not exist: {policy_path}")


class PreLoadedPolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    """Policy server with pre-loaded model using synchronous inference."""
    
    def __init__(self, config: PolicyServerConfig, policy_path: str, robot_config: dict, device: str = "cuda"):
        self.config = config
        self.policy_path = policy_path
        self.robot_config = robot_config  # Store robot config for lerobot_features
        self.device = device
        self.shutdown_event = threading.Event()
        self.logger = get_logger(f"PreLoadedPolicyServer-{config.port}")
        
        # Load the policy model immediately
        self.policy = self._load_policy()
        
        # Initialize lerobot_features from robot config
        self._init_lerobot_features()
        
        # Initialize FPS tracker for logging
        from lerobot.scripts.server.helpers import FPSTracker
        self.fps_tracker = FPSTracker(target_fps=config.fps)
        
        # No queues needed for synchronous operation
        self.current_observation = None
        self.observation_lock = threading.Lock()
        
        # Track episode state for automatic reset
        self.last_observation_time = 0
        self.episode_reset_timeout = 2.0  # Reset if no observation for 2 seconds
        self.needs_reset = True  # Start with reset needed
        
    def _init_lerobot_features(self):
        """Initialize lerobot_features from robot config."""
        from lerobot.scripts.server.helpers import map_robot_keys_to_lerobot_features
        from lerobot.robots import make_robot_from_config
        from lerobot.robots.so101_follower import SO101FollowerConfig
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        
        # Create robot config object
        robot_config = SO101FollowerConfig(
            port=self.robot_config["port"],
            id=self.robot_config["id"],
            cameras={
                name: OpenCVCameraConfig(
                    index_or_path=cam_config["index_or_path"],
                    width=cam_config["width"],
                    height=cam_config["height"],
                    fps=cam_config["fps"]
                )
                for name, cam_config in self.robot_config["cameras"].items()
            }
        )
        
        # Create temporary robot to get features (without connecting)
        temp_robot = make_robot_from_config(robot_config)
        self.lerobot_features = map_robot_keys_to_lerobot_features(temp_robot)
        
    def _load_policy(self):
        """Load and initialize the policy model."""
        self.logger.info(f"Loading policy from: {self.policy_path}")
        
        try:
            # Detect policy type from the model path
            policy_type = self._detect_policy_type(self.policy_path)
            self.logger.info(f"Detected policy type: {policy_type}")
            
            # Load the policy
            policy_class = get_policy_class(policy_type)
            policy = policy_class.from_pretrained(self.policy_path)
            policy.to(self.device)
            
            self.logger.info(f"Successfully loaded {policy_type} policy on {self.device}")
            return policy
            
        except Exception as e:
            self.logger.error(f"Failed to load policy from {self.policy_path}: {e}")
            raise
    
    def _detect_policy_type(self, policy_path: str) -> str:
        """Detect policy type from the model path."""
        path_lower = policy_path.lower()
        
        if "smolvla" in path_lower:
            return "smolvla"
        elif "act" in path_lower:
            return "act"
        elif "pi0" in path_lower:
            return "pi0"
        elif "diffusion" in path_lower:
            return "diffusion"
        elif "vqbet" in path_lower:
            return "vqbet"
        elif "tdmpc" in path_lower:
            return "tdmpc"
        else:
            # Default to smolvla if can't detect
            self.logger.warning(f"Could not detect policy type from path: {policy_path}. Defaulting to 'smolvla'")
            return "smolvla"
    
    @property
    def running(self):
        return not self.shutdown_event.is_set()
    
    @property
    def policy_image_features(self):
        return self.policy.config.image_features
    
    def Ready(self, request, context):  # noqa: N802
        client_id = context.peer()
        self.logger.info(f"Client {client_id} connected and ready. Policy already loaded: {self.policy_path}")
        return services_pb2.Empty()
    
    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Policy is already loaded, so just acknowledge the request."""
        client_id = context.peer()
        self.logger.info(f"Policy instructions received from {client_id}. Using pre-loaded policy: {self.policy_path}")
        return services_pb2.Empty()
    
    def SendObservations(self, request_iterator, context):  # noqa: N802
        """Receive observations from the robot client - synchronous version."""
        # Import the required components here to avoid circular imports
        from lerobot.scripts.server.helpers import TimedObservation
        from lerobot.transport.utils import receive_bytes_in_chunks
        import pickle
        
        client_id = context.peer()
        self.logger.debug(f"Receiving observation from {client_id}")

        receive_time = time.time()
        start_deserialize = time.perf_counter()
        received_bytes = receive_bytes_in_chunks(
            request_iterator, None, self.shutdown_event, self.logger
        )
        timed_observation = pickle.loads(received_bytes)  # nosec
        deserialize_time = time.perf_counter() - start_deserialize

        obs_timestep = timed_observation.get_timestep()
        obs_timestamp = timed_observation.get_timestamp()

        # Calculate FPS metrics
        fps_metrics = self.fps_tracker.calculate_fps_metrics(obs_timestamp)

        self.logger.info(
            f"Received observation #{obs_timestep} | "
            f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
            f"One-way latency: {(receive_time - obs_timestamp) * 1000:.2f}ms"
        )

        # Check if we need to reset policy state (new episode detection)
        current_time = time.perf_counter()
        time_since_last = current_time - self.last_observation_time
        
        if self.needs_reset or time_since_last > self.episode_reset_timeout:
            self.logger.info(f"Auto-resetting policy state (gap: {time_since_last:.2f}s, needs_reset: {self.needs_reset})")
            self.reset_policy_state()
            self.needs_reset = False
        
        self.last_observation_time = current_time

        # Store the observation for synchronous processing
        with self.observation_lock:
            self.current_observation = timed_observation

        return services_pb2.Empty()
    
    def GetActions(self, request, context):  # noqa: N802
        """Returns a single action to the robot client - synchronous version."""
        import pickle
        
        client_id = context.peer()
        self.logger.debug(f"Client {client_id} requesting action")

        try:
            # Get the current observation synchronously
            with self.observation_lock:
                obs = self.current_observation
                
            if obs is None:
                self.logger.warning("No observation available for inference")
                return services_pb2.Actions(data=b"")

            self.logger.info(f"Running inference for observation #{obs.get_timestep()}")

            start_time = time.perf_counter()
            action = self._predict_single_action(obs)
            inference_time = time.perf_counter() - start_time

            start_time = time.perf_counter()
            # Return single action instead of action chunk
            actions_bytes = pickle.dumps([action])  # Wrap in list for compatibility
            serialize_time = time.perf_counter() - start_time

            actions = services_pb2.Actions(data=actions_bytes)

            self.logger.info(
                f"Single action #{obs.get_timestep()} generated | "
                f"Inference time: {inference_time * 1000:.2f}ms | "
                f"Total time: {(inference_time + serialize_time) * 1000:.2f}ms"
            )

            return actions

        except Exception as e:
            self.logger.error(f"Error in GetActions: {e}")
            return services_pb2.Actions(data=b"")
    
    def _predict_single_action(self, observation_t):
        """Predict a single action using the pre-loaded policy - same as record script."""
        from lerobot.scripts.server.helpers import raw_observation_to_observation, TimedAction
        
        # Prepare observation for inference exactly like record script
        observation = raw_observation_to_observation(
            observation_t.get_observation(),
            self.lerobot_features,  # Use pre-initialized lerobot features
            self.policy_image_features,
            self.device,
        )
        
        # Use select_action() just like the record script does
        with torch.no_grad():
            action_tensor = self.policy.select_action(observation)
            
            # Remove batch dimension just like record script
            action_tensor = action_tensor.squeeze(0)
            
            # Move to CPU if not already there
            action_tensor = action_tensor.to("cpu")
        
        # Create single timed action
        timed_action = TimedAction(
            timestamp=observation_t.get_timestamp(),
            timestep=observation_t.get_timestep(),
            action=action_tensor
        )
        
        return timed_action
    
    def reset_policy_state(self):
        """
        Reset the policy's internal state without reloading weights.
        This clears action queues, hidden states, and memory buffers.
        """
        try:
            self.logger.info("Resetting policy internal state...")
            
            # Reset different types of policy states
            if hasattr(self.policy, 'reset'):
                # Some policies have explicit reset methods
                self.policy.reset()
            
            if hasattr(self.policy, 'action_queue'):
                # Clear action queue for policies that use them (like SmolVLA)
                self.policy.action_queue.clear()
                self.logger.debug("Cleared action queue")
            
            if hasattr(self.policy, '_action_queue'):
                # Alternative action queue naming
                self.policy._action_queue.clear()
                self.logger.debug("Cleared _action_queue")
            
            if hasattr(self.policy, 'hidden_state'):
                # Reset hidden states for RNN-based policies
                self.policy.hidden_state = None
                self.logger.debug("Reset hidden_state")
            
            if hasattr(self.policy, 'memory'):
                # Clear memory buffers
                if hasattr(self.policy.memory, 'clear'):
                    self.policy.memory.clear()
                else:
                    self.policy.memory = None
                self.logger.debug("Cleared memory")
            
            # For transformer-based policies, reset attention caches
            if hasattr(self.policy, 'model'):
                model = self.policy.model
                if hasattr(model, 'clear_cache'):
                    model.clear_cache()
                elif hasattr(model, 'reset_cache'):
                    model.reset_cache()
            
            # Reset any internal timestep counters
            if hasattr(self.policy, 'timestep'):
                self.policy.timestep = 0
            if hasattr(self.policy, 'step_count'):
                self.policy.step_count = 0
                
            self.logger.info("Policy state reset completed")
            
        except Exception as e:
            self.logger.error(f"Error resetting policy state: {e}")
            # Don't raise - just log the error and continue
    
    def stop(self):
        """Stop the server."""
        self.shutdown_event.set()
        self.logger.info("Server stopping...")


class MultiModelPolicyServerManager:
    """Manager class for multiple pre-loaded policy servers."""
    
    def __init__(self, config: MultiModelPolicyServerConfig):
        self.config = config
        self.processes = []
        self.shutdown_event = threading.Event()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("MultiModelPolicyServer")
        
    def _create_server_process(self, port: int, policy_path: str) -> multiprocessing.Process:
        """Create a server process for a specific port and policy."""
        
        def run_server(port: int, policy_path: str, config: MultiModelPolicyServerConfig):
            """Function to run a single policy server in a separate process."""
            # Create policy server config for this specific port
            server_config = PolicyServerConfig(
                host=config.host,
                port=port,
                fps=config.fps,
                inference_latency=config.inference_latency,
                obs_queue_timeout=config.obs_queue_timeout
            )
            
            # Create robot config dict for lerobot_features
            robot_config_dict = {
                "port": config.robot_port,
                "id": config.robot_id,
                "cameras": {
                    "up": {"index_or_path": 0, "width": 640, "height": 480, "fps": 30},
                    "side": {"index_or_path": 2, "width": 640, "height": 480, "fps": 30}
                }
            }
            
            # Setup logging for this process
            logging.basicConfig(
                level=logging.INFO,
                format=f'%(asctime)s - Server-{port} - %(name)s - %(levelname)s - %(message)s'
            )
            logger = logging.getLogger(f"PolicyServer-{port}")
            
            try:
                # Create the pre-loaded policy server instance
                policy_server = PreLoadedPolicyServer(server_config, policy_path, robot_config_dict, config.device)
                
                # Setup and start gRPC server
                server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
                services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
                server.add_insecure_port(f"{config.host}:{port}")
                
                logger.info(f"PolicyServer started on {config.host}:{port} with policy: {policy_path}")
                server.start()
                
                # Keep the server running
                try:
                    server.wait_for_termination()
                except KeyboardInterrupt:
                    logger.info(f"Server on port {port} received shutdown signal")
                finally:
                    policy_server.stop()
                    server.stop(grace=5)
                    logger.info(f"Server on port {port} terminated")
                    
            except Exception as e:
                logger.error(f"Error starting server on port {port} with policy {policy_path}: {e}")
                
        return multiprocessing.Process(target=run_server, args=(port, policy_path, self.config))
    
    def start_servers(self):
        """Start all policy servers with their respective models."""
        num_servers = len(self.config.policy_paths)
        self.logger.info(f"Starting {num_servers} policy servers with pre-loaded models...")
        
        # Create and start a process for each server
        for i, policy_path in enumerate(self.config.policy_paths):
            port = self.config.base_port + i
            process = self._create_server_process(port, policy_path)
            process.start()
            self.processes.append(process)
            self.logger.info(f"Started server {i+1}/{num_servers} on port {port} with policy: {Path(policy_path).name}")
            
        self.logger.info("All servers started successfully!")
        self.logger.info("Server configuration:")
        self.logger.info(f"  Robot Port: {self.config.robot_port}")
        self.logger.info(f"  Robot ID: {self.config.robot_id}")
        self.logger.info(f"  Robot Cameras: {self.config.robot_cameras}")
        self.logger.info("Server endpoints:")
        for i, policy_path in enumerate(self.config.policy_paths):
            port = self.config.base_port + i
            self.logger.info(f"  Server {i+1}: {self.config.host}:{port} -> {Path(policy_path).name}")
            
    def stop_servers(self):
        """Stop all policy servers."""
        self.logger.info("Stopping all servers...")
        self.shutdown_event.set()
        
        # Terminate all processes
        for i, process in enumerate(self.processes):
            if process.is_alive():
                self.logger.info(f"Terminating server process {i+1}")
                process.terminate()
                
        # Wait for all processes to finish
        for i, process in enumerate(self.processes):
            process.join(timeout=10)
            if process.is_alive():
                self.logger.warning(f"Force killing server process {i+1}")
                process.kill()
                process.join()
                
        self.logger.info("All servers stopped")
        
    def wait_for_shutdown(self):
        """Wait for shutdown signal and keep servers running."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down...")
            self.stop_servers()
            sys.exit(0)
            
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # Keep the main thread alive
            while not self.shutdown_event.is_set():
                # Check if any process has died unexpectedly
                for i, process in enumerate(self.processes):
                    if not process.is_alive():
                        self.logger.error(f"Server process {i+1} died unexpectedly")
                        
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt, shutting down...")
            self.stop_servers()


@draccus.wrap()
def serve_multi_models(cfg: MultiModelPolicyServerConfig):
    """Start multiple policy servers with pre-loaded models.

    Args:
        cfg: MultiModelPolicyServerConfig instance containing server configuration.
    """
    logging.info("Multi-Model Policy Server Configuration:")
    logging.info(pformat(asdict(cfg)))
    
    # Validate configuration
    if not cfg.policy_paths:
        raise ValueError("At least one policy path must be provided")
        
    if cfg.base_port <= 0 or cfg.base_port > 65535:
        raise ValueError("Base port must be between 1 and 65535")
        
    num_servers = len(cfg.policy_paths)
    if cfg.base_port + num_servers - 1 > 65535:
        raise ValueError(f"Port range exceeds maximum (base_port + num_servers - 1 = {cfg.base_port + num_servers - 1})")
    
    # Create and start the multi-server manager
    manager = MultiModelPolicyServerManager(cfg)
    
    try:
        manager.start_servers()
        manager.wait_for_shutdown()
    except Exception as e:
        logging.error(f"Error in multi-model policy server: {e}")
        manager.stop_servers()
        raise


if __name__ == "__main__":
    serve_multi_models()
