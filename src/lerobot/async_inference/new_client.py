import asyncio
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, Any, Optional

import grpc
import numpy as np
import torch

# Import LeRobot components
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import send_bytes_in_chunks
from lerobot.scripts.server.helpers import TimedObservation, RawObservation


class PolicyClient:
    """Client for communicating with a single policy server."""
    
    def __init__(self, server_address: str, timeout: float = 30.0):
        """
        Initialize the policy client.
        
        Args:
            server_address: Server address in format "host:port" (e.g., "127.0.0.1:8080")
            timeout: Timeout for gRPC operations in seconds
        """
        self.server_address = server_address
        self.timeout = timeout
        self.channel = None
        self.stub = None
        self.timestep = 0
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"PolicyClient-{server_address}")
        
    def connect(self):
        """Connect to the policy server."""
        try:
            self.channel = grpc.insecure_channel(self.server_address)
            self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
            
            # Send ready signal
            self.stub.Ready(services_pb2.Empty(), timeout=self.timeout)
            self.logger.info(f"Connected to policy server at {self.server_address}")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to server {self.server_address}: {e}")
            raise
            
    def disconnect(self):
        """Disconnect from the policy server."""
        if self.channel:
            self.channel.close()
            self.logger.info(f"Disconnected from {self.server_address}")
            
    def send_observation(self, observation: Dict[str, Any]) -> bool:
        """
        Send an observation to the policy server.
        
        Args:
            observation: Observation dictionary with keys like 'images', 'state', etc.
            
        Returns:
            True if observation was sent successfully
        """
        try:
            # Create timed observation
            timed_obs = TimedObservation(
                timestamp=time.time(),
                timestep=self.timestep,
                observation=RawObservation(observation),
                must_go=False
            )
            
            # Serialize the observation
            obs_bytes = pickle.dumps(timed_obs)
            
            # Send observation to server
            def generate_chunks():
                yield from send_bytes_in_chunks(
                    obs_bytes, 
                    services_pb2.Observation,
                    log_prefix="[CLIENT] Observation",
                    silent=True
                )
                
            response = self.stub.SendObservations(generate_chunks(), timeout=self.timeout)
            
            self.logger.debug(f"Sent observation #{self.timestep}")
            self.timestep += 1
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send observation: {e}")
            return False
            
    def get_actions(self) -> Optional[list]:
        """
        Get action predictions from the policy server.
        
        Returns:
            List of TimedAction objects or None if failed
        """
        try:
            response = self.stub.GetActions(services_pb2.Empty(), timeout=self.timeout)
            
            if response.data:
                # Deserialize the action chunk
                actions = pickle.loads(response.data)
                self.logger.debug(f"Received {len(actions)} actions from server")
                return actions
            else:
                self.logger.warning("Received empty action response")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get actions: {e}")
            return None


class MultiPolicyClient:
    """Client for managing connections to multiple policy servers."""
    
    def __init__(self, server_configs: Dict[str, str]):
        """
        Initialize multi-policy client.
        
        Args:
            server_configs: Dictionary mapping policy names to server addresses
                          e.g., {"pick_knife": "127.0.0.1:8080", "place_left": "127.0.0.1:8081"}
        """
        self.server_configs = server_configs
        self.clients = {}
        self.logger = logging.getLogger("MultiPolicyClient")
        
    def connect_all(self):
        """Connect to all policy servers."""
        for policy_name, server_address in self.server_configs.items():
            try:
                client = PolicyClient(server_address)
                client.connect()
                self.clients[policy_name] = client
                self.logger.info(f"Connected to {policy_name} policy at {server_address}")
            except Exception as e:
                self.logger.error(f"Failed to connect to {policy_name} at {server_address}: {e}")
                
    def disconnect_all(self):
        """Disconnect from all policy servers."""
        for policy_name, client in self.clients.items():
            try:
                client.disconnect()
            except Exception as e:
                self.logger.error(f"Error disconnecting from {policy_name}: {e}")
                
    def get_client(self, policy_name: str) -> Optional[PolicyClient]:
        """Get client for a specific policy."""
        return self.clients.get(policy_name)
    
    def reset_policy(self, policy_name: str):
        """
        Reset a specific policy's internal state.
        This is done by introducing a gap in observations to trigger auto-reset.
        """
        self.logger.info(f"Resetting policy '{policy_name}'...")
        
        client = self.get_client(policy_name)
        if not client:
            self.logger.error(f"No client found for policy: {policy_name}")
            return
        
        # Introduce a small delay to trigger the server's auto-reset mechanism
        # The server will reset if there's a gap > episode_reset_timeout (2s)
        time.sleep(2.1)
        self.logger.info(f"Policy '{policy_name}' should be reset on next observation")
        
    def predict_action(self, policy_name: str, observation: Dict[str, Any]) -> Optional[list]:
        """
        Get action prediction from a specific policy.
        
        Args:
            policy_name: Name of the policy to use
            observation: Observation dictionary
            
        Returns:
            List of predicted actions or None if failed
        """
        client = self.get_client(policy_name)
        if not client:
            self.logger.error(f"No client found for policy: {policy_name}")
            return None
            
        # Send observation and get actions
        if client.send_observation(observation):
            return client.get_actions()
        else:
            return None


def create_dummy_observation(image_shape=(3, 480, 640), state_dim=6):
    """Create a dummy observation for testing."""
    return {
        "images": {
            "up": np.random.randint(0, 255, image_shape, dtype=np.uint8),
            "side": np.random.randint(0, 255, image_shape, dtype=np.uint8),
        },
        "state": np.random.randn(state_dim).astype(np.float32),
    }
