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
import threading
from typing import Any

from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.teleoperators.config import RemoteTeleoperatorConfig
from lerobot.common.teleoperators.teleoperator import Teleoperator
from lerobot.common.transport.livekit_service import LiveKitService, LiveKitServiceHandler

try:
    from livekit import rtc
    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False

logger = logging.getLogger(__name__)


class RemoteTeleoperatorHandler(LiveKitServiceHandler):
    """
    Handler for RemoteTeleoperator LiveKit events.
    """
    
    def __init__(self, teleoperator: 'RemoteTeleoperator'):
        self.teleoperator = teleoperator
    
    def on_data_received(self, data: rtc.DataPacket) -> None:
        """
        Handle incoming action messages from LiveKit data channel.
        """
        try:
            # Check if this is the teleop_action topic
            if data.topic == "teleop_action":
                # Decode the JSON action message
                action_data = json.loads(data.data.decode('utf-8'))
                self.teleoperator._handle_action_message(action_data)
                
        except Exception as e:
            logger.error(f"Error processing data packet: {e}")


class RemoteTeleoperator(Teleoperator):
    """
    Remote teleoperator via WebRTC.
    
    This teleoperator enables receiving actions via WebRTC data channels,
    allowing operators to control robots over the internet with low latency.
    """
    
    config_class = RemoteTeleoperatorConfig
    
    def __init__(self, config: RemoteTeleoperatorConfig):
        super().__init__(config)
        
        if not LIVEKIT_AVAILABLE:
            raise ImportError("LiveKit SDK is required. Install with: pip install livekit")
            
        self.config = config
        
        # Create handler and LiveKit service
        self._handler = RemoteTeleoperatorHandler(self)
        self._livekit_service = LiveKitService(
            config.livekit_url,
            config.livekit_token,
            self._handler
        )
        
        # Action caching and validation
        self._cached_action: dict[str, Any] | None = None
        self._expected_action_shape: dict[str, type] | None = None
        self._action_lock = threading.Lock()

    @property
    def is_connected(self) -> bool:
        """
        Whether the teleoperator is currently connected or not.
        """
        return self._livekit_service.is_connected

    def connect(self, calibrate: bool = True) -> None:
        """
        Establish communication with the LiveKit server.
        
        Args:
            calibrate (bool): not used
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Get action features to determine expected message shape
        self._expected_action_shape = self.action_features.copy()
        
        # Connect using the LiveKit service
        try:
            self._livekit_service.connect(timeout=10.0)
            logger.info(f"{self} connected to LiveKit server")
        except ConnectionError as e:
            raise ConnectionError(f"Failed to connect {self}: {e}")

    def _handle_action_message(self, action_data: dict[str, Any]) -> None:
        """
        Handle incoming action messages and validate them.
        
        Args:
            action_data: The received action data dictionary
        """
        try:
            # Validate message shape against expected action features
            if self._expected_action_shape:
                self._validate_action_shape(action_data)
            
            # Cache the latest valid action
            with self._action_lock:
                self._cached_action = action_data.copy()
                
            logger.debug(f"Received valid action: {action_data}")
            
        except Exception as e:
            logger.error(f"Invalid action message received: {e}")

    def _validate_action_shape(self, action_data: dict[str, Any]) -> None:
        """
        Validate that the action message matches the expected shape.
        
        Args:
            action_data: The action data to validate
            
        Raises:
            ValueError: If the action shape doesn't match expectations
        """
        expected_keys = set(self._expected_action_shape.keys())
        received_keys = set(action_data.keys())
        
        if expected_keys != received_keys:
            missing_keys = expected_keys - received_keys
            extra_keys = received_keys - expected_keys
            
            error_msg = "Action message shape mismatch between leader and follower."
            if missing_keys:
                error_msg += f" Missing keys: {missing_keys}."
            if extra_keys:
                error_msg += f" Extra keys: {extra_keys}."
            error_msg += f" Expected: {list(expected_keys)}, Received: {list(received_keys)}"
            
            raise ValueError(error_msg)
        
        # Validate data types
        for key, expected_type in self._expected_action_shape.items():
            if key in action_data:
                received_value = action_data[key]
                if not isinstance(received_value, expected_type):
                    raise ValueError(
                        f"Type mismatch for key '{key}': expected {expected_type.__name__}, "
                        f"got {type(received_value).__name__}"
                    )

    def get_action(self) -> dict[str, Any]:
        """
        Retrieve the current action from the teleoperator.
        
        Returns:
            dict[str, Any]: A flat dictionary representing the teleoperator's current actions.
            
        Raises:
            DeviceNotConnectedError: If the teleoperator is not connected
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
            
        with self._action_lock:
            if self._cached_action is None:
                # Return a default action with zeros if no action received yet
                if self._expected_action_shape:
                    return {key: 0.0 for key in self._expected_action_shape.keys()}
                else:
                    return {}
            return self._cached_action.copy()

    def disconnect(self) -> None:
        """
        Disconnect from the LiveKit server and perform any necessary cleanup.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        logger.info(f"Disconnecting {self} from LiveKit server")
        
        # Disconnect using the LiveKit service
        try:
            self._livekit_service.disconnect()
        except RuntimeError:
            pass  # Already disconnected
        
        # Clean up state
        self._cached_action = None
        self._expected_action_shape = None
        
        logger.info(f"{self} disconnected from LiveKit server") 