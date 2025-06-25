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

import logging
import threading
from typing import Any

from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.robots.config import RemoteRobotConfig
from lerobot.common.robots.robot import Robot
from lerobot.common.transport.livekit_service import LiveKitService, LiveKitServiceHandler

try:
    from livekit import rtc
    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False

logger = logging.getLogger(__name__)


class RemoteRobotHandler(LiveKitServiceHandler):
    """
    Handler for RemoteRobot LiveKit events.
    """
    
    def __init__(self, robot: 'RemoteRobot'):
        self.robot = robot
    
    def on_data_received(self, data: rtc.DataPacket) -> None:
        """
        Handle data received from LiveKit data channel.
        RemoteRobot typically doesn't receive data, only publishes actions.
        """
        # RemoteRobot primarily publishes actions, doesn't typically receive data
        logger.debug(f"RemoteRobot received data on topic '{data.topic}' (not processed)")


class RemoteRobot(Robot):
    """
    LiveKit-based robot for remote control via WebRTC.
    
    This robot connects to a LiveKit room and publishes action commands
    received via the send_action method to a remote robot.
    """
    
    config_class = RemoteRobotConfig
    
    def __init__(self, config: RemoteRobotConfig):
        super().__init__(config)
        
        if not LIVEKIT_AVAILABLE:
            raise ImportError("LiveKit SDK is required. Install with: pip install livekit")
            
        self.config = config
        
        # Create handler and LiveKit service
        self._handler = RemoteRobotHandler(self)
        self._livekit_service = LiveKitService(
            config.livekit_url,
            config.livekit_token,
            self._handler
        )
        
        # Robot state
        self._last_observation: dict[str, Any] = {}
        self._last_action: dict[str, Any] = {}
        self._state_lock = threading.Lock()

    @property
    def observation_features(self) -> dict[str, type]:
        """
        Define observation features for this robot.
        Override in subclasses to provide specific observation structure.
        """
        # Default empty features - should be overridden by subclasses
        return {}

    @property
    def action_features(self) -> dict[str, type]:
        """
        Define action features for this robot.
        Override in subclasses to provide specific action structure.
        """
        # Default empty features - should be overridden by subclasses
        return {}

    @property
    def is_connected(self) -> bool:
        """
        Whether the robot is currently connected or not.
        """
        return self._livekit_service.is_connected

    @property
    def is_calibrated(self) -> bool:
        """
        LiveKit robot doesn't require calibration.
        """
        return True

    def calibrate(self) -> None:
        """
        No calibration needed for LiveKit robot.
        """
        pass

    def configure(self) -> None:
        """
        No configuration needed for LiveKit robot.
        """
        pass

    def connect(self, calibrate: bool = True) -> None:
        """
        Establish communication with the LiveKit server.
        
        Args:
            calibrate (bool): not used
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Connect using the LiveKit service
        try:
            self._livekit_service.connect(timeout=10.0)
            logger.info(f"{self} connected to LiveKit server")
        except ConnectionError as e:
            raise ConnectionError(f"Failed to connect {self}: {e}")

    def get_observation(self) -> dict[str, Any]:
        """
        Retrieve the current observation from the robot.
        
        Returns:
            dict[str, Any]: A flat dictionary representing the robot's current sensory state.
            
        Raises:
            DeviceNotConnectedError: If the robot is not connected
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
            
        with self._state_lock:
            return self._last_observation.copy()

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Send an action command to remote robot via LiveKit data channel.

        Args:
            action (dict[str, Any]): Dictionary representing the desired action.

        Returns:
            dict[str, Any]: The action that was sent (same as input for LiveKit robot).
            
        Raises:
            DeviceNotConnectedError: If the robot is not connected
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        try:
            # Store the action locally
            with self._state_lock:
                self._last_action = action.copy()

            # Publish action via LiveKit data channel
            try:
                self._livekit_service.publish_json_sync(
                    action, 
                    "teleop_action", 
                    reliable=False,  # Use unreliable for real-time action data
                    timeout=1.0
                )
            except Exception as e:
                logger.error(f"Error publishing action to LiveKit: {e}")

            logger.debug(f"Sent action: {action}")
            return action
            
        except Exception as e:
            logger.error(f"Failed to send action: {e}")
            return action

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
        self._last_observation = {}
        self._last_action = {}
        
        logger.info(f"{self} disconnected from LiveKit server") 