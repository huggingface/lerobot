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

import asyncio
import json
import logging
import threading
import time
from typing import Any

from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.robots.config import RemoteRobotConfig
from lerobot.common.robots.robot import Robot

try:
    from livekit import rtc
    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False

logger = logging.getLogger(__name__)


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
        self.livekit_url = config.livekit_url
        self.livekit_token = config.livekit_token
        
        # LiveKit connection state
        self.room: rtc.Room | None = None
        self._is_connected = False
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        
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
        return self._is_connected and self.room is not None

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

        # Start the async event loop in a separate thread
        self._loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._loop_thread.start()
        
        # Wait for connection to be established
        timeout = 10  # 10 second timeout for better reliability
        start_time = time.time()
        while not self._is_connected and (time.time() - start_time) < timeout:
            time.sleep(0.1)
            
        if not self._is_connected:
            raise ConnectionError(f"Failed to connect to LiveKit server within {timeout}s")
            
        logger.info(f"{self} connected to LiveKit server")

    def _run_event_loop(self) -> None:
        """
        Run the async event loop for LiveKit connection.
        """
        try:
            # Create new event loop following example.py pattern
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)
            
            # Create room instance
            self.room = rtc.Room(loop=self._event_loop)
            
            # Start the main async task
            asyncio.ensure_future(self._main_async_loop())
            
            # Run the event loop
            self._event_loop.run_forever()
            
        except Exception as e:
            logger.error(f"Error in LiveKit async loop: {e}")
            self._is_connected = False
        finally:
            if self._event_loop and not self._event_loop.is_closed():
                self._event_loop.close()

    async def _main_async_loop(self) -> None:
        """
        Main async loop following example.py pattern.
        """
        try:
            # Set up event handlers
            @self.room.on("participant_joined")
            def on_participant_joined(participant: rtc.RemoteParticipant):
                logger.info(f"Participant {participant.identity} joined the room")

            @self.room.on("participant_disconnected")
            def on_participant_disconnected(participant: rtc.RemoteParticipant):
                logger.info(f"Participant {participant.identity} disconnected")

            @self.room.on("connected")
            def on_connected():
                logger.info("Successfully connected to LiveKit room")
                self._is_connected = True

            @self.room.on("disconnected")
            def on_disconnected(reason):
                logger.info(f"Disconnected from LiveKit room: {reason}")
                self._is_connected = False

            # Connect to the room with proper options
            await self.room.connect(
                self.livekit_url,
                self.livekit_token,
                rtc.RoomOptions(auto_subscribe=True)
            )
            
            logger.info(f"Connected to room {self.room.name}")
            
            self._is_connected = True
            
            # Keep the event loop running while connected
            while self._is_connected:
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Failed to connect to LiveKit room: {e}")
            self._is_connected = False
            raise

    async def _cleanup(self) -> None:
        """
        Cleanup resources following example.py pattern.
        """
        logger.info("Cleaning up LiveKit connection...")
        self._is_connected = False
        
        if self.room:
            await self.room.disconnect()
        
        if self._event_loop:
            self._event_loop.stop()

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
            if self.room and self._event_loop:
                # Schedule the data publishing on the event loop
                future = asyncio.run_coroutine_threadsafe(
                    self._publish_action(action), 
                    self._event_loop
                )
                # Wait for the publish to complete (with a short timeout)
                try:
                    future.result(timeout=1.0)
                except asyncio.TimeoutError:
                    logger.warning("Timeout while publishing action to LiveKit")
                except Exception as e:
                    logger.error(f"Error publishing action to LiveKit: {e}")

            logger.debug(f"Sent action: {action}")
            return action
            
        except Exception as e:
            logger.error(f"Failed to send action: {e}")
            return action

    async def _publish_action(self, action: dict[str, Any]) -> None:
        """
        Publish action data to LiveKit data channel following example.py pattern.
        
        Args:
            action: The action dictionary to publish
        """
        try:
            # Convert action to JSON string
            action_json = json.dumps(action)
            payload = action_json.encode('utf-8')
            
            # Publish to the 'teleop_action' topic with reliable=False for low latency
            await self.room.local_participant.publish_data(
                payload,
                topic="teleop_action",
                reliable=False  # Use unreliable for real-time action data
            )
            
        except Exception as e:
            logger.error(f"Error publishing action data: {e}")
            raise

    def disconnect(self) -> None:
        """
        Disconnect from the LiveKit server and perform any necessary cleanup.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        logger.info(f"Disconnecting {self} from LiveKit server")
        
        self._is_connected = False
        
        # Trigger cleanup if event loop is running
        if self._event_loop and not self._event_loop.is_closed():
            asyncio.run_coroutine_threadsafe(self._cleanup(), self._event_loop)
        
        # Wait for the event loop thread to finish
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=10)  # Increased timeout for graceful shutdown
            
        # Clean up state
        self.room = None
        self._last_observation = {}
        self._last_action = {}
        
        logger.info(f"{self} disconnected from LiveKit server") 