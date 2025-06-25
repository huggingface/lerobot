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
from lerobot.common.teleoperators.config import RemoteTeleoperatorConfig
from lerobot.common.teleoperators.teleoperator import Teleoperator

try:
    from livekit import rtc
    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False

logger = logging.getLogger(__name__)


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
        self.livekit_url = config.livekit_url
        self.livekit_token = config.livekit_token
        
        # LiveKit connection state
        self.room: rtc.Room | None = None
        self._is_connected = False
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        
        # Action caching and validation
        self._cached_action: dict[str, Any] | None = None
        self._expected_action_shape: dict[str, type] | None = None
        self._action_lock = threading.Lock()

    @property
    def is_connected(self) -> bool:
        """
        Whether the teleoperator is currently connected or not.
        """
        return self._is_connected and self.room is not None

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

            @self.room.on("data_received")
            def on_data_received(data: rtc.DataPacket):
                try:
                    # Check if this is the teleop_action topic
                    if data.topic == "teleop_action":
                        # Decode the JSON action message
                        action_data = json.loads(data.data.decode('utf-8'))
                        self._handle_action_message(action_data)
                        
                except Exception as e:
                    logger.error(f"Error processing data packet: {e}")

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
        
        self._is_connected = False
        
        # Trigger cleanup if event loop is running
        if self._event_loop and not self._event_loop.is_closed():
            asyncio.run_coroutine_threadsafe(self._cleanup(), self._event_loop)
        
        # Wait for the event loop thread to finish
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=10)  # Increased timeout for graceful shutdown
            
        # Clean up state
        self.room = None
        self._cached_action = None
        self._expected_action_shape = None
        
        logger.info(f"{self} disconnected from LiveKit server") 