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
from abc import ABC, abstractmethod
from typing import Any, Callable

try:
    from livekit import rtc
    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False

logger = logging.getLogger(__name__)


class LiveKitServiceHandler(ABC):
    """
    Abstract base class for handling LiveKit service events.
    Implementations should provide specific logic for handling data received
    and other LiveKit room events.
    """
    
    @abstractmethod
    def on_data_received(self, data: rtc.DataPacket) -> None:
        """
        Handle data received from LiveKit data channel.
        
        Args:
            data: The data packet received from LiveKit
        """
        pass
    
    def on_participant_joined(self, participant: rtc.RemoteParticipant) -> None:
        """
        Handle participant joined event.
        
        Args:
            participant: The participant that joined
        """
        logger.info(f"Participant {participant.identity} joined the room")
    
    def on_participant_disconnected(self, participant: rtc.RemoteParticipant) -> None:
        """
        Handle participant disconnected event.
        
        Args:
            participant: The participant that disconnected
        """
        logger.info(f"Participant {participant.identity} disconnected")
    
    def on_connected(self) -> None:
        """
        Handle successful connection to LiveKit room.
        """
        logger.info("Successfully connected to LiveKit room")
    
    def on_disconnected(self, reason: str) -> None:
        """
        Handle disconnection from LiveKit room.
        
        Args:
            reason: The reason for disconnection
        """
        logger.info(f"Disconnected from LiveKit room: {reason}")


class LiveKitService:
    """
    Service class for managing LiveKit WebRTC connections.
    
    This class abstracts the common LiveKit connection logic used by both
    RemoteRobot and RemoteTeleoperator classes, providing a unified interface
    for establishing and managing LiveKit room connections.
    """
    
    def __init__(self, livekit_url: str, livekit_token: str, handler: LiveKitServiceHandler):
        """
        Initialize the LiveKit service.
        
        Args:
            livekit_url: The LiveKit server URL
            livekit_token: The LiveKit authentication token
            handler: Handler for LiveKit events and data
        """
        if not LIVEKIT_AVAILABLE:
            raise ImportError("LiveKit SDK is required. Install with: pip install livekit")
        
        self.livekit_url = livekit_url
        self.livekit_token = livekit_token
        self.handler = handler
        
        # LiveKit connection state
        self.room: rtc.Room | None = None
        self._is_connected = False
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
    
    @property
    def is_connected(self) -> bool:
        """
        Whether the service is currently connected to LiveKit or not.
        """
        return self._is_connected and self.room is not None
    
    def connect(self, timeout: float = 10.0) -> None:
        """
        Establish connection to the LiveKit server.
        
        Args:
            timeout: Connection timeout in seconds
            
        Raises:
            ConnectionError: If connection cannot be established
        """
        if self.is_connected:
            raise ConnectionError("LiveKit service is already connected")
        
        # Start the async event loop in a separate thread
        self._loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._loop_thread.start()
        
        # Wait for connection to be established
        start_time = time.time()
        while not self._is_connected and (time.time() - start_time) < timeout:
            time.sleep(0.1)
            
        if not self._is_connected:
            raise ConnectionError(f"Failed to connect to LiveKit server within {timeout}s")
            
        logger.info("LiveKit service connected to server")
    
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
                self.handler.on_participant_joined(participant)

            @self.room.on("participant_disconnected")
            def on_participant_disconnected(participant: rtc.RemoteParticipant):
                self.handler.on_participant_disconnected(participant)

            @self.room.on("data_received")
            def on_data_received(data: rtc.DataPacket):
                try:
                    self.handler.on_data_received(data)
                except Exception as e:
                    logger.error(f"Error processing data packet: {e}")

            @self.room.on("connected")
            def on_connected():
                self.handler.on_connected()
                self._is_connected = True

            @self.room.on("disconnected")
            def on_disconnected(reason):
                self.handler.on_disconnected(reason)
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
    
    async def publish_data(self, data: bytes, topic: str, reliable: bool = False) -> None:
        """
        Publish data to a LiveKit data channel.
        
        Args:
            data: The data to publish
            topic: The topic to publish to
            reliable: Whether to use reliable transport
            
        Raises:
            RuntimeError: If not connected to LiveKit
        """
        if not self.is_connected or not self.room:
            raise RuntimeError("Not connected to LiveKit room")
        
        try:
            await self.room.local_participant.publish_data(
                data,
                topic=topic,
                reliable=reliable
            )
        except Exception as e:
            logger.error(f"Error publishing data: {e}")
            raise
    
    def publish_data_sync(self, data: bytes, topic: str, reliable: bool = False, timeout: float = 1.0) -> None:
        """
        Synchronously publish data to a LiveKit data channel.
        
        Args:
            data: The data to publish
            topic: The topic to publish to
            reliable: Whether to use reliable transport
            timeout: Timeout for the operation
            
        Raises:
            RuntimeError: If not connected to LiveKit
        """
        if not self.is_connected or not self._event_loop:
            raise RuntimeError("Not connected to LiveKit room")
        
        # Schedule the data publishing on the event loop
        future = asyncio.run_coroutine_threadsafe(
            self.publish_data(data, topic, reliable), 
            self._event_loop
        )
        
        # Wait for the publish to complete
        try:
            future.result(timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout while publishing data to topic '{topic}'")
            raise
        except Exception as e:
            logger.error(f"Error publishing data to topic '{topic}': {e}")
            raise
    
    def publish_json_sync(self, data: dict[str, Any], topic: str, reliable: bool = False, timeout: float = 1.0) -> None:
        """
        Synchronously publish JSON data to a LiveKit data channel.
        
        Args:
            data: The dictionary to publish as JSON
            topic: The topic to publish to
            reliable: Whether to use reliable transport
            timeout: Timeout for the operation
        """
        json_data = json.dumps(data).encode('utf-8')
        self.publish_data_sync(json_data, topic, reliable, timeout)
    
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
    
    def disconnect(self) -> None:
        """
        Disconnect from the LiveKit server and perform cleanup.
        
        Raises:
            RuntimeError: If not connected to LiveKit
        """
        if not self.is_connected:
            raise RuntimeError("LiveKit service is not connected")

        logger.info("Disconnecting LiveKit service from server")
        
        self._is_connected = False
        
        # Trigger cleanup if event loop is running
        if self._event_loop and not self._event_loop.is_closed():
            asyncio.run_coroutine_threadsafe(self._cleanup(), self._event_loop)
        
        # Wait for the event loop thread to finish
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=10)  # Increased timeout for graceful shutdown
            
        # Clean up state
        self.room = None
        
        logger.info("LiveKit service disconnected from server") 