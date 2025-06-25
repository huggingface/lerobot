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
import numpy as np

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

        # Video track management for camera observations
        self._video_sources: dict[str, rtc.VideoSource] = {}
        self._video_tracks: dict[str, rtc.LocalVideoTrack] = {}
        self._video_lock = threading.Lock()

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

    def publish_observation(self, observation: dict[str, Any]) -> None:
        """
        Publish robot observation data to LiveKit for remote viewing.
        
        This method:
        1. Extracts values from observation that match the action features
        2. Publishes matching values as JSON data to the 'teleop_observation' topic
        3. Creates/updates video tracks for any camera frame observations
        
        Args:
            observation: The observation dictionary returned from Robot.get_observation()
            
        Raises:
            DeviceNotConnectedError: If the teleoperator is not connected
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        try:
            # Extract observation values that match action features for JSON publishing
            observation_data = {}
            camera_frames = {}
            
            for key, value in observation.items():
                # Check if this observation key matches any action feature
                if self._expected_action_shape and key in self._expected_action_shape:
                    observation_data[key] = value
                # Check if this is a camera frame (numpy array with 3 dimensions for H, W, C)
                elif isinstance(value, np.ndarray) and value.ndim == 3:
                    camera_frames[key] = value
                    
            # Publish matching observation data as JSON if we have any
            if observation_data:
                try:
                    self._livekit_service.publish_json_sync(
                        observation_data, 
                        "teleop_observation", 
                        reliable=False,
                        timeout=0.5
                    )
                    logger.debug(f"Published observation data: {list(observation_data.keys())}")
                except Exception as e:
                    logger.warning(f"Failed to publish observation data: {e}")
            
            # Handle video tracks for camera frames
            if camera_frames:
                self._publish_camera_frames(camera_frames)
                
        except Exception as e:
            logger.error(f"Error publishing observation: {e}")

    def _publish_camera_frames(self, camera_frames: dict[str, np.ndarray]) -> None:
        """
        Publish camera frames as video tracks to LiveKit.
        
        Args:
            camera_frames: Dictionary mapping camera names to image arrays (H, W, C)
        """
        with self._video_lock:
            for camera_name, frame_array in camera_frames.items():
                try:
                    # Ensure frame is in the correct format (H, W, C) with uint8 values
                    if frame_array.dtype != np.uint8:
                        # Convert to uint8 if needed
                        if frame_array.max() <= 1.0:
                            frame_array = (frame_array * 255).astype(np.uint8)
                        else:
                            frame_array = frame_array.astype(np.uint8)
                    
                    height, width, channels = frame_array.shape
                    
                    # Create video source and track if they don't exist
                    if camera_name not in self._video_sources:
                        self._video_sources[camera_name] = rtc.VideoSource(width, height)
                        self._video_tracks[camera_name] = rtc.LocalVideoTrack.create_video_track(
                            f"camera_{camera_name}", 
                            self._video_sources[camera_name]
                        )
                        
                        # Publish the video track
                        if self._livekit_service.room and self._livekit_service.room.local_participant:
                            options = rtc.TrackPublishOptions(
                                source=rtc.TrackSource.SOURCE_CAMERA,
                                simulcast=True,
                                video_encoding=rtc.VideoEncoding(
                                    max_framerate=30,
                                    max_bitrate=2_000_000,
                                ),
                                video_codec=rtc.VideoCodec.H264,
                            )
                            
                            # Schedule the track publishing asynchronously
                            import asyncio
                            if self._livekit_service._event_loop:
                                asyncio.run_coroutine_threadsafe(
                                    self._livekit_service.room.local_participant.publish_track(
                                        self._video_tracks[camera_name], options
                                    ),
                                    self._livekit_service._event_loop
                                )
                    
                    # Convert numpy array to video frame
                    if channels == 3:
                        # Assume RGB format, convert to RGBA
                        rgba_frame = np.zeros((height, width, 4), dtype=np.uint8)
                        rgba_frame[:, :, :3] = frame_array
                        rgba_frame[:, :, 3] = 255  # Alpha channel
                        frame_data = rgba_frame.flatten().tobytes()
                        buffer_type = rtc.VideoBufferType.RGBA
                    elif channels == 4:
                        # Already RGBA
                        frame_data = frame_array.flatten().tobytes()
                        buffer_type = rtc.VideoBufferType.RGBA
                    else:
                        logger.warning(f"Unsupported number of channels for camera {camera_name}: {channels}")
                        continue
                    
                    # Create and publish video frame
                    video_frame = rtc.VideoFrame(width, height, buffer_type, frame_data)
                    self._video_sources[camera_name].capture_frame(video_frame)
                    
                    logger.debug(f"Published video frame for camera: {camera_name}")
                    
                except Exception as e:
                    logger.error(f"Error publishing camera frame for {camera_name}: {e}")

    def disconnect(self) -> None:
        """
        Disconnect from the LiveKit server and perform any necessary cleanup.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        logger.info(f"Disconnecting {self} from LiveKit server")
        
        # Clean up video tracks
        with self._video_lock:
            self._video_sources.clear()
            self._video_tracks.clear()
        
        # Disconnect using the LiveKit service
        try:
            self._livekit_service.disconnect()
        except RuntimeError:
            pass  # Already disconnected
        
        # Clean up state
        self._cached_action = None
        self._expected_action_shape = None
        
        logger.info(f"{self} disconnected from LiveKit server") 