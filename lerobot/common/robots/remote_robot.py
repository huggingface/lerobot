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
from typing import Any

import numpy as np

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

    def __init__(self, robot: "RemoteRobot"):
        self.robot = robot

    def on_data_received(self, data: rtc.DataPacket) -> None:
        """
        Handle data received from LiveKit data channel.
        """
        try:
            # Check if this is the teleop_observation topic
            if data.topic == "teleop_observation":
                # Decode the JSON observation message
                observation_data = json.loads(data.data.decode("utf-8"))
                self.robot._handle_observation_message(observation_data)

        except Exception as e:
            logger.error(f"Error processing data packet: {e}")

    def on_participant_joined(self, participant: rtc.RemoteParticipant) -> None:
        """
        Handle participant joined event.
        """
        logger.info(f"Participant {participant.identity} joined the room")

        # Check if this is the leader participant
        if participant.identity == "leader":
            self.robot._setup_leader_subscriptions(participant)

    def on_participant_disconnected(self, participant: rtc.RemoteParticipant) -> None:
        """
        Handle participant disconnected event.
        """
        logger.info(f"Participant {participant.identity} disconnected")

        # Clean up video streams if leader disconnected
        if participant.identity == "leader":
            self.robot._cleanup_leader_subscriptions()


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
        self._livekit_service = LiveKitService(config.livekit_url, config.livekit_token, self._handler)

        # Robot state
        self._last_observation: dict[str, Any] = {}
        self._last_action: dict[str, Any] = {}
        self._state_lock = threading.Lock()

        # Video tracking and processing
        self._leader_participant: rtc.RemoteParticipant | None = None
        self._video_streams: dict[str, rtc.VideoStream] = {}
        self._video_tasks: dict[str, asyncio.Task] = {}
        self._video_lock = threading.Lock()

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

        # Connect using the LiveKit service and set up event handlers
        try:
            self._livekit_service.connect(timeout=10.0)
            self._setup_track_subscriptions()
            logger.info(f"{self} connected to LiveKit server")
        except ConnectionError as e:
            raise ConnectionError(f"Failed to connect {self}: {e}")

    def _setup_track_subscriptions(self) -> None:
        """
        Set up LiveKit room event handlers for track subscription.
        """
        if not self._livekit_service.room:
            return

        # Set up track subscription handler
        @self._livekit_service.room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            # Only process video tracks from the leader
            if participant.identity == "leader" and track.kind == rtc.TrackKind.KIND_VIDEO:
                asyncio.create_task(self._handle_video_track(track, participant))

        # Check for existing participants (if any joined before we connected)
        for participant in self._livekit_service.room.remote_participants.values():
            if participant.identity == "leader":
                self._setup_leader_subscriptions(participant)

    def _setup_leader_subscriptions(self, participant: rtc.RemoteParticipant) -> None:
        """
        Set up subscriptions for tracks from the leader participant.
        """
        self._leader_participant = participant

        # Subscribe to existing video tracks
        for publication in participant.track_publications.values():
            if publication.track and publication.track.kind == rtc.TrackKind.KIND_VIDEO:
                # Schedule the video track handling on the LiveKit service's event loop
                if self._livekit_service._event_loop:
                    asyncio.run_coroutine_threadsafe(
                        self._handle_video_track(publication.track, participant),
                        self._livekit_service._event_loop,
                    )

    def _cleanup_leader_subscriptions(self) -> None:
        """
        Clean up video streams when leader disconnects.
        """
        with self._video_lock:
            # Cancel all video processing tasks
            for task in self._video_tasks.values():
                if hasattr(task, "cancel"):
                    task.cancel()

            # Clear video streams and tasks
            self._video_streams.clear()
            self._video_tasks.clear()

        # Clear video frames from observation
        with self._state_lock:
            # Remove any camera frames from cached observations
            keys_to_remove = [
                key
                for key in self._last_observation.keys()
                if isinstance(self._last_observation.get(key), np.ndarray)
            ]
            for key in keys_to_remove:
                del self._last_observation[key]

    async def _handle_video_track(self, track: rtc.Track, participant: rtc.RemoteParticipant) -> None:
        """
        Handle a video track from the leader participant.
        """
        try:
            # Create a name for this video track (use track sid or a default name)
            track_name = getattr(track, "name", None) or f"camera_{track.sid}"

            # Create video stream
            video_stream = rtc.VideoStream(track)

            with self._video_lock:
                self._video_streams[track_name] = video_stream
                # Create task to process video frames - we're in an async context here
                self._video_tasks[track_name] = asyncio.create_task(
                    self._process_video_stream(track_name, video_stream)
                )

            logger.info(f"Started video stream for track: {track_name}")

        except Exception as e:
            logger.error(f"Error handling video track from {participant.identity}: {e}")

    async def _process_video_stream(self, track_name: str, video_stream: rtc.VideoStream) -> None:
        """
        Process incoming video frames from a video stream.
        """
        try:
            async for event in video_stream:
                # Convert LiveKit video frame to numpy array
                video_frame = event.frame

                # Convert video frame to numpy array (RGB format)
                frame_array = self._video_frame_to_numpy(video_frame)

                if frame_array is not None:
                    # Store the frame in observations
                    with self._state_lock:
                        self._last_observation[track_name] = frame_array

                    logger.debug(f"Received video frame for {track_name}: {frame_array.shape}")

        except Exception as e:
            logger.error(f"Error processing video stream {track_name}: {e}")
        finally:
            # Clean up this stream
            with self._video_lock:
                self._video_streams.pop(track_name, None)
                self._video_tasks.pop(track_name, None)

            await video_stream.aclose()

    def _video_frame_to_numpy(self, video_frame: rtc.VideoFrame) -> np.ndarray | None:
        """
        Convert LiveKit VideoFrame to numpy array.
        """
        try:
            # Convert to RGB format (3 channels)
            rgb_frame = video_frame.convert(rtc.VideoBufferType.RGB24)

            # Get frame dimensions
            width = rgb_frame.width
            height = rgb_frame.height

            # Convert data to numpy array
            frame_data = np.frombuffer(rgb_frame.data, dtype=np.uint8)

            # Reshape to (height, width, channels) format
            frame_array = frame_data.reshape((height, width, 3))

            return frame_array

        except Exception as e:
            logger.error(f"Error converting video frame to numpy: {e}")
            return None

    def _handle_observation_message(self, observation_data: dict[str, Any]) -> None:
        """
        Handle incoming observation messages from the teleop_observation topic.

        Args:
            observation_data: The received observation data dictionary
        """
        try:
            # Merge the observation data with our current cached observations
            with self._state_lock:
                # Update with new observation data (motor positions, etc.)
                self._last_observation.update(observation_data)

            logger.debug(f"Received observation data: {list(observation_data.keys())}")

        except Exception as e:
            logger.error(f"Invalid observation message received: {e}")

    def get_observation(self) -> dict[str, Any]:
        """
        Retrieve the current observation from the robot.

        Returns:
            dict[str, Any]: A flat dictionary representing the robot's current sensory state.
            This includes both video frames from cameras and action-matching data like motor positions.

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
                    timeout=1.0,
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

        # Clean up video subscriptions
        self._cleanup_leader_subscriptions()

        # Disconnect using the LiveKit service
        try:
            self._livekit_service.disconnect()
        except RuntimeError:
            pass  # Already disconnected

        # Clean up state
        self._last_observation = {}
        self._last_action = {}
        self._leader_participant = None

        logger.info(f"{self} disconnected from LiveKit server")
