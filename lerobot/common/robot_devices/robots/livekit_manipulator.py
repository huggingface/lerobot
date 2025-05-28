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

"""Contains logic for a Livekit-enabled manipulator robot that can stream video 
and receive remote control commands over WebRTC.
"""

import logging
import torch
import numpy as np
import time
import asyncio
import json
import threading
from livekit import rtc

from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot, ensure_safe_goal_position
from lerobot.common.robot_devices.robots.configs import LivekitManipulatorRobotConfig
from lerobot.common.robot_devices.utils import RobotDeviceNotConnectedError


class LivekitManipulatorRobot(ManipulatorRobot):
    """A ManipulatorRobot subclass that adds Livekit WebRTC capabilities for remote control
    and video streaming.
    
    This class extends the base ManipulatorRobot with Livekit integration capabilities.
    """
    
    def __init__(
        self,
        config: LivekitManipulatorRobotConfig,
        **kwargs
    ):
        """Initialize the LivekitManipulatorRobot.
        
        Args:
            config: Robot configuration including Livekit settings
            **kwargs: Additional arguments passed to base ManipulatorRobot
        """
        super().__init__(config, **kwargs)
        
        # Livekit configuration
        self.livekit_url = config.livekit_url
        self.livekit_token = config.livekit_token
        self.is_leader = config.is_leader
        
        # Initialize Livekit room
        self.room = rtc.Room()
        self.livekit_connected = False
        
        logging.info("LivekitManipulatorRobot initialized")

    def connect(self):
        """Override connect to also establish Livekit connection."""
        # Call parent connect method first
        super().connect()
        
        try:
            self._livekit_loop = asyncio.new_event_loop()
            def run_loop(loop):
                asyncio.set_event_loop(loop)
                loop.run_forever()
            self._livekit_thread = threading.Thread(target=run_loop, args=(self._livekit_loop,), daemon=True)
            self._livekit_thread.start()
            # Connect to Livekit room in the new loop
            fut = asyncio.run_coroutine_threadsafe(
                self._livekit_connect(self.livekit_url, self.livekit_token), self._livekit_loop
            )
            fut.result(timeout=10)
            logging.info(f"Connected to Livekit room at {self.livekit_url}")
        except Exception as e:
            logging.error(f"Failed to connect to Livekit: {e}")
            self.livekit_connected = False

   
    async def _livekit_connect(self, url, token):
        self.room = rtc.Room(loop=self._livekit_loop)
        
        # Set up data received handler for follower mode
        if not self.is_leader:
            self.room.on("data_received", self._on_data_received)
        
        await self.room.connect(url, token)
        self.livekit_connected = True

    def _on_data_received(self, data_packet: rtc.DataPacket):
        """Handle data received from Livekit room.
        
        Args:
            data_packet: The received data packet from Livekit
        """
        if data_packet.topic == "leader":
            try:
                # Decode JSON data
                json_str = data_packet.data.decode('utf-8')
                leader_data = json.loads(json_str)
                
                # Extract leader_arm_positions from the data structure
                if "leader_arm_positions" in leader_data:
                    # Convert back to tensors and set follower positions
                    self._set_follower_from_leader_data(leader_data["leader_arm_positions"])
            except Exception as e:
                logging.error(f"Error processing leader data: {e}")

    def _set_follower_from_leader_data(self, leader_data: dict):
        """Set follower arm positions based on received leader data.
        
        Args:
            leader_data: Dictionary mapping arm names to position lists
        """
        if not self.is_connected:
            logging.warning("Robot not connected, cannot set follower positions")
            return
        
        for arm_name, position_list in leader_data.items():
            if arm_name in self.follower_arms:
                try:
                    # Convert list back to tensor
                    goal_pos = torch.tensor(position_list, dtype=torch.float32)
                    
                    # Apply safety limits if configured
                    if self.config.max_relative_target is not None:
                        present_pos = self.follower_arms[arm_name].read("Present_Position")
                        present_pos = torch.from_numpy(present_pos)
                        goal_pos = ensure_safe_goal_position(
                            goal_pos, present_pos, self.config.max_relative_target
                        )
                    
                    # Send goal position to follower arm
                    goal_pos_np = goal_pos.numpy().astype(np.float32)
                    self.follower_arms[arm_name].write("Goal_Position", goal_pos_np)
                    
                except Exception as e:
                    logging.error(f"Error setting position for arm {arm_name}: {e}")

    def publish_leader_position(self, leader_pos: dict[str, torch.Tensor]) -> None:
        """Publish leader position data via Livekit.
        
        Args:
            leader_pos: Dictionary mapping arm names to position tensors
        """
        # Check if room is connected before trying to publish
        if not self.livekit_connected or not self.room:
            logging.debug("Livekit room not connected yet, skipping data publish")
            return
            
        try:
            leader_flat = {k: v.tolist() for k, v in leader_pos.items()}
            data = json.dumps({"leader_arm_positions": leader_flat}).encode("utf-8")
            
            async def send_packet():
                await self.room.local_participant.publish_data(
                    data, reliable=False, topic="leader"
                )
            
            if self._livekit_loop and not self._livekit_loop.is_closed():
                # Submit the coroutine and get the future
                future = asyncio.run_coroutine_threadsafe(send_packet(), self._livekit_loop)
                # Wait a short time for it to complete (non-blocking for real-time performance)
                # try:
                #     future.result(timeout=0.001)  # Very short timeout to avoid blocking
                #     print("Data sent successfully")
                # except Exception as e:
                #     print(f"Send failed: {e}")
        except Exception as e:
            logging.error(f"Failed to publish leader arm positions to Livekit: {e}")

    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )
        if not self.is_leader:
            return {}, {}

        # read leader position
        leader_pos = {}
        for name in self.leader_arms:
            before_lread_t = time.perf_counter()
            leader_pos[name] = self.leader_arms[name].read("Present_Position")
            leader_pos[name] = torch.from_numpy(leader_pos[name])
            self.logs[f"read_leader_{name}_pos_dt_s"] = time.perf_counter() - before_lread_t
        
        self.publish_leader_position(leader_pos)

        # TODO: renable recording 
        return {}, {}

