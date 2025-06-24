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

from typing import Any

from lerobot.common.teleoperators.config import LiveKitTeleoperatorConfig
from lerobot.common.teleoperators.teleoperator import Teleoperator


class LiveKitTeleoperator(Teleoperator):
    """
    LiveKit-based teleoperator for remote control via WebRTC.
    
    This teleoperator enables receiving actions via WebRTC data channels,
    allowing operators to control robots over the internet with low latency.
    """
    
    config_class = LiveKitTeleoperatorConfig
    
    def __init__(self, config: LiveKitTeleoperatorConfig):
        super().__init__(config)
        self.livekit_url = config.livekit_url
        self.livekit_token = config.livekit_token

    def is_connected(self) -> bool:
        """
        Whether the teleoperator is currently connected or not. If `False`, calling :pymeth:`get_action`
        or :pymeth:`send_feedback` should raise an error.
        """
        pass
        
    def connect(self, calibrate: bool = True) -> None:
        """
        Establish communication with the LiveKit server.
        
        Args:
            calibrate (bool): If True, automatically calibrate the teleoperator after connecting.
        """
        pass
    
    def get_action(self) -> dict[str, Any]:
        """
        Retrieve the current action from the teleoperator.
        
        Returns:
            dict[str, Any]: A flat dictionary representing the teleoperator's current actions.
        """
        pass
    
    def disconnect(self) -> None:
        """
        Disconnect from the LiveKit server and perform any necessary cleanup.
        """
        pass 