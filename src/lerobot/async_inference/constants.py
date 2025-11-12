# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Client side: The environment evolves with a time resolution equal to 1/fps"""

DEFAULT_FPS = 30

"""Server side: Running inference on (at most) 1/fps"""
DEFAULT_INFERENCE_LATENCY = 1 / DEFAULT_FPS

"""Server side: Timeout for observation queue in seconds"""
DEFAULT_OBS_QUEUE_TIMEOUT = 2

# All action chunking policies
SUPPORTED_POLICIES = ["act", "smolvla", "diffusion", "tdmpc", "vqbet", "pi0", "pi05"]


def get_supported_robots():
    """Get dynamically supported robots from registered RobotConfig subclasses.

    This function should be called after register_third_party_devices() has been called
    to ensure all third-party robots are registered.
    """
    try:
        from lerobot.robots.config import RobotConfig
        return list(RobotConfig.get_known_choices().keys())
    except ImportError:
        # Fallback to hardcoded list if RobotConfig not available
        return ["so100_follower", "so101_follower", "bi_so100_follower", "koch_follower"]


# SUPPORTED_ROBOTS is now computed dynamically in robot_client.py
# after register_third_party_devices() has been called
