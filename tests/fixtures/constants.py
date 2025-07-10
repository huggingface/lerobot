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
from lerobot.constants import HF_LEROBOT_HOME

LEROBOT_TEST_DIR = HF_LEROBOT_HOME / "_testing"
DUMMY_REPO_ID = "dummy/repo"
DUMMY_ROBOT_TYPE = "dummy_robot"
DUMMY_MOTOR_FEATURES = {
    "action": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
    },
    "state": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
    },
}
DUMMY_CAMERA_FEATURES = {
    "laptop": {"shape": (64, 96, 3), "names": ["height", "width", "channels"], "info": None},
    "phone": {"shape": (64, 96, 3), "names": ["height", "width", "channels"], "info": None},
}
DEFAULT_FPS = 30
DUMMY_VIDEO_INFO = {
    "video.fps": DEFAULT_FPS,
    "video.codec": "av1",
    "video.pix_fmt": "yuv420p",
    "video.is_depth_map": False,
    "has_audio": False,
}
DUMMY_CHW = (3, 96, 128)
DUMMY_HWC = (96, 128, 3)
