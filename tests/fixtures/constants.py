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
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME

LEROBOT_TEST_DIR = HF_LEROBOT_HOME / "_testing"
DUMMY_REPO_ID = "dummy/repo"
DUMMY_ROBOT_TYPE = "dummy_robot"
DUMMY_MOTOR_FEATURES = {
    ACTION: {
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
DEFAULT_FPS = 30
DUMMY_VIDEO_INFO = {
    "video.fps": DEFAULT_FPS,
    "video.codec": "av1",
    "video.pix_fmt": "yuv420p",
    "video.video_backend": "pyav",
    "video.extra_options": {},
    "video.g": 2,
    "video.crf": 30,
    "video.preset": 12,
    "video.fast_decode": 0,
    "is_depth_map": False,
    "has_audio": False,
}
DUMMY_CAMERA_FEATURES = {
    "laptop": {"shape": (64, 96, 3), "names": ["height", "width", "channels"], "info": DUMMY_VIDEO_INFO},
    "phone": {"shape": (64, 96, 3), "names": ["height", "width", "channels"], "info": DUMMY_VIDEO_INFO},
}
DUMMY_DEPTH_VIDEO_INFO = {
    **DUMMY_VIDEO_INFO,
    "is_depth_map": True,
}
DUMMY_DEPTH_VIDEO_INFO_FULL = {
    **{k: v for k, v in DUMMY_VIDEO_INFO.items() if k != "video.preset"},
    "video.codec": "hevc",
    "video.pix_fmt": "gray12le",
    "is_depth_map": True,
    "video.depth_min": 0.05,
    "video.depth_max": 8.0,
    "video.shift": 2.5,
    "video.use_log": True,
}
DUMMY_DEPTH_CAMERA_FEATURES = {
    "laptop_depth": {
        "shape": (64, 96, 1),
        "names": ["height", "width", "channels"],
        "info": DUMMY_DEPTH_VIDEO_INFO,
    },
}
DUMMY_CAMERA_FEATURES_WITH_DEPTH = {**DUMMY_CAMERA_FEATURES, **DUMMY_DEPTH_CAMERA_FEATURES}
DUMMY_CHW = (3, 96, 128)
DUMMY_HWC = (96, 128, 3)

# Default video feature set used by video-encoding persistence tests.
DUMMY_VIDEO_FEATURES = {
    "observation.images.cam": {
        "dtype": "video",
        "shape": (64, 96, 3),
        "names": ["height", "width", "channels"],
    },
    "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
}
DUMMY_VIDEO_KEY = "observation.images.cam"

DUMMY_DEPTH_FEATURES = {
    "observation.images.depth": {
        "dtype": "video",
        "shape": (64, 96, 1),
        "names": ["height", "width", "channels"],
        "info": {"is_depth_map": True},
    },
    "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
}
DUMMY_DEPTH_KEY = "observation.images.depth"
