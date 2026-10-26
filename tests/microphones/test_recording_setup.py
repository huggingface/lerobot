#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import pytest

from lerobot.microphones.portaudio import PortAudioMicrophoneConfig
from lerobot.robots.koch_follower import KochFollowerConfig
from lerobot.robots.so_follower import SOFollowerRobotConfig
from lerobot.utils.feature_utils import combine_feature_dicts, microphones_to_dataset_features


def test_microphones_to_dataset_features():
    microphones = {
        "wrist": PortAudioMicrophoneConfig(microphone_index=0, sample_rate=48000, channels=[1, 2]),
        "base": PortAudioMicrophoneConfig(microphone_index=1, sample_rate=16000, channels=[3]),
    }

    assert microphones_to_dataset_features(microphones) == {
        "observation.audio.wrist": {
            "dtype": "audio",
            "shape": (2,),
            "names": ["channel_1", "channel_2"],
            "info": {"audio.sample_rate": 48000},
        },
        "observation.audio.base": {
            "dtype": "audio",
            "shape": (1,),
            "names": ["channel_3"],
            "info": {"audio.sample_rate": 16000},
        },
    }


def test_combine_feature_dicts_keeps_audio_features():
    state = {"observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]}}
    audio = microphones_to_dataset_features(
        {"wrist": PortAudioMicrophoneConfig(microphone_index=0, sample_rate=48000, channels=[1, 2])}
    )

    combined = combine_feature_dicts(state, audio)

    assert combined["observation.audio.wrist"] == audio["observation.audio.wrist"]


def test_microphones_to_dataset_features_requires_sample_rate_and_channels():
    with pytest.raises(ValueError, match="sample_rate"):
        microphones_to_dataset_features({"wrist": PortAudioMicrophoneConfig(microphone_index=0)})


@pytest.mark.parametrize("robot_config_class", [SOFollowerRobotConfig, KochFollowerConfig])
@pytest.mark.parametrize(
    "microphone_kwargs, missing",
    [({"channels": [1]}, "sample_rate"), ({"sample_rate": 48000}, "channels")],
)
def test_robot_config_requires_microphone_sample_rate_and_channels(
    robot_config_class, microphone_kwargs, missing
):
    microphone = PortAudioMicrophoneConfig(microphone_index=0, **microphone_kwargs)
    with pytest.raises(ValueError, match=missing):
        robot_config_class(port="/dev/null", microphones={"wrist": microphone})


@pytest.mark.parametrize("robot_config_class", [SOFollowerRobotConfig, KochFollowerConfig])
def test_robot_config_accepts_microphones(robot_config_class):
    microphone = PortAudioMicrophoneConfig(microphone_index=0, sample_rate=48000, channels=[1, 2])
    config = robot_config_class(port="/dev/null", microphones={"wrist": microphone})
    assert config.microphones == {"wrist": microphone}
