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

from pathlib import Path

import pytest

from lerobot.microphones.portaudio import PortAudioMicrophoneConfig
from lerobot.robots.koch_follower import KochFollowerConfig
from lerobot.robots.so_follower import SOFollowerRobotConfig
from lerobot.utils.feature_utils import combine_feature_dicts, microphones_to_dataset_features

pytestmark = pytest.mark.hardware


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


class _FakeMicrophone:
    def __init__(self):
        self.is_recording = False

    def start_recording(self, output_file, multiprocessing=False, overwrite=True, barrier=None):
        if barrier is not None:
            barrier.wait()
        self.is_recording = True
        Path(output_file).write_bytes(b"audio")

    def stop_recording(self):
        self.is_recording = False


@pytest.mark.parametrize("error", [TimeoutError, KeyboardInterrupt])
def test_start_episode_audio_cleans_up_on_failure(monkeypatch, tmp_path, error):
    pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
    from lerobot.scripts import lerobot_record

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    monkeypatch.setattr(lerobot_record.tempfile, "mkdtemp", lambda **kwargs: str(audio_dir))

    def fail(microphone):
        raise error("no audio")

    monkeypatch.setattr(lerobot_record, "wait_for_first_sample_time", fail)
    microphones = {"wrist": _FakeMicrophone(), "base": _FakeMicrophone()}

    with pytest.raises(error, match="no audio"):
        lerobot_record.start_episode_audio(microphones)

    assert not any(mic.is_recording for mic in microphones.values())
    assert not audio_dir.exists()
