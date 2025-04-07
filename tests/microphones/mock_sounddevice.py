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
from functools import cache

from tests.fixtures.constants import DUMMY_AUDIO_CHANNELS, DEFAULT_SAMPLE_RATE

import numpy as np
from lerobot.common.utils.utils import capture_timestamp_utc
from threading import Thread, Event
import time

@cache
def _generate_sound(duration: float, sample_rate: int, channels: int):
    return np.random.uniform(-1, 1, size=(int(duration * sample_rate), channels)).astype(np.float32)

def query_devices(query_index: int):
    return {
            "name": "Mock Sound Device",
            "index": query_index,
            "max_input_channels": DUMMY_AUDIO_CHANNELS,
            "default_samplerate": DEFAULT_SAMPLE_RATE,
    }

class InputStream:
    def __init__(self, *args, **kwargs):
        self._mock_dict = {
            "channels": DUMMY_AUDIO_CHANNELS,
            "samplerate": DEFAULT_SAMPLE_RATE,
        }
        self._is_active = False
        self._audio_callback = kwargs.get("callback")

        self.callback_thread = None
        self.callback_thread_stop_event = None

    def _acquisition_loop(self):
        if self._audio_callback is not None:
            while not self.callback_thread_stop_event.is_set():
                # Simulate audio data acquisition
                time.sleep(0.01)
                self._audio_callback(_generate_sound(0.01, DEFAULT_SAMPLE_RATE, DUMMY_AUDIO_CHANNELS), 0.01*DEFAULT_SAMPLE_RATE, capture_timestamp_utc(), None)

    def start(self):
        self.callback_thread_stop_event = Event()
        self.callback_thread = Thread(target=self._acquisition_loop, args=())
        self.callback_thread.daemon = True
        self.callback_thread.start()

        self._is_active = True

    @property
    def active(self):
        return self._is_active
    
    def stop(self):
        if self.callback_thread_stop_event is not None:
            self.callback_thread_stop_event.set()
            self.callback_thread.join()
            self.callback_thread = None
            self.callback_thread_stop_event = None
        self._is_active = False

    def close(self):
        if self._is_active:
            self.stop()

    def __del__(self):
        if self._is_active:
            self.stop()


