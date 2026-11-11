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

"""
Helper to find the microphones available in your system, and optionally record a short test clip
from each of them.

Example:

```shell
lerobot-find-microphones
lerobot-find-microphones --record-time-s 3
```
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Any

from lerobot.microphones.portaudio import PortAudioMicrophone, PortAudioMicrophoneConfig
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


def find_all_portaudio_microphones() -> list[dict[str, Any]]:
    """
    Finds all available PortAudio microphones plugged into the system.

    Returns:
        A list of all available PortAudio microphones with their metadata.
    """
    logger.info("Searching for PortAudio microphones...")
    try:
        microphones = PortAudioMicrophone.find_microphones()
    except Exception as e:
        logger.error(f"Error finding PortAudio microphones: {e}")
        return []

    assert isinstance(microphones, list)  # nosec B101
    logger.info(f"Found {len(microphones)} PortAudio microphones.")
    return microphones


def record_from_all_microphones(output_dir: Path, record_time_s: float) -> None:
    """
    Lists all microphones and, if `record_time_s` is positive, records a WAV clip from each of them.

    Args:
        output_dir: Directory to save the clips to.
        record_time_s: Duration of each clip in seconds. Nothing is recorded if it is 0.
    """
    microphones_info = find_all_portaudio_microphones()
    if not microphones_info:
        logger.warning("No microphones detected.")
        return

    print("\n--- Detected Microphones ---")
    for i, microphone_info in enumerate(microphones_info):
        print(f"Microphone #{i}:")
        for key, value in microphone_info.items():
            print(f"  {key.replace('_', ' ').capitalize()}: {value}")
        print("-" * 20)

    if record_time_s <= 0:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for microphone_info in microphones_info:
        config = PortAudioMicrophoneConfig(microphone_index=microphone_info["index"])
        output_file = output_dir / f"portaudio_{microphone_info['index']}.wav"
        try:
            with PortAudioMicrophone(config) as microphone:
                microphone.start_recording(output_file)
                time.sleep(record_time_s)
                num_samples = len(microphone.read())
                microphone.stop_recording()
            logger.info(f"Recorded {num_samples} samples from {microphone_info['name']} to {output_file}")
        except Exception as e:
            logger.error(f"Error recording from {microphone_info['name']}: {e}")


def main() -> None:
    init_logging()

    parser = argparse.ArgumentParser(
        description="Microphone utility script for listing microphones and recording test clips."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="outputs/captured_audio",
        help="Directory to save audio clips. Default: outputs/captured_audio",
    )
    parser.add_argument(
        "--record-time-s",
        type=float,
        default=0.0,
        help="Duration of the test clip recorded from each microphone. Default: 0 (only list microphones).",
    )
    args = parser.parse_args()
    record_from_all_microphones(**vars(args))


if __name__ == "__main__":
    main()
