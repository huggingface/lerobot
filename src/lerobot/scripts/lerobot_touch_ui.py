#!/usr/bin/env python

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
Launch the Bimanual SO-101 Touch UI

Example usage:

```shell
lerobot-touch-ui \
    --follower-left-port=/dev/ttyUSB0 \
    --follower-right-port=/dev/ttyUSB1 \
    --leader-left-port=/dev/ttyUSB2 \
    --leader-right-port=/dev/ttyUSB3
```
"""

import argparse
import logging
from pathlib import Path

from lerobot.touch_ui import BimanualTouchUI
from lerobot.utils.utils import init_logging


def main():
    init_logging()

    parser = argparse.ArgumentParser(description="Bimanual SO-101 Touch UI")
    parser.add_argument(
        "--follower-left-port",
        type=str,
        default="/dev/ttyUSB0",
        help="USB port for left follower arm",
    )
    parser.add_argument(
        "--follower-right-port",
        type=str,
        default="/dev/ttyUSB1",
        help="USB port for right follower arm",
    )
    parser.add_argument(
        "--leader-left-port",
        type=str,
        default="/dev/ttyUSB2",
        help="USB port for left leader arm",
    )
    parser.add_argument(
        "--leader-right-port",
        type=str,
        default="/dev/ttyUSB3",
        help="USB port for right leader arm",
    )
    parser.add_argument(
        "--calibration-dir",
        type=str,
        default="~/.cache/lerobot/calibration",
        help="Directory for calibration files",
    )

    args = parser.parse_args()

    logging.info("Starting Bimanual SO-101 Touch UI")
    logging.info(f"Follower ports: {args.follower_left_port}, {args.follower_right_port}")
    logging.info(f"Leader ports: {args.leader_left_port}, {args.leader_right_port}")

    app = BimanualTouchUI(
        follower_left_port=args.follower_left_port,
        follower_right_port=args.follower_right_port,
        leader_left_port=args.leader_left_port,
        leader_right_port=args.leader_right_port,
        calibration_dir=args.calibration_dir,
    )

    try:
        app.run()
    except KeyboardInterrupt:
        logging.info("Touch UI stopped by user")
    except Exception as e:
        logging.error(f"Touch UI error: {e}")
        raise


if __name__ == "__main__":
    main()
