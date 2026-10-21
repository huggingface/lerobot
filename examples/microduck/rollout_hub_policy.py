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

"""Roll out an onboard RL policy from the Hugging Face Hub on a Microduck, and drive it with the keyboard.

The duck runs its ONNX policies onboard; this script asks updaterd to download one from the Hub into the
robot's policy library, loads it into a slot (``walk`` by default) and lets you drive it.

Forward both sockets first, and keep that terminal open:

    ssh -N -L /tmp/robotd.sock:/run/robotd.sock -L /tmp/updaterd.sock:/run/updaterd.sock microduck@<ip>

Then:

    python examples/microduck/rollout_hub_policy.py --ip <ip> --policy RemiFabre/microduck-flamingo-cycle

Have the robot on its stand the first time: nothing published on the Hub is verified by anybody.
"""

import argparse
import time

from lerobot.robots.microduck import MicroDuck, MicroDuckConfig
from lerobot.teleoperators.keyboard import KeyboardMicroDuckTeleop, KeyboardMicroDuckTeleopConfig
from lerobot.utils.robot_utils import precise_sleep

FPS = 30


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ip", required=True, help="Microduck IP on the LAN (`duckctl ip`).")
    parser.add_argument(
        "--policy", required=True, help="Hub repo `user/repo[@revision][:file]`, or a path on the robot."
    )
    parser.add_argument("--slot", default="walk", help="Policy slot to load into.")
    parser.add_argument("--robotd", default="/tmp/robotd.sock")  # nosec B108: forwarded socket path
    parser.add_argument("--updaterd", default="/tmp/updaterd.sock")  # nosec B108: forwarded socket path
    args = parser.parse_args()

    robot = MicroDuck(
        MicroDuckConfig(
            remote_ip=args.ip,
            robotd_address=args.robotd,
            updaterd_address=args.updaterd,
            use_camera=False,
        )
    )
    keyboard = KeyboardMicroDuckTeleop(KeyboardMicroDuckTeleopConfig())

    robot.connect()
    keyboard.connect()

    print(f"Loading {args.policy} into slot '{args.slot}'...")
    print(robot.load_policy(args.policy, slot=args.slot))

    print("Driving: W/S forward/back, A/D strafe, Z/X turn, I/J/K/L head, +/- speed. Ctrl+C to stop.")
    try:
        while True:
            t0 = time.perf_counter()
            robot.send_action(keyboard.get_action())
            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        pass
    finally:
        keyboard.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    main()
