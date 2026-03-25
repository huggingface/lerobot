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

"""run this command to open a viewer of the bi-so follower simulated robot:
python -m lerobot.simulations.bi_so.view_simulation --sim-root \lerobot\src\lerobot\robots\bi_so_follower_simulated\mujoco 
"""

from __future__ import annotations

import argparse
import time

from lerobot.simulations.bi_so.runtime import add_common_sim_args, build_sim_helper
from lerobot.utils.utils import init_logging


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_common_sim_args(parser)
    parser.set_defaults(launch_viewer=True)
    parser.add_argument("--fps", type=float, default=60.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    init_logging()

    sim_helper = build_sim_helper(args, sim_id="bimanual_so_follower_viewer")
    try:
        sim_helper.connect()
        while True:
            time.sleep(max(1.0 / float(args.fps), 0.001))
    except KeyboardInterrupt:
        return 0
    finally:
        try:
            sim_helper.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
