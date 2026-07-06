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

"""CLI/self-test shim for the live ``rt/smpl`` SONIC reference stream.

The implementation now lives in the installed package so it can be shared with the
``pico_headset`` teleoperator:
``lerobot.robots.unitree_g1.smpl_stream``. This example re-exports it (so
``from smpl_stream import SmplStream`` keeps working next to ``sonic.py``) and adds a
standalone smoke-test entrypoint.

Example:
    python examples/unitree_g1/smpl_stream.py --smpl-host 127.0.0.1 --smpl-port 5560
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from lerobot.robots.unitree_g1.smpl_stream import (
    DEFAULT_SMPL_HOST,
    DEFAULT_SMPL_PORT,
    SMPL_OBS_DIM,
    SMPL_TOPIC,
    SmplStream,
)

__all__ = [
    "DEFAULT_SMPL_HOST",
    "DEFAULT_SMPL_PORT",
    "SMPL_OBS_DIM",
    "SMPL_TOPIC",
    "SmplStream",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smpl-host", default=DEFAULT_SMPL_HOST)
    parser.add_argument("--smpl-port", type=int, default=DEFAULT_SMPL_PORT)
    parser.add_argument("--ticks", type=int, default=250, help="control ticks to sample")
    args = parser.parse_args()

    stream = SmplStream(host=args.smpl_host, port=args.smpl_port)
    print(f"Subscribed to {SMPL_TOPIC} @ tcp://{args.smpl_host}:{args.smpl_port}")
    print("Start pico_manager_thread_server.py --manager on the publisher host.")
    try:
        for t in range(args.ticks):
            w = stream.step()
            assert w.shape == (SMPL_OBS_DIM,), w.shape
            if t % 25 == 0:
                print(
                    f"  t={t} idx={stream._last_index} window_norm={np.linalg.norm(w):.3f} "
                    f"first={stream.has_data}"
                )
            time.sleep(1.0 / 50.0)
    finally:
        stream.close()
    print("OK: rt/smpl stream yields SONIC-format 720-vec windows.")


if __name__ == "__main__":
    main()
