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
Replay a dataset episode in EE frame using a browser-based URDF viewer.

Extracts ``observation.pose`` from the dataset, saves a trajectory JSON file,
then launches a local HTTP server and opens the replay viewer.  The trajectory
is re-centered so frame 0 starts at the OpenArm ``openarm_right_ee_target``
EE tip (zero-joint pose).

Usage:
    python replay.py
    python replay.py --episode 3 --repo-id myuser/mydata
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import threading
import webbrowser
from pathlib import Path

VIEWER_DIR = Path(__file__).resolve().parents[2] / "src/lerobot/robots/openarm_follower/urdf"
TRAJECTORY_FILENAME = "trajectory_ep0.json"


def extract_trajectory(repo_id: str, episode: int, output_path: Path) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id, episodes=[episode])
    poses = dataset.select_columns("observation.pose")
    actions = dataset.select_columns("action")

    frames = []
    for i in range(dataset.num_frames):
        p = poses[i]["observation.pose"]
        a = actions[i]["action"]
        frames.append(
            {
                "x": float(p[0]),
                "y": float(p[1]),
                "z": float(p[2]),
                "ax": float(p[3]),
                "ay": float(p[4]),
                "az": float(p[5]),
                "proximal": float(a[0]),
                "distal": float(a[1]),
            }
        )
    payload = {"fps": dataset.fps, "num_frames": dataset.num_frames, "frames": frames}
    with open(output_path, "w") as f:
        json.dump(payload, f)
    print(f"Extracted {dataset.num_frames} frames at {dataset.fps} FPS → {output_path}")
    return payload


# ---------------------------------------------------------------------------
# Viewer mode
# ---------------------------------------------------------------------------


def serve_and_open(directory: Path, port: int = 8765):
    os.chdir(directory)
    handler = http.server.SimpleHTTPRequestHandler
    httpd = http.server.HTTPServer(("", port), handler)
    url = f"http://localhost:{port}/replay_viewer.html"
    print(f"Serving at {url}")
    threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        httpd.server_close()


def run_viewer(args):
    trajectory_path = VIEWER_DIR / TRAJECTORY_FILENAME
    if not trajectory_path.exists() or args.force:
        extract_trajectory(args.repo_id, args.episode, trajectory_path)
    else:
        print(f"Using cached trajectory at {trajectory_path}  (pass --force to re-extract)")
    serve_and_open(VIEWER_DIR, args.port)


def main():
    parser = argparse.ArgumentParser(description="Replay a dataset episode in EE frame (URDF viewer)")
    parser.add_argument("--repo-id", default="glannuzel/grabette-dataset")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--force", action="store_true", help="Re-extract trajectory even if cached")
    args = parser.parse_args()
    run_viewer(args)


if __name__ == "__main__":
    main()
