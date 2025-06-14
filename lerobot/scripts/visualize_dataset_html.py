#!/usr/bin/env python

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
""" Visualize data of **all** frames of any episode of a dataset of type LeRobotDataset.

Note: The last frame of the episode doesnt always correspond to a final state.
That's because our datasets are composed of transition from state to state up to
the antepenultimate state associated to the ultimate action to arrive in the final state.
However, there might not be a transition from a final state to another state.

Note: This script aims to visualize the data used to train the neural networks.
~What you see is what you get~. When visualizing image modality, it is often expected to observe
lossly compression artifacts since these images have been decoded from compressed mp4 videos to
save disk space. The compression factor applied has been tuned to not affect success rate.

Example of usage:

- Visualize data stored on a local machine:
```bash
local$ python lerobot/scripts/visualize_dataset_html.py \
    --repo-id lerobot/pusht

local$ open http://localhost:9090
```

- Visualize data stored on a distant machine with a local viewer:
```bash
distant$ python lerobot/scripts/visualize_dataset_html.py \
    --repo-id lerobot/pusht

local$ ssh -L 9090:localhost:9090 distant  # create a ssh tunnel
local$ open http://localhost:9090
```

- Select episodes to visualize:
```bash
python lerobot/scripts/visualize_dataset_html.py \
    --repo-id lerobot/pusht \
    --episodes 7 3 5 1 4
```
"""

import argparse
import atexit
import contextlib
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from flask import Flask, jsonify, redirect, send_file, url_for

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import DEFAULT_PARQUET_PATH, DEFAULT_VIDEO_PATH, INFO_PATH
from lerobot.common.utils.utils import init_logging


def run_data_server(
    dataset: LeRobotDataset | None,
    host: str,
    port: int,
) -> Path | None:
    init_logging()

    data_server = Flask(__name__)
    data_server.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0  # specifying not to cache

    @data_server.route("/<string:dataset_namespace>/<string:dataset_name>")
    def show_first_episode(dataset_namespace, dataset_name):
        first_episode_id = 0
        return redirect(
            url_for(
                "show_episode",
                dataset_namespace=dataset_namespace,
                dataset_name=dataset_name,
                episode_id=first_episode_id,
            )
        )

    @data_server.route("/<string:dataset_namespace>/<string:dataset_name>/resolve/main/meta/info.json")
    def serve_info_json(dataset_namespace, dataset_name):
        try:
            return send_file(dataset.root / INFO_PATH, mimetype="application/json")
        except FileNotFoundError:
            return jsonify({"error": "File not found"}), 404
        except Exception as e:
            return jsonify({"error": f"Server error: {str(e)}"}), 500

    @data_server.route(
        "/<string:dataset_namespace>/<string:dataset_name>/resolve/main/data/chunk-<int:episode_chunk>/episode_<int:episode_index>.parquet"
    )
    def serve_parquet_file(dataset_namespace, dataset_name, episode_chunk, episode_index):
        try:
            # Format the path with the captured parameters
            file_path = DEFAULT_PARQUET_PATH.format(episode_chunk=episode_chunk, episode_index=episode_index)

            full_path = dataset.root / file_path

            return send_file(full_path, mimetype="application/octet-stream")
        except FileNotFoundError:
            return jsonify({"error": "File not found"}), 404
        except Exception as e:
            return jsonify({"error": f"Server error: {str(e)}"}), 500

    @data_server.route(
        "/<string:dataset_namespace>/<string:dataset_name>/resolve/main/videos/chunk-<int:episode_chunk>/<string:video_key>/episode_<int:episode_index>.mp4"
    )
    def serve_video_file(dataset_namespace, dataset_name, episode_chunk, video_key, episode_index):
        try:
            # Format the path with the captured parameters
            file_path = DEFAULT_VIDEO_PATH.format(
                episode_chunk=episode_chunk, video_key=video_key, episode_index=episode_index
            )

            # Assuming 'dataset' object has a 'root' attribute
            full_path = dataset.root / file_path

            return send_file(full_path, mimetype="video/mp4")
        except FileNotFoundError:
            return jsonify({"error": "Video file not found"}), 404
        except Exception as e:
            return jsonify({"error": f"Server error: {str(e)}"}), 500

    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)

    data_server.run(host=host, port=get_local_data_server_port(port))


def is_npm_available():
    npm_path = shutil.which("npm")
    if npm_path is None:
        return False
    try:
        subprocess.run([npm_path, "--version"], capture_output=True, text=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def build_react_app(script_dir: Path):
    next_dir = script_dir.parent / "html_dataset_visualizer"
    next_build_dir = next_dir / ".next"
    npm_path = shutil.which("npm")
    if npm_path is None:
        raise RuntimeError(
            "'npm' executable not found in PATH. Please ensure Node.js is installed and npm is available."
        )
    if not next_build_dir.exists() or not next_build_dir.is_dir():
        print("Building React.js app ...")
        subprocess.run([npm_path, "ci"], cwd=next_dir)
        subprocess.run([npm_path, "run", "build"], cwd=next_dir)

    package_json_path = next_dir / "package.json"
    build_id_path = next_build_dir / "BUILD_ID"
    with open(package_json_path) as f:
        package_data = json.load(f)
        package_version = package_data.get("version", "")
    with open(build_id_path) as f:
        build_id = f.read().strip()

    if package_version != build_id:
        print("Building React.js app ...")
        subprocess.run([npm_path, "ci"], cwd=next_dir)
        subprocess.run([npm_path, "run", "build"], cwd=next_dir)


def run_react_app(
    repo_id: str,
    script_dir: Path,
    load_from_hf_hub: bool,
    host: str,
    port: int,
    episodes: list[int] | None = None,
):
    next_dir = script_dir.parent / "html_dataset_visualizer"

    env = os.environ.copy()
    env["REPO_ID"] = repo_id
    if not load_from_hf_hub:
        env["DATASET_URL"] = f"http://{host}:{get_local_data_server_port(port)}"
    if episodes:
        env["EPISODES"] = " ".join(map(str, episodes))

    npm_path = shutil.which("npm")
    if npm_path is None:
        raise RuntimeError(
            "'npm' executable not found in PATH. Please ensure Node.js is installed and npm is available."
        )
    process = subprocess.Popen(
        [npm_path, "run", "start", "--", f"--port={port}"], cwd=next_dir, env=env, preexec_fn=os.setsid
    )

    def cleanup():
        if process.poll() is None:  # Process still running
            print("Cleaning up React server...")
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                # Force kill if graceful termination fails
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)

    def signal_handler(sig, frame):
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    atexit.register(cleanup)  # Also cleanup on normal exit

    return process


def get_local_data_server_port(port: str):
    """Returns the port used by the local data server."""
    return str(int(port) + 1)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-id",
        type=str,
        help="Name of hugging face repositery containing a LeRobotDataset dataset (e.g. `lerobot/pusht` for https://huggingface.co/datasets/lerobot/pusht).",
        required=True,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for a dataset stored locally (e.g. `--root data`). By default, the dataset will be loaded from hugging face cache folder, or downloaded from the hub if available.",
    )
    parser.add_argument(
        "--load-from-hf-hub",
        type=int,
        default=0,
        help="Load videos and parquet files from HF Hub rather than local system.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        default=None,
        help="Episode indices to visualize (e.g. `0 1 5 6` to load episodes of index 0, 1, 5 and 6). By default loads all episodes.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Web host used by the http server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9090,
        help="Web port used by the http server.",
    )
    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help=(
            "Tolerance in seconds used to ensure data timestamps respect the dataset fps value"
            "This is argument passed to the constructor of LeRobotDataset and maps to its tolerance_s constructor argument"
            "If not given, defaults to 1e-4."
        ),
    )

    args = parser.parse_args()
    kwargs = vars(args)
    repo_id = kwargs.pop("repo_id")
    load_from_hf_hub = kwargs.pop("load_from_hf_hub")
    root = kwargs.pop("root")
    tolerance_s = kwargs.pop("tolerance_s")
    host = kwargs.pop("host")
    port = kwargs.pop("port")
    episodes = kwargs.pop("episodes")

    if not is_npm_available():
        raise RuntimeError("npm is not available. Please install it to use this script.")

    script_dir = Path(__file__).parent.absolute()

    build_react_app(script_dir)
    run_react_app(repo_id, script_dir, load_from_hf_hub, host, port, episodes)

    if not load_from_hf_hub:
        dataset = LeRobotDataset(repo_id, root=root, tolerance_s=tolerance_s)
        run_data_server(dataset, host, port)


if __name__ == "__main__":
    main()
