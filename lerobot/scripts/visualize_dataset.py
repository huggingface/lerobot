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

Examples:

- Visualize data stored on a local machine:
```
local$ python lerobot/scripts/visualize_dataset.py \
    --repo-id lerobot/pusht

local$ open http://localhost:9090
```

- Visualize data stored on a distant machine with a local viewer:
```
distant$ python lerobot/scripts/visualize_dataset.py \
    --repo-id lerobot/pusht

local$ ssh -L 9090:localhost:9090 distant  # create a ssh tunnel
local$ open http://localhost:9090
```

- Select episodes to visualize:
```
python lerobot/scripts/visualize_dataset.py \
    --repo-id lerobot/pusht \
    --episode-indices 7 3 5 1 4
```
"""

import argparse
import logging
import shutil
import webbrowser
from typing import List
from pathlib import Path

import tqdm
from flask import Flask, render_template, url_for, redirect

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.utils.utils import init_logging


def run_server(dataset: LeRobotDataset, episode_indices: List[int], port: str, open: bool, static_folder: Path, template_folder: Path):
    app = Flask(__name__, static_folder=static_folder.resolve(), template_folder=template_folder.resolve())
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0 # specifying not to cache

    @app.route('/')
    def index():
        # home page redirects to the first episode page
        first_episode_id = episode_indices[0]
        return redirect(url_for('show_episode', episode_id=first_episode_id))

    @app.route('/episode_<int:episode_id>')
    def show_episode(episode_id):
        dataset_info = {
            "repo_id": dataset.repo_id,
            'num_samples': dataset.num_samples,
            'num_episodes': dataset.num_episodes,
            'fps': dataset.fps,
        }
        video_paths = get_episode_video_paths(dataset, episode_id)
        videos_info = [{"url": url_for('static', filename=video_path), "filename": Path(video_path).name} for video_path in video_paths]
        ep_csv_url = url_for('static', filename=get_ep_csv_fname(episode_id))
        return render_template('visualize_dataset_template.html', episode_id=episode_id, episode_indices=episode_indices, dataset_info=dataset_info, videos_info=videos_info, ep_csv_url=ep_csv_url)

    if open:
        webbrowser.open_new_tab(f"http://127.0.0.1:{port}")
    app.run(port=port)


def get_ep_csv_fname(episode_id: int):
    ep_csv_fname = f"episode_{episode_id}.csv"
    return ep_csv_fname

def write_episode_data_csv(output_dir: Path, file_name: str, episode_index: int, dataset: LeRobotDataset):
    """Write a csv file containg timeseries data of an episode (e.g. state and action).
    This file will be loaded by Dygraph javascript to plot data in real time."""
    from_idx = dataset.episode_data_index["from"][episode_index]
    to_idx = dataset.episode_data_index["to"][episode_index]

    has_state = "observation.state" in dataset.hf_dataset.features
    has_action = "action" in dataset.hf_dataset.features

    # init header of csv with state and action names
    header = ["timestamp"]
    if has_state:
        dim_state = len(dataset.hf_dataset["observation.state"][0])
        header += [f"state_{i}" for i in range(dim_state)]
    if has_action:
        dim_action = len(dataset.hf_dataset["action"][0])
        header += [f"action_{i}" for i in range(dim_action)]

    columns = ["timestamp"]
    if has_state:
        columns += ["observation.state"]
    if has_action:
        columns += ["action"]

    rows = []
    data = dataset.hf_dataset.select_columns(columns)
    for i in range(from_idx, to_idx):
        row = [data[i]["timestamp"].item()]
        if has_state:
            row += data[i]["observation.state"].tolist()
        if has_action:
            row += data[i]["action"].tolist()
        rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / file_name, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            row_str = [str(col) for col in row]
            f.write(",".join(row_str) + "\n")


def get_episode_video_paths(dataset: LeRobotDataset, ep_index: int) -> List[str]:
    # get first frame of episode (hack to get video_path of the episode)
    first_frame_idx = dataset.episode_data_index["from"][ep_index].item()
    return [dataset.hf_dataset.select_columns(key)[first_frame_idx][key]["path"] for key in dataset.video_frame_keys]


def visualize_dataset(
    repo_id: str,
    episode_indices: list[int] = None,
    output_dir: Path | None = None,
    serve: bool = True,
    open: bool = True,
    port: int = 9090,
) -> Path | None:
    init_logging()

    logging.info("Loading dataset")
    dataset = LeRobotDataset(repo_id)

    if not dataset.video:
        raise NotImplementedError(f"Image datasets ({dataset.video=}) are currently not supported.")

    if output_dir is None:
        output_dir = f"outputs/visualize_dataset/{repo_id}"

    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a simlink from the dataset video folder containg mp4 files to the output directory
    # so that the http server can get access to the mp4 files.
    static_dir = output_dir / 'static'
    static_dir.mkdir(parents=True, exist_ok=True)
    ln_videos_dir = static_dir / 'videos'
    ln_videos_dir.symlink_to(dataset.videos_dir.resolve())

    template_dir = Path(__file__).resolve().parent

    if episode_indices is None:
        episode_indices = list(range(dataset.num_episodes))

    logging.info("Writing sensor data CSV files")
    for episode_idx in tqdm.tqdm(episode_indices):
        # write states and actions in a csv
        write_episode_data_csv(static_dir, get_ep_csv_fname(episode_idx), episode_idx, dataset)

    if serve:
        run_server(dataset, episode_indices, port, open, static_dir, template_dir)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Name of hugging face repositery containing a LeRobotDataset dataset (e.g. `lerobot/pusht`).",
    )
    parser.add_argument(
        "--episode-indices",
        type=int,
        nargs="*",
        default=None,
        help="Episode indices to visualize (e.g. `0 1 5 6` to load episodes of index 0, 1, 5 and 6). By default loads all episodes.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory path to write html files and kickoff a web server. By default write them to 'outputs/visualize_dataset/REPO_ID'.",
    )
    parser.add_argument(
        "--serve",
        type=int,
        default=1,
        help="Launch web server.",
    )
    parser.add_argument(
        "--open",
        type=int,
        default=1,
        help="Launch web browser.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9090,
        help="Web port used by the http server.",
    )

    args = parser.parse_args()
    visualize_dataset(**vars(args))


if __name__ == "__main__":
    main()
