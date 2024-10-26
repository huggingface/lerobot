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
import csv
import logging
import shutil
from io import StringIO
from pathlib import Path
import re
import tempfile
import os

import numpy as np
from flask import Flask, redirect, render_template, url_for, request
from datasets import load_dataset

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.utils.utils import init_logging
from lerobot import available_datasets


def run_server(
    dataset: LeRobotDataset | dict | None,
    episodes: list[int] | None,
    host: str,
    port: str,
    static_folder: Path,
    template_folder: Path,
):
    app = Flask(__name__, static_folder=static_folder.resolve(), template_folder=template_folder.resolve())
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0  # specifying not to cache

    @app.route("/")
    def hommepage(dataset=dataset):
        if dataset:
            dataset_namespace, dataset_name = (
                dataset.repo_id if isinstance(dataset, LeRobotDataset) else dataset["repo_id"]
            ).split("/")
            return redirect(
                url_for(
                    "show_episode",
                    dataset_namespace=dataset_namespace,
                    dataset_name=dataset_name,
                    episode_id=0,
                )
            )

        dataset_param, episode_param, time_param = None, None, None
        all_params = request.args
        if "dataset" in all_params:
            dataset_param = all_params["dataset"]
        if "episode" in all_params:
            episode_param = int(all_params["episode"])
        if "t" in all_params:
            time_param = all_params["t"]

        if dataset_param:
            dataset_namespace, dataset_name = dataset_param.split("/")
            return redirect(
                url_for(
                    "show_episode",
                    dataset_namespace=dataset_namespace,
                    dataset_name=dataset_name,
                    episode_id=episode_param if episode_param is not None else 0,
                )
            )

        featured_datasets = [
            "lerobot/aloha_static_cups_open",
            "lerobot/columbia_cairlab_pusht_real",
            "lerobot/taco_play",
        ]
        return render_template(
            "visualize_dataset_homepage.html",
            featured_datasets=featured_datasets,
            lerobot_datasets=available_datasets,
        )

    @app.route("/<string:dataset_namespace>/<string:dataset_name>")
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

    @app.route("/<string:dataset_namespace>/<string:dataset_name>/episode_<int:episode_id>")
    def show_episode(dataset_namespace, dataset_name, episode_id, dataset=dataset, episodes=episodes):
        repo_id = f"{dataset_namespace}/{dataset_name}"
        try:
            if dataset is None:
                dataset = get_dataset_info(repo_id)
        except FileNotFoundError:
            return "Make sure your convert your LeRobotDataset to v2 & above."
        dataset_version = (
            dataset._version if isinstance(dataset, LeRobotDataset) else dataset["codebase_version"]
        )
        match = re.search(r"v(\d+)\.", dataset_version)
        if match:
            major_version = int(match.group(1))
            if major_version < 2:
                return "Make sure your convert your LeRobotDataset to v2 & above."

        episode_data_csv_str = get_episode_data_csv_str(dataset, episode_id)
        dataset_info = {
            "repo_id": f"{dataset_namespace}/{dataset_name}",
            "num_samples": dataset.num_samples
            if isinstance(dataset, LeRobotDataset)
            else dataset["total_frames"],
            "num_episodes": dataset.num_episodes
            if isinstance(dataset, LeRobotDataset)
            else dataset["total_episodes"],
            "fps": dataset.fps if isinstance(dataset, LeRobotDataset) else dataset["fps"],
        }
        if isinstance(dataset, LeRobotDataset):
            video_paths = [dataset.get_video_file_path(episode_id, key) for key in dataset.video_keys]
            videos_info = [
                {"url": url_for("static", filename=video_path), "filename": video_path.name}
                for video_path in video_paths
            ]
            tasks = dataset.episode_dicts[episode_id]["tasks"]
        else:
            videos_info = [
                {
                    "url": f"https://huggingface.co/datasets/{repo_id}/resolve/main/"
                    + dataset["videos"]["videos_path"].format(
                        episode_chunk=int(episode_id) // dataset["chunks_size"],
                        video_key=video_key,
                        episode_index=episode_id,
                    ),
                    "filename": video_key,
                }
                for video_key in dataset["video_keys"]
            ]
            tasks_jsonl = load_dataset(
                "json",
                data_files=f"https://huggingface.co/datasets/{repo_id}/resolve/main/meta/episodes.jsonl",
                split="train",
            )
            filtered_tasks_jsonl = tasks_jsonl.filter(lambda x: x["episode_index"] == episode_id)
            tasks = filtered_tasks_jsonl["tasks"][0]

        videos_info[0]["language_instruction"] = tasks

        if episodes is None:
            episodes = list(
                range(
                    dataset.num_episodes if isinstance(dataset, LeRobotDataset) else dataset["total_episodes"]
                )
            )

        return render_template(
            "visualize_dataset_template.html",
            episode_id=episode_id,
            episodes=episodes,
            dataset_info=dataset_info,
            videos_info=videos_info,
            has_policy=False,
            episode_data_csv_str=episode_data_csv_str,
        )

    app.run(host=host, port=port)


def get_ep_csv_fname(episode_id: int):
    ep_csv_fname = f"episode_{episode_id}.csv"
    return ep_csv_fname


def get_episode_data_csv_str(dataset: LeRobotDataset | dict, episode_index):
    """Get a csv str containing timeseries data of an episode (e.g. state and action).
    This file will be loaded by Dygraph javascript to plot data in real time."""
    has_state = "observation.state" in (
        dataset.hf_dataset.features if isinstance(dataset, LeRobotDataset) else dataset["keys"]
    )
    has_action = "action" in (
        dataset.hf_dataset.features if isinstance(dataset, LeRobotDataset) else dataset["keys"]
    )

    # init header of csv with state and action names
    header = ["timestamp"]
    if has_state:
        dim_state = (dataset.shapes if isinstance(dataset, LeRobotDataset) else dataset["shapes"])[
            "observation.state"
        ]
        header += [f"state_{i}" for i in range(dim_state)]
    if has_action:
        dim_action = (dataset.shapes if isinstance(dataset, LeRobotDataset) else dataset["shapes"])["action"]
        header += [f"action_{i}" for i in range(dim_action)]

    if isinstance(dataset, LeRobotDataset):
        from_idx = dataset.episode_data_index["from"][episode_index]
        to_idx = dataset.episode_data_index["to"][episode_index]
        columns = ["timestamp"]
        if has_state:
            columns += ["observation.state"]
        if has_action:
            columns += ["action"]
        data = dataset.hf_dataset.select(range(from_idx, to_idx)).select_columns(columns).with_format("numpy")
        rows = np.hstack(
            (np.expand_dims(data["timestamp"], axis=1), *[data[col] for col in columns[1:]])
        ).tolist()
    else:
        repo_id = dataset["repo_id"]
        columns = ["timestamp"]
        if "observation.state" in dataset["keys"]:
            columns.append("observation.state")
        if "action" in dataset["keys"]:
            columns.append("action")
        episode_parquet = load_dataset(
            "parquet",
            data_files=f"https://huggingface.co/datasets/{repo_id}/resolve/main/"
            + dataset["data_path"].format(
                episode_chunk=int(episode_index) // dataset["chunks_size"], episode_index=episode_index
            ),
            split="train",
        )
        d = episode_parquet.select_columns(columns).with_format("numpy")
        data = d.to_pandas()
        rows = np.hstack(
            (np.expand_dims(data["timestamp"], axis=1), *[np.vstack(data[col]) for col in columns[1:]])
        ).tolist()

    # Convert data to CSV string
    csv_buffer = StringIO()
    csv_writer = csv.writer(csv_buffer)
    # Write header
    csv_writer.writerow(header)
    # Write data rows
    csv_writer.writerows(rows)
    csv_string = csv_buffer.getvalue()

    return csv_string


def get_episode_video_paths(dataset: LeRobotDataset, ep_index: int) -> list[str]:
    # get first frame of episode (hack to get video_path of the episode)
    first_frame_idx = dataset.episode_data_index["from"][ep_index].item()
    return [
        dataset.hf_dataset.select_columns(key)[first_frame_idx][key]["path"] for key in dataset.video_keys
    ]


def get_episode_language_instruction(dataset: LeRobotDataset, ep_index: int) -> list[str]:
    # check if the dataset has language instructions
    if "language_instruction" not in dataset.hf_dataset.features:
        return None

    # get first frame index
    first_frame_idx = dataset.episode_data_index["from"][ep_index].item()

    language_instruction = dataset.hf_dataset[first_frame_idx]["language_instruction"]
    # TODO (michel-aractingi) hack to get the sentence, some strings in openx are badly stored
    # with the tf.tensor appearing in the string
    return language_instruction.removeprefix("tf.Tensor(b'").removesuffix("', shape=(), dtype=string)")


def get_dataset_info(repo_id: str) -> dict:
    dataset_info = load_dataset(
        "json",
        data_files=f"https://huggingface.co/datasets/{repo_id}/resolve/main/meta/info.json",
        split="train",
    )[0]
    dataset_info["repo_id"] = repo_id
    return dataset_info


def visualize_dataset_html(
    repo_id: str | None = None,
    root: Path | None = None,
    load_from_hf_hub: bool = False,
    episodes: list[int] | None = None,
    output_dir: Path | None = None,
    serve: bool = True,
    host: str = "127.0.0.1",
    port: int = 9090,
    force_override: bool = False,
) -> Path | None:
    init_logging()

    template_dir = Path(__file__).resolve().parent.parent / "templates"

    if output_dir is None:
        # Create a temporary directory that will be automatically cleaned up
        temp_base = tempfile.mkdtemp(prefix="lerobot_visualize_dataset_")
        output_dir = temp_base if not repo_id else os.path.join(temp_base, repo_id)

    output_dir = Path(output_dir)
    if output_dir.exists():
        if force_override:
            shutil.rmtree(output_dir)
        else:
            logging.info(f"Output directory already exists. Loading from it: '{output_dir}'")

    output_dir.mkdir(parents=True, exist_ok=True)

    static_dir = output_dir / "static"
    static_dir.mkdir(parents=True, exist_ok=True)

    if not repo_id:
        if serve:
            run_server(
                dataset=None,
                episodes=None,
                host=host,
                port=port,
                static_folder=static_dir,
                template_folder=template_dir,
            )
    else:
        dataset = LeRobotDataset(repo_id, root=root) if not load_from_hf_hub else get_dataset_info(repo_id)

        image_keys = dataset.image_keys if isinstance(dataset, LeRobotDataset) else dataset["image_keys"]
        if len(image_keys) > 0:
            raise NotImplementedError(f"Image keys ({image_keys=}) are currently not supported.")

        # Create a simlink from the dataset video folder containg mp4 files to the output directory
        # so that the http server can get access to the mp4 files.
        if isinstance(dataset, LeRobotDataset):
            ln_videos_dir = static_dir / "videos"
            if not ln_videos_dir.exists():
                ln_videos_dir.symlink_to((dataset.root / "videos").resolve())

        if serve:
            run_server(dataset, episodes, host, port, static_dir, template_dir)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Name of hugging face repositery containing a LeRobotDataset dataset (e.g. `lerobot/pusht` for https://huggingface.co/datasets/lerobot/pusht).",
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
        "--output-dir",
        type=Path,
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
        "--force-override",
        type=int,
        default=0,
        help="Delete the output directory if it exists already.",
    )

    args = parser.parse_args()
    visualize_dataset_html(**vars(args))


if __name__ == "__main__":
    main()
