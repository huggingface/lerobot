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
local$ python lerobot/scripts/visualize_dataset_html.py \
    --repo-id lerobot/pusht

local$ open http://localhost:9090
```

- Visualize data stored on a distant machine with a local viewer:
```
distant$ python lerobot/scripts/visualize_dataset_html.py \
    --repo-id lerobot/pusht

local$ ssh -L 9090:localhost:9090 distant  # create a ssh tunnel
local$ open http://localhost:9090
```

- Select episodes to visualize:
```
python lerobot/scripts/visualize_dataset_html.py \
    --repo-id lerobot/pusht \
    --episodes 7 3 5 1 4
```
"""

import argparse
import http.server
import logging
import shutil
import warnings
from pathlib import Path
from typing import List

import torch
import tqdm
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from flask import Flask, render_template, url_for, redirect

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import init_hydra_config, init_logging


class EpisodeSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, episode_index):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self):
        return len(self.frame_ids)


class NoCacheHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def run_server(dataset: LeRobotDataset, episodes: List[int], port: str, static_folder: Path, template_folder: Path):
    app = Flask(__name__, static_folder=static_folder.resolve(), template_folder=template_folder.resolve())
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0 # specifying not to cache

    @app.route('/')
    def index():
        # home page redirects to the first episode page
        first_episode_id = episodes[0]
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
        return render_template('visualize_dataset_template.html', episode_id=episode_id, episodes=episodes, dataset_info=dataset_info, videos_info=videos_info, ep_csv_url=ep_csv_url)

    app.run(port=port)


def get_ep_csv_fname(episode_id: int):
    ep_csv_fname = f"episode_{episode_id}.csv"
    return ep_csv_fname


def write_episode_data_csv(output_dir, file_name, episode_index, dataset, inference_results=None):
    """Write a csv file containg timeseries data of an episode (e.g. state and action).
    This file will be loaded by Dygraph javascript to plot data in real time."""
    from_idx = dataset.episode_data_index["from"][episode_index]
    to_idx = dataset.episode_data_index["to"][episode_index]

    has_state = "observation.state" in dataset.hf_dataset.features
    has_action = "action" in dataset.hf_dataset.features
    has_inference = inference_results is not None

    # init header of csv with state and action names
    header = ["timestamp"]
    if has_state:
        dim_state = len(dataset.hf_dataset["observation.state"][0])
        header += [f"state_{i}" for i in range(dim_state)]
    if has_action:
        dim_action = len(dataset.hf_dataset["action"][0])
        header += [f"action_{i}" for i in range(dim_action)]
    if has_inference:
        if "actions" in inference_results:
            dim_pred_action = inference_results["actions"].shape[2]
            header += [f"pred_action_{i}" for i in range(dim_pred_action)]
        if "loss" in inference_results:
            header += ["loss"]

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

    if has_inference:
        num_frames = len(rows)
        if "actions" in inference_results:
            assert num_frames == inference_results["actions"].shape[0]
            for i in range(num_frames):
                rows[i] += inference_results["actions"][i, 0].tolist()
        if "loss" in inference_results:
            assert num_frames == inference_results["loss"].shape[0]
            for i in range(num_frames):
                rows[i] += [inference_results["loss"][i].item()]

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


def run_inference(
    dataset, episode_index, policy, policy_method="select_action", num_workers=4, batch_size=32, device="cuda"
):
    if policy_method not in ["select_action", "forward"]:
        raise ValueError(
            f"`policy_method` is expected to be 'select_action' or 'forward', but '{policy_method}' is provided instead."
        )

    policy.eval()
    policy.to(device)

    logging.info("Loading dataloader")
    episode_sampler = EpisodeSampler(dataset, episode_index)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        sampler=episode_sampler,
    )

    warned_ndim_eq_0 = False
    warned_ndim_gt_2 = False

    logging.info("Running inference")
    inference_results = {}
    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.inference_mode():
            if policy_method == "select_action":
                output_dict = {"action": policy.select_action(batch)}
            elif policy_method == "forward":
                output_dict = policy.forward(batch)

        for key in output_dict:
            if output_dict[key].ndim == 0:
                if not warned_ndim_eq_0:
                    warnings.warn(
                        f"Ignore output key '{key}'. Its value is a scalar instead of a vector. It might have been aggregated over the batch dimension (e.g. `loss.mean()`).",
                        stacklevel=1,
                    )
                    warned_ndim_eq_0 = True
                continue

            if output_dict[key].ndim > 2:
                if not warned_ndim_gt_2:
                    warnings.warn(
                        f"Ignore output key '{key}'. Its value is a tensor of {output_dict[key].ndim} dimensions instead of a vector.",
                        stacklevel=1,
                    )
                    warned_ndim_gt_2 = True
                continue

            if key not in inference_results:
                inference_results[key] = []
            inference_results[key].append(output_dict[key].to("cpu"))

    for key in inference_results:
        inference_results[key] = torch.cat(inference_results[key])

    return inference_results


def visualize_dataset_html(
    repo_id: str,
    episodes: list[int] = None,
    output_dir: Path | None = None,
    serve: bool = True,
    port: int = 9090,
    force_override: bool = True,
    policy_repo_id: str | None = None,
    policy_ckpt_path: Path | None = None,
    policy_method: str = "select_action",
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = "cuda",
) -> Path | None:
    init_logging()

    has_policy = policy_repo_id or policy_ckpt_path

    if has_policy:
        logging.info("Loading policy")
        if policy_repo_id:
            pretrained_policy_path = Path(snapshot_download(policy_repo_id))
        elif policy_ckpt_path:
            pretrained_policy_path = Path(policy_ckpt_path)

        cfg = init_hydra_config(pretrained_policy_path / "config.yaml")
        dataset = make_dataset(cfg)
        policy = make_policy(cfg, pretrained_policy_path)

        if policy_method == "select_action":
            # Do not load previous observations or future actions, to simulate that the observations come from
            # an environment.
            dataset.delta_timestamps = None
    else:
        dataset = LeRobotDataset(repo_id)

    if not dataset.video:
        raise NotImplementedError(f"Image datasets ({dataset.video=}) are currently not supported.")

    if output_dir is None:
        output_dir = f"outputs/visualize_dataset_html/{repo_id}"

    output_dir = Path(output_dir)
    if force_override and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a simlink from the dataset video folder containg mp4 files to the output directory
    # so that the http server can get access to the mp4 files.
    static_dir = output_dir / 'static'
    static_dir.mkdir(parents=True, exist_ok=True)
    ln_videos_dir = static_dir / 'videos'
    if not ln_videos_dir.exists():
        ln_videos_dir.symlink_to(dataset.videos_dir.resolve())

    template_dir = Path(__file__).resolve().parent

    if episodes is None:
        episodes = list(range(dataset.num_episodes))

    logging.info("Writing CSV files")
    for episode_index in tqdm.tqdm(episodes):
        inference_results = None
        if has_policy:
            inference_results_path = output_dir / f"episode_{episode_index}.safetensors"
            if inference_results_path.exists():
                inference_results = load_file(inference_results_path)
            else:
                inference_results = run_inference(
                    dataset, episode_index, policy, policy_method, num_workers, batch_size, device
                )
                save_file(inference_results, inference_results_path)

        # write states and actions in a csv
        write_episode_data_csv(static_dir, get_ep_csv_fname(episode_index), episode_index, dataset, inference_results)

    if serve:
        run_server(dataset, episodes, port, static_dir, template_dir)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Name of hugging face repositery containing a LeRobotDataset dataset (e.g. `lerobot/pusht` for https://huggingface.co/datasets/lerobot/pusht).",
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

    parser.add_argument(
        "--policy-repo-id",
        type=str,
        default=None,
        help="Name of hugging face repositery containing a pretrained policy (e.g. `lerobot/diffusion_pusht` for https://huggingface.co/lerobot/diffusion_pusht).",
    )
    parser.add_argument(
        "--policy-ckpt-path",
        type=str,
        default=None,
        help="Path hugging face repositery containing a pretrained policy (e.g. `lerobot/diffusion_pusht` for https://huggingface.co/lerobot/diffusion_pusht).",
    )
    parser.add_argument(
        "--policy-method",
        type=str,
        default="select_action",
        choices=["select_action", "forward"],
        help="Python method used to run the inference. It can be `forward` used during training to compute the loss, or `select_action` used during evaluation to output the sequence of actions.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size loaded by DataLoader.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of processes of Dataloader for loading the data.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device used to run inference.",
    )

    args = parser.parse_args()
    visualize_dataset_html(**vars(args))


if __name__ == "__main__":
    main()