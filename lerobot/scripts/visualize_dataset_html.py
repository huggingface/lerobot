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
import logging
import shutil
import warnings
from pathlib import Path

import torch
import tqdm
from flask import Flask, redirect, render_template, url_for

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import init_hydra_config, init_logging
from lerobot.scripts.eval import get_pretrained_policy_path


class EpisodeSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, episode_index):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self):
        return len(self.frame_ids)


def run_server(
    dataset: LeRobotDataset,
    episodes: list[int],
    host: str,
    port: str,
    static_folder: Path,
    template_folder: Path,
    has_policy = False,
):
    app = Flask(__name__, static_folder=static_folder.resolve(), template_folder=template_folder.resolve())
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0  # specifying not to cache

    @app.route("/")
    def index():
        # home page redirects to the first episode page
        [dataset_namespace, dataset_name] = dataset.repo_id.split("/")
        first_episode_id = episodes[0]
        return redirect(
            url_for(
                "show_episode",
                dataset_namespace=dataset_namespace,
                dataset_name=dataset_name,
                episode_id=first_episode_id,
            )
        )

    @app.route("/<string:dataset_namespace>/<string:dataset_name>/episode_<int:episode_id>")
    def show_episode(dataset_namespace, dataset_name, episode_id):
        dataset_info = {
            "repo_id": dataset.repo_id,
            "num_samples": dataset.num_frames,
            "num_episodes": dataset.num_episodes,
            "fps": dataset.fps,
        }
        video_paths = [dataset.meta.get_video_file_path(episode_id, key) for key in dataset.meta.video_keys]
        tasks = dataset.meta.episodes[episode_id]["tasks"]
        videos_info = [
            {"url": url_for("static", filename=video_path), "filename": video_path.name}
            for video_path in video_paths
        ]
        videos_info[0]["language_instruction"] = tasks

        ep_csv_url = url_for("static", filename=get_ep_csv_fname(episode_id))
        return render_template(
            "visualize_dataset_template.html",
            episode_id=episode_id,
            episodes=episodes,
            dataset_info=dataset_info,
            videos_info=videos_info,
            ep_csv_url=ep_csv_url,
            has_policy = has_policy,
        )

    app.run(host=host, port=port)


def run_inference(
    dataset, episode_index, policy, policy_method="select_action", num_workers=4, batch_size=32, device="mps"
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
        # When using `select_action`, we set batch size 1 so that we feed 1 frame at a time, in a continuous fashion.
        batch_size=1 if policy_method == "select_action" else batch_size,
        sampler=episode_sampler,
        drop_last=False,
    )

    warned_ndim_eq_0 = False
    warned_ndim_gt_2 = False

    logging.info("Running inference")
    inference_results = {}
    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.inference_mode():
            if policy_method == "select_action":
                gt_action = batch.pop("action")
                output_dict = {"action": policy.select_action(batch)}
                batch["action"] = gt_action
            elif policy_method == "forward":
                output_dict = policy.forward(batch)
                # TODO(rcadene): Save and display all predicted actions at a given timestamp
                # Save predicted action for the next timestamp only
                output_dict["action"] = output_dict["action"][:, 0, :]

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


def get_ep_csv_fname(episode_id: int):
    ep_csv_fname = f"episode_{episode_id}.csv"
    return ep_csv_fname


def write_episode_data_csv(output_dir, file_name, episode_index, dataset, policy=None):
    """Write a csv file containg timeseries data of an episode (e.g. state and action).
    This file will be loaded by Dygraph javascript to plot data in real time."""

    if policy is not None:
        inference_results = run_inference(
            dataset,
            episode_index,
            policy,
            policy_method="select_action",
            # num_workers=hydra_cfg.training.num_workers,
            # batch_size=hydra_cfg.training.batch_size,
            # device=hydra_cfg.device,
        )

    from_idx = dataset.episode_data_index["from"][episode_index]
    to_idx = dataset.episode_data_index["to"][episode_index]

    has_state = "observation.state" in dataset.features
    has_action = "action" in dataset.features

    # init header of csv with state and action names
    header = ["timestamp"]
    if has_state:
        dim_state = dataset.meta.shapes["observation.state"][0]
        header += [f"state_{i}" for i in range(dim_state)]
    if has_action:
        dim_action = dataset.meta.shapes["action"][0]
        header += [f"action_{i}" for i in range(dim_action)]
    if policy is not None:
        dim_action = dataset.meta.shapes["action"][0]
        header += [f"pred_action_{i}" for i in range(dim_action)]

    columns = ["timestamp"]
    if has_state:
        columns += ["observation.state"]
    if has_action:
        columns += ["action"]
    data = dataset.hf_dataset.select_columns(columns)

    rows = []
    for i in range(from_idx, to_idx):
        row = [data[i]["timestamp"].item()]
        if has_state:
            row += data[i]["observation.state"].tolist()
        if has_action:
            row += data[i]["action"].tolist()
        if policy is not None:
            row += inference_results["action"][i].tolist()
        rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / file_name, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            row_str = [str(col) for col in row]
            f.write(",".join(row_str) + "\n")


def get_episode_video_paths(dataset: LeRobotDataset, ep_index: int) -> list[str]:
    # get first frame of episode (hack to get video_path of the episode)
    first_frame_idx = dataset.episode_data_index["from"][ep_index].item()
    return [
        dataset.hf_dataset.select_columns(key)[first_frame_idx][key]["path"]
        for key in dataset.meta.video_keys
    ]


def visualize_dataset_html(
    dataset: LeRobotDataset,
    episodes: list[int] = None,
    output_dir: Path | None = None,
    serve: bool = True,
    host: str = "127.0.0.1",
    port: int = 9090,
    force_override: bool = False,
    policy_method: str = "select_action",
    pretrained_policy_name_or_path: str | None = None,
    policy_overrides: list[str] | None = None,
) -> Path | None:
    init_logging()

    if len(dataset.meta.image_keys) > 0:
        raise NotImplementedError(f"Image keys ({dataset.meta.image_keys=}) are currently not supported.")

    if output_dir is None:
        output_dir = f"outputs/visualize_dataset_html/{dataset.repo_id}"

    output_dir = Path(output_dir)
    if output_dir.exists():
        if force_override:
            shutil.rmtree(output_dir)
        else:
            logging.info(f"Output directory already exists. Loading from it: '{output_dir}'")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a simlink from the dataset video folder containg mp4 files to the output directory
    # so that the http server can get access to the mp4 files.
    static_dir = output_dir / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    ln_videos_dir = static_dir / "videos"
    if not ln_videos_dir.exists():
        ln_videos_dir.symlink_to((dataset.root / "videos").resolve())

    template_dir = Path(__file__).resolve().parent.parent / "templates"

    if episodes is None:
        episodes = list(range(dataset.num_episodes))

    pretrained_policy_name_or_path = "aliberts/act_reachy_test_model"

    policy = None
    if pretrained_policy_name_or_path is not None:
        logging.info("Loading policy")
        pretrained_policy_path = get_pretrained_policy_path(pretrained_policy_name_or_path)

        hydra_cfg = init_hydra_config(pretrained_policy_path / "config.yaml", overrides=["device=mps"])
        # dataset = make_dataset(hydra_cfg)
        policy = make_policy(hydra_cfg, pretrained_policy_name_or_path=pretrained_policy_path)

        if policy_method == "select_action":
            # Do not load previous observations or future actions, to simulate that the observations come from
            # an environment.
            dataset.delta_timestamps = None

    logging.info("Writing CSV files")
    for episode_index in tqdm.tqdm(episodes):
        # write states and actions in a csv (it can be slow for big datasets)
        ep_csv_fname = get_ep_csv_fname(episode_index)
        # TODO(rcadene): speedup script by loading directly from dataset, pyarrow, parquet, safetensors?
        write_episode_data_csv(static_dir, ep_csv_fname, episode_index, dataset, policy=policy)

    if serve:
        run_server(dataset, episodes, host, port, static_dir, template_dir, has_policy=policy is not None)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Name of hugging face repositery containing a LeRobotDataset dataset (e.g. `lerobot/pusht` for https://huggingface.co/datasets/lerobot/pusht).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for a dataset stored locally (e.g. `--root data`). By default, the dataset will be loaded from hugging face cache folder, or downloaded from the hub if available.",
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
    kwargs = vars(args)
    repo_id = kwargs.pop("repo_id")
    # root = kwargs.pop("root")
    dataset = LeRobotDataset(repo_id)
    visualize_dataset_html(dataset, **kwargs)


if __name__ == "__main__":
    main()
