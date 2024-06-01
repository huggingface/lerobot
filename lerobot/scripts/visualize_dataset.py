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
import http.server
import logging
import os
import shutil
import socketserver
from pathlib import Path

import torch
import tqdm
import yaml
from bs4 import BeautifulSoup
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.utils.utils import init_logging


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


def run_server(path, port):
    # Change directory to serve 'index.html` as front page
    os.chdir(path)

    with socketserver.TCPServer(("", port), NoCacheHTTPRequestHandler) as httpd:
        logging.info(f"Serving HTTP on 0.0.0.0 port {port} (http://0.0.0.0:{port}/) ...")
        httpd.serve_forever()


def create_html_page(page_title: str):
    """Create a html page with beautiful soop with default doctype, meta, header and title."""
    soup = BeautifulSoup("", "html.parser")

    doctype = soup.new_tag("!DOCTYPE html")
    soup.append(doctype)

    html = soup.new_tag("html", lang="en")
    soup.append(html)

    head = soup.new_tag("head")
    html.append(head)

    meta_charset = soup.new_tag("meta", charset="UTF-8")
    head.append(meta_charset)

    meta_viewport = soup.new_tag(
        "meta", attrs={"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    )
    head.append(meta_viewport)

    title = soup.new_tag("title")
    title.string = page_title
    head.append(title)

    body = soup.new_tag("body")
    html.append(body)

    main_div = soup.new_tag("div")
    body.append(main_div)
    return soup, head, body


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
        assert "actions" in inference_results
        assert "loss" in inference_results
        dim_pred_action = inference_results["actions"].shape[2]
        header += [f"pred_action_{i}" for i in range(dim_pred_action)]
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
        assert num_frames == inference_results["actions"].shape[0]
        assert num_frames == inference_results["loss"].shape[0]
        for i in range(num_frames):
            rows[i] += inference_results["actions"][i, 0].tolist()
            rows[i] += [inference_results["loss"][i].item()]

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / file_name, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            row_str = [str(col) for col in row]
            f.write(",".join(row_str) + "\n")


def write_episode_data_js(output_dir, file_name, ep_csv_fname, dataset):
    """Write a javascript file containing logic to synchronize camera feeds and timeseries."""
    s = ""
    s += "document.addEventListener('DOMContentLoaded', function () {\n"
    for i, key in enumerate(dataset.video_frame_keys):
        s += f"  const video{i} = document.getElementById('video_{key}');\n"
    s += "  const slider = document.getElementById('videoControl');\n"
    s += "  const playButton = document.getElementById('playButton');\n"
    s += f"  const dygraph = new Dygraph(document.getElementById('graph'), '{ep_csv_fname}', " + "{\n"
    s += "    pixelsPerPoint: 0.01,\n"
    s += "    legend: 'always',\n"
    s += "    labelsDiv: document.getElementById('labels'),\n"
    s += "    labelsSeparateLines: true,\n"
    s += "    labelsKMB: true,\n"
    s += "    highlightCircleSize: 1.5,\n"
    s += "    highlightSeriesOpts: {\n"
    s += "        strokeWidth: 1.5,\n"
    s += "        strokeBorderWidth: 1,\n"
    s += "        highlightCircleSize: 3\n"
    s += "    }\n"
    s += "  });\n"
    s += "\n"
    s += "  // Function to play both videos\n"
    s += "  playButton.addEventListener('click', function () {\n"
    for i in range(len(dataset.video_frame_keys)):
        s += f"    video{i}.play();\n"
    s += "    // playButton.disabled = true; // Optional: disable button after playing\n"
    s += "  });\n"
    s += "\n"
    s += "  // Update the video time when the slider value changes\n"
    s += "  slider.addEventListener('input', function () {\n"
    s += "    const sliderValue = slider.value;\n"
    for i in range(len(dataset.video_frame_keys)):
        s += f"    const time{i} = (video{i}.duration * sliderValue) / 100;\n"
    for i in range(len(dataset.video_frame_keys)):
        s += f"    video{i}.currentTime = time{i};\n"
    s += "  });\n"
    s += "\n"
    s += "  // Synchronize slider with the video's current time\n"
    s += "  const syncSlider = (video) => {\n"
    s += "    video.addEventListener('timeupdate', function () {\n"
    s += "      if (video.duration) {\n"
    s += "        const pc = (100 / video.duration) * video.currentTime;\n"
    s += "        slider.value = pc;\n"
    s += "        const index = Math.floor(pc * dygraph.numRows() / 100);\n"
    s += "        dygraph.setSelection(index, undefined, true, true);\n"
    s += "      }\n"
    s += "    });\n"
    s += "  };\n"
    s += "\n"
    for i in range(len(dataset.video_frame_keys)):
        s += f"  syncSlider(video{i});\n"
    s += "\n"
    s += "});\n"

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / file_name, "w", encoding="utf-8") as f:
        f.write(s)


def write_episode_data_html(output_dir, file_name, js_fname, ep_index, dataset):
    """Write an html file containg video feeds and timeseries associated to an episode."""
    soup, head, body = create_html_page("")

    css_style = soup.new_tag("style")
    css_style.string = ""
    css_style.string += "#labels > span.highlight {\n"
    css_style.string += "  border: 1px solid grey;\n"
    css_style.string += "}"
    head.append(css_style)

    # Add videos from camera feeds

    videos_control_div = soup.new_tag("div")
    body.append(videos_control_div)

    videos_div = soup.new_tag("div")
    videos_control_div.append(videos_div)

    def create_video(id, src):
        video = soup.new_tag("video", id=id, width="320", height="240", controls="")
        source = soup.new_tag("source", src=src, type="video/mp4")
        video.string = "Your browser does not support the video tag."
        video.append(source)
        return video

    # get first frame of episode (hack to get video_path of the episode)
    first_frame_idx = dataset.episode_data_index["from"][ep_index].item()

    for key in dataset.video_frame_keys:
        # Example of video_path: 'videos/observation.image_episode_000004.mp4'
        video_path = dataset.hf_dataset.select_columns(key)[first_frame_idx][key]["path"]
        videos_div.append(create_video(f"video_{key}", video_path))

    # Add controls for videos and graph

    control_div = soup.new_tag("div")
    videos_control_div.append(control_div)

    button_div = soup.new_tag("div")
    control_div.append(button_div)

    button = soup.new_tag("button", id="playButton")
    button.string = "Play Videos"
    button_div.append(button)

    slider_div = soup.new_tag("div")
    control_div.append(slider_div)

    slider = soup.new_tag("input", type="range", id="videoControl", min="0", max="100", value="0", step="1")
    control_div.append(slider)

    # Add graph of states/actions, and its labels

    graph_labels_div = soup.new_tag("div", style="display: flex;")
    body.append(graph_labels_div)

    graph_div = soup.new_tag("div", id="graph", style="flex: 1; width: 85%")
    graph_labels_div.append(graph_div)

    labels_div = soup.new_tag("div", id="labels", style="flex: 1; width: 15%")
    graph_labels_div.append(labels_div)

    # add dygraph library
    script = soup.new_tag("script", type="text/javascript", src=js_fname)
    body.append(script)

    script_dygraph = soup.new_tag(
        "script",
        type="text/javascript",
        src="https://cdn.jsdelivr.net/npm/dygraphs@2.1.0/dist/dygraph.min.js",
    )
    body.append(script_dygraph)

    link_dygraph = soup.new_tag(
        "link", rel="stylesheet", href="https://cdn.jsdelivr.net/npm/dygraphs@2.1.0/dist/dygraph.min.css"
    )
    body.append(link_dygraph)

    # Write as a html file

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / file_name, "w", encoding="utf-8") as f:
        f.write(soup.prettify())


def write_episodes_list_html(output_dir, file_name, ep_indices, ep_html_fnames, dataset):
    """Write an html file containing information related to the dataset and a list of links to
    html pages of episodes."""
    soup, head, body = create_html_page("TODO")

    h3 = soup.new_tag("h3")
    h3.string = "TODO"
    body.append(h3)

    ul_info = soup.new_tag("ul")
    body.append(ul_info)

    li_info = soup.new_tag("li")
    li_info.string = f"Number of samples/frames: {dataset.num_samples}"
    ul_info.append(li_info)

    li_info = soup.new_tag("li")
    li_info.string = f"Number of episodes: {dataset.num_episodes}"
    ul_info.append(li_info)

    li_info = soup.new_tag("li")
    li_info.string = f"Frames per second: {dataset.fps}"
    ul_info.append(li_info)

    # li_info = soup.new_tag("li")
    # li_info.string = f"Size: {format_big_number(dataset.hf_dataset.info.size_in_bytes)}B"
    # ul_info.append(li_info)

    ul = soup.new_tag("ul")
    body.append(ul)

    for ep_idx, ep_html_fname in zip(ep_indices, ep_html_fnames, strict=False):
        li = soup.new_tag("li")
        ul.append(li)

        a = soup.new_tag("a", href=ep_html_fname)
        a.string = f"Episode number {ep_idx}"

        li.append(a)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / file_name, "w", encoding="utf-8") as f:
        f.write(soup.prettify())


def run_inference(dataset, episode_index, policy, num_workers=4, batch_size=32, device="cuda"):
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

    logging.info("Running inference")
    inference_results = {}
    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.inference_mode():
            output_dict = policy.forward(batch)

        for key in output_dict:
            if key not in inference_results:
                inference_results[key] = []
            inference_results[key].append(output_dict[key].to("cpu"))

    for key in inference_results:
        inference_results[key] = torch.cat(inference_results[key])

    return inference_results


def visualize_dataset(
    repo_id: str,
    episode_indices: list[int] = None,
    output_dir: Path | None = None,
    serve: bool = True,
    port: int = 9090,
    force_overwrite: bool = True,
    policy_repo_id: str | None = None,
    policy_ckpt_path: Path | None = None,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Path | None:
    init_logging()

    has_policy = policy_repo_id or policy_ckpt_path

    if has_policy:
        logging.info("Loading policy")
        if policy_repo_id:
            pretrained_policy_path = Path(snapshot_download(policy_repo_id))
        elif policy_ckpt_path:
            pretrained_policy_path = Path(policy_ckpt_path)
        policy = ACTPolicy.from_pretrained(pretrained_policy_path)
        with open(pretrained_policy_path / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        delta_timestamps = cfg["training"]["delta_timestamps"]
    else:
        delta_timestamps = None

    logging.info("Loading dataset")
    dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)

    if not dataset.video:
        raise NotImplementedError(f"Image datasets ({dataset.video=}) are currently not supported.")

    if output_dir is None:
        output_dir = f"outputs/visualize_dataset/{repo_id}"

    output_dir = Path(output_dir)
    if force_overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a simlink from the dataset video folder containg mp4 files to the output directory
    # so that the http server can get access to the mp4 files.
    ln_videos_dir = output_dir / "videos"
    if not ln_videos_dir.exists():
        ln_videos_dir.symlink_to(dataset.videos_dir.resolve())

    if episode_indices is None:
        episode_indices = list(range(dataset.num_episodes))

    logging.info("Writing html")
    ep_html_fnames = []
    for episode_index in tqdm.tqdm(episode_indices):
        inference_results = None
        if has_policy:
            inference_results_path = output_dir / f"episode_{episode_index}.safetensors"
            if inference_results_path.exists():
                inference_results = load_file(inference_results_path)
            else:
                inference_results = run_inference(dataset, episode_index, policy)
                save_file(inference_results, inference_results_path)

        # write states and actions in a csv
        ep_csv_fname = f"episode_{episode_index}.csv"
        write_episode_data_csv(output_dir, ep_csv_fname, episode_index, dataset, inference_results)

        js_fname = f"episode_{episode_index}.js"
        write_episode_data_js(output_dir, js_fname, ep_csv_fname, dataset)

        # write a html page to view videos and timeseries
        ep_html_fname = f"episode_{episode_index}.html"
        write_episode_data_html(output_dir, ep_html_fname, js_fname, episode_index, dataset)
        ep_html_fnames.append(ep_html_fname)

    write_episodes_list_html(output_dir, "index.html", episode_indices, ep_html_fnames, dataset)

    if serve:
        run_server(output_dir, port)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Name of hugging face repositery containing a LeRobotDataset dataset (e.g. `lerobot/pusht` for https://huggingface.co/datasets/lerobot/pusht).",
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
        "--port",
        type=int,
        default=9090,
        help="Web port used by the http server.",
    )
    parser.add_argument(
        "--force-overwrite",
        type=int,
        default=1,
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
        help="Name of hugging face repositery containing a pretrained policy (e.g. `lerobot/diffusion_pusht` for https://huggingface.co/lerobot/diffusion_pusht).",
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

    args = parser.parse_args()
    visualize_dataset(**vars(args))


if __name__ == "__main__":
    main()
