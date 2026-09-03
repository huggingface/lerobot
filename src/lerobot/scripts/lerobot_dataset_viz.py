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
""" Visualize data of **all** frames of any episode(s) of a dataset of type LeRobotDataset.

Multi-episode extension: Supports single episode, multiple specific episodes,
a range of episodes, or all episodes — each as a separate Rerun recording.

Requires: pip install 'lerobot[dataset_viz]'  (includes dataset + viz extras)

Note: The last frame of the episode doesn't always correspond to a final state.
That's because our datasets are composed of transition from state to state up to
the antepenultimate state associated to the ultimate action to arrive in the final state.
However, there might not be a transition from a final state to another state.

Note: This script aims to visualize the data used to train the neural networks.
~What you see is what you get~. When visualizing image modality, it is often expected to observe
lossy compression artifacts since these images have been decoded from compressed mp4 videos to
save disk space. The compression factor applied has been tuned to not affect success rate.

Examples:

- Visualize a single episode:
```
local$ python viz_multi.py \
    --repo-id lerobot/pusht \
    --episode-index 0
```

- Visualize multiple specific episodes:
```
local$ python viz_multi.py \
    --repo-id lerobot/pusht \
    --episode-indices 0,1,2,3
```

- Visualize a range of episodes:
```
local$ python viz_multi.py \
    --repo-id lerobot/pusht \
    --episode-start 0 \
    --episode-end 20
```

- Visualize all episodes:
```
local$ python viz_multi.py \
    --repo-id lerobot/pusht \
    --all
```

- Visualize data stored on a distant machine with a local viewer:
```
distant$ python viz_multi.py \
    --repo-id lerobot/pusht \
    --episode-index 0 \
    --save 1 \
    --output-dir path/to/directory

local$ scp distant:path/to/directory/lerobot_pusht_episode_0.rrd .
local$ rerun lerobot_pusht_episode_0.rrd
```

- Visualize data stored on a distant machine through streaming:
```
distant$ python viz_multi.py \
    --repo-id lerobot/pusht \
    --episode-index 0 \
    --mode distant \
    --grpc-port 9876

local$ rerun rerun+http://IP:GRPC_PORT/proxy
```

- Visualize data in Foxglove with a seekable, scrubbable timeline:
```
local$ python viz_multi.py \
    --repo-id lerobot/pusht \
    --episode-index 0 \
    --display-mode foxglove

# then open the Foxglove app and connect to ws://127.0.0.1:8765
```
This starts a Foxglove WebSocket server that serves the episode on demand from the on-disk dataset,
so you can play/pause and scrub anywhere in the episode using Foxglove's playback controls.

"""

import argparse
import gc
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
import tqdm

from lerobot.configs import DEPTH_MILLIMETER_UNIT
from lerobot.datasets import LeRobotDataset
from lerobot.utils.constants import ACTION, DONE, OBS_STATE, REWARD, SUCCESS
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)

DEFAULT_FOXGLOVE_PORT = 8765
DEFAULT_RERUN_PORT = 9090


def get_episode_indices(args, repo_id, root, tolerance_s) -> list[int]:
    """Resolve which episodes to visualize based on CLI arguments.

    Priority: --all > --episode-indices > --episode-start+--episode-end > --episode-index > default [0].
    """
    # --all: load dataset metadata to get total episode count
    if args.all:
        tmp_dataset = LeRobotDataset(
            repo_id,
            root=root,
            tolerance_s=tolerance_s,
        )
        return list(range(tmp_dataset.num_episodes))

    # --episode-indices: parse comma-separated list, e.g. "0,3,5"
    if args.episode_indices is not None:
        return [int(x) for x in args.episode_indices.split(",")]

    # --episode-start + --episode-end: inclusive range
    if args.episode_start is not None and args.episode_end is not None:
        return list(range(args.episode_start, args.episode_end + 1))

    # --episode-index: single episode (original behaviour)
    if args.episode_index is not None:
        return [args.episode_index]

    # Default: show episode 0
    return [0]


def get_feature_names(dataset: LeRobotDataset, key: str) -> list[str]:
    """Return per-dimension names for a feature from the dataset metadata.

    Only flat-list ``names`` metadata is used. Dict-style ``names`` and missing names fall back to ``{key}_{i}`` indices.
    """
    feature = dataset.features[key]
    dim = feature["shape"][-1]

    names = feature.get("names")
    if isinstance(names, list) and len(names) == dim:
        return [str(name) for name in names]

    return [f"{key}_{d}" for d in range(dim)]


def check_chw_float32(frame: torch.Tensor) -> None:
    """
    Check if a frame is a channel-first, float32 tensor.
    """
    assert frame.dtype == torch.float32
    assert frame.ndim == 3
    c, h, w = frame.shape
    assert c < h and c < w, f"expect channel first images, but instead {frame.shape}"


def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    check_chw_float32(chw_float32_torch)
    hwc_uint8_numpy = (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()
    return hwc_uint8_numpy


def to_hwc_float32_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    check_chw_float32(chw_float32_torch)
    hwc_float32_numpy = chw_float32_torch.permute(1, 2, 0).numpy()
    return hwc_float32_numpy


def build_blueprint_from_dataset(dataset: LeRobotDataset):
    """Build a Rerun blueprint laying out camera images and time series for the given dataset.

    Camera images and scalar signals (action, state, reward, done, success) are arranged in a grid.
    The per-dimension series names for ``action`` and ``state`` are applied directly
    via blueprint overrides.
    """
    import rerun as rr
    import rerun.blueprint as rrb

    views = [rrb.Spatial2DView(origin=key, name=key) for key in dataset.meta.camera_keys]

    # Style multi-dimensional signals (action, state) with per-dimension names.
    for origin, key in ((ACTION, ACTION), ("state", OBS_STATE)):
        if key in dataset.features:
            names = get_feature_names(dataset, key)
            styling = rr.SeriesLines(names=names)
            views.append(rrb.TimeSeriesView(origin=origin, name=origin, overrides={origin: styling}))
    for key in (DONE, REWARD, SUCCESS):
        if key in dataset.features:
            views.append(rrb.TimeSeriesView(origin=key, name=key))

    return rrb.Blueprint(rrb.Grid(*views))


def log_episode_frames(
    dataloader,
    recording: "RecordingStream",
    depth_meter: float,
    depth_ranges: dict[str, tuple[float, float]],
    camera_keys: list[str],
    depth_keys: set[str],
    display_compressed_images: bool,
):
    """Log all frames of a single episode to the given recording stream.

    Uses recording methods directly to avoid any ambiguity about which recording
    receives the data.
    """
    import rerun as rr

    first_index = None
    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        if first_index is None:
            first_index = batch["index"][0].item()

        # iterate over the batch
        for i in range(len(batch["index"])):
            recording.set_time("frame_index", sequence=batch["index"][i].item() - first_index)
            recording.set_time("timestamp", timestamp=batch["timestamp"][i].item())

            # display each camera image (or depth map)
            for key in camera_keys:
                if key in depth_keys:
                    depth = to_hwc_float32_numpy(batch[key][i])
                    depth_entity = rr.DepthImage(
                        depth,
                        meter=depth_meter,
                        colormap=rr.components.Colormap.Viridis,
                        depth_range=depth_ranges.get(key),
                    )
                    recording.log(key, depth_entity)
                else:
                    img = to_hwc_uint8_numpy(batch[key][i])
                    if display_compressed_images:
                        img_entity = rr.Image(img).compress()
                    else:
                        img_entity = rr.Image(img)
                    recording.log(key, img_entity)

            # display the action space (e.g. actuators command)
            if ACTION in batch:
                recording.log(ACTION, rr.Scalars(batch[ACTION][i].numpy()))

            # display the observed state space (e.g. agent position in joint space)
            if OBS_STATE in batch:
                recording.log("state", rr.Scalars(batch[OBS_STATE][i].numpy()))

            if DONE in batch:
                recording.log(DONE, rr.Scalars(batch[DONE][i].item()))

            if REWARD in batch:
                recording.log(REWARD, rr.Scalars(batch[REWARD][i].item()))

            if SUCCESS in batch:
                recording.log(SUCCESS, rr.Scalars(batch[SUCCESS][i].item()))


def visualize_dataset(
    dataset: LeRobotDataset,
    episode_index: int,
    batch_size: int = 32,
    num_workers: int = 0,
    mode: str = "local",
    web_port: int | None = None,
    grpc_port: int = 9876,
    save: bool = False,
    output_dir: Path | None = None,
    display_compressed_images: bool = False,
    display_mode: str = "rerun",
    host: str = "127.0.0.1",
    autoplay: bool = True,
    **kwargs,
) -> Path | None:
    """Log a single episode to Rerun as its own recording.

    Returns the path to the saved .rrd file if ``save=True``, otherwise None.
    """
    if display_mode == "foxglove":
        from lerobot.utils.foxglove_visualization import serve_foxglove_dataset_playback

        logging.info("Starting Foxglove server")
        serve_foxglove_dataset_playback(
            dataset,
            episode_index,
            host=host,
            port=web_port if web_port is not None else DEFAULT_FOXGLOVE_PORT,
            compress_images=display_compressed_images,
            autoplay=autoplay,
        )
        return None

    if save:
        assert output_dir is not None, (
            "Set an output directory where to write .rrd files with `--output-dir path/to/directory`."
        )

    repo_id = dataset.repo_id

    logging.info("Loading dataloader")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    if mode not in ["local", "distant"]:
        raise ValueError(mode)

    from lerobot.utils.import_utils import require_package

    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr
    from rerun import RecordingStream

    # Create a dedicated recording for this episode.
    # Using RecordingStream directly (not rr.init()) allows us to create multiple
    # distinct recordings within the same process without triggering the cleanup
    # that rr.init() does (which flushes and destroys previous recordings).
    recording = RecordingStream(
        repo_id,
        recording_id=f"episode_{episode_index}",
        make_default=True,
    )

    if mode == "distant":
        recording.connect_grpc(f"rerun+http://127.0.0.1:{grpc_port}/proxy")
        logging.info(f"Connect to a Rerun Server: rerun rerun+http://127.0.0.1:{grpc_port}/proxy")

    # Build and send the blueprint
    blueprint = build_blueprint_from_dataset(dataset)
    recording.send_blueprint(blueprint)

    # Manually call python garbage collector after creating the recording to avoid
    # hanging in a blocking flush when iterating on a dataloader with num_workers > 0
    gc.collect()

    # Depth frames and stats are dequantized to the dataset's depth_output_unit on load.
    depth_meter = 1000.0 if dataset.depth_output_unit == DEPTH_MILLIMETER_UNIT else 1.0

    # Use the dataset's q01/q99 depth statistics for robust depth range bounds
    depth_ranges = {}
    for key in dataset.meta.depth_keys:
        stats = (dataset.meta.stats or {}).get(key)
        if not stats:
            continue
        lo = stats["q01"] if "q01" in stats else stats["min"]
        hi = stats["q99"] if "q99" in stats else stats["max"]
        depth_ranges[key] = (float(np.asarray(lo).item()), float(np.asarray(hi).item()))

    logging.info("Logging to Rerun")

    log_episode_frames(
        dataloader,
        recording,
        depth_meter,
        depth_ranges,
        dataset.meta.camera_keys,
        set(dataset.meta.depth_keys),
        display_compressed_images,
    )

    # save .rrd locally
    if mode == "local" and save:
        output_dir.mkdir(parents=True, exist_ok=True)
        repo_id_str = repo_id.replace("/", "_")
        rrd_path = output_dir / f"{repo_id_str}_episode_{episode_index}.rrd"
        recording.save(str(rrd_path))
        return rrd_path

    elif mode == "distant":
        # Keep the process alive while it serves the gRPC/web connection.
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Ctrl-C received. Exiting.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Name of hugging face repository containing a LeRobotDataset dataset (e.g. `lerobot/pusht`).",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=None,
        help="Visualize a single episode (e.g. `--episode-index 0`).",
    )
    parser.add_argument(
        "--episode-indices",
        type=str,
        default=None,
        help="Visualize multiple episodes (e.g. `--episode-indices 0,1,2,3`).",
    )
    parser.add_argument(
        "--episode-start",
        type=int,
        default=None,
        help="Start episode index for a range (use with `--episode-end`).",
    )
    parser.add_argument(
        "--episode-end",
        type=int,
        default=None,
        help="End episode index for a range (use with `--episode-start`).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Visualize all episodes in the dataset.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for the dataset stored locally (e.g. `--root data`). By default, the dataset will be loaded from hugging face cache folder, or downloaded from the hub if available.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory path to write a .rrd file when `--save 1` is set.",
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
        "--mode",
        type=str,
        default="local",
        help=(
            "Mode of viewing between 'local' or 'distant'. "
            "'local' requires data to be on a local machine. It spawns a viewer to visualize the data locally. "
            "'distant' creates a server on the distant machine where the data is stored. "
            "Visualize the data by connecting to the server with `rerun rerun+http://IP:GRPC_PORT/proxy` on the local machine."
        ),
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=None,
        help=(
            "Web/WebSocket port. For rerun `--mode distant` it is the web viewer port (default 9090); "
            "for `--display-mode foxglove` it is the server bind port (default 8765)."
        ),
    )
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=9876,
        help="gRPC port for rerun.io when `--mode distant` is set.",
    )
    parser.add_argument(
        "--save",
        type=int,
        default=0,
        help=(
            "Save a .rrd file in the directory provided by `--output-dir`. "
            "It also deactivates the spawning of a viewer. "
            "Visualize the data by running `rerun path/to/file.rrd` on your local machine."
        ),
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

    parser.add_argument(
        "--display-compressed-images",
        action="store_true",
        help="If set, display compressed (JPEG) images instead of uncompressed ones.",
    )

    parser.add_argument(
        "--display-mode",
        type=str,
        default="rerun",
        choices=["rerun", "foxglove"],
        help=(
            "Visualization backend. 'rerun' uses the Rerun viewer (--mode/--save/--*-port apply). "
            "'foxglove' starts a Foxglove WebSocket server that serves the episode as a seekable, "
            "scrubbable timeline; connect the Foxglove app to ws://HOST:PORT (--host/--web-port)."
        ),
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help=(
            "Host to bind the Foxglove WebSocket server to when `--display-mode foxglove` is set "
            "(127.0.0.1 for local only, 0.0.0.0 for all interfaces)."
        ),
    )
    parser.add_argument(
        "--no-autoplay",
        dest="autoplay",
        action="store_false",
        help=(
            "For `--display-mode foxglove`: don't start playing automatically when a client "
            "connects; wait for play to be pressed in the Foxglove app instead."
        ),
    )

    args = parser.parse_args()

    if args.display_mode == "foxglove":
        rerun_only = ("mode", "save", "output_dir", "grpc_port", "batch_size", "num_workers")
        ignored = [name for name in rerun_only if getattr(args, name) != parser.get_default(name)]
        if ignored:
            logging.warning(
                "These flags only apply to `--display-mode rerun` and are ignored with "
                "`--display-mode foxglove`: %s.",
                ", ".join(f"--{name.replace('_', '-')}" for name in ignored),
            )

        # Foxglove only supports a single episode
        if args.all or args.episode_indices is not None or (args.episode_start is not None and args.episode_end is not None):
            raise ValueError(
                "Foxglove display mode only supports a single episode. "
                "Use `--episode-index N` or switch to `--display-mode rerun`."
            )

    init_logging()
    logging.info("Loading dataset")

    repo_id = args.repo_id
    root = args.root
    tolerance_s = args.tolerance_s

    episode_indices = get_episode_indices(args, repo_id, root, tolerance_s)
    logging.info(f"Loading episodes: {episode_indices}")

    # Build kwargs dict for visualize_dataset (excluding episode-selection args)
    viz_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "mode": args.mode,
        "web_port": args.web_port,
        "grpc_port": args.grpc_port,
        "save": args.save,
        "output_dir": args.output_dir,
        "display_compressed_images": args.display_compressed_images,
        "display_mode": args.display_mode,
        "host": args.host,
        "autoplay": args.autoplay,
    }

    # Import rerun once
    from lerobot.utils.import_utils import require_package

    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr
    from rerun import RecordingStream

    # Keep a reference to all RecordingStream objects to prevent Python GC
    # from destroying them while other episodes are still logging.
    all_recordings_pool: list[RecordingStream] = []

    for i, ep_idx in enumerate(episode_indices):
        logging.info(f"Episode {ep_idx}/{episode_indices[-1]}")

        dataset = LeRobotDataset(
            repo_id,
            episodes=[ep_idx],
            root=root,
            tolerance_s=tolerance_s,
        )

        if args.mode == "local" and not args.save and i == 0:
            # First episode in local mode: rr.init() with spawn=True opens the viewer
            # and creates the first recording in one call.
            rr.init(
                repo_id,
                recording_id=f"episode_{ep_idx}",
                spawn=True,
                default_blueprint=build_blueprint_from_dataset(dataset),
            )
            gc.collect()
            recording = rr.get_data_recording()
        elif args.mode == "local" and not args.save and i > 0:
            # Subsequent episodes: create a RecordingStream that connects to the
            # already-running viewer via gRPC (default rerun+http://127.0.0.1:9876/proxy).
            rec = RecordingStream(
                repo_id,
                recording_id=f"episode_{ep_idx}",
                make_default=True,
            )
            rec.connect_grpc()
            rec.send_blueprint(build_blueprint_from_dataset(dataset))
            gc.collect()
            recording = rec
            all_recordings_pool.append(rec)
        else:
            # save mode or distant mode: use visualize_dataset helper
            visualize_dataset(
                dataset,
                episode_index=ep_idx,
                **viz_kwargs,
            )
            continue

        # Build dataloader for this episode
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
        )

        # Depth config
        depth_meter = 1000.0 if dataset.depth_output_unit == DEPTH_MILLIMETER_UNIT else 1.0
        depth_ranges = {}
        for key in dataset.meta.depth_keys:
            stats = (dataset.meta.stats or {}).get(key)
            if not stats:
                continue
            lo = stats["q01"] if "q01" in stats else stats["min"]
            hi = stats["q99"] if "q99" in stats else stats["max"]
            depth_ranges[key] = (float(np.asarray(lo).item()), float(np.asarray(hi).item()))

        log_episode_frames(
            dataloader,
            recording,
            depth_meter,
            depth_ranges,
            dataset.meta.camera_keys,
            set(dataset.meta.depth_keys),
            args.display_compressed_images,
        )


if __name__ == "__main__":
    main()