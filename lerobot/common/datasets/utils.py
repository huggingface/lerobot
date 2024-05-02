import json
from pathlib import Path

import datasets
import torch
from datasets import load_dataset, load_from_disk
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image as PILImage
from safetensors.torch import load_file
from torchvision import transforms


def flatten_dict(d, parent_key="", sep="/"):
    """Flatten a nested dictionary structure by collapsing nested keys into one key with a separator.

    For example:
    ```
    >>> dct = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}`
    >>> print(flatten_dict(dct))
    {"a/b": 1, "a/c/d": 2, "e": 3}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d, sep="/"):
    outdict = {}
    for key, value in d.items():
        parts = key.split(sep)
        d = outdict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return outdict


def hf_transform_to_torch(items_dict):
    """Get a transform function that convert items from Hugging Face dataset (pyarrow)
    to torch tensors. Importantly, images are converted from PIL, which corresponds to
    a channel last representation (h w c) of uint8 type, to a torch image representation
    with channel first (c h w) of float32 type in range [0,1].
    """
    for key in items_dict:
        first_item = items_dict[key][0]
        if isinstance(first_item, PILImage.Image):
            to_tensor = transforms.ToTensor()
            items_dict[key] = [to_tensor(img) for img in items_dict[key]]
        elif isinstance(first_item, dict) and "path" in first_item and "timestamp" in first_item:
            # video frame will be processed downstream
            pass
        else:
            items_dict[key] = [torch.tensor(x) for x in items_dict[key]]
    return items_dict


def load_hf_dataset(repo_id, version, root, split) -> datasets.Dataset:
    """hf_dataset contains all the observations, states, actions, rewards, etc."""
    if root is not None:
        hf_dataset = load_from_disk(str(Path(root) / repo_id / split))
    else:
        hf_dataset = load_dataset(repo_id, revision=version, split=split)
    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset


def load_episode_data_index(repo_id, version, root) -> dict[str, torch.Tensor]:
    """episode_data_index contains the range of indices for each episode

    Example:
    ```python
    from_id = episode_data_index["from"][episode_id].item()
    to_id = episode_data_index["to"][episode_id].item()
    episode_frames = [dataset[i] for i in range(from_id, to_id)]
    ```
    """
    if root is not None:
        path = Path(root) / repo_id / "meta_data" / "episode_data_index.safetensors"
    else:
        path = hf_hub_download(
            repo_id, "meta_data/episode_data_index.safetensors", repo_type="dataset", revision=version
        )

    return load_file(path)


def load_stats(repo_id, version, root) -> dict[str, dict[str, torch.Tensor]]:
    """stats contains the statistics per modality computed over the full dataset, such as max, min, mean, std

    Example:
    ```python
    normalized_action = (action - stats["action"]["mean"]) / stats["action"]["std"]
    ```
    """
    if root is not None:
        path = Path(root) / repo_id / "meta_data" / "stats.safetensors"
    else:
        path = hf_hub_download(repo_id, "meta_data/stats.safetensors", repo_type="dataset", revision=version)

    stats = load_file(path)
    return unflatten_dict(stats)


def load_info(repo_id, version, root) -> dict:
    """info contains useful information regarding the dataset that are not stored elsewhere

    Example:
    ```python
    print("frame per second used to collect the video", info["fps"])
    ```
    """
    if root is not None:
        path = Path(root) / repo_id / "meta_data" / "info.json"
    else:
        path = hf_hub_download(repo_id, "meta_data/info.json", repo_type="dataset", revision=version)

    with open(path) as f:
        info = json.load(f)
    return info


def load_videos(repo_id, version, root) -> Path:
    if root is not None:
        path = Path(root) / repo_id / "videos"
    else:
        # TODO(rcadene): we download the whole repo here. see if we can avoid this
        repo_dir = snapshot_download(repo_id, repo_type="dataset", revision=version)
        path = Path(repo_dir) / "videos"

    return path


def load_previous_and_future_frames(
    item: dict[str, torch.Tensor],
    hf_dataset: datasets.Dataset,
    episode_data_index: dict[str, torch.Tensor],
    delta_timestamps: dict[str, list[float]],
    tolerance_s: float,
) -> dict[torch.Tensor]:
    """
    Given a current item in the dataset containing a timestamp (e.g. 0.6 seconds), and a list of time differences of
    some modalities (e.g. delta_timestamps={"observation.image": [-0.8, -0.2, 0, 0.2]}), this function computes for each
    given modality (e.g. "observation.image") a list of query timestamps (e.g. [-0.2, 0.4, 0.6, 0.8]) and loads the closest
    frames in the dataset.

    Importantly, when no frame can be found around a query timestamp within a specified tolerance window, this function
    raises an AssertionError. When a timestamp is queried before the first available timestamp of the episode or after
    the last available timestamp, the violation of the tolerance doesnt raise an AssertionError, and the function
    populates a boolean array indicating which frames are outside of the episode range. For instance, this boolean array
    is useful during batched training to not supervise actions associated to timestamps coming after the end of the
    episode, or to pad the observations in a specific way. Note that by default the observation frames before the start
    of the episode are the same as the first frame of the episode.

    Parameters:
    - item (dict): A dictionary containing all the data related to a frame. It is the result of `dataset[idx]`. Each key
      corresponds to a different modality (e.g., "timestamp", "observation.image", "action").
    - hf_dataset (datasets.Dataset): A dictionary containing the full dataset. Each key corresponds to a different
      modality (e.g., "timestamp", "observation.image", "action").
    - episode_data_index (dict): A dictionary containing two keys ("from" and "to") associated to dataset indices.
      They indicate the start index and end index of each episode in the dataset.
    - delta_timestamps (dict): A dictionary containing lists of delta timestamps for each possible modality to be
      retrieved. These deltas are added to the item timestamp to form the query timestamps.
    - tolerance_s (float, optional): The tolerance level (in seconds) used to determine if a data point is close enough to the query
      timestamp by asserting `tol > difference`. It is suggested to set `tol` to a smaller value than the
      smallest expected inter-frame period, but large enough to account for jitter.

    Returns:
    - The same item with the queried frames for each modality specified in delta_timestamps, with an additional key for
      each modality (e.g. "observation.image_is_pad").

    Raises:
    - AssertionError: If any of the frames unexpectedly violate the tolerance level. This could indicate synchronization
      issues with timestamps during data collection.
    """
    # get indices of the frames associated to the episode, and their timestamps
    ep_id = item["episode_index"].item()
    ep_data_id_from = episode_data_index["from"][ep_id].item()
    ep_data_id_to = episode_data_index["to"][ep_id].item()
    ep_data_ids = torch.arange(ep_data_id_from, ep_data_id_to, 1)

    # load timestamps
    ep_timestamps = hf_dataset.select_columns("timestamp")[ep_data_id_from:ep_data_id_to]["timestamp"]
    ep_timestamps = torch.stack(ep_timestamps)

    # we make the assumption that the timestamps are sorted
    ep_first_ts = ep_timestamps[0]
    ep_last_ts = ep_timestamps[-1]
    current_ts = item["timestamp"].item()

    for key in delta_timestamps:
        # get timestamps used as query to retrieve data of previous/future frames
        delta_ts = delta_timestamps[key]
        query_ts = current_ts + torch.tensor(delta_ts)

        # compute distances between each query timestamp and all timestamps of all the frames belonging to the episode
        dist = torch.cdist(query_ts[:, None], ep_timestamps[:, None], p=1)
        min_, argmin_ = dist.min(1)

        # TODO(rcadene): synchronize timestamps + interpolation if needed

        is_pad = min_ > tolerance_s

        # check violated query timestamps are all outside the episode range
        assert ((query_ts[is_pad] < ep_first_ts) | (ep_last_ts < query_ts[is_pad])).all(), (
            f"One or several timestamps unexpectedly violate the tolerance ({min_} > {tolerance_s=}) inside episode range."
            "This might be due to synchronization issues with timestamps during data collection."
        )

        # get dataset indices corresponding to frames to be loaded
        data_ids = ep_data_ids[argmin_]

        # load frames modality
        item[key] = hf_dataset.select_columns(key)[data_ids][key]

        if isinstance(item[key][0], dict) and "path" in item[key][0]:
            # video mode where frame are expressed as dict of path and timestamp
            item[key] = item[key]
        else:
            item[key] = torch.stack(item[key])

        item[f"{key}_is_pad"] = is_pad

    return item


def cycle(iterable):
    """The equivalent of itertools.cycle, but safe for Pytorch dataloaders.

    See https://github.com/pytorch/pytorch/issues/23900 for information on why itertools.cycle is not safe.
    """
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)
