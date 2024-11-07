from typing import Dict

import numpy as np
import torch
import zarr
from datasets import Sequence, Value, Dataset, Features

from lerobot.common.datasets.utils import hf_transform_to_torch
from lerobot.common.datasets.video_utils import VideoFrame


def zarr_data_generator(zarr_dict: Dict[str, zarr.core.Array]):
    data_size = _check_sanity_and_get_data_size(zarr_dict)
    for i in range(data_size):
        record = {k: v[i] if not isinstance(v[i], np.ndarray) else torch.from_numpy(v[i]) for k, v in zarr_dict.items()}
        record['index'] = i
        yield record


def _check_sanity_and_get_data_size(zarr_dict: Dict[str, zarr.core.Array]):
    array_lengths = [v.shape[0] for v in zarr_dict.values()]
    for p, n in zip(array_lengths, array_lengths[1:]):
        assert p == n
    return array_lengths[0]


def to_hf_dataset(zarr_dict: Dict[str, zarr.core.Array]):
    features = {}

    features["observation.image"] = VideoFrame()
    features["observation.state"] = Sequence(
        length=zarr_dict["observation.state"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["action"] = Sequence(
        length=zarr_dict["action"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["episode_index"] = Value(dtype="int64", id=None)
    features["frame_index"] = Value(dtype="int64", id=None)
    features["timestamp"] = Value(dtype="float32", id=None)
    features["next.reward"] = Value(dtype="float32", id=None)
    features["next.done"] = Value(dtype="bool", id=None)
    features["next.success"] = Value(dtype="bool", id=None)
    features["index"] = Value(dtype="int64", id=None)

    hf_dataset = Dataset.from_generator(zarr_data_generator,
                                        gen_kwargs={'zarr_dict': zarr_dict},
                                        features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset


def read_data_from_zarr(zarr_group) -> Dict[str, zarr.core.Array]:
    zarr_dict = {k: v for k, v in zarr_group.items()}
    return zarr_dict
