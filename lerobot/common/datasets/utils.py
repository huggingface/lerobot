from copy import deepcopy
from math import ceil

import datasets
import einops
import torch
import tqdm


def load_previous_and_future_frames(
    item: dict[str, torch.Tensor],
    hf_dataset: datasets.Dataset,
    delta_timestamps: dict[str, list[float]],
    tol: float = 0.04,
) -> dict[torch.Tensor]:
    """
    Given a current item in the dataset containing a timestamp (e.g. 0.6 seconds), and a list of time differences of some modalities (e.g. delta_timestamps={"observation.image": [-0.8, -0.2, 0, 0.2]}),
    this function computes for each given modality a list of query timestamps (e.g. [-0.2, 0.4, 0.6, 0.8]) and loads the closest frames in the dataset.

    Importantly, when no frame can be found around a query timestamp within a specified tolerance window (e.g. tol=0.04), this function raises an AssertionError.
    When a timestamp is queried before the first available timestamp of the episode or after the last available timestamp,
    the violation of the tolerance doesnt raise an AssertionError, and the function populates a boolean array indicating which frames are outside of the episode range.
    For instance, this boolean array is useful during batched training to not supervise actions associated to timestamps coming after the end of the episode,
    or to pad the observations in a specific way. Note that by default the observation frames before the start of the episode are the same as the first frame of the episode.

    Parameters:
    - item (dict): A dictionary containing all the data related to a frame. It is the result of `dataset[idx]`. Each key corresponds to a different modality (e.g., "timestamp", "observation.image", "action").
    - hf_dataset (datasets.Dataset): A dictionary containing the full dataset. Each key corresponds to a different modality (e.g., "timestamp", "observation.image", "action").
    - delta_timestamps (dict): A dictionary containing lists of delta timestamps for each possible modality to be retrieved. These deltas are added to the item timestamp to form the query timestamps.
    - tol (float, optional): The tolerance level used to determine if a data point is close enough to the query timestamp. Defaults to 0.04.

    Returns:
    - The same item with the queried frames for each modality specified in delta_timestamps, with an additional key for each modality (e.g. "observation.image_is_pad").

    Raises:
    - AssertionError: If any of the frames unexpectedly violate the tolerance level. This could indicate synchronization issues with timestamps during data collection.
    """
    # get indices of the frames associated to the episode, and their timestamps
    ep_data_id_from = item["episode_data_index_from"].item()
    ep_data_id_to = item["episode_data_index_to"].item()
    ep_data_ids = torch.arange(ep_data_id_from, ep_data_id_to, 1)

    # load timestamps
    ep_timestamps = hf_dataset.select_columns("timestamp")[ep_data_id_from:ep_data_id_to]["timestamp"]

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

        is_pad = min_ > tol

        # check violated query timestamps are all outside the episode range
        assert ((query_ts[is_pad] < ep_first_ts) | (ep_last_ts < query_ts[is_pad])).all(), (
            f"One or several timestamps unexpectedly violate the tolerance ({min_} > {tol=}) inside episode range."
            "This might be due to synchronization issues with timestamps during data collection."
        )

        # get dataset indices corresponding to frames to be loaded
        data_ids = ep_data_ids[argmin_]

        # load frames modality
        item[key] = hf_dataset.select_columns(key)[data_ids][key]
        item[f"{key}_is_pad"] = is_pad

    return item


def get_stats_einops_patterns(dataset):
    """These einops patterns will be used to aggregate batches and compute statistics."""
    stats_patterns = {
        "action": "b c -> c",
        "observation.state": "b c -> c",
    }
    for key in dataset.image_keys:
        stats_patterns[key] = "b c h w -> c 1 1"
    return stats_patterns


def compute_stats(dataset, batch_size=32, max_num_samples=None):
    if max_num_samples is None:
        max_num_samples = len(dataset)
    else:
        raise NotImplementedError("We need to set shuffle=True, but this violate an assert for now.")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=batch_size,
        shuffle=False,
        # pin_memory=cfg.device != "cpu",
        drop_last=False,
    )

    # get einops patterns to aggregate batches and compute statistics
    stats_patterns = get_stats_einops_patterns(dataset)

    # mean and std will be computed incrementally while max and min will track the running value.
    mean, std, max, min = {}, {}, {}, {}
    for key in stats_patterns:
        mean[key] = torch.tensor(0.0).float()
        std[key] = torch.tensor(0.0).float()
        max[key] = torch.tensor(-float("inf")).float()
        min[key] = torch.tensor(float("inf")).float()

    # Note: Due to be refactored soon. The point of storing `first_batch` is to make sure we don't get
    # surprises when rerunning the sampler.
    first_batch = None
    running_item_count = 0  # for online mean computation
    for i, batch in enumerate(
        tqdm.tqdm(dataloader, total=ceil(max_num_samples / batch_size), desc="Compute mean, min, max")
    ):
        this_batch_size = len(batch["index"])
        running_item_count += this_batch_size
        if first_batch is None:
            first_batch = deepcopy(batch)
        for key, pattern in stats_patterns.items():
            batch[key] = batch[key].float()
            # Numerically stable update step for mean computation.
            batch_mean = einops.reduce(batch[key], pattern, "mean")
            # Hint: to update the mean we need x̄ₙ = (Nₙ₋₁x̄ₙ₋₁ + Bₙxₙ) / Nₙ, where the subscript represents
            # the update step, N is the running item count, B is this batch size, x̄ is the running mean,
            # and x is the current batch mean. Some rearrangement is then required to avoid risking
            # numerical overflow. Another hint: Nₙ₋₁ = Nₙ - Bₙ. Rearrangement yields
            # x̄ₙ = x̄ₙ₋₁ + Bₙ * (xₙ - x̄ₙ₋₁) / Nₙ
            mean[key] = mean[key] + this_batch_size * (batch_mean - mean[key]) / running_item_count
            max[key] = torch.maximum(max[key], einops.reduce(batch[key], pattern, "max"))
            min[key] = torch.minimum(min[key], einops.reduce(batch[key], pattern, "min"))

        if i == ceil(max_num_samples / batch_size) - 1:
            break

    first_batch_ = None
    running_item_count = 0  # for online std computation
    for i, batch in enumerate(
        tqdm.tqdm(dataloader, total=ceil(max_num_samples / batch_size), desc="Compute std")
    ):
        this_batch_size = len(batch["index"])
        running_item_count += this_batch_size
        # Sanity check to make sure the batches are still in the same order as before.
        if first_batch_ is None:
            first_batch_ = deepcopy(batch)
            for key in stats_patterns:
                assert torch.equal(first_batch_[key], first_batch[key])
        for key, pattern in stats_patterns.items():
            batch[key] = batch[key].float()
            # Numerically stable update step for mean computation (where the mean is over squared
            # residuals).See notes in the mean computation loop above.
            batch_std = einops.reduce((batch[key] - mean[key]) ** 2, pattern, "mean")
            std[key] = std[key] + this_batch_size * (batch_std - std[key]) / running_item_count

        if i == ceil(max_num_samples / batch_size) - 1:
            break

    for key in stats_patterns:
        std[key] = torch.sqrt(std[key])

    stats = {}
    for key in stats_patterns:
        stats[key] = {
            "mean": mean[key],
            "std": std[key],
            "max": max[key],
            "min": min[key],
        }

    return stats


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
