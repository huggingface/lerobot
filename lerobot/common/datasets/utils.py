import io
import zipfile
from pathlib import Path

import requests
import torch
import tqdm


def download_and_extract_zip(url: str, destination_folder: Path) -> bool:
    print(f"downloading from {url}")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get("content-length", 0))
        progress_bar = tqdm.tqdm(total=total_size, unit="B", unit_scale=True)

        zip_file = io.BytesIO()
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                zip_file.write(chunk)
                progress_bar.update(len(chunk))

        progress_bar.close()

        zip_file.seek(0)

        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(destination_folder)
        return True
    else:
        return False


def euclidean_distance_matrix(mat0, mat1):
    # Compute the square of the distance matrix
    sq0 = torch.sum(mat0**2, dim=1, keepdim=True)
    sq1 = torch.sum(mat1**2, dim=1, keepdim=True)
    distance_sq = sq0 + sq1.transpose(0, 1) - 2 * mat0 @ mat1.transpose(0, 1)

    # Taking the square root to get the euclidean distance
    distance = torch.sqrt(torch.clamp(distance_sq, min=0))
    return distance


def is_contiguously_true_or_false(bool_vector):
    assert bool_vector.ndim == 1
    assert bool_vector.dtype == torch.bool

    # Compare each element with its neighbor to find changes
    changes = bool_vector[1:] != bool_vector[:-1]

    # Count the number of changes
    num_changes = changes.sum().item()

    # If there's more than one change, the list is not contiguous
    return num_changes <= 1

    # examples = [
    #     ([True, False, True, False, False, False], False),
    #     ([True, True, True, False, False, False], True),
    #     ([False, False, False, False, False, False], True)
    # ]
    # for bool_list, expected in examples:
    #     result = is_contiguously_true_or_false(bool_list)


def load_data_with_delta_timestamps(
    data_dict, data_ids_per_episode, delta_timestamps, key, current_ts, episode
):
    # get indices of the frames associated to the episode, and their timestamps
    ep_data_ids = data_ids_per_episode[episode]
    ep_timestamps = data_dict["timestamp"][ep_data_ids]

    # get timestamps used as query to retrieve data of previous/future frames
    delta_ts = delta_timestamps[key]
    query_ts = current_ts + torch.tensor(delta_ts)

    # compute distances between each query timestamp and all timestamps of all the frames belonging to the episode
    dist = euclidean_distance_matrix(query_ts[:, None], ep_timestamps[:, None])
    min_, argmin_ = dist.min(1)

    # get the indices of the data that are closest to the query timestamps
    data_ids = ep_data_ids[argmin_]
    # closest_ts = ep_timestamps[argmin_]

    # get the data
    data = data_dict[key][data_ids].clone()

    # TODO(rcadene): synchronize timestamps + interpolation if needed

    tol = 0.02
    is_pad = min_ > tol

    assert is_contiguously_true_or_false(is_pad), (
        "One or several timestamps unexpectedly violate the tolerance."
        "This might be due to synchronization issues with timestamps during data collection."
    )

    return data, is_pad
