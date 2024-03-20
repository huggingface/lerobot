"""
This script is designed to facilitate the creation of a subset of an existing dataset by selecting a specific number of frames from the original dataset.
This subset can then be used for running quick unit tests.
The script takes an input directory containing the original dataset and an output directory where the subset of the dataset will be saved.
Additionally, the number of frames to include in the subset can be specified.
The script ensures that the subset is a representative sample of the original dataset by copying the specified number of frames and retaining the structure and format of the data.

Usage:
    Run the script with the following command, specifying the path to the input data directory,
    the path to the output data directory, and optionally the number of frames to include in the subset dataset:

    `python tests/scripts/mock_dataset.py --in-data-dir path/to/input_data --out-data-dir path/to/output_data`

Example:
    `python tests/scripts/mock_dataset.py --in-data-dir data/pusht --out-data-dir tests/data/pusht`
"""

import argparse
import shutil

from tensordict import TensorDict
from pathlib import Path


def mock_dataset(in_data_dir, out_data_dir, num_frames):
    in_data_dir = Path(in_data_dir)
    out_data_dir = Path(out_data_dir)

    # load full dataset as a tensor dict
    in_td_data = TensorDict.load_memmap(in_data_dir / "replay_buffer")

    # use 1 frame to know the specification of the dataset
    # and copy it over `n` frames in the test artifact directory
    out_td_data = in_td_data[0].expand(num_frames).memmap_like(out_data_dir / "replay_buffer")

    # copy the first `n` frames so that we have real data
    out_td_data[:num_frames] = in_td_data[:num_frames].clone()

    # make sure everything has been properly written
    out_td_data.lock_()

    # copy the full statistics of dataset since it's pretty small
    in_stats_path = in_data_dir / "stats.pth"
    out_stats_path = out_data_dir / "stats.pth"
    shutil.copy(in_stats_path, out_stats_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create a dataset with a subset of frames for quick testing.")

    parser.add_argument("--in-data-dir", type=str, help="Path to input data")
    parser.add_argument("--out-data-dir", type=str, help="Path to save the output data")
    parser.add_argument("--num-frames", type=int, default=50, help="Number of frames to copy over")

    args = parser.parse_args()

    mock_dataset(args.in_data_dir, args.out_data_dir, args.num_frames)