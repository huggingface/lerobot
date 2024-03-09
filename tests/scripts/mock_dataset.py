"""
    usage: `python tests/scripts/mock_dataset.py --in-data-dir data/pusht --out-data-dir tests/data/pusht`
"""

import argparse
import shutil

from tensordict import TensorDict
from pathlib import Path


def mock_dataset(in_data_dir, out_data_dir, num_frames=50):
    # load full dataset as a tensor dict
    in_td_data = TensorDict.load_memmap(in_data_dir)

    # use 1 frame to know the specification of the dataset
    # and copy it over `n` frames in the test artifact directory
    out_td_data = in_td_data[0].expand(num_frames).memmap_like(out_data_dir)

    # copy the first `n` frames so that we have real data
    out_td_data[:num_frames] = in_td_data[:num_frames].clone()

    # make sure everything has been properly written
    out_td_data.lock_()

    # copy the full statistics of dataset since it's pretty small
    in_stats_path = Path(in_data_dir) / "stats.pth"
    out_stats_path = Path(out_data_dir) / "stats.pth"
    shutil.copy(in_stats_path, out_stats_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create dataset")

    parser.add_argument("--in-data-dir", type=str, help="Path to input data")
    parser.add_argument("--out-data-dir", type=str, help="Path to save the output data")

    args = parser.parse_args()

    mock_dataset(args.in_data_dir, args.out_data_dir)