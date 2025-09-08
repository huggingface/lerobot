#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
Simple script to download OpenX datasets using TensorFlow Datasets.

Usage:
    python examples/port_datasets/download_openx_datasets.py
    python examples/port_datasets/download_openx_datasets.py --download-dir /path/to/datasets
    python examples/port_datasets/download_openx_datasets.py --datasets fractal20220817_data kuka bridge
"""

import argparse
from pathlib import Path

import tensorflow_datasets as tfds
import tqdm

# Full list of OpenX dataset names
# Optionally replace with filtered datasets from the Google Sheet
DATASET_NAMES = [
    "fractal20220817_data",
    "kuka",
    "bridge",
    "taco_play",
    "jaco_play",
    "berkeley_cable_routing",
    "roboturk",
    "nyu_door_opening_surprising_effectiveness",
    "viola",
    "berkeley_autolab_ur5",
    "toto",
    "language_table",
    "columbia_cairlab_pusht_real",
    "stanford_kuka_multimodal_dataset_converted_externally_to_rlds",
    "nyu_rot_dataset_converted_externally_to_rlds",
    "stanford_hydra_dataset_converted_externally_to_rlds",
    "austin_buds_dataset_converted_externally_to_rlds",
    "nyu_franka_play_dataset_converted_externally_to_rlds",
    "maniskill_dataset_converted_externally_to_rlds",
    "furniture_bench_dataset_converted_externally_to_rlds",
    "cmu_franka_exploration_dataset_converted_externally_to_rlds",
    "ucsd_kitchen_dataset_converted_externally_to_rlds",
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds",
    "austin_sailor_dataset_converted_externally_to_rlds",
    "austin_sirius_dataset_converted_externally_to_rlds",
    "bc_z",
    "usc_cloth_sim_converted_externally_to_rlds",
    "utokyo_pr2_opening_fridge_converted_externally_to_rlds",
    "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds",
    "utokyo_saytap_converted_externally_to_rlds",
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds",
    "utokyo_xarm_bimanual_converted_externally_to_rlds",
    "robo_net",
    "berkeley_mvp_converted_externally_to_rlds",
    "berkeley_rpt_converted_externally_to_rlds",
    "kaist_nonprehensile_converted_externally_to_rlds",
    "stanford_mask_vit_converted_externally_to_rlds",
    "tokyo_u_lsmo_converted_externally_to_rlds",
    "dlr_sara_pour_converted_externally_to_rlds",
    "dlr_sara_grid_clamp_converted_externally_to_rlds",
    "dlr_edan_shared_control_converted_externally_to_rlds",
    "asu_table_top_converted_externally_to_rlds",
    "stanford_robocook_converted_externally_to_rlds",
    "eth_agent_affordances",
    "imperialcollege_sawyer_wrist_cam",
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds",
    "uiuc_d3field",
    "utaustin_mutex",
    "berkeley_fanuc_manipulation",
    "cmu_food_manipulation",
    "cmu_play_fusion",
    "cmu_stretch",
    "berkeley_gnm_recon",
    "berkeley_gnm_cory_hall",
    "berkeley_gnm_sac_son",
]

DEFAULT_DOWNLOAD_DIR = "~/tensorflow_datasets"


def download_datasets(datasets, download_dir):
    """Download the specified datasets to the given directory."""
    download_dir = Path(download_dir).expanduser().resolve()
    print(f"Downloading {len(datasets)} datasets to {download_dir}")

    # Create directory if it doesn't exist
    download_dir.mkdir(parents=True, exist_ok=True)

    failed_downloads = []

    for dataset_name in tqdm.tqdm(datasets, desc="Downloading datasets"):
        try:
            print(f"\nDownloading {dataset_name}...")
            _ = tfds.load(dataset_name, data_dir=str(download_dir), download=True)
            print(f"✓ Successfully downloaded {dataset_name}")
        except Exception as e:
            print(f"✗ Failed to download {dataset_name}: {e}")
            failed_downloads.append((dataset_name, str(e)))

    # Summary
    print(f"\n{'=' * 60}")
    print("Download Summary:")
    print(f"  Total datasets: {len(datasets)}")
    print(f"  Successfully downloaded: {len(datasets) - len(failed_downloads)}")
    print(f"  Failed downloads: {len(failed_downloads)}")

    if failed_downloads:
        print("\nFailed downloads:")
        for dataset_name, error in failed_downloads:
            print(f"  - {dataset_name}: {error}")

    print(f"\nDatasets saved to: {download_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download OpenX datasets using TensorFlow Datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all OpenX datasets to default directory
  python download_openx_datasets.py

  # Download to specific directory
  python download_openx_datasets.py --download-dir /path/to/datasets

  # Download only specific datasets
  python download_openx_datasets.py --datasets fractal20220817_data kuka bridge

  # Download RT-1 dataset only
  python download_openx_datasets.py --datasets fractal20220817_data
        """,
    )

    parser.add_argument(
        "--download-dir",
        type=str,
        default=DEFAULT_DOWNLOAD_DIR,
        help=f"Directory to download datasets to (default: {DEFAULT_DOWNLOAD_DIR})",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Specific datasets to download. If not provided, downloads all OpenX datasets.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List all available dataset names and exit",
    )

    args = parser.parse_args()

    if args.list_datasets:
        print("Available OpenX datasets:")
        for i, dataset in enumerate(DATASET_NAMES, 1):
            print(f"  {i:2d}. {dataset}")
        print(f"\nTotal: {len(DATASET_NAMES)} datasets")
        return

    # Determine which datasets to download
    if args.datasets:
        datasets_to_download = args.datasets
        # Validate dataset names
        invalid_datasets = [d for d in datasets_to_download if d not in DATASET_NAMES]
        if invalid_datasets:
            print(f"Warning: Unknown datasets: {invalid_datasets}")
            print("Use --list-datasets to see available datasets")
    else:
        datasets_to_download = DATASET_NAMES

    download_datasets(datasets_to_download, args.download_dir)


if __name__ == "__main__":
    main()
