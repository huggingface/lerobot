# dataset_tool_cli.py
# Copyright 2024-2025 The HuggingFace Inc. team and contributors.
# Licensed under the Apache-2.0 license.

"""
Lerobot Dataset Tool - CLI Interface

Example usage:
  python dataset_tool_cli.py merge \\
      --datasets "/path/to/datasetA /path/to/datasetB" \\
      --output_dir /path/to/merged_dataset

  python dataset_tool_cli.py delete \\
      --dataset_dir /path/to/dataset_to_modify \\
      --episode_id 32 \\
      --verbose
"""

import argparse
from pathlib import Path

# Import the manager class from the other file
from lerobot.common.datasets.dataset_manager import CHUNK_NAME_DEFAULT, DatasetManager


def main_cli():
    parser = argparse.ArgumentParser(
        description="Lerobot Dataset Management Tool.",
        formatter_class=argparse.RawTextHelpFormatter,  # To preserve newline in help text
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available commands. Use <command> -h for details."
    )

    # --- Merge command ---
    parser_merge = subparsers.add_parser(
        "merge",
        help="Merge multiple datasets into one.",
        description=(
            "Merges multiple Lerobot datasets into a new output directory. \n"
            "Episode indices are renumbered, Parquet files are updated, \n"
            "meta files are concatenated/updated, and videos are copied and re-indexed."
        ),
    )
    parser_merge.add_argument(
        "--datasets",
        type=str,
        required=True,
        help='Space-separated string of paths to datasets to merge. e.g., "/path/A /path/B"',
    )
    parser_merge.add_argument(
        "--output_dir", type=Path, required=True, help="Directory where the merged dataset will be saved."
    )
    parser_merge.add_argument(
        "--chunk_name",
        type=str,
        default=CHUNK_NAME_DEFAULT,
        help=f"Name of the data chunk (default: {CHUNK_NAME_DEFAULT}).",
    )
    parser_merge.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output.")

    # --- Delete command ---
    parser_delete = subparsers.add_parser(
        "delete",
        help="Delete an episode from a dataset.",
        description=(
            "Deletes a specific episode from a dataset and renumbers all subsequent episodes and their associated files.\n"
            "This operation modifies the dataset IN-PLACE."
        ),
    )
    parser_delete.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="Path to the dataset to modify (operation is in-place).",
    )
    parser_delete.add_argument("--episode_id", type=int, required=True, help="ID of the episode to delete.")
    parser_delete.add_argument(
        "--chunk_name",
        type=str,
        default=CHUNK_NAME_DEFAULT,
        help=f"Name of the data chunk (default: {CHUNK_NAME_DEFAULT}).",
    )
    parser_delete.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output.")

    args = parser.parse_args()
    manager = DatasetManager()

    if args.command == "merge":
        manager.merge_datasets(args.datasets, args.output_dir, args.chunk_name, args.verbose)
    elif args.command == "delete":
        manager.delete_episode_from_dataset(args.dataset_dir, args.episode_id, args.chunk_name, args.verbose)
    else:
        parser.print_help()  # Should not be reached due to `required=True` on subparsers


if __name__ == "__main__":
    main_cli()
