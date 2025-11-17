#!/usr/bin/env python

import argparse
import logging
from pathlib import Path
from typing import List
import json
import shlex

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.utils.utils import init_logging


def combine_datasets(
    dataset_paths: List[str],
    output_path: str
) -> None:
    """
    Combine multiple LeRobot datasets into a single dataset using the aggregate function.

    Args:
        dataset_paths: List of dataset repo IDs to combine (e.g., "repo_id/dataset_name")
        output_path: Output repo ID for the combined dataset (e.g., "repo_id/dataset_name")
    """
    logging.info(f"Combining {len(dataset_paths)} datasets into {output_path}")

    aggregate_datasets(
        repo_ids=dataset_paths,
        aggr_repo_id=output_path
    )

    logging.info(f"Successfully combined datasets into {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Combine multiple LeRobot datasets into a single dataset")
    parser.add_argument(
        "--dataset_paths",
        type=str,
        required=True,
        help="Dataset paths to combine. Can be space-separated (e.g., 'repo1/dataset1 repo2/dataset2') or JSON array format (e.g., '[\"repo1/dataset1\", \"repo2/dataset2\"]').",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path/repo ID for the combined dataset (e.g., `repo_id/combined_dataset`).",
    )

    args = parser.parse_args()

    # Parse dataset paths (same logic as the original combine_dataset.py)
    dataset_paths = _parse_dataset_paths(args.dataset_paths)

    logging.info(f"Parsed dataset paths: {dataset_paths}")

    # Run the combination using aggregate_datasets
    combine_datasets(
        dataset_paths=dataset_paths,
        output_path=args.output_path
    )

    logging.info(f"Successfully combined datasets into {args.output_path}")

def _parse_dataset_paths(dataset_paths_arg: str) -> List[str]:
    """
    Parse dataset paths from command line argument.
    Supports both space-separated format and JSON array format.
    """
    # Try to parse as JSON array first
    if dataset_paths_arg.strip().startswith('[') and dataset_paths_arg.strip().endswith(']'):
        try:
            # Clean up the JSON string by removing trailing commas
            cleaned_arg = dataset_paths_arg.strip()
            # Remove trailing comma before the closing bracket
            cleaned_arg = cleaned_arg.replace(',]', ']').replace(', ]', ']')

            parsed_paths = json.loads(cleaned_arg)
            # Filter out empty strings and None values
            filtered_paths = [path for path in parsed_paths if path and isinstance(path, str) and path.strip()]
            return filtered_paths
        except json.JSONDecodeError as e:
            logging.warning(f"Failed to parse as JSON array: {e}")
            # If JSON parsing fails, fall back to space-separated parsing
            pass

    # Parse as space-separated arguments
    parsed_paths = shlex.split(dataset_paths_arg)
    # Filter out empty strings and brackets
    filtered_paths = [path for path in parsed_paths if path and path.strip() and path not in ['[', ']']]
    return filtered_paths


if __name__ == "__main__":
    init_logging()
    main()
