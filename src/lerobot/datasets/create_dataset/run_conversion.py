#!/usr/bin/env python3

"""Command-line interface for dataset conversion."""

import argparse
import sys

from lerobot.datasets.create_dataset.config.dataset_config import (
    DatasetConfig,
    create_sample_config,
    load_config,
)
from lerobot.datasets.create_dataset.config.defaults import DEFAULT_CONFIG
from lerobot.datasets.create_dataset.converter.convert_to_lerobot_dataset import DatasetConverter


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Convert custom datasets to LeRobotDataset format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert using config file
    python -m lerobot.datasets.create_dataset.cli --config my_config.yaml

    # Create sample config
    python -m lerobot.datasets.create_dataset.cli --create-sample-config sample_config.yaml

    # Quick conversion with minimal config
    python -m lerobot.datasets.create_dataset.cli --repo-id "user/dataset" --input-dir "/data" --fps 30
        """,
    )

    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--create-sample-config", type=str, help="Create sample config file")

    # Quick setup arguments
    parser.add_argument("--repo-id", type=str, help="Repository ID for the dataset")
    parser.add_argument("--input-dir", type=str, help="Input directory containing raw data")
    parser.add_argument("--fps", type=int, help="Frames per second")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--test-mode", action="store_true", help="Process only first few episodes")

    args = parser.parse_args()

    # Create sample config if requested
    if args.create_sample_config:
        create_sample_config(args.create_sample_config, DEFAULT_CONFIG)
        print(f"Sample configuration created: {args.create_sample_config}")
        return 0

    # Load configuration
    try:
        if args.config:
            config = load_config(args.config)
        elif args.repo_id and args.input_dir and args.fps:
            # Quick setup
            config = DatasetConfig(
                repo_id=args.repo_id,
                input_dir=args.input_dir,
                fps=args.fps,
                debug=args.debug or False,
                test_mode=args.test_mode or False,
            )
        else:
            # Use defaults from DEFAULT_CONFIG
            config = DatasetConfig(**DEFAULT_CONFIG)
            print("Using default configuration:")
            print(f"  Repo ID: {config.repo_id}")
            print(f"  Input dir: {config.input_dir}")
            print(f"  FPS: {config.fps}")
            print(f"  Use videos: {config.use_videos}")
            print(f"  Debug: {config.debug}")
            print(f"  Test mode: {config.test_mode}")
            print(f"  Push to hub: {config.push_to_hub}")

        # Override config with command line arguments
        if args.debug:
            config.debug = False
        if args.test_mode:
            config.test_mode = True

        # Run conversion
        converter = DatasetConverter(config)
        dataset = converter.convert()
        print("✅ Conversion completed successfully!")
        print(f"📁 Dataset saved to: {config.output_dir}")
        print(f"📊 Total episodes: {dataset.num_episodes}")
        print(f"🎞️  Total frames: {dataset.num_frames}")
        return 0

    except Exception as e:
        print(f"❌ Conversion failed: {e}", file=sys.stderr)
        if config.debug:
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
