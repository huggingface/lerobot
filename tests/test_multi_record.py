#!/usr/bin/env python3
"""
Simple test to verify the multi_record function and configurations work correctly.
"""

import sys

sys.path.insert(0, "src")

from lerobot.record import DatasetRecordConfig, MultiDatasetRecordConfig


def test_multi_dataset_config():
    """Test that MultiDatasetRecordConfig creates correctly."""

    # Create test dataset configurations
    dataset1 = DatasetRecordConfig(repo_id="test/dataset1", single_task="Test task 1")

    dataset2 = DatasetRecordConfig(repo_id="test/dataset2", single_task="Test task 2")

    # Test multi-dataset config creation
    multi_config = MultiDatasetRecordConfig(datasets=[dataset1, dataset2], stage_switch_keys=["space", "tab"])

    assert len(multi_config.datasets) == 2
    assert multi_config.stage_switch_keys == ["space", "tab"]
    print("✓ MultiDatasetRecordConfig creation test passed")


def test_multi_dataset_config_defaults():
    """Test that default stage switch keys are generated correctly."""

    dataset1 = DatasetRecordConfig(repo_id="test/dataset1", single_task="Test task 1")
    dataset2 = DatasetRecordConfig(repo_id="test/dataset2", single_task="Test task 2")
    dataset3 = DatasetRecordConfig(repo_id="test/dataset3", single_task="Test task 3")

    # Test with default keys
    multi_config = MultiDatasetRecordConfig(datasets=[dataset1, dataset2, dataset3])

    assert multi_config.stage_switch_keys == ["space", "tab", "enter"]
    print("✓ Default stage switch keys test passed")


def test_validation():
    """Test configuration validation."""

    # Test empty datasets list
    try:
        MultiDatasetRecordConfig(datasets=[])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "At least one dataset configuration must be provided" in str(e)
        print("✓ Empty datasets validation test passed")

    # Test insufficient stage keys
    dataset1 = DatasetRecordConfig(repo_id="test/dataset1", single_task="Test task 1")
    dataset2 = DatasetRecordConfig(repo_id="test/dataset2", single_task="Test task 2")

    try:
        MultiDatasetRecordConfig(
            datasets=[dataset1, dataset2],
            stage_switch_keys=["space"],  # Only one key for two datasets
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Number of stage_switch_keys" in str(e)
        print("✓ Insufficient stage keys validation test passed")


if __name__ == "__main__":
    print("Running multi-dataset recording configuration tests...")

    test_multi_dataset_config()
    test_multi_dataset_config_defaults()
    test_validation()

    print("\n✅ All tests passed! Multi-dataset recording configuration is working correctly.")
