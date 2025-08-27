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
Test script for RLearN evaluation metrics.

This script tests the VOC-S and success/failure detection metrics with synthetic data
to ensure they work correctly before running on real datasets.
"""

import numpy as np

from lerobot.policies.rlearn.evaluation import (
    compute_success_failure_detection,
    compute_voc_s,
    generate_mismatched_languages,
)


def test_voc_s():
    """Test VOC-S computation with synthetic data."""
    print("Testing VOC-S computation...")

    # Test case 1: Perfect positive correlation (0 -> 1)
    perfect_positive = [np.linspace(0, 1, 20) for _ in range(10)]
    results = compute_voc_s(perfect_positive)

    print("Perfect positive correlation:")
    print(f"  Mean: {results['voc_s_mean']:.4f} (should be ~1.0)")
    print(f"  IQM:  {results['voc_s_iqm']:.4f} (should be ~1.0)")
    assert results["voc_s_mean"] > 0.95, f"Expected >0.95, got {results['voc_s_mean']}"

    # Test case 2: Perfect negative correlation (1 -> 0)
    perfect_negative = [np.linspace(1, 0, 20) for _ in range(10)]
    results = compute_voc_s(perfect_negative)

    print("Perfect negative correlation:")
    print(f"  Mean: {results['voc_s_mean']:.4f} (should be ~-1.0)")
    print(f"  IQM:  {results['voc_s_iqm']:.4f} (should be ~-1.0)")
    assert results["voc_s_mean"] < -0.95, f"Expected <-0.95, got {results['voc_s_mean']}"

    # Test case 3: No correlation (random)
    np.random.seed(42)
    random_rewards = [np.random.random(20) for _ in range(50)]
    results = compute_voc_s(random_rewards)

    print("Random correlation:")
    print(f"  Mean: {results['voc_s_mean']:.4f} (should be ~0.0)")
    print(f"  IQM:  {results['voc_s_iqm']:.4f} (should be ~0.0)")
    assert abs(results["voc_s_mean"]) < 0.3, f"Expected ~0, got {results['voc_s_mean']}"

    # Test case 4: Mixed correlations
    mixed = []
    mixed.extend([np.linspace(0, 1, 15) for _ in range(5)])  # Positive
    mixed.extend([np.linspace(1, 0, 15) for _ in range(5)])  # Negative
    mixed.extend([np.random.random(15) for _ in range(5)])  # Random

    results = compute_voc_s(mixed)
    print("Mixed correlations:")
    print(f"  Mean: {results['voc_s_mean']:.4f}")
    print(f"  IQM:  {results['voc_s_iqm']:.4f}")
    print(f"  Std:  {results['voc_s_std']:.4f}")

    print("✓ VOC-S tests passed!\n")


def test_success_failure_detection():
    """Test success/failure detection with synthetic data."""
    print("Testing Success/Failure Detection...")

    # Test case 1: Clear separation (correct > incorrect)
    correct_rewards = [np.linspace(0, 1, 20) for _ in range(20)]  # Always increasing
    incorrect_rewards = [np.linspace(0, 0.3, 20) for _ in range(20)]  # Lower final values

    results = compute_success_failure_detection(correct_rewards, incorrect_rewards)

    print("Clear separation test:")
    print(f"  Detection accuracy: {results['detection_accuracy']:.4f} (should be 1.0)")
    print(f"  Mean correct:       {results['mean_correct_final']:.4f}")
    print(f"  Mean incorrect:     {results['mean_incorrect_final']:.4f}")
    print(f"  Separation score:   {results['separation_score']:.4f}")
    assert results["detection_accuracy"] == 1.0, f"Expected 1.0, got {results['detection_accuracy']}"

    # Test case 2: No separation (same distributions with some randomness)
    np.random.seed(42)
    same_rewards_1 = [np.random.normal(0.5, 0.05, 15) for _ in range(20)]
    same_rewards_2 = [np.random.normal(0.5, 0.05, 15) for _ in range(20)]

    results = compute_success_failure_detection(same_rewards_1, same_rewards_2)

    print("No separation test:")
    print(f"  Detection accuracy: {results['detection_accuracy']:.4f} (should be ~0.5)")
    print(f"  Separation score:   {results['separation_score']:.4f} (should be ~0.0)")
    # Relax the assertion since random data can vary
    assert 0.2 <= results["detection_accuracy"] <= 0.8, (
        f"Expected ~0.5 (±0.3), got {results['detection_accuracy']}"
    )

    # Test case 3: Partial separation
    np.random.seed(42)
    partial_correct = [np.random.normal(0.7, 0.1, 10) for _ in range(20)]
    partial_incorrect = [np.random.normal(0.4, 0.1, 10) for _ in range(20)]

    results = compute_success_failure_detection(partial_correct, partial_incorrect)

    print("Partial separation test:")
    print(f"  Detection accuracy: {results['detection_accuracy']:.4f}")
    print(f"  Separation score:   {results['separation_score']:.4f}")

    print("✓ Success/Failure Detection tests passed!\n")


def test_mismatch_generation():
    """Test mismatch language generation."""
    print("Testing mismatch language generation...")

    original_languages = [
        "pick up the red ball",
        "put the cup on the table",
        "open the drawer",
        "close the door",
    ]

    # Test with default templates
    mismatched = generate_mismatched_languages(original_languages)

    print(f"Original languages: {len(original_languages)}")
    print(f"Mismatched languages: {len(mismatched)}")
    assert len(mismatched) == len(original_languages)

    # Ensure they're actually different
    for orig, mismatch in zip(original_languages, mismatched, strict=False):
        print(f"  '{orig}' -> '{mismatch}'")
        assert orig != mismatch, "Mismatch should be different from original"

    # Test with custom templates
    custom_templates = ["dance", "sing", "jump"]
    mismatched_custom = generate_mismatched_languages(original_languages, custom_templates)

    print("\nWith custom templates:")
    for orig, mismatch in zip(original_languages, mismatched_custom, strict=False):
        print(f"  '{orig}' -> '{mismatch}'")
        assert mismatch in custom_templates

    print("✓ Mismatch generation tests passed!\n")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("Testing edge cases...")

    # Empty input
    empty_results = compute_voc_s([])
    assert empty_results["num_episodes"] == 0
    assert empty_results["voc_s_mean"] == 0.0

    # Single frame episodes (should be skipped)
    single_frame = [np.array([0.5]) for _ in range(5)]
    results = compute_voc_s(single_frame)
    assert results["num_episodes"] == 0, "Single-frame episodes should be skipped"

    # Constant rewards (should give correlation = 0)
    constant_rewards = [np.ones(10) * 0.5 for _ in range(5)]
    results = compute_voc_s(constant_rewards)
    print(f"Constant rewards correlation: {results['voc_s_mean']:.4f} (should be 0.0)")
    assert results["voc_s_mean"] == 0.0

    # Mismatched array lengths for detection
    try:
        compute_success_failure_detection([np.array([1, 2])], [])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected

    print("✓ Edge case tests passed!\n")
