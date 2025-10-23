#!/bin/bash

# Example script to run RTC evaluation on dataset
# This shows different usage scenarios

set -e  # Exit on error

POLICY_PATH="lerobot/smolvla_base"
DATASET="lerobot/pusht"
DEVICE="cuda"  # Change to "cpu" or "mps" if needed

echo "========================================"
echo "RTC Dataset Evaluation Examples"
echo "========================================"

# Example 1: Quick evaluation (100 samples, every step)
echo -e "\n[Example 1] Quick evaluation - 100 samples, every step"
python examples/rtc/evaluate_rtc_on_dataset.py \
    --policy.path="${POLICY_PATH}" \
    --dataset.repo_id="${DATASET}" \
    --num_iterations=100 \
    --skip_steps=1 \
    --device="${DEVICE}" \
    --output_path="results/rtc_eval_quick.json"

# Example 2: Simulating realistic inference delay (every 3rd step)
echo -e "\n[Example 2] Realistic inference delay - 200 samples, every 3rd step"
python examples/rtc/evaluate_rtc_on_dataset.py \
    --policy.path="${POLICY_PATH}" \
    --dataset.repo_id="${DATASET}" \
    --num_iterations=200 \
    --skip_steps=3 \
    --rtc.execution_horizon=10 \
    --device="${DEVICE}" \
    --output_path="results/rtc_eval_delay3.json"

# Example 3: Higher inference delay (every 5th step)
echo -e "\n[Example 3] High inference delay - 200 samples, every 5th step"
python examples/rtc/evaluate_rtc_on_dataset.py \
    --policy.path="${POLICY_PATH}" \
    --dataset.repo_id="${DATASET}" \
    --num_iterations=200 \
    --skip_steps=5 \
    --rtc.execution_horizon=12 \
    --device="${DEVICE}" \
    --output_path="results/rtc_eval_delay5.json"

# Example 4: Testing different RTC configurations
echo -e "\n[Example 4] Different RTC config - LINEAR schedule"
python examples/rtc/evaluate_rtc_on_dataset.py \
    --policy.path="${POLICY_PATH}" \
    --dataset.repo_id="${DATASET}" \
    --num_iterations=100 \
    --skip_steps=3 \
    --rtc.execution_horizon=8 \
    --rtc.prefix_attention_schedule=LINEAR \
    --rtc.max_guidance_weight=5.0 \
    --device="${DEVICE}" \
    --output_path="results/rtc_eval_linear.json"

# Example 5: Verbose mode for debugging
echo -e "\n[Example 5] Verbose mode - 20 samples with detailed output"
python examples/rtc/evaluate_rtc_on_dataset.py \
    --policy.path="${POLICY_PATH}" \
    --dataset.repo_id="${DATASET}" \
    --num_iterations=20 \
    --skip_steps=3 \
    --device="${DEVICE}" \
    --verbose=true \
    --output_path="results/rtc_eval_verbose.json"

echo -e "\n========================================"
echo "All evaluations completed!"
echo "Results saved in results/ directory"
echo "========================================"
