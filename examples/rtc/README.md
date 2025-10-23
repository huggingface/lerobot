# Real-Time Chunking (RTC) Examples

This directory contains examples and evaluation scripts for Real-Time Chunking (RTC), a technique for improving action chunking policies in real-time robot control.

## Overview

Real-Time Chunking addresses the challenge of maintaining consistency and reactivity when using action chunking policies with non-negligible inference latency. It uses a guidance technique during diffusion sampling to blend new action predictions with previously planned actions.

**Key Benefits:**

- Maintains consistency between consecutive action chunks
- Reduces jitter and improves smoothness
- Adapts to inference delays dynamically

**Reference:** [Physical Intelligence - Real-Time Chunking](https://www.physicalintelligence.company/download/real_time_chunking.pdf)

## Scripts

### 1. `real_time_chunking_evaluate.py`

Real-time evaluation on physical robots or simulation environments.

**Features:**

- Run policy with RTC on real robot or simulation
- Compare RTC vs non-RTC actions in real-time
- Multi-threaded action execution and inference
- Support for torch.compile() optimization

**Usage:**

```bash
# With real robot
uv run python examples/rtc/real_time_chunking_evaluate.py \
    --policy.path=lerobot/smolvla_base \
    --robot.type=so100 \
    --task="pick up the cup"

# With simulation environment
uv run python examples/rtc/real_time_chunking_evaluate.py \
    --policy.path=lerobot/smolvla_base \
    --env.type=pusht \
    --duration=60.0

# Disable verbose comparison (faster)
uv run python examples/rtc/real_time_chunking_evaluate.py \
    --policy.path=lerobot/smolvla_base \
    --robot.type=so100 \
    --verbose_rtc_comparison=false

# With policy compilation (CUDA only, not MPS)
uv run python examples/rtc/real_time_chunking_evaluate.py \
    --policy.path=lerobot/smolvla_base \
    --robot.type=so100 \
    --compile_policy=true \
    --compile_mode=max-autotune
```

**Key Parameters:**

- `--policy.path`: Path to pretrained policy
- `--robot.type` or `--env.type`: Robot or environment to use
- `--rtc.execution_horizon`: Number of steps to maintain consistency (default: 10)
- `--rtc.max_guidance_weight`: Maximum guidance weight (default: 1.0)
- `--rtc.prefix_attention_schedule`: Schedule type (ZEROS, ONES, LINEAR, EXP)
- `--verbose_rtc_comparison`: Enable detailed RTC comparison logging (default: true)
- `--duration`: How long to run (seconds, default: 30.0)
- `--fps`: Action execution frequency (Hz, default: 10.0)

### 2. `evaluate_rtc_on_dataset.py`

Offline evaluation on dataset samples to measure RTC effectiveness.

**Features:**

- Evaluate RTC on dataset without running robot
- Compare RTC vs non-RTC predictions
- Measure consistency and ground truth alignment
- Simulate different inference delays
- Save detailed metrics to JSON

**Usage:**

```bash
# Basic evaluation
uv run python examples/rtc/evaluate_rtc_on_dataset.py \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=lerobot/pusht \
    --num_iterations=100

# Simulate inference delay (every 3rd step)
uv run python examples/rtc/evaluate_rtc_on_dataset.py \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=lerobot/pusht \
    --num_iterations=200 \
    --skip_steps=3

# Custom RTC configuration
uv run python examples/rtc/evaluate_rtc_on_dataset.py \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=lerobot/pusht \
    --num_iterations=100 \
    --rtc.execution_horizon=12 \
    --rtc.max_guidance_weight=5.0 \
    --rtc.prefix_attention_schedule=LINEAR

# Save results to file
uv run python examples/rtc/evaluate_rtc_on_dataset.py \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=lerobot/pusht \
    --num_iterations=100 \
    --output_path=results/rtc_evaluation.json

# Verbose mode with detailed logging
uv run python examples/rtc/evaluate_rtc_on_dataset.py \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=lerobot/pusht \
    --num_iterations=50 \
    --verbose=true
```

**Key Parameters:**

- `--policy.path`: Path to pretrained policy
- `--dataset.repo_id`: Dataset to evaluate on
- `--num_iterations`: Number of samples to evaluate (default: 100)
- `--skip_steps`: Steps to skip between inferences, simulates inference delay (default: 1)
- `--start_episode`: Episode to start from (default: 0)
- `--output_path`: Path to save results JSON
- `--verbose`: Enable detailed per-sample logging
- `--device`: Device to use (cuda, cpu, mps, auto)

**Metrics Reported:**

- **RTC vs Ground Truth MSE**: How close RTC predictions are to actual actions
- **No-RTC vs Ground Truth MSE**: Baseline without RTC
- **RTC Improvement**: Absolute and relative improvement over baseline
- **RTC Consistency**: How well RTC maintains consistency in prefix region
  - Prefix MSE
  - Mean/Max error in overlap region

### 3. `run_dataset_evaluation.sh`

Convenience script with multiple evaluation scenarios.

**Usage:**

```bash
# Edit the script to set your policy and dataset
# Then run all examples:
./examples/rtc/run_dataset_evaluation.sh

# Or run individual examples from the script
```

## Understanding RTC Parameters

### `execution_horizon`

Number of timesteps from previous chunk to maintain consistency with. Higher values mean more consistency but potentially less reactivity.

**Typical values:** 8-12 steps

### `max_guidance_weight`

Upper bound on guidance strength. Higher values give stronger consistency but may over-constrain new predictions.

**Typical values:** 1.0-10.0

### `prefix_attention_schedule`

How to weight consistency across the overlap region:

- `ZEROS`: Binary (full weight up to inference_delay, then zero)
- `ONES`: Full weight across entire execution_horizon
- `LINEAR`: Linear decay from inference_delay to execution_horizon
- `EXP`: Exponential decay (recommended)

**Recommended:** `EXP`

### `skip_steps` (evaluation only)

Simulates inference delay by evaluating every N-th step. This helps understand how RTC performs with realistic delays.

**Example:** `skip_steps=3` means policy infers every 3 steps, simulating 3x action execution frequency vs inference frequency.

## Output Format (Dataset Evaluation)

When using `--output_path`, results are saved in JSON format:

```json
{
  "summary": {
    "rtc_vs_ground_truth_mse": {
      "mean": 0.00123,
      "std": 0.00045,
      "min": 0.00012,
      "max": 0.00456
    },
    "improvement": {
      "absolute": 0.00034,
      "relative_percent": 12.5
    },
    ...
  },
  "config": {
    "num_iterations": 100,
    "skip_steps": 3,
    "execution_horizon": 10,
    ...
  },
  "detailed_results": [
    {
      "sample_idx": 0,
      "rtc_vs_ground_truth_mse": 0.00112,
      "no_rtc_vs_ground_truth_mse": 0.00145,
      ...
    },
    ...
  ]
}
```

## Tips

1. **Start with dataset evaluation** to understand RTC behavior before running on robot
2. **Use verbose mode** for debugging unexpected behavior
3. **Tune execution_horizon** based on your inference latency and action frequency
4. **Monitor consistency metrics** - very low consistency might indicate execution_horizon is too small
5. **Compare different schedules** - EXP usually works best but LINEAR can be more interpretable

## Troubleshooting

### High RTC vs No-RTC difference but no improvement

- Try reducing `max_guidance_weight`
- Check if `execution_horizon` is too large

### Poor consistency metrics

- Increase `execution_horizon`
- Check that `skip_steps` is not larger than your action chunk size
- Verify episodes are being reset correctly

### RTC worse than No-RTC

- RTC may not help if inference is faster than action execution
- Try different `prefix_attention_schedule`
- Ensure `execution_horizon` matches your use case

## Examples Results

Example output from dataset evaluation:

```
================================================================================
EVALUATION SUMMARY
================================================================================

Ground Truth Alignment:
  RTC MSE:        0.001234 ± 0.000456
  No-RTC MSE:     0.001567 ± 0.000512

RTC Improvement:
  Absolute:       0.000333
  Relative:       21.23%

RTC vs No-RTC Difference:
  MSE:            0.000112 ± 0.000034

RTC Consistency (Prefix Region):
  MSE:            0.000089 ± 0.000023
  Mean Error:     0.007654 ± 0.002341
  Max Error:      0.023456 ± 0.008765
```

## Related Documentation

- [RTC Implementation](../../src/lerobot/policies/rtc/modeling_rtc.py)
- [RTC Configuration](../../src/lerobot/policies/rtc/configuration_rtc.py)
- [Physical Intelligence Paper](https://www.physicalintelligence.company/download/real_time_chunking.pdf)
