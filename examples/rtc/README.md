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

### 1. `eval_dataset.py`

Offline evaluation on dataset samples with detailed visualization and validation.

**Features:**

- Compare RTC vs non-RTC predictions on two random dataset samples
- Validate RTC behavior (delay region, blend region, post-horizon region)
- Generate debug visualizations:
  - Denoising step comparisons (x_t, v_t, x1_t, corrections)
  - Final action predictions comparison
- Support for torch.compile() optimization
- Memory-efficient sequential policy loading for large models

**Usage:**

```bash
# Basic usage with SmolVLA policy
uv run python examples/rtc/eval_dataset.py \
    --policy.path=helper2424/smolvla_check_rtc_last3 \
    --dataset.repo_id=helper2424/check_rtc \
    --rtc.execution_horizon=8 \
    --device=mps \
    --rtc.max_guidance_weight=10.0 \
    --seed=10

# With Pi0.5 policy on CUDA
uv run python examples/rtc/eval_dataset.py \
    --policy.path=lerobot/pi05_libero_finetuned \
    --dataset.repo_id=HuggingFaceVLA/libero \
    --rtc.execution_horizon=8 \
    --device=cuda

# With Pi0 policy
uv run python examples/rtc/eval_dataset.py \
    --policy.path=lerobot/pi0_libero_finetuned \
    --dataset.repo_id=HuggingFaceVLA/libero \
    --rtc.execution_horizon=8 \
    --device=cuda

# With torch.compile for faster inference
uv run python examples/rtc/eval_dataset.py \
    --policy.path=helper2424/smolvla_check_rtc_last3 \
    --dataset.repo_id=helper2424/check_rtc \
    --rtc.execution_horizon=8 \
    --device=cuda \
    --use_torch_compile=true \
    --torch_compile_mode=max-autotune

# Enable CUDA graphs (advanced - may cause tensor aliasing errors)
uv run python examples/rtc/eval_dataset.py \
    --policy.path=helper2424/smolvla_check_rtc_last3 \
    --dataset.repo_id=helper2424/check_rtc \
    --use_torch_compile=true \
    --torch_compile_backend=inductor \
    --torch_compile_mode=max-autotune \
    --torch_compile_disable_cudagraphs=false
```

**Key Parameters:**

- `--policy.path`: Path to pretrained policy
- `--dataset.repo_id`: Dataset to evaluate on
- `--rtc.execution_horizon`: Number of steps to maintain consistency (default: 20)
- `--rtc.max_guidance_weight`: Maximum guidance weight (default: 10.0)
- `--rtc.prefix_attention_schedule`: Schedule type (ZEROS, ONES, LINEAR, EXP)
- `--inference_delay`: Inference delay for RTC (default: 4)
- `--seed`: Random seed for reproducibility (default: 42)
- `--output_dir`: Directory to save visualizations (default: rtc_debug_output)
- `--device`: Device to use (cuda, cpu, mps, auto)
- `--use_torch_compile`: Enable torch.compile() for faster inference

**Output:**

The script generates several visualization files in `rtc_debug_output/`:

- `denoising_xt_comparison.png` - Noisy state evolution during denoising
- `denoising_vt_comparison.png` - Velocity predictions during denoising
- `denoising_x1t_comparison.png` - Predicted final states during denoising
- `denoising_correction_comparison.png` - RTC guidance corrections applied
- `final_actions_comparison.png` - Final action predictions (prev_chunk, no_rtc, rtc)

The script also validates RTC behavior and reports:

- ✅ Delay region [0:inference_delay]: RTC = prev_chunk
- ✅ Blend region [inference_delay:execution_horizon]: prev_chunk ≤ RTC ≤ no_rtc
- ✅ Post-horizon [execution_horizon:]: RTC = no_rtc

### 2. `eval_with_real_robot.py`

Real-time evaluation on physical robots or simulation environments.

**Features:**

- Run policy with RTC on real robot or simulation
- Multi-threaded action execution and inference
- Action queue management with proper timing
- Latency tracking and adaptive inference delay
- Support for both robots and gym environments
- Support for torch.compile() optimization

**Usage:**

```bash
# With real robot
uv run python examples/rtc/eval_with_real_robot.py \
    --policy.path=lerobot/smolvla_base \
    --robot.type=so100 \
    --task="pick up the cup" \
    --duration=30.0

# With simulation environment
uv run python examples/rtc/eval_with_real_robot.py \
    --policy.path=lerobot/smolvla_base \
    --env.type=pusht \
    --duration=60.0

# With policy compilation (CUDA only, not MPS)
uv run python examples/rtc/eval_with_real_robot.py \
    --policy.path=lerobot/smolvla_base \
    --robot.type=so100 \
    --use_torch_compile=true \
    --torch_compile_mode=max-autotune
```

**Key Parameters:**

- `--policy.path`: Path to pretrained policy
- `--robot.type` or `--env.type`: Robot or environment to use
- `--task`: Task description (for VLA models)
- `--rtc.execution_horizon`: Number of steps to maintain consistency (default: 10)
- `--rtc.max_guidance_weight`: Maximum guidance weight (default: 1.0)
- `--rtc.prefix_attention_schedule`: Schedule type (ZEROS, ONES, LINEAR, EXP)
- `--duration`: How long to run (seconds, default: 30.0)
- `--fps`: Action execution frequency (Hz, default: 10.0)
- `--action_queue_size_to_get_new_actions`: Queue size threshold to request new actions (default: 30)
- `--device`: Device to use (cuda, cpu, mps, auto)
- `--use_torch_compile`: Enable torch.compile() for faster inference

## Understanding RTC Parameters

### `execution_horizon`

Number of timesteps from previous chunk to maintain consistency with. Higher values mean more consistency but potentially less reactivity.

**Typical values:** 8-12 steps for dataset evaluation, 10 steps for real-time execution

### `max_guidance_weight`

Upper bound on guidance strength. Higher values give stronger consistency but may over-constrain new predictions.

**Typical values:**

- Dataset evaluation: 10.0-100.0 (can be higher for analysis)
- Real-time execution: 1.0-10.0 (more conservative)

### `prefix_attention_schedule`

How to weight consistency across the overlap region:

- `ZEROS`: Binary (full weight up to inference_delay, then zero)
- `ONES`: Full weight across entire execution_horizon
- `LINEAR`: Linear decay from inference_delay to execution_horizon
- `EXP`: Exponential decay (recommended)

**Recommended:** `EXP`

### `inference_delay`

Number of timesteps from the prefix to use for guidance. Typically calculated dynamically based on inference latency in real-time execution, but fixed for dataset evaluation.

**Typical values:** 3-5 steps for dataset evaluation

### `action_queue_size_to_get_new_actions` (real-time only)

Threshold for requesting new action chunks. Should be higher than `inference_delay + execution_horizon` to ensure smooth operation.

**Typical values:** 20-30 steps

## Validation Rules (Dataset Evaluation)

The dataset evaluation script validates that RTC behavior matches expectations:

1. **Delay Region [0:inference_delay]**: RTC actions should equal previous chunk
   - Ensures consistency during the inference delay period

2. **Blend Region [inference_delay:execution_horizon]**: RTC should be between prev_chunk and no_rtc
   - Smooth transition from previous plan to new predictions

3. **Post-Horizon [execution_horizon:]**: RTC should equal no_rtc
   - Full adoption of new predictions after execution horizon

## Tips

1. **Start with dataset evaluation** (`eval_dataset.py`) to understand RTC behavior and tune parameters before running on robot
2. **Use visualizations** to debug unexpected behavior - check denoising steps and final actions
3. **Tune execution_horizon** based on your inference latency and action frequency
4. **Monitor validation output** - failures indicate potential implementation issues or misconfigured parameters
5. **Compare different schedules** - EXP usually works best but LINEAR can be more interpretable

## Troubleshooting

### Validation fails in delay region

- Check that `prev_chunk_left_over` is properly passed to the policy
- Verify RTC guidance is being applied during denoising
- Look at denoising visualizations to see where guidance diverges

### Validation fails in post-horizon region

- RTC and no_rtc use different noise - verify same noise is being used for comparison
- Check that weights are correctly zeroed out after execution horizon
- Review prefix_attention_schedule visualization

### Poor performance on real robot

- Increase `action_queue_size_to_get_new_actions` if you see warnings
- Reduce `max_guidance_weight` if robot is too conservative
- Try different `prefix_attention_schedule` values
- Enable torch.compile() for faster inference (CUDA only)

### Memory issues with large models

- The dataset evaluation script loads policies sequentially to minimize memory
- For real-time execution, only one policy is loaded
- Use smaller batch sizes if needed

## Related Documentation

- [RTC Implementation](../../src/lerobot/policies/rtc/modeling_rtc.py)
- [RTC Configuration](../../src/lerobot/policies/rtc/configuration_rtc.py)
- [Action Queue](../../src/lerobot/policies/rtc/action_queue.py)
- [Physical Intelligence Paper](https://www.physicalintelligence.company/download/real_time_chunking.pdf)
