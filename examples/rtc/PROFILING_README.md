# RTC Profiling Toolkit

Complete toolkit for profiling Pi0 with RTC to identify performance bottlenecks.

## 📦 What's Included

### Scripts

1. **`eval_with_real_robot_profiled.py`**
   - Profiled version of the real robot eval script
   - Adds timing measurements throughout execution
   - Works with actual robot hardware
   - Same usage as original but with profiling output

2. **`profile_rtc_comparison.py`**
   - Side-by-side comparison of RTC vs no-RTC
   - No robot needed (uses mock observations)
   - Shows clear verdict on whether RTC is helping
   - Great for quick performance checks

3. **`profile_pi0_rtc_detailed.py`**
   - Most detailed profiling available
   - Can enable RTC method-level profiling
   - Provides insights and recommendations
   - Perfect for deep-dive investigations

4. **`add_rtc_profiling.py`**
   - Monkey-patching utility for RTC internals
   - Profiles individual RTC operations
   - Can be applied without modifying source
   - Shows exactly where RTC spends time

### Utilities

5. **`src/lerobot/utils/profiling.py`**
   - Core profiling utilities
   - Decorators for method profiling
   - Context managers for code blocks
   - Statistics collection and reporting

### Documentation

6. **`PROFILING_GUIDE.md`** - Comprehensive guide
7. **`PROFILING_QUICK_START.md`** - Quick reference

## 🚀 Quick Start

### Step 1: Compare Performance

Run this first to see if RTC is actually slower:

```bash
uv run examples/rtc/profile_rtc_comparison.py \
    --policy_path=helper2424/pi05_check_rtc \
    --device=mps \
    --num_iterations=50 \
    --execution_horizon=20
```

**Expected output:**
```
COMPARISON SUMMARY
================================================================================
Metric                         Without RTC        With RTC      Difference
--------------------------------------------------------------------------------
Mean time (ms)                       150.23         165.45          +15.22
Throughput (iter/s)                    6.66           6.05           -0.61
================================================================================
VERDICT
✗ RTC is SLOWER by 10.1%
  Mean time increased by 15.22 ms
  
  Possible reasons:
  - RTC overhead exceeds benefits at current execution horizon
  - No torch.compile enabled
```

### Step 2: Identify Bottleneck

If RTC is slower, find out why:

```bash
uv run examples/rtc/profile_pi0_rtc_detailed.py \
    --policy_path=helper2424/pi05_check_rtc \
    --device=mps \
    --num_iterations=20 \
    --execution_horizon=20 \
    --enable_rtc_profiling
```

**Expected output:**
```
PROFILING SUMMARY
================================================================================
Function                                             Count    Mean (ms)    Total (s)
------------------------------------------------------------------------------------
iteration.policy_inference                              20      150.23         3.00
rtc.denoise_step.guidance_computation                  200       15.67         3.13
rtc.denoise_step.autograd_correction                   200        8.23         1.65
iteration.preprocessing                                 20       12.45         0.25
================================================================================

KEY INSIGHTS
================================================================================
Time breakdown:
  Policy inference:  150.23 ms (87.2%)
  Preprocessing:     12.45 ms (7.2%)
  Postprocessing:    2.10 ms (1.2%)

RTC breakdown:
  Base denoising:    120.45 ms
  Guidance compute:  15.67 ms
  Autograd correct:  8.23 ms
  RTC overhead:      23.90 ms (19.8% of base)

Recommendations:
  ⚠ RTC autograd overhead is significant
    → This is expected, but consider increasing execution_horizon
    → Try torch.compile if not already enabled
  💡 torch.compile not enabled
    → Try --use_torch_compile for potential speedup
================================================================================
```

### Step 3: Try Optimizations

Based on recommendations:

```bash
# Try with torch.compile
uv run examples/rtc/profile_rtc_comparison.py \
    --policy_path=helper2424/pi05_check_rtc \
    --device=mps \
    --num_iterations=50 \
    --execution_horizon=20 \
    --use_torch_compile

# Try larger execution horizon
uv run examples/rtc/profile_rtc_comparison.py \
    --policy_path=helper2424/pi05_check_rtc \
    --device=mps \
    --num_iterations=50 \
    --execution_horizon=30
```

### Step 4: Profile Real Robot (Optional)

Test with actual hardware:

```bash
uv run examples/rtc/eval_with_real_robot_profiled.py \
    --policy.path=helper2424/pi05_check_rtc \
    --policy.device=mps \
    --rtc.enabled=true \
    --rtc.execution_horizon=20 \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58FA0834591 \
    --robot.cameras="{...}" \
    --task="Pick up object" \
    --duration=30
```

## 🎯 Common Scenarios

### "RTC is 2x slower!"

This usually means:
- RTC overhead is high but not getting benefits
- Need to enable torch.compile
- Execution horizon too small
- Inference delay not calculated correctly

**Try:**
1. `--use_torch_compile`
2. Increase `--execution_horizon` to 30+
3. Check inference_delay calculation

### "RTC is only slightly slower"

This is expected! RTC overhead is about 10-30% typically.
The benefit comes during **execution**, not single inference:
- Actions are reused across chunks
- Overall system latency is reduced
- Robot gets smoother actions

### "Want to optimize specific part"

Use the profiling utilities:

```python
from lerobot.utils.profiling import enable_profiling, profile_section, print_profiling_summary

enable_profiling()

with profile_section("my_custom_operation"):
    # Your code here
    pass

print_profiling_summary()
```

## 📊 Understanding Results

### Key Metrics

**Policy Inference Time**
- Time for forward pass through model
- Should be largest component (70-90%)
- Includes RTC guidance if enabled

**Preprocessing Time**
- Image normalization, resizing
- Should be < 20% of total
- If high: reduce image resolution

**RTC Guidance Overhead**
- Extra time for RTC guidance computation
- Typically 10-30% of base inference
- If > 50%: RTC may not be beneficial at current settings

**Autograd Correction**
- Time computing gradients for RTC
- Usually 5-15% of base inference
- Can be reduced with torch.compile

### Expected Ranges (Apple Silicon MPS)

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| Policy inference | 100-150ms | 150-250ms | >250ms |
| Preprocessing | <20ms | 20-50ms | >50ms |
| RTC overhead | 10-30% | 30-50% | >50% |

## 🔧 Optimization Guide

### If RTC overhead is too high:

1. **Enable compilation:**
   ```bash
   --use_torch_compile
   ```
   Expected improvement: 20-40% faster

2. **Increase execution horizon:**
   ```bash
   --execution_horizon=30  # or higher
   ```
   Amortizes RTC cost over more actions

3. **Check guidance weight:**
   ```python
   # In config
   rtc.max_guidance_weight=1.0  # try 0.5 for less overhead
   ```

### If preprocessing is slow:

1. **Reduce image resolution:**
   ```python
   # In robot config
   cameras={
       "gripper": {"width": 320, "height": 240}  # instead of 640x480
   }
   ```

2. **Use fewer cameras:**
   - Profile which cameras are essential
   - Remove unnecessary views

### If inference is generally slow:

1. Use torch.compile (if not already)
2. Check device is correct (MPS vs CUDA)
3. Verify model is in eval mode
4. Check for unnecessary gradient tracking

## 🐛 Troubleshooting

### Empty profiling output
```python
# Make sure to enable profiling!
from lerobot.utils.profiling import enable_profiling
enable_profiling()
```

### Inconsistent timings
- Run more iterations (50-100)
- Check thermal throttling
- Close background apps
- Use `--warmup_iterations=10`

### Can't find bottleneck
1. Start with `profile_rtc_comparison.py`
2. Then run `profile_pi0_rtc_detailed.py --enable_rtc_profiling`
3. Compare with/without RTC
4. Check each component separately

## 📖 Full Documentation

- **`PROFILING_GUIDE.md`** - Complete reference with examples
- **`PROFILING_QUICK_START.md`** - Quick commands and tips

## 🤝 Getting Help

If you're still experiencing issues:

1. Run comparison script and save output
2. Run detailed profiling and save output
3. Include:
   - Policy path
   - Device type
   - RTC settings (execution_horizon, etc.)
   - Hardware specs
   - Full profiling output

## 🎓 Learning More

### Profiling your own code:

```python
from lerobot.utils.profiling import profile_method, enable_profiling

enable_profiling()

@profile_method
def my_function():
    # Automatically profiled
    pass
```

### RTC internals:

```python
from examples.rtc.add_rtc_profiling import monkey_patch_rtc_profiling

enable_profiling()
monkey_patch_rtc_profiling()

# Now RTC methods are profiled
policy.predict_action_chunk(...)
```

## ✨ Next Steps

1. Run `profile_rtc_comparison.py` to establish baseline
2. Use `profile_pi0_rtc_detailed.py` to find bottlenecks
3. Apply optimizations (torch.compile, larger horizon)
4. Re-run comparison to verify improvements
5. Test with real robot using profiled version

Happy profiling! 🚀

