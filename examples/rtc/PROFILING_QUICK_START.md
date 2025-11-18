# RTC Profiling - Quick Start

Quick reference for profiling Pi0 with RTC to identify performance bottlenecks.

## 🚀 Quick Commands

### 1. Profile with Real Robot

```bash
# With RTC enabled (profiled version)
uv run examples/rtc/eval_with_real_robot_profiled.py \
    --policy.path=helper2424/pi05_check_rtc \
    --policy.device=mps \
    --rtc.enabled=true \
    --rtc.execution_horizon=20 \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58FA0834591 \
    --robot.cameras="{ gripper: {type: opencv, index_or_path: 0}, front: {type: opencv, index_or_path: 1}}" \
    --task="Pick up object" \
    --duration=30
```

### 2. Compare RTC vs No-RTC (No Robot Needed)

```bash
uv run examples/rtc/profile_rtc_comparison.py \
    --policy_path=helper2424/pi05_check_rtc \
    --device=mps \
    --num_iterations=50 \
    --execution_horizon=20
```

### 3. Detailed RTC Method Profiling

```bash
uv run examples/rtc/profile_pi0_rtc_detailed.py \
    --policy_path=helper2424/pi05_check_rtc \
    --device=mps \
    --num_iterations=20 \
    --execution_horizon=20 \
    --enable_rtc_profiling
```

## 📊 What Each Tool Does

| Tool | Purpose | Needs Robot? |
|------|---------|--------------|
| `eval_with_real_robot_profiled.py` | Profile actual robot execution with RTC | ✅ Yes |
| `profile_rtc_comparison.py` | Compare RTC vs no-RTC side-by-side | ❌ No |
| `profile_pi0_rtc_detailed.py` | Deep dive into RTC internals | ❌ No |

## 🔍 Key Metrics to Watch

### Overall Performance
- **iteration.policy_inference** - Total policy inference time
- **iteration.preprocessing** - Image preprocessing time
- **iteration.postprocessing** - Action denormalization time

### RTC-Specific (with `--enable_rtc_profiling`)
- **rtc.denoise_step.base_denoising** - Time without RTC overhead
- **rtc.denoise_step.autograd_correction** - Gradient computation time
- **rtc.denoise_step.guidance_computation** - Total RTC guidance overhead

### Robot Communication
- **robot.get_observation** - Time to get robot state
- **robot.send_action** - Time to send action command

## 🎯 Quick Diagnosis

### RTC is slower than expected?

1. **Check if torch.compile is enabled**
   ```bash
   # Add this flag
   --use_torch_compile
   ```

2. **Try larger execution horizon**
   ```bash
   # Increase to amortize RTC overhead
   --rtc.execution_horizon=30
   ```

3. **Profile to find bottleneck**
   ```bash
   uv run examples/rtc/profile_pi0_rtc_detailed.py \
       --policy_path=helper2424/pi05_check_rtc \
       --device=mps \
       --enable_rtc_profiling
   ```

### Preprocessing is slow?

- Reduce image resolution in robot config
- Use fewer cameras
- Check camera FPS settings

### Policy inference is slow?

- Enable torch.compile
- Check device (MPS vs CUDA vs CPU)
- Try smaller model if available

## 📈 Expected Performance

### Typical timings on Apple Silicon (MPS):

| Component | Time (ms) | Notes |
|-----------|-----------|-------|
| Policy inference | 100-200 | Depends on model size |
| Preprocessing | 5-20 | Depends on #cameras |
| Postprocessing | 1-5 | Usually fast |
| RTC overhead | 10-50 | Should be < 50% of base |

### When RTC helps:
- ✅ Execution horizon ≥ 10
- ✅ Inference time > action execution rate
- ✅ Using torch.compile
- ✅ Proper inference_delay calculation

### When RTC might not help:
- ❌ Very fast inference already
- ❌ Small execution horizon (< 5)
- ❌ No compilation (interpreted mode)
- ❌ Inference delay not accounted for

## 🛠️ Adding Profiling to Your Code

### Quick snippet:

```python
from lerobot.utils.profiling import enable_profiling, print_profiling_summary, profile_section

# Enable at start
enable_profiling()

# Profile sections
with profile_section("my_operation"):
    # ... your code ...
    pass

# Print at end
print_profiling_summary()
```

### Profile specific methods:

```python
from lerobot.utils.profiling import profile_method

@profile_method
def my_slow_function():
    # ... your code ...
    pass
```

## 📝 Example Output

```
PROFILING SUMMARY
================================================================================
Function                                                    Count    Mean (ms)
--------------------------------------------------------------------------------
iteration.policy_inference                                    20       150.23
iteration.preprocessing                                       20        12.45
rtc.denoise_step.guidance_computation                        200        15.67
rtc.denoise_step.autograd_correction                         200         8.23
rtc.denoise_step.base_denoising                             200       120.45
================================================================================
```

## 🚨 Common Issues

### "No profiling data available"
- Did you call `enable_profiling()`?
- Running enough iterations?

### Inconsistent results
- Increase `--num_iterations`
- Check for thermal throttling
- Close other applications

### Can't find bottleneck
- Enable `--enable_rtc_profiling` for detailed breakdown
- Check both preprocessing and inference
- Compare with and without RTC

## 📖 More Details

See `PROFILING_GUIDE.md` for comprehensive documentation.

## 🤔 Still Slow?

1. Run comparison: `profile_rtc_comparison.py`
2. Run detailed profiling: `profile_pi0_rtc_detailed.py --enable_rtc_profiling`
3. Share output for help (include device, model, settings)

## ✅ Quick Checklist

Before asking for help, verify:

- [ ] Ran comparison script (with/without RTC)
- [ ] Tried torch.compile
- [ ] Tested different execution horizons (10, 20, 30)
- [ ] Profiled with detailed RTC profiling
- [ ] Checked preprocessing vs inference split
- [ ] Verified hardware (device type, thermal state)

