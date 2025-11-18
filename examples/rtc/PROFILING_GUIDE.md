# RTC Profiling Guide

This guide explains how to profile RTC (Real-Time Chunking) performance to identify bottlenecks and understand why RTC might be slower than expected.

## Quick Start

### 1. Profile with Real Robot (Profiled Version)

Use `eval_with_real_robot_profiled.py` to profile actual robot execution:

```bash
# With RTC enabled
uv run examples/rtc/eval_with_real_robot_profiled.py \
    --policy.path=helper2424/pi05_check_rtc \
    --policy.device=mps \
    --rtc.enabled=true \
    --rtc.execution_horizon=20 \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58FA0834591 \
    --robot.id=so100_follower \
    --robot.cameras="{ gripper: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
    --task="Move green small object into the purple platform" \
    --duration=30

# Without RTC for comparison
uv run examples/rtc/eval_with_real_robot_profiled.py \
    --policy.path=helper2424/pi05_check_rtc \
    --policy.device=mps \
    --rtc.enabled=false \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58FA0834591 \
    --robot.id=so100_follower \
    --robot.cameras="{ gripper: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
    --task="Move green small object into the purple platform" \
    --duration=30
```

**Output**: At the end of execution, you'll see a detailed breakdown of timing for each component:
- `get_actions.policy_inference` - Time spent in policy inference
- `get_actions.preprocessing` - Time spent preprocessing observations
- `get_actions.postprocessing` - Time spent postprocessing actions
- `get_actions.action_queue_merge` - Time spent merging actions with RTC
- `robot.get_observation` - Time to get observations from robot
- `robot.send_action` - Time to send actions to robot
- And more...

### 2. Profile Without Robot (Comparison Script)

Use `profile_rtc_comparison.py` to profile just the policy inference without needing a robot:

```bash
uv run examples/rtc/profile_rtc_comparison.py \
    --policy_path=helper2424/pi05_check_rtc \
    --device=mps \
    --num_iterations=50 \
    --execution_horizon=20
```

**Output**: Side-by-side comparison of performance with and without RTC, including:
- Mean/min/max inference times
- Throughput (iterations per second)
- Verdict on whether RTC is faster or slower

### 3. Enable Detailed Method-Level Profiling

For even more granular profiling, add the `--enable_detailed_profiling` flag:

```bash
uv run examples/rtc/profile_rtc_comparison.py \
    --policy_path=helper2424/pi05_check_rtc \
    --device=mps \
    --num_iterations=50 \
    --execution_horizon=20 \
    --enable_detailed_profiling
```

This will show timing for individual methods within the policy.

## Understanding the Output

### Key Metrics to Look At

1. **get_actions.policy_inference** - This should be the largest component
   - If RTC is enabled, this includes the RTC guidance overhead
   - Compare this with/without RTC to see the overhead

2. **get_actions.preprocessing** - Image preprocessing and normalization
   - Should be relatively fast
   - If slow, consider optimizing image processing

3. **get_actions.postprocessing** - Action denormalization
   - Should be minimal
   - If slow, check postprocessor implementation

4. **get_actions.action_queue_merge** - RTC-specific merging logic
   - Only present when RTC is enabled
   - If this is taking significant time, the RTC algorithm may need optimization

5. **robot.get_observation** - Robot communication overhead
   - If slow, check camera/sensor latency
   - Consider reducing image resolution

6. **robot.send_action** - Action execution overhead
   - Should be very fast
   - If slow, check robot communication

### Expected Performance

For a typical Pi0 policy on Apple Silicon (MPS):
- **Without RTC**: ~100-200ms per inference
- **With RTC**: Should be similar or slightly faster due to action reuse
- **Preprocessing**: ~5-20ms depending on number of cameras
- **Postprocessing**: ~1-5ms

If RTC is significantly slower, likely causes:
1. **RTC overhead exceeds benefits** - The guidance computation is expensive
2. **Execution horizon too small** - Not reusing enough actions to amortize overhead
3. **No compilation** - Try with `--use_torch_compile`
4. **Large prev_actions buffer** - Copying/processing previous actions is slow

## Profiling Your Own Code

### Using the Profiling Decorator

Add profiling to your own methods:

```python
from lerobot.utils.profiling import profile_method, enable_profiling, print_profiling_summary

# Enable profiling
enable_profiling()

# Decorate methods you want to profile
@profile_method
def my_slow_function(x):
    # ... your code ...
    return result

# At end of execution
print_profiling_summary()
```

### Using Profile Context Manager

For profiling specific code blocks:

```python
from lerobot.utils.profiling import profile_section, enable_profiling

enable_profiling()

with profile_section("data_loading"):
    data = load_data()

with profile_section("model_inference"):
    output = model(data)
```

### Adding Profiling to Policy Methods

To profile specific parts of the Pi0 policy, you can add decorators:

```python
# In src/lerobot/policies/pi0/modeling_pi0.py
from lerobot.utils.profiling import profile_method, profile_section

class Pi0Policy:
    @profile_method
    def predict_action_chunk(self, obs, inference_delay=0, prev_chunk_left_over=None):
        # ... existing code ...
        pass

    def _generate_actions_with_rtc(self, ...):
        with profile_section("rtc.guidance_computation"):
            # ... guidance code ...
            pass
        
        with profile_section("rtc.action_merging"):
            # ... merging code ...
            pass
```

## Analyzing Results

### Comparison Checklist

When comparing RTC vs non-RTC performance, check:

- [ ] Is `policy_inference` time higher with RTC?
- [ ] Is `action_queue_merge` taking significant time?
- [ ] Are you running enough iterations to amortize warmup?
- [ ] Is torch.compile enabled for fair comparison?
- [ ] Is the execution horizon large enough? (should be >= 10-20)
- [ ] Are you testing on the same hardware/device?

### Common Bottlenecks

1. **Image preprocessing dominates** 
   - Solution: Reduce image resolution, use fewer cameras, or optimize preprocessing

2. **Action queue operations are slow**
   - Solution: Review queue implementation, consider using ring buffer

3. **RTC guidance is expensive**
   - Solution: Reduce guidance weight, simplify guidance computation, use torch.compile

4. **Robot communication is slow**
   - Solution: Increase baud rate, reduce action frequency, optimize protocol

5. **Memory allocation overhead**
   - Solution: Pre-allocate buffers, reuse tensors, avoid unnecessary copies

## Advanced: Adding Custom Metrics

You can add custom timing metrics to the profiled script:

```python
from lerobot.utils.profiling import record_timing

start = time.perf_counter()
# ... your code ...
duration = time.perf_counter() - start
record_timing("my_custom_metric", duration)
```

## Troubleshooting

### Profiling shows RTC is slower by >50%

1. Check if torch.compile is enabled: `--use_torch_compile`
2. Increase execution horizon: `--rtc.execution_horizon=30`
3. Verify inference_delay is calculated correctly
4. Profile with `--enable_detailed_profiling` to find exact bottleneck

### Profiling output is empty

1. Make sure profiling is enabled with `enable_profiling()`
2. Verify you're running enough iterations (at least 10)
3. Check that code is actually executing (not short-circuited)

### Inconsistent results between runs

1. Run more iterations: `--num_iterations=100`
2. Increase warmup iterations
3. Check for thermal throttling on device
4. Ensure no other processes competing for resources

## Next Steps

1. Run both profiling scripts (with/without robot)
2. Compare timing breakdowns
3. Identify the largest bottleneck
4. Focus optimization efforts on that component
5. Re-run profiling to verify improvements

## Questions?

If profiling reveals unexpected bottlenecks or you need help interpreting results, please share:
- The full profiling output
- Your configuration (RTC enabled/disabled, execution horizon, etc.)
- Hardware specs (device type, memory, etc.)
- Policy type and size

