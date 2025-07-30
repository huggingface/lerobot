# Control Loop Frequency Control: How `busy_wait` Works

## 🎯 **The Control Loop Timing Mechanism**

### **Core Formula:**
```python
dt_s = time.perf_counter() - start_loop_t  # Actual processing time
busy_wait(1 / fps - dt_s)                  # Wait for remaining time
```

---

## 📊 **Step-by-Step Breakdown**

### **1. Target Timing Calculation**
```python
target_fps = 30              # Your setting
target_loop_time = 1/fps     # 1/30 = 0.03333s = 33.33ms
```

### **2. Actual Loop Execution**
```python
start_loop_t = time.perf_counter()
# ... do work (observation, inference, action) ...
dt_s = time.perf_counter() - start_loop_t  # Measure actual time
```

### **3. Wait Calculation**
```python
wait_time = 1/fps - dt_s  # How long to wait
busy_wait(wait_time)      # Execute the wait
```

---

## 🔄 **Three Scenarios Explained**

### **Scenario A: Loop is FAST (Ideal Case)**
```
Target: 33.33ms (30 Hz)
Actual: 25.00ms processing
Wait:   8.33ms
Result: 33.33ms total → 30.0 Hz ✅
```

### **Scenario B: Loop is SLOW (Your Current Case)**
```
Target: 33.33ms (30 Hz)  
Actual: 37.50ms processing (cameras + inference taking too long)
Wait:   -4.17ms (negative!)
Result: 37.50ms total → 26.7 Hz ❌
```

### **Scenario C: Loop is EXACTLY on Target**
```
Target: 33.33ms (30 Hz)
Actual: 33.33ms processing  
Wait:   0.00ms
Result: 33.33ms total → 30.0 Hz ✅
```

---

## 🖥️ **Platform-Specific `busy_wait` Behavior**

### **On macOS (Your System):**
```python
def busy_wait(seconds):
    if platform.system() == "Darwin":
        # Uses CPU-intensive busy loop for precision
        end_time = time.perf_counter() + seconds
        while time.perf_counter() < end_time:
            pass  # Busy waiting - consumes CPU cycles
```

**Why busy loop on Mac?**
- `time.sleep()` is not precise enough on macOS
- Busy loop gives microsecond precision
- **Trade-off**: Higher CPU usage but better timing accuracy

### **On Linux:**
```python
else:
    # Uses efficient sleep for timing
    if seconds > 0:
        time.sleep(seconds)  # More efficient, less CPU usage
```

---

## 📈 **Why Your Frequency Modulates (26.7 Hz vs 30 Hz)**

### **Root Cause Analysis:**

#### **1. Processing Time Exceeds Target**
```
Your actual measurements:
- Inference: 13.5ms
- Camera reads: ~15-20ms (estimated)
- Motor I/O: ~3-5ms  
- Python overhead: ~2-3ms
- TOTAL: ~35-40ms

Target: 33.33ms
Shortfall: 2-7ms per loop
```

#### **2. When `busy_wait` Receives Negative Values**
```python
# Example from your system:
dt_s = 0.0375           # 37.5ms actual processing
target = 1/30           # 33.33ms target
wait_time = target - dt_s  # 33.33 - 37.5 = -4.17ms

busy_wait(-4.17ms)      # Can't wait negative time!
# Results in immediate continuation → lower frequency
```

#### **3. Variable Processing Times**
Different components have variable timing:
- **Camera reads**: 15-25ms (USB bandwidth, lighting)
- **Neural network**: 10-15ms (model complexity, GPU load)
- **Motor communication**: 2-6ms (serial bus congestion)

---

## 🎛️ **Frequency Control Strategies**

### **Strategy 1: Reduce Processing Time**
```bash
# Lower camera resolution to reduce processing time
--robot.cameras='{
  "gripper": {"width": 640, "height": 480},  # Instead of 1280x720
  "top": {"width": 640, "height": 480}
}'
# Expected gain: 5-10ms per loop → higher frequency
```

### **Strategy 2: Lower Target FPS**
```bash
# Set achievable target instead of aspirational
--dataset.fps=25  # 40ms target instead of 33.33ms
# Result: Stable 25 Hz instead of variable 26-27 Hz
```

### **Strategy 3: Adaptive Frequency (Advanced)**
```python
# Modify record_loop to adapt to actual performance
actual_fps = 1 / dt_s
adaptive_fps = min(target_fps, actual_fps * 0.95)  # 5% safety margin
busy_wait(1 / adaptive_fps - dt_s)
```

---

## 📊 **Real-Time Monitoring with Your New Feature**

Your control loop frequency logging now shows:

### **Console Output:**
```
🔄 CONTROL LOOP: 26.7 Hz  ← Real-time frequency
```

### **CSV Data:**
```csv
timestamp,step,control_loop_freq_hz,...
0.000,1,0.0,...          ← First step (no previous timing)
0.037,2,26.7,...         ← 37ms since last step = 26.7 Hz  
0.075,3,26.3,...         ← 38ms since last step = 26.3 Hz
0.112,4,27.0,...         ← 37ms since last step = 27.0 Hz
```

---

## 🔧 **Debugging Your Specific Case**

### **Expected vs Actual:**
```
Target Loop Time: 33.33ms (30 Hz)
Your Loop Time:   37.50ms (26.7 Hz)
Deficit:          4.17ms per loop

busy_wait calculation:
wait_time = 33.33ms - 37.50ms = -4.17ms
busy_wait(-4.17ms) → immediate continuation
```

### **To Achieve Stable 30 Hz:**
You need to **reduce processing time by 4.17ms**:
- Camera optimization: -3 to -8ms
- Lower resolution: -2 to -5ms  
- System optimization: -1 to -2ms
- **Total potential gain**: -6 to -15ms

---

## 🎯 **Quick Fix Command**

Try this optimized command to reduce processing time:

```bash
python -m lerobot.record \
  --robot.type=so100_follower \
  --robot.id=so100_follow \
  --robot.port=/dev/tty.usbmodem58760434091 \
  --robot.cameras='{
    "gripper": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30},
    "top": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}
  }' \
  --dataset.fps=25 \
  --policy.path=adungus/pi0_BatteryPickPlace_10k \
  --log=true
```

**Expected result**: Stable 25 Hz instead of variable 26-27 Hz 