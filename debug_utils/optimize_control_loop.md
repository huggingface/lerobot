# Control Loop Frequency Optimization Guide

## 🎯 **Target: Constant 30 Hz Control Loop**

### **Current Performance Analysis:**
- **Target**: 30 Hz (33.33ms per loop)
- **Actual**: 26.7 Hz (37.5ms per loop) 
- **Gap**: 4.17ms extra processing time

---

## 🚀 **Optimization Strategies**

### **1. Camera Optimization (Biggest Impact)**
```bash
# Reduce camera resolution (saves 5-10ms per camera)
--robot.cameras='{
  "gripper": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30},
  "top":     {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}
}'

# Or even lower for fastest performance
--robot.cameras='{
  "gripper": {"type": "opencv", "index_or_path": 0, "width": 320, "height": 240, "fps": 30},
  "top":     {"type": "opencv", "index_or_path": 1, "width": 320, "height": 240, "fps": 30}
}'
```

### **2. Motor Communication Optimization**
```python
# In robot config, reduce sync read frequency
# Read only essential motor data per loop
sync_read_motors = ["Present_Position"]  # Skip temperature, voltage, etc.
```

### **3. Policy Inference Optimization**
```bash
# Use smaller model or optimize inference
--policy.path=adungus/pi0_BatteryPickPlace_smaller  # If available
# Or use mixed precision for faster inference
--policy.use_amp=true
```

### **4. System-Level Optimizations**

#### **A. CPU Priority (Linux/macOS)**
```bash
# Run with higher priority
sudo nice -n -10 python -m lerobot.record ...
```

#### **B. USB Optimization**
- Use USB 3.0 ports for cameras
- Separate cameras on different USB controllers
- Disable USB power management:
```bash
# macOS: System Preferences > Energy Saver > Prevent computer from sleeping
# Linux: echo 'on' | sudo tee /sys/bus/usb/devices/*/power/control
```

#### **C. Python Performance**
```bash
# Use faster Python (if available)
python3.11 -m lerobot.record ...  # Python 3.11+ has better performance
```

---

## 🔧 **Alternative Approach: Adaptive Frequency**

Instead of forcing 30 Hz, adapt to actual performance:

### **Option A: Lower Target FPS**
```bash
# Set realistic target based on actual performance
--dataset.fps=25  # More achievable than 30
```

### **Option B: Variable FPS Control**
Modify the control loop to use dynamic frequency adjustment:

```python
# In record_loop, replace fixed fps with adaptive
target_fps = min(fps, 1.0 / actual_loop_time)  # Adapt to actual performance
busy_wait(1 / target_fps - dt_s)
```

---

## 📊 **Monitoring Performance**

### **Real-time Monitoring**
Your control loop frequency logging now shows:
```
🔄 CONTROL LOOP: 26.7 Hz  # Real-time feedback
```

### **CSV Analysis**
Check `control_loop_freq_hz` column for:
- **Average frequency**
- **Frequency stability** (standard deviation)
- **Drop patterns** (when does frequency drop?)

---

## 🎯 **Recommended Quick Fixes**

### **Immediate (5-minute fixes):**
1. **Lower camera resolution**: 1280x720 → 640x480
2. **Reduce dataset FPS**: 30 → 25 Hz
3. **Close background apps**: Free up CPU/USB bandwidth

### **Advanced (requires code changes):**
1. **Async camera reads**: Read cameras in parallel
2. **Motor read optimization**: Only read essential data
3. **Threading**: Separate inference from motor control

---

## 🔍 **Expected Results**

| Optimization | Expected Gain | Final FPS |
|--------------|---------------|-----------|
| Lower camera res | +3-5ms | 28-29 Hz |
| Reduce FPS target | N/A | 25 Hz (stable) |
| USB optimization | +1-2ms | 27-28 Hz |
| All combined | +5-8ms | 30+ Hz |

---

## ⚡ **Best Practice Commands**

### **Optimized Recording Command**
```bash
python -m lerobot.record \
  --robot.type=so100_follower \
  --robot.id=so100_follow \
  --robot.port=/dev/tty.usbmodem58760434091 \
  --robot.cameras='{
    "gripper": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30},
    "top":     {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}
  }' \
  --display_data=true \
  --dataset.repo_id=adungus/eval_BPP-v1 \
  --dataset.single_task="Put the battery in the cup" \
  --dataset.fps=25 \
  --policy.path=adungus/pi0_BatteryPickPlace_10k \
  --log=true
```

### **Performance Monitoring Command**
```bash
# Monitor system resources during recording
top -pid $(pgrep -f "lerobot.record")
``` 