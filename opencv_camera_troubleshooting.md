# OpenCVCamera(/dev/video6) 读取帧失败排查

## 错误信息

```
WARNING:lerobot.cameras.opencv.camera_opencv:Error reading frame in background thread for OpenCVCamera(/dev/video6): OpenCVCamera(/dev/video6) read failed (status=False).
RuntimeError: OpenCVCamera(/dev/video6) exceeded maximum consecutive read failures.
```

## 原因分析

后台读取线程连续读取帧失败超过 10 次（`camera_opencv.py:458`），线程被终止。常见原因：

1. **设备不存在** — `/dev/video6` 没有对应的物理相机
2. **被其他进程占用** — 有程序正在使用该设备
3. **USB 连接不稳定** — 相机掉线或带宽不足
4. **视频格式不匹配** — 设备不支持当前配置的分辨率/帧率

## 排查步骤

```bash
# 1. 查看有哪些视频设备
ls -l /dev/video*

# 2. 列出所有 V4L2 设备及其路径
v4l2-ctl --list-devices

# 3. 检查设备是否被其他进程占用
fuser /dev/video6

# 4. 查看设备支持的视频格式和分辨率
v4l2-ctl -d /dev/video6 --list-formats-ext

# 5. 查看设备当前配置
v4l2-ctl -d /dev/video6 --all
```

## 解决方案

| 原因 | 解决方法 |
|------|---------|
| 设备不存在 | 确认相机已连接，检查 USB 线缆，重新插拔后查看 `dmesg` 输出 |
| 被占用 | `fuser -k /dev/video6` 终止占用进程，或关闭其他使用相机的程序 |
| USB 不稳定 | 更换 USB 线缆/接口，避免使用 USB 集线器，检查 `dmesg` 是否有断连日志 |
| 格式不匹配 | 使用 `v4l2-ctl --list-formats-ext` 查看支持的格式，调整配置中的分辨率和帧率 |

## 相关代码

- 读取循环: `src/lerobot/cameras/opencv/camera_opencv.py:443` (`_read_loop`)
- 硬件读取: `src/lerobot/cameras/opencv/camera_opencv.py:346` (`_read_from_hardware`)
- 最大重试次数: 10 次 (`camera_opencv.py:458`)

---

# NERO Follower + SmolVLA 推理指南

## 1. 环境准备

### 登录 HuggingFace

```bash
huggingface-cli login
```

Token 从 https://huggingface.co/settings/tokens 获取（Read 权限即可）。

### 下载 SmolVLA 预训练模型

网络不好时使用镜像站下载到本地：

```bash
# 方式1：huggingface_hub 下载
HF_ENDPOINT=https://hf-mirror.com python3 -c "from huggingface_hub import snapshot_download; snapshot_download('lerobot/smolvla_base', local_dir='$HOME/models/smolvla_base')"

# 方式2：git clone
git lfs install
git clone https://hf-mirror.com/lerobot/smolvla_base ~/models/smolvla_base
```

模型目录下必须包含以下文件：

```
config.json
model.safetensors
policy_postprocessor.json
policy_postprocessor_step_0_unnormalizer_processor.safetensors
policy_preprocessor.json
policy_preprocessor_step_5_normalizer_processor.safetensors
```

## 2. 零样本推理命令

```bash
lerobot-record \
    --robot.type=nero_follower \
    --robot.channel=can0 \
    --robot.interface=socketcan \
    --robot.firmeware_version=default \
    --robot.speed_percent=10 \
    --robot.disable_torque_on_disconnect=False \
    --robot.cameras='{ front: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: "MJPG"} }' \
    --policy.path=~/models/smolvla_base \
    --dataset.repo_id=yuhang/nero_smolvla_test \
    --dataset.num_episodes=1 \
    --dataset.single_task="pick up the object" \
    --dataset.push_to_hub=false \
    --display_data=true
```

> 注意：无微调的 `smolvla_base` 是通用基础模型，在 NERO Follower 上效果可能很差，需微调后才能有好的表现。
