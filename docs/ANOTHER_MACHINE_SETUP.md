# 在另一台电脑上部署 HIL-SERL 训练环境

本指南说明如何在另一台电脑上使用已有的 GitHub Fork (`LIJianxuanLeo/lerobot`) 进行 SO-ARM 101 真实机器人的数据采集和 RL 训练。

---

## 部署方案选择

有三种部署方式，根据你的硬件条件选择：

| 方案 | 适用场景 | 电脑要求 |
|------|---------|---------|
| **方案 A：单机部署** | 一台电脑完成所有工作 | 连接机器人 + 有 GPU（或 Apple Silicon） |
| **方案 B：双机部署（推荐）** | Actor 和 Learner 分离 | Mac 连机器人（Actor），GPU 服务器训练（Learner） |
| **方案 C：先采集后训练** | 没有 GPU 服务器 | 任何电脑采集 demo，之后在 GPU 上训练 |

---

## 第一步：克隆仓库 & 安装环境

以下步骤适用于**任何新电脑**（Linux / macOS / Windows WSL）。

### 1.1 安装 Conda（如果没有）

```bash
# macOS (Apple Silicon)
curl -L -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh

# Linux (x86_64)
curl -L -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### 1.2 创建 Conda 环境

```bash
conda create -n hilserl python=3.12 -y
conda activate hilserl
```

### 1.3 安装 FFmpeg（必须 v7.x）

```bash
conda install ffmpeg=7.1.1 -c conda-forge

# 验证
ffmpeg -version           # 确认是 7.x
ffmpeg -encoders | grep libsvtav1  # 确认有 svtav1 编码器
```

### 1.4 克隆 Fork 仓库

```bash
git clone https://github.com/LIJianxuanLeo/lerobot.git
cd lerobot
```

### 1.5 安装 LeRobot

根据这台电脑的角色，选择不同的安装方式：

```bash
# 方案一：连接机器人的电脑（需要 feetech 驱动 + HIL-SERL）
pip install -e ".[feetech,hilserl]"

# 方案二：只做训练的 GPU 服务器（不需要 feetech）
pip install -e ".[hilserl]"
```

### 1.6 修复已知兼容性问题

```bash
# transformers 5.x 会破坏 ResNet10 加载，必须降级
pip install "transformers>=4.45,<5.0"
```

### 1.7 验证安装

```bash
python -c "
import torch
import lerobot
print(f'LeRobot version: {lerobot.__version__}')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'MPS available: {torch.backends.mps.is_available()}')
"
```

---

## 第二步：配置文件说明

仓库已包含所有需要的配置文件：

```
lerobot/
├── configs/
│   ├── sim/                          # 仿真配置（已完成）
│   │   ├── env_config.json
│   │   ├── train_config.json
│   │   └── record_config.json
│   └── real/                         # 真实机器人配置
│       ├── env_config.json           # 环境 + 录制配置
│       └── train_config.json         # SAC 训练配置
├── SO101/
│   └── so101_new_calib.urdf          # 逆运动学 URDF
└── docs/
    ├── SIM_TO_REAL_PLAN.md           # 完整部署方案
    └── HIL-SERL_REPRODUCTION_GUIDE.md
```

### 需要修改的参数

使用前**必须**修改配置文件中的以下占位符：

| 占位符 | 替换为 | 如何获取 |
|--------|--------|---------|
| `/dev/tty.usbmodemFOLLOWER` | Follower 的 USB 端口 | `lerobot-find-port` |
| `/dev/tty.usbmodemLEADER` | Leader 的 USB 端口 | `lerobot-find-port` |
| `my_follower` | Follower 的校准 ID | 校准时指定 |
| `my_leader` | Leader 的校准 ID | 校准时指定 |
| `index_or_path: 0, 1` | 摄像头设备号 | 见下方测试方法 |
| `end_effector_bounds` | 实际 EE 边界 | `lerobot-find-joint-limits` |
| `fixed_reset_joint_positions` | 实际复位姿态角度 | 手动测量 |

**测试摄像头设备号：**
```bash
python -c "
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        print(f'Camera {i}: {frame.shape if ret else \"no frame\"}')
        cap.release()
"
```

---

## 第三步：硬件准备（连接机器人的电脑）

### 3.1 连接硬件

```bash
# 连接 Follower 和 Leader USB
# 连接两个 USB 摄像头

# 查找端口
lerobot-find-port
```

### 3.2 设置电机（仅新机械臂首次需要）

```bash
lerobot-setup-motors \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodemXXXX
```

### 3.3 校准两个机械臂

```bash
# 校准 Follower
lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodemXXXX \
  --robot.id=my_follower

# 校准 Leader
lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodemYYYY \
  --teleop.id=my_leader
```

校准文件保存在：`~/.cache/huggingface/lerobot/calibration/`

### 3.4 测试遥操作

```bash
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodemXXXX \
  --robot.id=my_follower \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodemYYYY \
  --teleop.id=my_leader \
  --display-cameras 1
```

确认：Leader 移动时 Follower 跟随，摄像头画面正常显示。

### 3.5 获取关节限位和末端执行器边界

```bash
lerobot-find-joint-limits \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodemXXXX \
  --robot.id=my_follower \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodemYYYY \
  --teleop.id=my_leader \
  --urdf_path=SO101/so101_new_calib.urdf \
  --target_frame_name=gripper_frame_link \
  --teleop_time_s=30
```

**操作：** 30 秒内用 Leader 手臂遍历整个工作空间。记录输出的 `min_ee` 和 `max_ee`。

**重要：** SO-101 的 EE 边界可能偏小（[#1387](https://github.com/huggingface/lerobot/issues/1387)），建议将边界在各方向扩大 ~20%。

### 3.6 更新配置文件

用实际值替换 `configs/real/env_config.json` 和 `configs/real/train_config.json` 中的占位符。

---

## 第四步：数据采集（录制演示）

### 4.1 录制 Demo

```bash
cd lerobot

# 修改 configs/real/env_config.json 中的端口和参数后：
python -m lerobot.rl.gym_manipulator --config_path configs/real/env_config.json
```

### 4.2 录制操作指南

| 按键 | 功能 |
|------|------|
| `Space` | 切换 Leader 控制（开始/释放遥操作） |
| `s` | 标记当前 episode 成功 |
| `Esc` | 标记当前 episode 失败/丢弃 |

**录制流程：**
1. 程序启动后，机器人自动移到复位姿态
2. 按 `Space` 开始遥操作
3. 用 Leader 手臂控制 Follower 抓取方块
4. 抓取成功后，按 `s` 标记成功
5. 失败则按 `Esc` 丢弃重来
6. 重复录制 **25+ 个成功 episode**

**录制技巧：**
- 从上方接近，缓慢下降
- 夹爪对准方块中心
- 抓稳后直接向上提起
- 动作保持平滑一致
- 每次录制前把方块放回同一位置

### 4.3 处理数据集 — ROI 裁剪

```bash
python -m lerobot.rl.crop_dataset_roi --repo-id LIJianxuanLeo/real_pick_cube
```

用鼠标框选每个摄像头的工作区域，排除背景杂物。按 `c` 确认，`r` 重选。

将输出的裁剪参数更新到配置文件：
```json
"crop_params_dict": {
    "observation.images.front": [top, left, height, width],
    "observation.images.wrist": [top, left, height, width]
}
```

### 4.4 训练 Reward Classifier（可选但推荐）

```bash
python -m lerobot.rl.train_reward_classifier \
  --repo-id LIJianxuanLeo/real_pick_cube \
  --output-dir outputs/reward_classifier
```

准确率应达到 >95%。训练完成后，将路径更新到 `train_config.json`:
```json
"reward_classifier": {
    "pretrained_path": "outputs/reward_classifier/checkpoint/",
    "success_threshold": 0.5,
    "success_reward": 1.0
}
```

---

## 第五步：RL 训练

### 方案 A：单机训练（所有操作在一台电脑上）

```bash
# 终端 1：启动 Learner（训练进程）
python -m lerobot.rl.learner --config_path configs/real/train_config.json

# 终端 2：启动 Actor（机器人控制进程）
python -m lerobot.rl.actor --config_path configs/real/train_config.json
```

**设备选择：**
- Apple Silicon Mac：`"device": "mps"`
- NVIDIA GPU：`"device": "cuda"`
- 无 GPU：`"device": "cpu"`（非常慢，不推荐）

---

### 方案 B：双机训练（推荐）

**适用场景：** Mac 连接机器人（Actor），GPU 服务器训练（Learner）

#### 在 GPU 服务器上（Learner）：

```bash
# 1. 克隆仓库 & 安装
git clone https://github.com/LIJianxuanLeo/lerobot.git
cd lerobot
conda create -n hilserl python=3.12 -y
conda activate hilserl
conda install ffmpeg=7.1.1 -c conda-forge
pip install -e ".[hilserl]"
pip install "transformers>=4.45,<5.0"

# 2. 修改 train_config.json 的两个关键参数：
#    "device": "cuda"                    ← 使用 NVIDIA GPU
#    "learner_host": "0.0.0.0"           ← 监听所有网络接口
#    "storage_device": "cpu"             ← replay buffer 存 CPU（节省显存）

# 3. 如果 demo 数据在 Mac 上录制，需要先传输到 GPU 服务器：
scp -r user@mac_ip:~/lerobot/data/LIJianxuanLeo/real_pick_cube \
    ~/.cache/huggingface/lerobot/LIJianxuanLeo/real_pick_cube

# 4. 启动 Learner
python -m lerobot.rl.learner --config_path configs/real/train_config.json
```

#### 在 Mac 上（Actor，连接机器人）：

```bash
# 修改 train_config.json 的关键参数：
#    "learner_host": "GPU服务器的IP地址"   ← 连接远程 Learner
#    "device": "cpu"                      ← Actor 用 CPU 即可

# 启动 Actor
python -m lerobot.rl.actor --config_path configs/real/train_config.json
```

#### 网络要求：
- 两台电脑在同一局域网，或做端口转发
- 需要开放 TCP 端口 **50051**（gRPC）
- 如果有防火墙，需要放行：
  ```bash
  # Linux GPU 服务器
  sudo ufw allow 50051/tcp

  # 或用 SSH 隧道
  ssh -L 50051:localhost:50051 user@gpu_server
  ```

---

### 方案 C：先采集后训练

**适用场景：** 没有随时可用的 GPU 服务器

```bash
# 第一阶段：在连接机器人的电脑上录制 demo
python -m lerobot.rl.gym_manipulator --config_path configs/real/env_config.json

# 将数据集上传到 HuggingFace Hub
# 修改 env_config.json: "push_to_hub": true, "repo_id": "你的用户名/real_pick_cube"
# 或者手动 scp 传输数据到 GPU 服务器

# 第二阶段：在 GPU 服务器上训练（离线，不连机器人）
# 注意：纯离线训练无法做 human-in-the-loop 干预
# 但可以先用 demo 数据做离线 SAC 预训练
```

---

## 第六步：训练过程中的人类干预

训练启动后，你需要在 Actor 端进行人类干预：

| 阶段 | Episode | 操作 |
|------|---------|------|
| 自由探索 | 1-5 | **不要干预**，让策略自由探索 |
| 轻度引导 | 5-20 | 用 Leader 手臂做**短暂纠正**（1-2秒） |
| 放手 | 20+ | 减少干预，让策略自主完成 |

**操作方式：**
- `Space` — 按住 Leader 接管控制
- 松开 `Space` — 释放控制给策略
- `s` — 标记 episode 成功
- `Esc` — 标记 episode 失败

**关键原则：**
- 只做短暂纠正推力，**不要完成整个任务**
- 长时间接管会导致 Q 函数过估计，反而损害学习效果
- 干预率应随时间下降

---

## 第七步：监控训练

**终端输出关键指标：**
- `Episodic reward` — 每个 episode 的总奖励（应逐渐上升）
- `Intervention rate` — 人类干预比例（应逐渐下降）
- `Policy frequency [Hz]` — 策略推理频率（应 ~10 Hz）

**启用 Weights & Biases 监控（可选）：**
```json
// 在 train_config.json 中修改
"wandb": {
    "enable": true,
    "project": "so101_real_pick"
}
```

```bash
# 首次使用需要登录
wandb login
```

**预期结果：**

| 指标 | 目标值 |
|------|--------|
| 训练时间 | 1-3 小时主动监控 |
| 抓取成功率 | ~70%+ |
| 最终干预率 | <10% |

---

## 调参指南

如果训练效果不好，按以下顺序调整：

| 问题 | 参数 | 调整 |
|------|------|------|
| 机器人几乎不动 | `temperature_init` | 0.01 → 0.1（增加探索） |
| 策略更新太慢 | `policy_parameters_push_frequency` | 4 → 1（秒） |
| 训练不稳定 | `critic_lr` / `actor_lr` | 3e-4 → 1e-4 |
| Learner 跟不上 | `utd_ratio` | 2 → 1 |
| 夹爪反复开合 | `gripper_penalty` | -0.02 → -0.05 |

---

## 常见问题排查

### 安装问题

| 问题 | 解决方案 |
|------|---------|
| `pip install` 失败 | 确认 Python 3.12，`conda activate hilserl` |
| FFmpeg 编码器缺失 | `conda install ffmpeg=7.1.1 -c conda-forge` |
| `transformers` 报错 | `pip install "transformers>=4.45,<5.0"` |
| `placo` 安装失败 | Linux: `sudo apt install cmake`; Mac: `brew install cmake` |

### 硬件问题

| 问题 | 解决方案 |
|------|---------|
| 找不到 USB 端口 | 检查 USB 线连接，`ls /dev/tty.usbmodem*`（Mac）或 `ls /dev/ttyUSB*`（Linux） |
| 校准失败 | 删除 `~/.cache/huggingface/lerobot/calibration/` 重新校准 |
| 电机不转 | 检查电源，LED 灯状态，用 `lerobot-setup-motors` 重新设置 |
| 摄像头打不开 | 确认设备号，Mac 需要授权摄像头权限 |

### 训练问题

| 问题 | 解决方案 |
|------|---------|
| Actor 连不上 Learner | 检查 IP、端口 50051、防火墙 |
| CUDA OOM | 减小 `batch_size`（256 → 128），`storage_device: "cpu"` |
| MPS 内存不足 | `storage_device: "cpu"`，减小 buffer capacity |
| 策略频率太低 | 降低图像分辨率（128→64），或用更快的 GPU |

### Linux USB 权限（非 root 用户）

```bash
# 添加用户到 dialout 组
sudo usermod -aG dialout $USER
# 重新登录生效

# 或临时修改权限
sudo chmod 666 /dev/ttyUSB0
```

---

## 快速启动清单

在新电脑上从零开始的完整命令序列：

```bash
# ===== 环境安装 =====
conda create -n hilserl python=3.12 -y
conda activate hilserl
conda install ffmpeg=7.1.1 -c conda-forge
git clone https://github.com/LIJianxuanLeo/lerobot.git
cd lerobot
pip install -e ".[feetech,hilserl]"
pip install "transformers>=4.45,<5.0"

# ===== 硬件设置 =====
lerobot-find-port
# 记下 Follower 和 Leader 的端口

lerobot-calibrate --robot.type=so101_follower --robot.port=<FOLLOWER_PORT> --robot.id=my_follower
lerobot-calibrate --teleop.type=so101_leader --teleop.port=<LEADER_PORT> --teleop.id=my_leader

lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=<FOLLOWER_PORT> --robot.id=my_follower \
  --teleop.type=so101_leader --teleop.port=<LEADER_PORT> --teleop.id=my_leader \
  --display-cameras 1

# ===== 修改配置文件中的端口号 =====
# 编辑 configs/real/env_config.json 和 configs/real/train_config.json
# 替换所有 /dev/tty.usbmodemFOLLOWER 和 /dev/tty.usbmodemLEADER

# ===== 获取工作空间边界 =====
lerobot-find-joint-limits \
  --robot.type=so101_follower --robot.port=<FOLLOWER_PORT> --robot.id=my_follower \
  --teleop.type=so101_leader --teleop.port=<LEADER_PORT> --teleop.id=my_leader \
  --urdf_path=SO101/so101_new_calib.urdf --target_frame_name=gripper_frame_link --teleop_time_s=30
# 将输出的 EE bounds 更新到配置文件

# ===== 录制 Demo =====
python -m lerobot.rl.gym_manipulator --config_path configs/real/env_config.json

# ===== ROI 裁剪 =====
python -m lerobot.rl.crop_dataset_roi --repo-id LIJianxuanLeo/real_pick_cube
# 更新 crop_params_dict 到配置文件

# ===== 训练 =====
# 终端 1
python -m lerobot.rl.learner --config_path configs/real/train_config.json
# 终端 2
python -m lerobot.rl.actor --config_path configs/real/train_config.json
```
