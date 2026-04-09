# HIL-SERL LeRobot 完整部署指南（Ubuntu 22.04 + NVIDIA RTX 3060）

本指南面向一台全新的 Ubuntu 22.04 + NVIDIA RTX 3060 电脑，从零开始完成 HIL-SERL 项目的部署。

包含两个独立方案：
- **方案 A**：MuJoCo 仿真训练（无需机器人硬件）
- **方案 B**：真实 SO-ARM 101 六自由度机械臂部署

| 维度 | 方案 A（仿真） | 方案 B（真实机器人） |
|------|--------------|-------------------|
| 机器人 | Franka Panda（MuJoCo） | SO-ARM 101（6-DOF + 夹爪） |
| 控制方式 | 键盘 | Leader 手臂遥操作 |
| 额外硬件 | 无 | 2x SO-ARM 101 + 2x USB 摄像头 |
| 安装命令 | `pip install -e ".[hilserl]"` | `pip install -e ".[feetech,hilserl]"` |
| 动作空间 | 3D (dx, dy, dz) | 4D (dx, dy, dz, gripper) |

---

## 共用环境搭建（方案 A 和方案 B 都需要）

### 1. 系统要求确认

```bash
# 确认 Ubuntu 版本
lsb_release -a
# 期望输出：Ubuntu 22.04.x LTS

# 确认 NVIDIA GPU
lspci | grep -i nvidia
# 期望输出：NVIDIA Corporation GA106 [GeForce RTX 3060] 或类似

# 确认磁盘空间（至少 50GB 可用）
df -h /
```

**最低硬件要求：**
- GPU：NVIDIA RTX 3060（12GB VRAM）
- 内存：16GB RAM（推荐 32GB）
- 磁盘：50GB 可用空间
- 网络：能访问 GitHub 和 HuggingFace Hub

### 2. 安装 NVIDIA 驱动

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装推荐的 NVIDIA 驱动
sudo apt install -y nvidia-driver-535

# 重启
sudo reboot
```

重启后验证：
```bash
nvidia-smi
```

**期望输出：**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xxx       Driver Version: 535.xxx    CUDA Version: 12.x      |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
|   0  NVIDIA GeForce RTX 3060  |   00000000:01:00.0  On |                  N/A |
+-------------------------------+----------------------+----------------------+
```

**关键确认：** `CUDA Version` 显示 12.x 即可。PyTorch 会自带 CUDA 运行时，不需要单独安装 CUDA Toolkit。

> **注意：** 如果 `nvidia-smi` 报错，尝试 `sudo apt install -y nvidia-driver-550` 或更新的版本。

### 3. 安装 Miniconda + 创建 Python 环境

```bash
# 下载 Miniconda（Linux x86_64）
curl -L -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# 按提示完成安装，选择 yes 初始化 conda

# 重新打开终端，或手动激活
source ~/.bashrc

# 创建 Python 3.12 环境
conda create -n hilserl python=3.12 -y
conda activate hilserl
```

### 4. 安装 Ubuntu 系统依赖包

```bash
sudo apt install -y \
  build-essential \
  git \
  curl \
  cmake \
  pkg-config \
  ninja-build \
  libglib2.0-0 \
  libegl1-mesa-dev \
  libgl1-mesa-dev \
  libglfw3-dev \
  libusb-1.0-0-dev \
  portaudio19-dev \
  libgeos-dev \
  v4l-utils \
  libavformat-dev \
  libavcodec-dev \
  libavdevice-dev \
  libavutil-dev \
  libswscale-dev \
  libswresample-dev \
  libavfilter-dev
```

### 5. 安装 FFmpeg v7.x

**必须使用 Conda 安装 FFmpeg v7.x**（Ubuntu 22.04 自带的是 v4.x，不兼容）：

```bash
conda install ffmpeg=7.1.1 -c conda-forge
```

验证：
```bash
ffmpeg -version
# 确认版本是 7.x

ffmpeg -encoders 2>/dev/null | grep libsvtav1
# 期望输出包含 libsvtav1（视频编码器，LeRobot 必需）
```

### 6. 克隆 Fork 仓库

```bash
cd ~
git clone https://github.com/LIJianxuanLeo/lerobot.git
cd lerobot
```

### 7. 安装 LeRobot + 修复兼容性

**方案 A（仿真）：**
```bash
pip install -e ".[hilserl]"
```

**方案 B（真实机器人）：**
```bash
pip install -e ".[feetech,hilserl]"
```

**必须执行的兼容性修复：**
```bash
# transformers 5.x 会导致 ResNet10 视觉编码器加载失败，必须降级
pip install "transformers>=4.45,<5.0"
```

### 8. 验证安装

```bash
python -c "
import torch
import lerobot
print(f'LeRobot version: {lerobot.__version__}')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
"
```

**期望输出：**
```
LeRobot version: 0.5.1
PyTorch version: 2.10.0+cu126
CUDA available: True
GPU: NVIDIA GeForce RTX 3060
VRAM: 12.0 GB
```

如果 `CUDA available: False`，请检查：
1. NVIDIA 驱动是否正确安装（`nvidia-smi`）
2. 是否在 conda 环境中（`conda activate hilserl`）
3. PyTorch 是否安装了 CUDA 版本（`pip install torch --index-url https://download.pytorch.org/whl/cu126`）

---

## 方案 A：MuJoCo 仿真训练

本方案在 MuJoCo 物理引擎中训练 Franka Panda 机械臂执行抓取任务，无需任何实体机器人。

### A1. MuJoCo 渲染后端配置

Linux 上 MuJoCo 需要手动指定渲染后端（macOS 上自动选择）：

```bash
# 有桌面显示器（推荐，可以看到 MuJoCo 窗口）
export MUJOCO_GL=glfw

# 无显示器（纯 SSH 连接的服务器，无法显示窗口）
export MUJOCO_GL=egl

# 建议写入 ~/.bashrc 自动加载
echo 'export MUJOCO_GL=glfw' >> ~/.bashrc
source ~/.bashrc
```

**与 macOS 的区别：** Linux 上不需要 `mjpython`，直接使用 `python` 即可。

验证 MuJoCo 安装：
```bash
python -c "import mujoco; print(f'MuJoCo version: {mujoco.__version__}')"
python -c "import gym_hil; print('gym_hil OK')"
```

### A2. 配置文件

仓库已包含 CUDA 版仿真训练配置：

```
configs/sim/
├── env_config.json            # 环境配置（录制 Demo 用）
├── record_config.json         # 录制配置
├── train_config.json          # 训练配置（macOS MPS 版）
└── train_config_cuda.json     # 训练配置（Ubuntu CUDA 版）  ← 使用这个
```

`train_config_cuda.json` 与 `train_config.json` 的唯一区别是 `"device": "cuda"`（替换 `"mps"`）。

### A3. 验证仿真环境

先打开 MuJoCo 仿真窗口，确认环境正常：

```bash
export MUJOCO_GL=glfw
python -m lerobot.rl.gym_manipulator --config_path configs/sim/env_config.json
```

**操作方式（键盘控制）：**

| 按键 | 功能 |
|------|------|
| 方向键 ←→ | X 轴移动 |
| 方向键 ↑↓ | Y 轴移动 |
| Shift + ↑↓ | Z 轴移动（上下） |
| Ctrl | 夹爪开合 |
| Enter | 标记成功 |
| Backspace | 标记失败 |

应该能看到 Franka Panda 机械臂和一个方块。

### A4. 录制仿真 Demo（可选）

可以自己录制 Demo，也可以使用官方数据集（推荐）。

**录制 Demo：**
```bash
export MUJOCO_GL=glfw
python -m lerobot.rl.gym_manipulator --config_path configs/sim/record_config.json
```

**使用官方数据集（推荐）：**

`train_config_cuda.json` 已配置使用官方数据集 `aractingi/franka_sim_pick_lift_5`（30 个成功抓取 episode）。首次运行会自动从 HuggingFace Hub 下载。

### A5. 启动 Actor-Learner 训练（单机）

打开**两个终端**，都激活 conda 环境：

**终端 1 — 启动 Learner（训练进程，使用 CUDA）：**
```bash
conda activate hilserl
cd ~/lerobot
python -m lerobot.rl.learner --config_path configs/sim/train_config_cuda.json
```

**终端 2 — 启动 Actor（仿真交互进程）：**
```bash
conda activate hilserl
cd ~/lerobot
export MUJOCO_GL=glfw
python -m lerobot.rl.actor --config_path configs/sim/train_config_cuda.json
```

**启动顺序很重要：先启动 Learner，再启动 Actor。** Actor 通过 gRPC 连接 Learner（默认 `127.0.0.1:50051`）。

启动后：
- Learner 终端显示训练损失和优化步数
- Actor 终端显示 episode 奖励和交互步数
- MuJoCo 窗口显示机械臂在自主尝试抓取方块

### A6. 双机训练（可选）

如果要将 Learner 和 Actor 分开到两台机器：

**GPU 服务器（Learner）：**
修改 `train_config_cuda.json` 中 `actor_learner_config.learner_host` 为 `"0.0.0.0"`，然后：
```bash
python -m lerobot.rl.learner --config_path configs/sim/train_config_cuda.json
```

**另一台机器（Actor）：**
修改 `train_config_cuda.json` 中 `actor_learner_config.learner_host` 为 GPU 服务器 IP：
```bash
export MUJOCO_GL=glfw
python -m lerobot.rl.actor --config_path configs/sim/train_config_cuda.json
```

**防火墙：** 需要开放 TCP 端口 50051：
```bash
sudo ufw allow 50051/tcp
```

### A7. 训练性能预期

| 指标 | Apple Silicon (MPS) | RTX 3060 (CUDA) |
|------|-------------------|-----------------|
| Learner 优化速率 | ~0.17 Hz | ~10-50 Hz |
| 100k 步训练时间 | ~158 小时 | **~3-15 小时** |
| 20k 步训练时间 | ~32 小时 | **~0.6-3 小时** |

RTX 3060 的 CUDA 加速约为 Apple Silicon MPS 的 **50-300 倍**，是可实际完成训练的关键。

### A8. 训练监控

**检查点保存位置：** `outputs/train/<日期>/<时间>_franka_sim_sac/`

**启用 Weights & Biases（可选）：**
```bash
# 首次使用需要注册和登录
pip install wandb
wandb login
```

修改 `train_config_cuda.json`：
```json
"wandb": {
    "enable": true,
    "project": "franka_sim"
}
```

**后台训练（SSH 断连不中断）：**
```bash
# 使用 tmux
sudo apt install -y tmux
tmux new -s learner
# 在 tmux 中启动 learner，按 Ctrl+B 然后 D 分离
tmux new -s actor
# 在 tmux 中启动 actor

# 重新连接
tmux attach -t learner
```

**训练成功标志：**
- Episode reward 逐渐上升
- 策略开始尝试接近方块
- 100k 步后应有 >80% 的抓取成功率

### A9. 仿真方案快速启动清单

完整命令序列（假设已完成共用环境搭建 1-8）：

```bash
conda activate hilserl
cd ~/lerobot

# 验证 MuJoCo
export MUJOCO_GL=glfw
python -c "import mujoco; import gym_hil; print('OK')"

# 终端 1：Learner
python -m lerobot.rl.learner --config_path configs/sim/train_config_cuda.json

# 终端 2：Actor
export MUJOCO_GL=glfw
python -m lerobot.rl.actor --config_path configs/sim/train_config_cuda.json
```

---

## 方案 B：真实 SO-ARM 101 六自由度机械臂部署

本方案在真实 SO-ARM 101 机器人上执行 HIL-SERL 训练，通过 Leader 手臂进行人类干预。

### B1. 额外安装 Feetech 驱动

如果在共用环境搭建第 7 步中使用了 `pip install -e ".[hilserl]"`（仿真版），需要补装 Feetech 驱动：

```bash
pip install -e ".[feetech,hilserl]"
```

验证：
```bash
python -c "from lerobot.robots.so_follower import SO101FollowerConfig; print('Feetech OK')"
```

### B2. Linux USB 串口权限

SO-ARM 101 通过 USB 转串口通信。Linux 默认普通用户无串口权限。

```bash
# 添加当前用户到 dialout 组
sudo usermod -aG dialout $USER

# 必须重新登录（或重启）才生效
# 验证方式 1：重新登录后
groups
# 输出应包含 dialout

# 验证方式 2：检查串口权限
ls -la /dev/ttyACM*
# 应显示 crw-rw---- ... dialout
```

> **重要：** `sudo usermod -aG dialout $USER` 之后必须**完全退出登录再重新登录**，仅 `source ~/.bashrc` 不够。

**Linux 与 macOS 的端口命名区别：**

| 系统 | 端口格式 | 示例 |
|------|---------|------|
| macOS | `/dev/tty.usbmodemXXXX` | `/dev/tty.usbmodem58760432981` |
| Linux | `/dev/ttyACM*` 或 `/dev/ttyUSB*` | `/dev/ttyACM0`, `/dev/ttyACM1` |

### B3. udev 规则：持久化设备命名

Linux 上 USB 设备的编号（`/dev/ttyACM0`, `/dev/ttyACM1`）在重启或重新插拔后可能会变。建议创建 udev 规则来固定设备名。

**步骤 1：查找设备属性**

插入 Follower 手臂的 USB，运行：
```bash
# 找到当前端口
ls /dev/ttyACM*

# 查看设备属性（替换为实际端口）
udevadm info -a /dev/ttyACM0 | grep -E "idVendor|idProduct|serial"
```

记录输出的 `idVendor`、`idProduct` 和 `serial` 值。对 Leader 手臂重复同样操作。

**步骤 2：创建 udev 规则**

```bash
sudo nano /etc/udev/rules.d/99-lerobot.rules
```

写入以下内容（替换为你的实际值）：
```
# SO-ARM 101 Follower
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d4", ATTRS{serial}=="FOLLOWER_SERIAL", SYMLINK+="ttyACM_follower", MODE="0666"

# SO-ARM 101 Leader
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d4", ATTRS{serial}=="LEADER_SERIAL", SYMLINK+="ttyACM_leader", MODE="0666"
```

**步骤 3：重载规则**

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

验证：
```bash
ls -la /dev/ttyACM_follower /dev/ttyACM_leader
# 应显示指向 /dev/ttyACM* 的符号链接
```

之后在配置文件中可以使用稳定的 `/dev/ttyACM_follower` 和 `/dev/ttyACM_leader`。

> **如果两个手臂的 serial 相同**（部分芯片没有唯一序列号），可以用 `KERNELS` 属性（USB 物理端口位置）来区分。运行 `udevadm info -a /dev/ttyACM0 | grep KERNELS` 找到不同的内核路径。

### B4. 摄像头设置与验证

**列出已连接的摄像头：**
```bash
# 使用 v4l2（Video4Linux2）
v4l2-ctl --list-devices

# 或使用 LeRobot 内置工具
lerobot-find-cameras opencv
```

**常见输出：**
```
/dev/video0   ← 前方摄像头（或笔记本内置）
/dev/video1   ← 控制流（跳过）
/dev/video2   ← 腕部摄像头
/dev/video3   ← 控制流（跳过）
```

> **注意：** Linux 上每个摄像头通常占用**两个** `/dev/video*` 设备号（偶数是视频流，奇数是元数据）。实际使用偶数编号。

**验证摄像头画面：**
```bash
python -c "
import cv2
for i in range(0, 10, 2):  # 只检查偶数
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f'/dev/video{i}: {frame.shape[1]}x{frame.shape[0]}')
        cap.release()
"
```

**在配置文件中使用设备号或路径：**
```json
"cameras": {
    "front": {
        "type": "opencv",
        "index_or_path": 0,
        "width": 640,
        "height": 480,
        "fps": 30
    },
    "wrist": {
        "type": "opencv",
        "index_or_path": 2,
        "width": 640,
        "height": 480,
        "fps": 30
    }
}
```

### B5. 电机设置与校准

**查找端口：**
```bash
lerobot-find-port
# Linux 输出示例：
# Found ports:
#   /dev/ttyACM0
#   /dev/ttyACM1
```

**设置电机 ID（仅新机械臂首次需要）：**
```bash
lerobot-setup-motors \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0
```

按提示逐个设置 6 个电机的 ID（1-6）。

**校准 Follower：**
```bash
lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_follower
```

**校准 Leader：**
```bash
lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=my_leader
```

**校准过程：**
1. 程序释放所有电机扭矩
2. 手动将手臂移到中间位置，按 Enter
3. 缓慢遍历每个关节的全范围
4. 校准数据保存到 `~/.cache/huggingface/lerobot/calibration/`

### B6. 遥操作验证

```bash
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_follower \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=my_leader \
  --display-cameras 1
```

**验证项：**
- [ ] Leader 手臂移动时 Follower 实时跟随
- [ ] 夹爪开合正常
- [ ] 摄像头画面显示正确
- [ ] 运动流畅无明显抖动

### B7. 获取关节限位 & URDF

**URDF 已包含在仓库中：** `SO101/so101_new_calib.urdf`

**获取工作空间边界：**
```bash
lerobot-find-joint-limits \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_follower \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=my_leader \
  --urdf_path=SO101/so101_new_calib.urdf \
  --target_frame_name=gripper_frame_link \
  --teleop_time_s=30
```

**操作：** 在 30 秒内用 Leader 手臂遍历整个工作空间。记录输出的 `min_ee` 和 `max_ee`。

**重要：** SO-101 的 EE 边界可能偏小（[#1387](https://github.com/huggingface/lerobot/issues/1387)），建议将边界在各方向扩大 ~20%：
```python
# 示例：如果输出 min_ee=[0.05, -0.10, 0.02], max_ee=[0.20, 0.10, 0.25]
# 扩大 20%:
# x 范围 0.15, 扩大 0.03: min_x=0.02, max_x=0.23
# y 范围 0.20, 扩大 0.04: min_y=-0.14, max_y=0.14
# z 范围 0.23, 扩大 0.046: min_z=-0.003 → 0.01(不低于桌面), max_z=0.296
```

同时记录一个安全的复位姿态（手臂抬起在桌面上方）：
```bash
# 遥操作时将手臂移到安全位置，读取关节角度
# 记录为 fixed_reset_joint_positions
```

### B8. 配置文件修改

仓库包含 Linux 专用配置模板：

```
configs/real/
├── env_config.json            # macOS 版环境配置
├── env_config_linux.json      # Linux 版环境配置  ← 使用这个
├── train_config.json          # macOS 版训练配置
└── train_config_cuda.json     # Linux CUDA 版训练配置  ← 使用这个
```

**必须修改的参数（在 `env_config_linux.json` 和 `train_config_cuda.json` 中）：**

| 参数 | 替换为 | 来源 |
|------|--------|------|
| `robot.port` | 你的 Follower 端口 | `lerobot-find-port`，如 `/dev/ttyACM0` |
| `teleop.port` | 你的 Leader 端口 | 如 `/dev/ttyACM1` |
| `robot.id` | 校准时的 ID | 如 `my_follower` |
| `teleop.id` | 校准时的 ID | 如 `my_leader` |
| `cameras.front.index_or_path` | 前方摄像头设备号 | `v4l2-ctl --list-devices` |
| `cameras.wrist.index_or_path` | 腕部摄像头设备号 | 同上 |
| `end_effector_bounds` | 实际 EE 边界 | `lerobot-find-joint-limits` 的输出 |
| `fixed_reset_joint_positions` | 安全复位姿态 | 遥操作时记录 |

### B9. 录制 Demo（25+ episodes）

```bash
cd ~/lerobot
python -m lerobot.rl.gym_manipulator --config_path configs/real/env_config_linux.json
```

**录制流程：**
1. 机器人自动移到复位姿态
2. 按 `Space` 开始遥操作
3. 用 Leader 手臂控制 Follower 抓取方块
4. 抓取成功后按 `s` 标记
5. 失败按 `Esc` 丢弃
6. 重复至少 **25 个成功 episode**

**录制技巧：**
- 从上方接近，缓慢下降
- 夹爪对准方块中心
- 抓稳后直接向上提起
- 保持动作平滑一致
- 每次把方块放回同一位置

### B10. ROI 裁剪

```bash
python -m lerobot.rl.crop_dataset_roi --repo-id LIJianxuanLeo/real_pick_cube
```

用鼠标在每个摄像头画面上框选工作区域：
- **包含：** 方块、夹爪、桌面工作区
- **排除：** 背景墙壁、线缆、其他物品
- 按 `c` 确认，`r` 重选

将输出的裁剪参数更新到 `env_config_linux.json` 和 `train_config_cuda.json`：
```json
"crop_params_dict": {
    "observation.images.front": [top, left, height, width],
    "observation.images.wrist": [top, left, height, width]
}
```

### B11. 训练 Reward Classifier（推荐）

Reward Classifier 是一个视觉分类器，自动判断"方块是否被成功抓起"，用于在训练中自动提供奖励信号。

**参考：** `examples/tutorial/rl/reward_classifier_example.py`

```bash
python -m lerobot.rl.train_reward_classifier \
  --repo-id LIJianxuanLeo/real_pick_cube \
  --output-dir outputs/reward_classifier
```

训练完成后，将路径写入 `train_config_cuda.json`：
```json
"reward_classifier": {
    "pretrained_path": "outputs/reward_classifier/checkpoint/",
    "success_threshold": 0.5,
    "success_reward": 1.0
}
```

准确率应达到 >95%。

**替代方案：** 不训练 Reward Classifier，在训练过程中手动按 `s`/`Esc` 标记成功/失败（更慢但更简单）。

### B12. Actor-Learner 训练（单机 CUDA）

在**同一台 Ubuntu 机器**上（连接机器人 + 有 GPU），打开两个终端：

**终端 1 — Learner（训练进程，使用 CUDA）：**
```bash
conda activate hilserl
cd ~/lerobot
python -m lerobot.rl.learner --config_path configs/real/train_config_cuda.json
```

**终端 2 — Actor（机器人控制进程）：**
```bash
conda activate hilserl
cd ~/lerobot
python -m lerobot.rl.actor --config_path configs/real/train_config_cuda.json
```

**先启动 Learner，再启动 Actor。**

### B13. 人类干预策略

训练启动后，你需要通过 Leader 手臂进行人类干预。这是 HIL-SERL 的核心：

| 阶段 | Episode 数 | 操作 |
|------|-----------|------|
| 自由探索 | 1-5 | **不要干预**，让策略自由探索 |
| 轻度引导 | 5-20 | 用 Leader 手臂做**短暂纠正推力**（1-2 秒） |
| 放手 | 20+ | 减少干预，让策略自主完成大部分任务 |

**操作方式：**

| 按键 | 功能 |
|------|------|
| `Space` | 切换 Leader 控制（接管 / 释放） |
| `s` | 标记 episode 成功 |
| `Esc` | 标记 episode 失败 |

**关键原则：**
- **只做短暂纠正**（1-2 秒），然后松开 Leader
- **不要完成整个任务** — 只是推一下让机器人朝正确方向移动
- 长时间接管会导致 Q 函数过估计，反而损害学习
- 干预率应随训练进行而逐渐下降

### B14. 训练监控与调参

**关键指标：**
- `Episodic reward` — 每个 episode 的总奖励（应逐渐上升）
- `Intervention rate` — 人类干预比例（应逐渐下降）
- `Policy frequency [Hz]` — 策略推理频率（应 ~10 Hz）

**调参速查表：**

| 问题现象 | 参数 | 调整方向 |
|---------|------|---------|
| 机器人几乎不动 | `temperature_init` | 0.01 → 0.1（增加探索） |
| 策略更新太慢 | `policy_parameters_push_frequency` | 4 → 1（秒） |
| 训练不稳定 | `critic_lr` / `actor_lr` | 3e-4 → 1e-4 |
| Learner 跟不上 | `utd_ratio` | 2 → 1 |
| 夹爪反复开合 | `gripper_penalty` | -0.02 → -0.05 |
| CUDA OOM | `batch_size` | 256 → 128 |

**预期结果：**

| 指标 | 目标值 |
|------|--------|
| Demo 录制时间 | ~30 分钟（25 episodes） |
| 训练时间 | 1-3 小时主动监控 |
| 抓取成功率 | ~70%+ |
| 最终干预率 | <10% |

**后台训练（推荐使用 tmux）：**
```bash
tmux new -s learner
# 启动 learner，Ctrl+B D 分离

tmux new -s actor
# 启动 actor，Ctrl+B D 分离

# 重新查看
tmux attach -t learner
tmux attach -t actor
```

### B15. 真实机器人方案快速启动清单

```bash
# ===== 1. 额外安装 =====
pip install -e ".[feetech,hilserl]"
pip install "transformers>=4.45,<5.0"

# ===== 2. USB 权限 =====
sudo usermod -aG dialout $USER
# 重新登录！

# ===== 3. 硬件设置 =====
lerobot-find-port
# 记下 /dev/ttyACM0 (follower) 和 /dev/ttyACM1 (leader)

lerobot-setup-motors --robot.type=so101_follower --robot.port=/dev/ttyACM0
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=my_leader

# ===== 4. 验证遥操作 =====
lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower \
  --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=my_leader \
  --display-cameras 1

# ===== 5. 获取工作空间边界 =====
lerobot-find-joint-limits \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower \
  --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=my_leader \
  --urdf_path=SO101/so101_new_calib.urdf --target_frame_name=gripper_frame_link --teleop_time_s=30
# 记录 EE bounds，更新到配置文件

# ===== 6. 修改配置文件 =====
# 编辑 configs/real/env_config_linux.json 和 configs/real/train_config_cuda.json
# 替换端口、摄像头编号、EE bounds、复位姿态

# ===== 7. 录制 Demo =====
python -m lerobot.rl.gym_manipulator --config_path configs/real/env_config_linux.json

# ===== 8. ROI 裁剪 =====
python -m lerobot.rl.crop_dataset_roi --repo-id LIJianxuanLeo/real_pick_cube

# ===== 9. 训练 =====
# 终端 1
python -m lerobot.rl.learner --config_path configs/real/train_config_cuda.json
# 终端 2
python -m lerobot.rl.actor --config_path configs/real/train_config_cuda.json
```

---

## 附录

### Ubuntu 常见问题排查

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| `nvidia-smi` 报错 | 驱动未安装或内核更新后失效 | `sudo apt install nvidia-driver-535` + 重启 |
| `CUDA available: False` | PyTorch 未安装 CUDA 版 | `pip install torch --index-url https://download.pytorch.org/whl/cu126` |
| MuJoCo 窗口打不开 | 渲染后端未设置 | `export MUJOCO_GL=glfw`（需要显示器） |
| `libEGL` 相关错误 | 缺少 EGL 库 | `sudo apt install libegl1-mesa-dev` |
| SSH 远程无法显示窗口 | X11 转发未启用 | `ssh -X user@host` 或使用 `MUJOCO_GL=egl` |
| USB 权限拒绝 | 未加入 dialout 组 | `sudo usermod -aG dialout $USER` + 重新登录 |
| `/dev/ttyACM*` 不存在 | USB 线松动或驱动问题 | 重新插拔 USB，`dmesg | tail` 查看 |
| 摄像头编号变化 | USB 插拔顺序变了 | 创建 udev 规则（见 B3） |
| `placo` 安装失败 | 缺少 CMake | `sudo apt install cmake` |
| `FFmpeg encoder not found` | 系统 FFmpeg 被使用 | 确认 conda 环境激活：`which ffmpeg` 应指向 conda 路径 |
| CUDA OOM (显存不足) | batch_size 或 buffer 太大 | `"storage_device": "cpu"`, 减小 `batch_size` |
| Actor 连不上 Learner | 端口或防火墙问题 | 检查 50051 端口，`sudo ufw allow 50051/tcp` |
| `transformers` 报错 | 版本不兼容 | `pip install "transformers>=4.45,<5.0"` |

### 关键文件路径参考

| 文件 | 用途 |
|------|------|
| `configs/sim/train_config_cuda.json` | 仿真 CUDA 训练配置 |
| `configs/real/env_config_linux.json` | 真实机器人 Linux 环境配置 |
| `configs/real/train_config_cuda.json` | 真实机器人 CUDA 训练配置 |
| `SO101/so101_new_calib.urdf` | 逆运动学 URDF |
| `src/lerobot/rl/actor.py` | Actor 进程 |
| `src/lerobot/rl/learner.py` | Learner 进程 |
| `src/lerobot/rl/gym_manipulator.py` | 环境创建与处理管线 |
| `examples/tutorial/rl/hilserl_example.py` | Actor-Learner 完整示例 |
| `examples/tutorial/rl/reward_classifier_example.py` | Reward Classifier 训练示例 |

### 参考链接

- HIL-SERL 论文：https://hil-serl.github.io/
- LeRobot HIL-SERL 文档：https://huggingface.co/docs/lerobot/en/hilserl
- LeRobot 仿真训练文档：https://huggingface.co/docs/lerobot/en/hilserl_sim
- SO-101 组装指南：https://huggingface.co/docs/lerobot/en/so101
- SO-101 真实抓取博客：https://ggando.com/blog/so101-hil-serl/
- 社区 SO-101 Fork：https://github.com/ubi-coro/lerobot-hil-serl
