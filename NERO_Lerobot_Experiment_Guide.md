# NERO 机械臂 LeRobot 实验指导书

本文档指导从零开始搭建 NERO 机械臂 + LeRobot + SmolVLA 推理的完整流程。

---

## 目录

1. [硬件准备与连接](#1-硬件准备与连接)
2. [软件环境安装](#2-软件环境安装)
3. [CAN 通信配置](#3-can-通信配置)
4. [相机配置与检测](#4-相机配置与检测)
5. [键盘遥操作](#5-键盘遥操作)
6. [SmolVLA 模型部署与推理](#6-smolvla-模型部署与推理)
7. [常见问题与踩坑记录](#7-常见问题与踩坑记录)

---

## 1. 硬件准备与连接

### 所需硬件

| 设备 | 说明 |
|------|------|
| NERO 机械臂 | 含电源适配器（100-240V） |
| CAN-to-USB 模块 | 含 Type-C 转 USB-A 线缆 |
| USB 相机 x3 | 前视 / 侧视 / 后视 |
| Ubuntu PC | 推荐 22.04，需有 USB3.0 接口 |

### 连接步骤

1. 将 CAN-to-USB 模块连接到 Nero 的 CAN 线（航空插头侧，剥出铜芯）：黄线接 H，蓝线接 L，拧紧端子螺丝
2. 将 CAN-to-USB 模块插入 PC 的 USB 接口
3. 连接 Nero 电源：XT30 接头插到机械臂线缆，接通电源
4. 等待机械臂指示灯变绿
5. 将 3 个 USB 相机插入 PC 的 USB3.0 接口（避免使用 USB 集线器）

### 首次启动：启用 CAN 推送

机械臂首次使用需启用 CAN 主动推送，两种方式任选其一：

**方式 A：Web UI 启用**
- 等待指示灯变绿后，连接 Nero 的 Web 管理界面，启用 CAN Push

**方式 B：代码启用**
```bash
conda activate lerobot
python3 -c "
from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, NeroFW
import time
robot_cfg = create_agx_arm_config(robot=ArmModel.NERO, firmeware_version=NeroFW.DEFAULT, channel='can0')
robot = AgxArmFactory.create_arm(robot_cfg)
robot.connect()
while not robot.enable():
    robot.set_normal_mode()
    time.sleep(0.01)
print('CAN push enabled!')
"
```

验证 CAN 通信是否正常：
```bash
candump can0
# 应看到持续的数据流输出
```

---

## 2. 软件环境安装

### 2.1 安装 Miniconda

```bash
mkdir -p ~/miniconda3
cd ~/miniconda3
wget https://repo.anaconda.com/miniconda3/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all
```

重新打开终端后继续。

### 2.2 创建 conda 环境

```bash
conda create -y -n lerobot python=3.12
conda activate lerobot
```

### 2.3 克隆 LeRobot 仓库

```bash
cd ~/projects
git clone https://github.com/huggingface/lerobot.git
cd lerobot
```

### 2.4 安装 LeRobot 及依赖

```bash
pip install -e ".[dataset,smolvla]"
```

### 2.5 安装 pyAgxArm SDK

```bash
cd ~/projects
git clone https://github.com/agilexrobotics/pyAgxArm.git
cd pyAgxArm
pip install -e .
```

### 2.6 安装 ffmpeg

```bash
conda install ffmpeg=7.1.1 -c conda-forge
```

### 2.7 安装 CAN 工具

```bash
sudo apt install can-utils
```

### 2.8 验证安装

```bash
python3 -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python3 -c "import lerobot; print('LeRobot:', lerobot.__file__)"
python3 -c "from pyAgxArm import AgxArmFactory; print('pyAgxArm: OK')"
python3 -c "from lerobot.robots.nero_follower.config_nero_follower import NEOFollowerRobotConfig; print('NERO: OK')"
```

---

## 3. CAN 通信配置

**每次重启电脑后需要重新拉起 CAN 接口：**

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
```

验证：
```bash
ip link show can0
# 应看到 state UP
```

> 提示：可将上述命令写入开机脚本自动执行。

---

## 4. 相机配置与检测

### 4.1 查找相机索引

```bash
lerobot-find-cameras opencv
```

记下每个相机的索引号（如 `/dev/video6` 索引为 6）。

### 4.2 手动检查相机

```bash
# 列出所有视频设备
ls -l /dev/video*

# 查看某相机支持的格式和分辨率
v4l2-ctl -d /dev/video6 --list-formats-ext

# 测试读取
python3 -c "
import cv2
cap = cv2.VideoCapture('/dev/video6')
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
ret, frame = cap.read()
print('OK' if ret else 'FAILED', frame.shape if ret else '')
cap.release()
"
```

### 4.3 相机参数选择

| 格式 | 分辨率 | 帧率 | 说明 |
|------|--------|------|------|
| MJPG | 640x480 | 30fps | **推荐**，压缩格式，USB 带宽占用低 |
| MJPG | 1920x1080 | 30fps | 高分辨率，需 USB3.0 |
| YUYV | 640x480 | 30fps | 未压缩，多相机时带宽可能不足 |
| YUYV | 1280x720 | 10fps | 仅 10fps，不推荐 |

**建议统一使用 `fourcc: "MJPG"` + `640x480` + `30fps`。**

---

## 5. 键盘遥操作

**必须在本机终端运行**（SSH / 桌面终端均可），不能通过脚本或管道执行。

### 5.1 无相机遥操作

```bash
conda activate lerobot

# 确保 CAN 接口已拉起
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up

lerobot-teleoperate \
    --robot.type=nero_follower \
    --robot.channel=can0 \
    --robot.interface=socketcan \
    --robot.firmeware_version=default \
    --robot.speed_percent=10 \
    --robot.disable_torque_on_disconnect=False \
    --teleop.type=keyboard_joint \
    --teleop.joint_step=0.05 \
    --teleop.gripper_step=2.0 \
    --fps=30 \
    --display_data=true
```

### 5.2 带相机遥操作

将 `index_or_path` 替换为你的实际相机索引：

```bash
conda activate lerobot

lerobot-teleoperate \
    --robot.type=nero_follower \
    --robot.channel=can0 \
    --robot.interface=socketcan \
    --robot.firmeware_version=default \
    --robot.speed_percent=10 \
    --robot.disable_torque_on_disconnect=False \
    --robot.cameras='{ front: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, back: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30, fourcc: "MJPG"} }' \
    --teleop.type=keyboard_joint \
    --teleop.joint_step=0.05 \
    --teleop.gripper_step=2.0 \
    --fps=30 \
    --display_data=true
```

### 5.3 按键映射

| 按键 | 功能 | 按键 | 功能 |
|------|------|------|------|
| Q | joint1 + | A | joint1 - |
| W | joint2 + | S | joint2 - |
| E | joint3 + | D | joint3 - |
| R | joint4 + | F | joint4 - |
| T | joint5 + | G | joint5 - |
| Y | joint6 + | H | joint6 - |
| U | joint7 + | J | joint7 - |
| 1 | 夹爪开 | 2 | 夹爪合 |
| **ESC** | **退出** | | |

### 5.4 注意事项

- 机械臂断开时**不会下电**（`disable_torque_on_disconnect=False`），防止不自锁掉落
- 速度默认 10%，如需调整改 `--robot.speed_percent`
- 关节增量 0.05 rad/步，夹爪增量 2.0/步

---

## 6. SmolVLA 模型部署与推理

### 6.1 登录 HuggingFace

```bash
huggingface-cli login
```

Token 从 https://huggingface.co/settings/tokens 获取（Read 权限即可）。

### 6.2 下载 SmolVLA 预训练模型

网络不好时使用镜像站：

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

### 6.3 运行零样本推理

SmolVLA 模型期望 3 个相机（camera1/camera2/camera3），NERO 的 3 个相机（front/side/back）需要用 `--dataset.rename_map` 映射。`--dataset.single_task` 是传给模型的语言指令，直接影响模型输出动作。

将 `index_or_path` 替换为你的实际相机索引：

```bash
conda activate lerobot

HF_HUB_OFFLINE=1 lerobot-record \
    --robot.type=nero_follower \
    --robot.channel=can0 \
    --robot.interface=socketcan \
    --robot.firmeware_version=default \
    --robot.speed_percent=10 \
    --robot.disable_torque_on_disconnect=False \
    --robot.cameras='{ front: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, back: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30, fourcc: "MJPG"} }' \
    --policy.path=$HOME/models/smolvla_base \
    --dataset.rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.side": "observation.images.camera2", "observation.images.back": "observation.images.camera3"}' \
    --dataset.repo_id=$USER/eval_nero_smolvla_test \
    --dataset.num_episodes=1 \
    --dataset.single_task="pick up the green cube" \
    --dataset.push_to_hub=false \
    --display_data=false
```

参数说明：

| 参数 | 说明 |
|------|------|
| `--policy.path` | 模型本地路径 |
| `--dataset.rename_map` | 相机名称映射：NERO 名称 → 模型期望名称 |
| `--dataset.repo_id` | 数据集名称，**必须以 `eval_` 开头** |
| `--dataset.single_task` | 语言指令，描述要执行的任务 |
| `--dataset.push_to_hub` | 是否上传到 HF Hub |
| `HF_HUB_OFFLINE=1` | 离线模式，避免网络超时 |

> **重要提示**：无微调的 `smolvla_base` 是通用基础模型，在 NERO Follower 上效果可能很差。需要在自己的数据集上微调后才能获得好的表现。

---

## 7. 常见问题与踩坑记录

### Q1: 相机读取帧失败（连续读取失败超限）

**报错**：
```
RuntimeError: OpenCVCamera(/dev/video6) exceeded maximum consecutive read failures.
```

**原因**：YUYV 格式在高分辨率下帧率不足（如 1280x720 仅 10fps），请求 30fps 会失败。

**解决**：使用 MJPG 压缩格式，配置中加 `fourcc: "MJPG"`。

排查命令：
```bash
v4l2-ctl -d /dev/video6 --list-formats-ext   # 查看支持的格式和帧率
fuser /dev/video6                              # 检查是否被占用
```

### Q2: `nero_follower` 未注册（invalid choice）

**报错**：
```
lerobot-record: error: argument --robot.type: invalid choice: 'nero_follower'
```

**原因**：`nero_follower` 模块未在脚本中 import，导致 draccus 无法识别。

**解决**：在 `src/lerobot/scripts/lerobot_record.py` 的 import 中添加：
```python
from lerobot.robots.nero_follower.config_nero_follower import NEOFollowerRobotConfig  # noqa: F401
```

### Q3: 数据集名称不以 `eval_` 开头

**报错**：
```
ValueError: Your dataset name does not begin with 'eval_' (nero_smolvla_test), but a policy is provided.
```

**解决**：使用策略推理时，`--dataset.repo_id` 必须以 `eval_` 开头，如 `yuhang/eval_nero_smolvla_test`。

### Q4: HuggingFace 下载模型超时

**报错**：
```
httpx.ConnectTimeout: [Errno 110] Connection timed out
```

**解决**：
- 使用镜像站：`HF_ENDPOINT=https://hf-mirror.com`
- 或手动下载模型到本地，使用 `--policy.path=/path/to/local/model`
- 推理时加 `HF_HUB_OFFLINE=1` 避免联网

### Q5: 相机名称不匹配（feature names inconsistent）

**报错**：
```
Please ensure your dataset and policy use consistent feature names.
```

**原因**：SmolVLA 模型期望 `camera1/camera2/camera3`，但 NERO 的相机叫 `front/side/back`。

**解决**：使用 `--dataset.rename_map` 映射：
```bash
--dataset.rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.side": "observation.images.camera2", "observation.images.back": "observation.images.camera3"}'
```

如果相机数量不足 3 个，可加 `--policy.empty_cameras=1` 补空相机。

### Q6: CAN 接口未拉起

**现象**：机械臂无响应，`candump can0` 无输出。

**解决**：每次重启后需重新拉起 CAN：
```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
```

### Q7: arboard 剪贴板警告

**报错**：
```
[WARN arboard::platform::linux::x11] Could not hand the clipboard contents over to the clipboard manager.
```

**解决**：不影响功能，可安全忽略。

### Q8: torch_dtype deprecated 警告

**报错**：
```
`torch_dtype` is deprecated! Use `dtype` instead!
```

**解决**：仅为警告，不影响推理。
