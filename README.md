# NexArm 6-DOF 机械臂 — LeRobot 完整使用指南

NexArm 是一款基于 ESP32 + AT32F421 协处理器的高性能 6 自由度桌面机械臂，采用 HX-30HM 高性能串行总线舵机，具备 12 位精度与 1 Mbps 高速通信，响应迅速、扭矩强劲。本文档涵盖从硬件连接、遥操作测试、数据采集、模型训练到策略推理的完整流程。

---

## 目录

- [硬件概述](#硬件概述)
- [环境安装](#环境安装)
- [第一步：查找串口](#第一步查找串口)
- [第二步：查找摄像头](#第二步查找摄像头)
- [第三步：遥操作 / 标定](#第三步遥操作--标定)
- [第四步：采集数据集](#第四步采集数据集)
- [第五步：训练策略模型](#第五步训练策略模型)
- [第六步：策略推理部署](#第六步策略推理部署)
- [代码架构说明](#代码架构说明)
- [PR 提交指南](#pr-提交指南)
- [常见问题排查](#常见问题排查)

---

## 硬件概述

### 硬件组成

| 组件 | 说明 |
|------|------|
| **主臂 (Leader)** | ESP32 开发板，直接驱动 6 个 HX-30HM 舵机。用于遥操作，操作者自由拖动此臂 |
| **从臂 (Follower)** | ESP32 开发板 + AT32F421 协处理器，驱动 6 个 HX-30HM 舵机。镜像主臂动作或执行策略输出 |
| **舵机** | HX-30HM 高性能串行总线舵机，12 位精度 (0–4095)，1 Mbps 波特率，高扭矩、快响应 |
| **摄像头** | 2 个 USB 摄像头 — "front"（工作区俯视）和 "wrist"（末端执行器近景），640×480 @ 30 FPS |

### 关节布局 (6 DOF)

| 关节编号 | 名称 | 说明 |
|----------|------|------|
| 1 | `shoulder_pan` | 底座旋转 |
| 2 | `shoulder_lift` | 大臂俯仰（leader 与 follower 装配方向相反，由校准的 drive_mode/homing_offset 自动对齐） |
| 3 | `elbow_flex` | 肘关节 |
| 4 | `wrist_flex` | 腕部俯仰 |
| 5 | `wrist_roll` | 腕部旋转 |
| 6 | `gripper` | 夹爪开合（leader 与 follower 物理行程不同，由校准 range_min/range_max 自动对齐） |

### 通信协议

NexArm 使用自定义 CommProtocol 通过 USB 串口通信：

```
帧格式: [0xFF][0xFF][ID][LEN][CMD][ARGS...][CHECKSUM]
```

| 命令 | 功能 | 方向 |
|------|------|------|
| CMD 68 | 进入/退出 LeRobot 桥接模式（仅从臂） | 主机 → 从臂 |
| CMD 96 | 读取 6 个舵机位置（回复：12 字节） | 主机 → 设备 → 主机 |
| CMD 97 | 写入 6 个舵机位置（12 字节，无回复） | 主机 → 设备 |
| CMD 98 | 使能/失能力矩 | 主机 → 设备 |

---

## 环境安装

### 1. 创建环境并安装

推荐使用 conda 创建独立环境：

```bash
git clone https://github.com/liangfuyuan581-creator/lerobot-nexarm.git
cd lerobot-nexarm

conda create -n nexarm python=3.12 -y
conda activate nexarm

# 安装 lerobot（editable 模式，包含所有必要依赖）
pip install -e ".[nexarm]"
```

如果需要遥操作时实时可视化（rerun 界面），额外安装：

```bash
pip install -e ".[nexarm,viz]"
```

> **注意：** 不安装 `viz` 也可以正常使用遥操作、采集、训练和推理，只是没有实时画面显示。配置文件中已默认关闭 `display_data`。

也可以使用 venv 代替 conda：

```bash
git clone https://github.com/liangfuyuan581-creator/lerobot-nexarm.git
cd lerobot-nexarm

python -m venv .venv

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -e ".[nexarm]"
```

### 2. 验证安装

```bash
python -c "import lerobot.rollout; print('OK')"
```

### 3. 连接硬件

- 将 **从臂 (Follower/Slave)** 的 ESP32 通过 USB 连接到电脑
- 将 **主臂 (Leader/Master)** 的 ESP32 通过 USB 连接到电脑
- 插入两个 USB 摄像头（front + wrist）

### 平台兼容性

| 平台 | 状态 | 说明 |
|------|------|------|
| Windows 10/11 | 已验证 | CH340 驱动需安装，串口格式 COM19 |
| Ubuntu 20.04+ | 已验证 | 串口格式 /dev/ttyUSB0，需 dialout 权限 |
| macOS | 应兼容 | 串口格式 /dev/tty.usbserial-xxx |

---

## 第一步：查找串口

确定主臂和从臂各自对应哪个串口。

```bash
python -m lerobot.scripts.lerobot_find_port
```

Windows 上的典型输出：

| 端口 | 设备 |
|------|------|
| COM18 | 主臂 (Leader/Master) ESP32 |
| COM19 | 从臂 (Follower/Slave) ESP32 |

Linux 上通常是 `/dev/ttyUSB0` 和 `/dev/ttyUSB1`。

> **提示：** 如果无法确定哪个是主臂哪个是从臂，可以只插一个试试，或查看设备管理器中的 USB 设备描述。

---

## 第二步：查找摄像头

确定哪个摄像头索引对应"前方"和"腕部"。

```bash
python -m lerobot.scripts.lerobot_find_cameras opencv
```

或者手动扫描并拍照确认：

```python
import cv2

for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"cam_{i}.png", frame)
            print(f"摄像头 {i}: 可用")
        cap.release()
```

查看保存的图片，确定：
- **front**（前方摄像头）：拍到整个工作区的俯视画面
- **wrist**（腕部摄像头）：拍到末端执行器/夹爪的近景

> **注意：** 摄像头索引会随 USB 插拔顺序变化。每次重新插拔后建议重新扫描确认。

---

## 第三步：遥操作 / 标定

验证主臂-从臂的遥操作链路是否正常。主臂力矩关闭，操作者自由拖动主臂，从臂实时镜像跟随。

> **关于标定：** NexArm 沿用 lerobot 官方标定流程（与 SO-100/SO-101 一致），主臂和从臂各跑一次：先把臂摆到机械中点，再依次拨动每个关节走完整行程。标定会写入 `~/.cache/huggingface/lerobot/calibration/{robots,teleoperators}/{nexarm_follower,nexarm_leader}/{id}.json`。第一次 `lerobot_teleoperate` 启动时会自动进入这个流程；之后会直接读取缓存。
>
> 标定记录的是每个关节真实的 raw min/max + homing_offset，主从臂之间的镜像/重映射不再写死在代码里，而是由 calibration 自动吸收（drive_mode + offset + range）。这意味着：换舵机、换臂、装配公差变化，**只需要重跑一次标定**，不用改任何代码。

### 方式一：使用配置文件

先根据你的实际情况修改 `examples/nexarm/teleoperate.yaml` 中的串口和摄像头索引，然后运行：

```bash
python -m lerobot.scripts.lerobot_teleoperate --config_path=examples/nexarm/teleoperate.yaml
```

### 方式二：使用命令行参数

```bash
python -m lerobot.scripts.lerobot_teleoperate --robot.type=nexarm_follower --robot.port=COM19 --teleop.type=nexarm_leader --teleop.port=COM18 --fps=30
```

### 标定流程要点

每个关节走完整行程时尽量缓慢、覆盖到机械极限附近（但不要硬撞）。

- 错把 leader 当 follower 标定 → 数值范围还在但方向会反，可以删掉 `~/.cache/.../<arm>/main.json` 重新标
- 标定中途想中止 → Ctrl+C，已有的缓存不受影响
- 多套臂共存 → 在 yaml 里加 `robot.id: arm_a` / `arm_b`，校准文件会按 id 分别落盘

### 验证要点

- 拖动主臂时从臂应该实时跟随
- 各个关节方向一致（标定的 drive_mode 自动处理装配反向）
- 夹爪开合应该自然映射，主臂全行程 → 从臂全行程

> **提示：** 如果遥操作感觉卡顿，检查是否有其他程序（如 Arduino IDE 串口监视器）占用了 COM 端口。

---

## 第四步：采集数据集

通过遥操作录制示范数据，用于后续模仿学习训练。操作者拖动主臂执行任务，从臂跟随执行，同时录制摄像头画面和关节位置。

### 运行采集

先修改 `examples/nexarm/record.yaml` 中的配置（串口、摄像头、数据集名称等），然后运行：

```bash
python -m lerobot.scripts.lerobot_record --config_path=examples/nexarm/record.yaml
```

### 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `dataset.repo_id` | `local/nexarm_pick` | 数据集名称（本地保存） |
| `dataset.num_episodes` | 50 | 录制的 episode 数量 |
| `dataset.episode_time_s` | 10 | 每个 episode 的时长（秒） |
| `dataset.reset_time_s` | 10 | episode 之间用于重置场景的等待时间 |
| `dataset.fps` | 30 | 录制帧率 |
| `dataset.push_to_hub` | false | 是否上传到 HuggingFace Hub |

### 采集流程

1. 脚本启动后自动连接双臂和摄像头
2. 每个 episode：
   - 按 Enter 开始录制
   - 拖动主臂执行任务（如抓取物体）
   - 到达 `episode_time_s` 后自动结束录制
   - 在 `reset_time_s` 内将物体重置到初始位置
3. 全部 episode 录完后，数据集保存到本地并可选上传到 HuggingFace Hub

### 仅本地保存（不上传）

```bash
python -m lerobot.scripts.lerobot_record --config_path=examples/nexarm/record.yaml --dataset.repo_id=local/nexarm_pick --dataset.push_to_hub=false
```

### 数据质量建议

- **至少录制 50 个 episode** 才能获得可用的训练效果
- 每次录制保持动作一致性（相同的起始位置、相似的运动轨迹）
- 确保摄像头画面清晰、光照稳定
- 录制完成后可用 `lerobot_dataset_viz` 查看数据集质量

---

## 第五步：训练策略模型

使用 ACT（Action Chunking with Transformers）策略在采集的数据集上训练。

### 安装训练依赖

训练需要额外的依赖（accelerate、wandb 等），安装方式：

```bash
pip install -e ".[nexarm,training]"
```

### 运行训练

```bash
python -m lerobot.scripts.lerobot_train --dataset.repo_id=local/nexarm_pick --policy.type=act --output_dir=outputs/train/nexarm_act --batch_size=32 --steps=100000 --save_freq=25000
```

> **注意：** 不建议使用 `--config_path=examples/nexarm/train.yaml` 加 `--dataset.repo_id` 覆盖的方式，可能会报 JSON 格式错误。直接使用命令行参数即可。

### 训练建议

| 项目 | 建议值 | 说明 |
|------|--------|------|
| 训练设备 | CUDA GPU | 推荐 RTX 3090 或以上 |
| 示范 episode 数 | 50+ | 越多越好，至少 50 个 |
| 训练步数 | 100,000 | 根据 loss 收敛情况调整 |
| 批量大小 | 32 | 默认值，显存不足可降到 16 |
| 保存频率 | 25,000 | 每 25k 步保存一个检查点 |

> **关于 episode 时长：** 每个 episode 可以是 10-30 秒。ACT 训练时从 episode 中随机采样起始帧，取接下来 chunk_size=100 帧（3.3秒）作为预测目标。episode 长度不影响 chunk_size 设置。关键是每个 episode 内包含完整的任务动作（如完整的一次抓取-放置）。

### 训练输出

训练完成后，模型检查点保存在：

```
outputs/train/nexarm_act/checkpoints/last/pretrained_model/
├── config.json                    # 策略配置
├── model.safetensors              # 模型权重
├── policy_preprocessor.json       # 输入预处理流水线
├── policy_preprocessor_step_3_normalizer_processor.safetensors
├── policy_postprocessor.json      # 输出后处理流水线
├── policy_postprocessor_step_0_unnormalizer_processor.safetensors
└── train_config.json              # 训练配置记录
```

---

## 第六步：策略推理部署

将训练好的策略部署到真实机器人上。从臂自主执行模型预测的动作，无需主臂。

### 运行推理

```bash
python -m lerobot.scripts.lerobot_rollout --config_path=examples/nexarm/inference.yaml --policy.path=outputs/train/nexarm_act/checkpoints/last/pretrained_model
```

### 重要说明

- **不需要传 `--teleop` 参数** — 策略替代人类操作者
- 数据集名称必须以 `rollout_` 开头（配置文件中已设置为 `rollout_nexarm_pick`）
- 每次推理会自动在 repo_id 后追加时间戳，生成唯一目录，不会冲突
- 推理结果会录制成数据集，方便后续分析评估
- 配置文件中 `device: cuda`，有 GPU 时使用 GPU 推理；无 GPU 时系统自动退回 CPU
- 推理结束后机械臂会回到初始位置并保持力矩（不掉电）

### 推理性能参考

| 设备 | 大约帧率 | 说明 |
|------|----------|------|
| NVIDIA GPU | 30 Hz | 满速实时控制 |
| CPU (i7/i5) | 20–30 Hz | ACT 策略利用 action chunking，模型每 100 帧推理一次，其余帧从队列弹出，CPU 也能实时 |

> **ACT 策略在 CPU 上可以正常实时运行。** chunk_size=100 意味着模型推理（~700ms）只在队列为空时触发，其余 99 帧只需 ~1ms 弹出动作，整体帧率可达 20-30 Hz。

### 不同策略模式

| 策略 | 命令 | 说明 |
|------|------|------|
| `base` | `--strategy.type=base` | 仅推理，不录制 |
| `sentry` | `--strategy.type=sentry` | 连续录制 + 自动保存（推荐用于评估） |
| `highlight` | `--strategy.type=highlight` | 环形缓冲区，按键保存精彩片段 |
| `dagger` | `--strategy.type=dagger` | 人机协作，需要主臂 |

---

## 代码架构说明

### 文件结构

```
src/lerobot/
├── motors/nexarm/                          # 串口驱动层
│   ├── __init__.py                         # 导出 NexArmMotorsBus
│   └── nexarm.py                           # CommProtocol 帧构建/解析、
│                                           #   位置读写、力矩控制、桥接模式
├── robots/nexarm_follower/                 # 从臂机器人
│   ├── __init__.py                         # 导出 NexArmFollower, NexArmFollowerConfig
│   ├── config_nexarm_follower.py           # RobotConfig 子类（端口、摄像头、波特率）
│   └── nexarm_follower.py                  # Robot 子类（连接、观测、动作执行）
└── teleoperators/nexarm_leader/            # 主臂遥操作
    ├── __init__.py                         # 导出 NexArmLeader, NexArmLeaderConfig
    ├── config_nexarm_leader.py             # TeleoperatorConfig 子类（端口、波特率）
    └── nexarm_leader.py                    # Teleoperator 子类（读取动作、主从映射）
```

### 修改的已有文件

| 文件 | 修改内容 |
|------|----------|
| `src/lerobot/robots/utils.py` | 在 `make_robot_from_config()` 中添加 `nexarm_follower` 分支 |
| `src/lerobot/teleoperators/utils.py` | 在 `make_teleoperator_from_config()` 中添加 `nexarm_leader` 分支 |
| `pyproject.toml` | 在 `[project.optional-dependencies]` 中添加 `nexarm = ["lerobot[pyserial-dep]", "lerobot[av-dep]"]` |

### 数据流向

```
遥操作模式:
  主臂 → sync_read("Present_Position", normalize=True)
        ↓ leader calibration: raw → DEGREES / RANGE_0_100
        ↓ follower calibration: DEGREES / RANGE_0_100 → raw
  从臂 → sync_write("Goal_Position", normalize=True) → write_positions()

策略推理模式:
  摄像头 + sync_read("Present_Position") → ACT 模型推理 → 预测动作
        ↓ follower calibration: DEGREES / RANGE_0_100 → raw (clamped to range_min/range_max)
  从臂 → sync_write("Goal_Position", normalize=True)
```

> **不再需要手写镜像/重映射函数。** 每个关节的方向、零点、机械限位都由标定的 `MotorCalibration(drive_mode, homing_offset, range_min, range_max)` 决定，校准后 leader 输出与 follower 输入都在同一个归一化空间。

---

## PR 提交指南

### 步骤 1：Fork 并克隆

```bash
# Fork huggingface/lerobot 到你自己的 GitHub 账号
git clone https://github.com/你的用户名/lerobot.git
cd lerobot
git remote add upstream https://github.com/huggingface/lerobot.git
git fetch upstream
git checkout -b add-nexarm-support upstream/main
```

### 步骤 2：复制 NexArm 文件

将本仓库中的 NexArm 相关文件复制到你 fork 的仓库中：

```bash
# 克隆本仓库
git clone https://github.com/liangfuyuan581-creator/lerobot-nexarm.git

# 从本仓库复制 NexArm 相关文件到你 fork 的 lerobot
cp -r lerobot-nexarm/src/lerobot/motors/nexarm           lerobot/src/lerobot/motors/
cp -r lerobot-nexarm/src/lerobot/robots/nexarm_follower  lerobot/src/lerobot/robots/
cp -r lerobot-nexarm/src/lerobot/teleoperators/nexarm_leader  lerobot/src/lerobot/teleoperators/
cp -r lerobot-nexarm/examples/nexarm                     lerobot/examples/
cp -r lerobot-nexarm/tests/motors/test_nexarm.py         lerobot/tests/motors/
cp -r lerobot-nexarm/tests/robots/test_nexarm_follower.py  lerobot/tests/robots/
cp -r lerobot-nexarm/tests/teleoperators/test_nexarm_leader.py  lerobot/tests/teleoperators/
```

### 步骤 3：修改已有文件

参考本仓库中已修改的文件，对你 fork 的仓库做同样的修改：

1. `src/lerobot/robots/utils.py` — 添加 `nexarm_follower` 分支
2. `src/lerobot/teleoperators/utils.py` — 添加 `nexarm_leader` 分支
3. `pyproject.toml` — 添加 `nexarm` 依赖项

### 步骤 4：安装并测试

```bash
pip install -e ".[nexarm,dev]"

# 代码风格检查
pre-commit install
pre-commit run --all-files

# 运行测试
pytest tests/ -v
```

### 步骤 5：提交 PR

```bash
git add -A
git commit -m "Add NexArm 6-DOF robot arm support (follower + leader)"
git push origin add-nexarm-support
```

在 GitHub 上创建 Pull Request，标题和描述建议：

- **标题：** `Add NexArm 6-DOF robot arm support`
- **描述：** 包含硬件简介、测试结果截图（遥操作、训练曲线、推理视频链接）

---

## 常见问题排查

### 串口权限不足 (Linux)

```bash
sudo usermod -a -G dialout $USER
# 需要注销并重新登录
```

### 找不到摄像头

- 运行 `lerobot_find_cameras` 扫描可用索引
- Windows 上关闭其他占用摄像头的程序（OBS、浏览器等）
- 重新插拔 USB 后摄像头索引可能会变

### 遥操作时从臂不动

1. 检查从臂的 COM 端口是否正确
2. 确认从臂 ESP32 固件支持 CMD 68（LeRobot 桥接模式）
3. 尝试重新给从臂上电

### 采集数据时主臂报 TimeoutError

```
TimeoutError: No position reply from NexArm
```

**原因 1：** 旧版主臂固件在 LeRobot 模式下仍会通过 Serial 打印 `[WARN]`、`[POS]` 调试信息，这些文本会污染协议帧；同时 LeRobot 模式下还会跑 ESPNow 读舵机循环，与 CMD 96 抢占 Serial1。

**解决方案：** 本仓库已修复主臂固件 `Nex_Arm_sys/Nex_Arm.ino`，在 LeRobot 模式下直接 `return`，让出 Serial1 和 USB Serial。重新烧录主臂固件即可。

**原因 2：** 旧版从臂固件 `Nex_Arm/system_task_handle.cpp` 使用 `volatile bool servo_uart_busy` 自旋等待 AT32 总线，两个核心同时进入会形成竞态，AT32 返回的帧错乱被 PC 端解析失败。

**解决方案：** 本仓库已把锁换成 FreeRTOS mutex（`servo_uart_lock()` / `servo_uart_unlock()`）。重新烧录从臂固件即可。

修复后单次读 < 25ms，30 Hz 帧率有充足余量。

### 训练 loss 不下降

- 确保有足够的示范 episode（建议 50+ 个）
- 检查数据集中的摄像头画面是否正常（非黑屏/模糊）
- 尝试提高学习率（如 `--policy.optimizer_lr=1e-4`）

### 推理时机械臂犹犹豫豫/动作幅度很小

这通常是模型"坍缩到均值"——输出的动作接近训练数据的平均值，没有学到完整的时序轨迹。排查和调整方法：

**1. 检查训练数据质量**

```bash
# 查看数据集统计，确认 observation.state 和 action 的值范围接近
python -c "
from safetensors.torch import load_file
stats = load_file('outputs/train/nexarm_act/checkpoints/last/pretrained_model/policy_preprocessor_step_3_normalizer_processor.safetensors')
print('state mean:', stats['observation.state.mean'].tolist())
print('action mean:', stats['action.mean'].tolist())
# 两者应该很接近（差距 < 100），否则说明采集时位置反馈有问题
"
```

**2. 增大 batch_size**

batch_size 越大，VAE 的 latent space 学得越稳定，模型越不容易坍缩。

```bash
--batch_size=32    # 默认值，显存不足可用 16
--batch_size=64    # 如果显存够（24GB+），效果更好
```

**3. 调整 kl_weight**

kl_weight 控制 VAE 正则化强度。太大会让模型输出过于保守（接近均值），太小会让动作不连贯。

```bash
--policy.kl_weight=10.0   # 默认值
--policy.kl_weight=5.0    # 如果动作太保守，尝试降低
--policy.kl_weight=1.0    # 进一步降低，让模型更大胆
```

**4. 提高学习率**

默认 1e-5 比较保守，可以适当提高：

```bash
--policy.optimizer_lr=5e-5     # 适度提高
--policy.optimizer_lr=1e-4     # 更激进，注意观察 loss 是否震荡
```

**5. 增加训练步数**

如果 loss 还在下降但还没收敛：

```bash
--steps=200000    # 从 100k 增加到 200k
```

**6. 增加训练数据**

50 个 episode 是下限。如果任务复杂（多步骤、多物体），建议：

```bash
--dataset.num_episodes=100    # 或更多
```

**7. 确保采集数据动作一致**

- 每个 episode 都应包含完整的任务动作（伸手→抓取→抬起→放置）
- 起始位置尽量一致
- 避免 episode 中有大量空闲/等待时间
- 动作速度保持均匀，不要忽快忽慢

### CPU 推理帧率低于预期

ACT 策略使用 action chunking（chunk_size=100），正常情况下 CPU 也能达到 20-30 Hz。如果帧率异常低：

- 确认 `inference.yaml` 中设置了 `device: cpu`
- 检查是否有其他程序占用 CPU（如后台编码、杀毒软件）
- 双摄像头采集本身需要 ~45ms，这是正常开销

### 串口通信校验和错误

主臂固件存在一个已知的校验和 bug（`tx_packet_complete` 使用了 `rx_packet.elements.length` 而非 `tx_packet.elements.length`）。驱动程序同时接受正确和错误的校验和以保持兼容性。
