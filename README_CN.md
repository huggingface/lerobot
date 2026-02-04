# AgileX Piper 机械臂 LeRobot 扩展框架

[Hugging Face 文档](http://huggingface.co/docs/lerobot)

## 1. 环境创建

```bash
uv venv
uv sync
```

## 2. 测试相机

使用 `lerobot-find-cameras` 来识别设备节点。

注意：两个相机不能连接到电脑的同一个 USB 集线器（Hub），否则可能会读取失败。

```bash
sudo apt install guvcview    # 安装 Guvcview
guvcview --device=/dev/video0  # 测试 wrist 相机
guvcview --device=/dev/video2  # 测试 ground 相机
```

## 3. 连接机械臂

注意："3-7.1:1.0" 应根据输出结果更改为您实际的 CAN 端口号。

```bash
conda activate lerobot
bash find_all_can_port.sh
bash can_activate.sh can_master 1000000 "1-8.2:1.0"
bash can_activate.sh can_follower 1000000 "1-8.3:1.0"
```

## 3.5 硬件级遥操作设置

我们的实验装置由 **四台 Piper 机械臂** 组成，分为两组（左臂组和右臂组），每组包含一台 Leader（主臂）和一台 Follower（从臂）。

- **硬件连接**: 每一对 Leader 和 Follower 机械臂通过 CAN 总线连接。
- **遥操作原理**: 利用 Piper SDK 提供的硬件级遥操作功能，配置为主/从（Master/Slave）模式后，Leader 和 Follower 可以直接通信进行控制，无需 PC 参与计算。
- **数据采集**: Follower 机械臂通过 USB 连接到 PC。在数据录制期间，PC 读取 Follower (Slave) 的状态数据，同时操作员手动操纵 Leader 进行遥操作。
- **推理与回放**: 进行模型推理或回放时，我们 **切断 Leader 机械臂的电源**。此时 PC 直接通过 USB 发送控制信号给 Follower 机械臂以执行动作。

有关 SDK 的更多详细信息，请参阅官方文档和 API 函数。

## 3.6 软件级遥操作设置 (4 个 CAN 端口)

如果您不想使用 SDK 的硬件主从模式，可以使用 `piper_dual_teleop` 插件进行 **软件级遥操作**。这需要 4 个独立的 CAN 端口：

```bash
# 激活 4 个 CAN 端口
bash can_activate.sh can_left_leader 1000000 "<usb_port1>"
bash can_activate.sh can_left_follower 1000000 "<usb_port2>"
bash can_activate.sh can_right_leader 1000000 "<usb_port3>"
bash can_activate.sh can_right_follower 1000000 "<usb_port4>"
```

**软件遥操作流程**:

- **数据采集期间**: 软件读取 Leader 关节位置 → 写入 Follower 关节
- **推理/回放期间**: 设置 `use_teleop=false`，只需要 2 个 Follower CAN 端口

## 4. 遥操作

```bash
lerobot-teleoperate \
    --robot.type=piper_follower \
    --robot.id=my_follower_arm \
    --teleop.type=piper_leader \
    --teleop.id=my_leader_arm \
    --display_data=true
```

## 5. 登录 Hugging Face (可选)

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

通过运行以下命令将您的 token 添加到 CLI：

```bash
hf auth login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
```

验证登录

```bash
HF_USER=$(hf auth whoami | head -n 1)
echo $HF_USER
```

上传数据集到 Hugging Face

```bash
hf upload jokeru/pick_and_place ~/.cache/huggingface/lerobot/jokeru/pick_and_place \
  --repo-type dataset \
  --revision "v3.0"
```

## 6. 数据采集

记录从 Leader 机械臂到 Follower 机械臂的遥操作数据。

```bash
uv run lerobot-record \
  --robot.type=piper_dual \
  --robot.left_port=can_left \
  --robot.right_port=can_right \
  --robot.read_only=true \
  --robot.cameras='{
      "left": {
        "type": "opencv",
        "index_or_path": "/dev/video12",
        "width": 640,
        "height": 480,
        "fps": 30,
        "rotation": 0
      },
      "right": {
        "type": "opencv",
        "index_or_path": "/dev/video4",
        "width": 640,
        "height": 480,
        "fps": 30,
        "rotation": 0
      },
      "middle": {
        "type": "opencv",
        "index_or_path": "/dev/video6",
        "width": 640,
        "height": 480,
        "fps": 30,
        "rotation": 0
      }
    }' \
  --dataset.repo_id=local/lerobot_new_dataset \
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=30 \
  --dataset.single_task="Dual arm manipulation task." \
  --display_data=true \
  --dataset.push_to_hub=false
```

_注意：调整 `episode_time_s` 以匹配您的任务长度，因为在无头模式（headless mode）下无法使用键盘快捷键。_

### 软件遥操作采集 (4 个 CAN 端口)

使用 `piper_dual_teleop` 插件，软件读取 Leader 位置并写入 Follower：

```bash
uv run lerobot-record \
  --robot.type=piper_dual_teleop \
  --robot.left_leader_port=can_left_leader \
  --robot.left_follower_port=can_left_follower \
  --robot.right_leader_port=can_right_leader \
  --robot.right_follower_port=can_right_follower \
  --robot.use_teleop=true \
  --robot.cameras='{"left":{"type":"opencv","index_or_path":"/dev/video4","width":640,"height":480,"fps":30},"right":{"type":"opencv","index_or_path":"/dev/video12","width":640,"height":480,"fps":30},"middle":{"type":"opencv","index_or_path":"/dev/video6","width":640,"height":480,"fps":30}}' \
  --dataset.repo_id=local/dual_teleop_dataset \
  --dataset.num_episodes=50 \
  --dataset.single_task="Dual arm manipulation task." \
  --display_data=true \
  --dataset.push_to_hub=false
```

### 其他可选参数:

```
  --dataset.episode_time_s=60 每次录制的持续时间（默认 60 秒），可以提前结束。
  --dataset.reset_time_s=60 每次录制后重置环境的持续时间（默认 60 秒）。
  --dataset.num_episodes=50 要录制的总次数（默认 50）。
```

### 使用键盘快捷键控制数据采集

按右箭头(→):提前停止当前事件,或重置时间,然后切换到下一个

按左箭头(→):取消当前事件并重新录制

按ESC:立即停止会话,编码视频并上传数据集

### 合并数据集

```bash
# 合并多个数据集（要求所有数据集特征完全一致）
HF_HUB_OFFLINE=1 lerobot-edit-dataset \
  --repo_id jokeru/pick_and_place \
  --operation.type merge \
  --operation.repo_ids "['jokeru/record_apple', 'jokeru/record_banana','jokeru/record_watermelon','jokeru/record_tape']" \
  --push_to_hub false
```

### 从数据集中删除片段

```
HF_LEROBOT_HOME=$HOME/.cache/huggingface/lerobot uv run lerobot-edit-dataset \
  --repo_id local/lerobot_new_dataset \
  --new_repo_id local/lerobot_new_dataset_filtered \
  --operation.type delete_episodes \
  --operation.episode_indices "[2]"
```

### 输出片段数量

```
HF_LEROBOT_HOME=$HOME/.cache/huggingface/lerobot uv run python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; ds = LeRobotDataset('local/lerobot_new_dataset_filtered'); print(f'Episodes: {ds.meta.total_episodes}')"
```

## 7. 数据集可视化

使用 rerun.io 验证录制的数据（相机和关节位置）。这将打开一个 rerun 窗口，您可以在其中检查 `observation/images` 和 `observation/state`。这使用的是 lerobot。

```bash

uv run lerobot-dataset-viz --repo-id local/lerobot_new_dataset --root ~/.cache/huggingface/lerobot/local/lerobot_new_dataset --episode-index 0
```

### 其他 Piper 可视化工具

```bash
# 可视化特定数据集片段
uv run python tools/viz/visualize_episode.py --repo_id local/lerobot_pick_and_place --episode 0
```

```bash
# 可视化整个 LeRobot 数据集（加速播放）
uv run python tools/viz/visualize_dataset.py --repo_id local/lerobot_pick_and_place
```

```bash
# 可视化 ACT 模型动作块（chunk）
uv run python tools/viz/visualize_action_chunk.py \
    --pretrained_path /path/to/model \
    --repo_id lerobot_pick_and_place \
    --episode_index 0 \
    --frame_index 90
```

## 回放片段 (Replaying Episode)

```bash
lerobot-replay \
    --robot.type=piper_dual \
    --robot.left_port=can_left \
    --robot.right_port=can_right \
    --dataset.repo_id=local/lerobot_new_dataset \
    --robot.cameras='{"left":{"type":"opencv","index_or_path":"/dev/video10","width":640,"height":480,"fps":30,"rotation":0},"right":{"type":"opencv","index_or_path":"/dev/video4","width":640,"height":480,"fps":30,"rotation":0},"middle":{"type":"opencv","index_or_path":"/dev/video12","width":640,"height":480,"fps":30,"rotation":0}}' \
    --dataset.episode=0
```

### 软件遥操作回放 (2 个 CAN 端口)

```bash
uv run lerobot-replay \
    --robot.type=piper_dual_teleop \
    --robot.left_follower_port=can_left_follower \
    --robot.right_follower_port=can_right_follower \
    --robot.use_teleop=false \
    --robot.cameras='{"left":{"type":"opencv","index_or_path":"/dev/video10","width":640,"height":480,"fps":30},"right":{"type":"opencv","index_or_path":"/dev/video4","width":640,"height":480,"fps":30},"middle":{"type":"opencv","index_or_path":"/dev/video12","width":640,"height":480,"fps":30}}' \
    --dataset.repo_id=local/dual_teleop_dataset \
    --dataset.episode=0
```

### 软件遥操作回放 (2 个 CAN 端口) - 另一组相机配置

```bash
uv run lerobot-replay \
    --robot.type=piper_dual_teleop \
    --robot.left_follower_port=can_left_follower \
    --robot.right_follower_port=can_right_follower \
    --robot.use_teleop=false \
    --robot.cameras='{"left":{"type":"opencv","index_or_path":"/dev/video4","width":640,"height":480,"fps":30},"right":{"type":"opencv","index_or_path":"/dev/video12","width":640,"height":480,"fps":30},"middle":{"type":"opencv","index_or_path":"/dev/video6","width":640,"height":480,"fps":30}}' \
    --dataset.repo_id=local/dual_teleop_dataset \
    --dataset.episode=0
```

## 8. 全部失能 (Disable All)

```bash
python utils/teleop_disable.py
```

## 9. Action Chunking Transformer (ACT) 模型

### 训练

在新收集的数据集上训练 ACT 策略。超参数说明：

- `chunk_size`: 控制合并为模型单个输入的动作步数（即 50 或 100）
- `n_action_steps`: 预测的动作步数 `<= chunk_size`。训练期间设置 `n_action_steps=chunk_size`，推理期间可修改。
- `steps`: 训练步数
- `save_freq`: 保存检查点的频率
- `dataset.image_transforms.enable`: 启用图像变换（对观测应用随机噪声，建议用于较大的数据集）

```bash
uv run lerobot-train \
  --policy.type=act \
  --dataset.repo_id=local/lerobot_pick_and_place \
  --output_dir=outputs/train/lerobot_pick_and_place \
  --job_name=act_piper \
  --wandb.mode=offline \
  --policy.push_to_hub=false \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --steps=100000 \
  --save_freq=10000 \
  --dataset.image_transforms.enable=true
```

### 上传 model或checkpoints 到huggingface

上传model

```
hf upload jokeru/pick_and_place ~/.cache/huggingface/lerobot/jokeru/pick_and_place \
  --repo-type model \
  --revision "main"
```

### 真机测试 (Testing on Real Robot)

应要求，使用 `lerobot-record` 测试策略（推理）。这将运行策略并记录结果。

```bash
# 使用 lerobot-record 测试策略
uv run lerobot-record \
  --robot.type=piper_dual \
  --robot.left_port=can_left \
  --robot.right_port=can_right \
  --robot.cameras='{"left":{"type":"opencv","index_or_path":"/dev/video10","width":640,"height":480,"fps":30,"rotation":0},"right":{"type":"opencv","index_or_path":"/dev/video4","width":640,"height":480,"fps":30,"rotation":0},"middle":{"type":"opencv","index_or_path":"/dev/video12","width":640,"height":480,"fps":30,"rotation":0}}' \
  --dataset.repo_id=local/eval_recording_test \
  --dataset.num_episodes=2 \
  --policy.type=act \
  --policy.pretrained_path=/home/droplab/workspace/lerobot_piper/outputs/train/lerobot_pick_and_place_100chunksz_jitter/checkpoints/last/pretrained_model \
  --dataset.single_task="Dual arm evaluation task" \
  --display_data=true \
  --dataset.push_to_hub=false
```

### 软件遥操作推理 (4 CAN → 2 CAN)

进行推理时，设置 `use_teleop=false`，只需要 2 个 Follower CAN 端口：

```bash
uv run lerobot-record \
  --robot.type=piper_dual_teleop \
  --robot.left_follower_port=can_left_follower \
  --robot.right_follower_port=can_right_follower \
  --robot.use_teleop=false \
  --robot.cameras='{"left":{"type":"opencv","index_or_path":"/dev/video4","width":640,"height":480,"fps":30},"right":{"type":"opencv","index_or_path":"/dev/video12","width":640,"height":480,"fps":30},"middle":{"type":"opencv","index_or_path":"/dev/video6","width":640,"height":480,"fps":30}}' \
  --dataset.repo_id=local/eval_test \
  --dataset.num_episodes=2 \
  --policy.type=act \
  --policy.pretrained_path=<path_to_model> \
  --dataset.single_task="Dual arm evaluation task" \
  --display_data=true \
  --dataset.push_to_hub=false
```

_(Note: You can also use `lerobot-eval` for pure evaluation without recording if desired, but this matches your request to use `lerobot-record`)_

## 10.openpi

### 环境安装

安装lerobot的pi相关依赖

```
pip install -e ".[pi]"
```

### 训练

```
python src/lerobot/scripts/lerobot_train.py\
    --dataset.repo_id=jokeru/record2 \
    --policy.type=pi05 \
    --output_dir=./outputs/pi05_training \
    --job_name=pi05_training \
    --policy.repo_id=jokeru/pi05 \
    --policy.pretrained_path=lerobot/pi05_libero \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=false \
    --policy.dtype=bfloat16 \
    --steps=3000 \
    --policy.device=cuda \
    --batch_size=32
```

pi05_base或pi05_libero 会下载在如 ~/.cache/huggingface/hub/models--lerobot--pi05_base

### 多卡训练

可用 tests/training/test_multi_gpu.py 测试

需要先安装依赖 pytest

```
pip install pytest
```

```
nohup accelerate launch --num_processes=8 \
  src/lerobot/scripts/lerobot_train.py\
    --dataset.repo_id=jokeru/record2 \
    --policy.type=pi05 \
    --output_dir=./outputs/pi05_training \
    --job_name=pi05_training \
    --policy.repo_id=jokeru/pi05 \
    --policy.pretrained_path=lerobot/pi05_libero \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=false \
    --policy.dtype=bfloat16 \
    --steps=3000 \
    --policy.device=cuda \
    --batch_size=32 > outputs/pi05_training.log 2>&1 &
```

### 本地推理

#### RTC

预训练

```
python examples/rtc/eval_with_real_robot.py \
  --policy.path=lerobot/pi05_base \
  --robot.type=piper_follower \
  --robot.cameras='{
    "wrist": {
      "type": "opencv",
      "index_or_path": "/dev/video0",
      "width": 480,
      "height": 640,
      "fps": 30,
      "rotation": -90
    },
    "ground": {
      "type": "opencv",
      "index_or_path": "/dev/video2",
      "width": 480,
      "height": 640,
      "fps": 30,
      "rotation": 90
    }
  }' \
  --task="Pick up it and put it into the basket." \
  --duration=120 \
  --action_queue_size_to_get_new_actions=30 \
  --fps=50 \
  --rtc.execution_horizon=5 \
  --display_data=true \
  --device=cuda
```

```
python examples/rtc/eval_with_real_robot.py \
  --policy.path=jokeru/pi05_pick_and_place \
  --robot.type=piper_follower \
  --robot.cameras='{
    "wrist": {
      "type": "opencv",
      "index_or_path": "/dev/video0",
      "width": 480,
      "height": 640,
      "fps": 30,
      "rotation": -90
    },
    "ground": {
      "type": "opencv",
      "index_or_path": "/dev/video2",
      "width": 480,
      "height": 640,
      "fps": 30,
      "rotation": 90
    }
  }' \
  --task="Pick up it and put it into the basket." \
  --duration=120 \
  --action_queue_size_to_get_new_actions=30 \
  --fps=50 \
  --rtc.execution_horizon=5 \
  --display_data=true \
  --device=cuda
```

## 11.异步推理（本地推理显存不够）

### 安装

```
pip install -e ".[async]"
```

### 启用远程推理服务器

用 CUDA_VISIBLE_DEVICES 设置用空闲的 GPU 推理，否则会默认用 GPU0

```
CUDA_VISIBLE_DEVICES=1 python -m src.lerobot.async_inference.policy_server \
    --host=127.0.0.1 \
    --port=8080 \
    --fps=30 \
    --inference_latency=0.033 \
    --obs_queue_timeout=1
```

### 若端口未开放需建立转发端口

在客户端建立端口转发，通过SSH把本地电脑的 8080 端口转发到远程服务器的 8080 端口,从而访问服务器上运行的服务

```
ssh -L 8080:127.0.0.1:8080 服务器用户名@服务器地址 -N
```

验证端口转发建立成功

```
nc -zv 127.0.0.1 8080
```

### 客户端接入

```
python -m src.lerobot.async_inference.robot_client \
    --server_address=127.0.0.1:8080 \
    --robot.type=piper_follower \
    --robot.cameras='{"wrist": {"type": "opencv", "index_or_path": "/dev/video6", "width": 480, "height": 640, "fps": 30, "rotation": 90}, "ground": {"type": "opencv", "index_or_path": "/dev/video0", "width": 480, "height": 640, "fps": 30, "rotation": -90}}' \
    --task="Pick up the apple and put it into the basket." \
    --policy_type=pi05 \
    --pretrained_name_or_path=jokeru/pi05_apple \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```
