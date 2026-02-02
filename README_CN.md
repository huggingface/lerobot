# piper_lerobot数据集v3格式

[huggingface文档](http://huggingface.co/docs/lerobot)

## 1.环境创建

uv venv
uv sync

## 2.测试相机

注意两个相机不能从同一个扩展坞连接电脑,否则可能读取会出问题

```
sudo apt install guvcview    #安装Guvcview
guvcview --device=/dev/video0  # 测试wrist相机
guvcview --device=/dev/video2  # 测试ground相机
```

## 3.连接机械臂

"3-7.1:1.0"根据输出的can端口号改为自己的

```
conda activate lerobot
bash find_all_can_port.sh
bash can_activate.sh can_master 1000000 "1-8.2:1.0"
bash can_activate.sh can_follower 1000000 "1-8.3:1.0"
```

## 3.5 硬件遥操作配置 (Hardware Teleoperation)

我们的实验环境包含 **四台 Piper 机械臂**，分为两组（左臂组和右臂组），每组包含一台 Leader 和一台 Follower。

- **硬件连接**: 每一对 Leader 和 Follower 机械臂通过 CAN 总线连接在一起。
- **遥操作原理**: 利用 Piper SDK 提供的硬件级遥操作功能， Leader 和 Follower 只要设置好主从模式（Master/Slave），即可直接通信进行控制，无需电脑参与计算。
- **数据采集**: 我们将 Follower 机械臂通过 USB 连接到电脑。在录制数据时，电脑读取 Follower (Slave) 的状态数据，同时操作人员手动操作 Leader 进行动作示范。
- **推理与回放**: 在进行模型推理或回放时，我们会 **切断 Leader 机械臂的电源**，此时电脑直接通过 USB 发送控制信号给 Follower 机械臂执行动作。

更多关于 SDK 的细节，请参考 SDK 的官方文档及 API 函数。

## 4.遥操作

```
lerobot-teleoperate \
    --robot.type=piper_follower \
    --robot.id=my_follower_arm \
    --teleop.type=piper_leader \
    --teleop.id=my_leader_arm \
    --display_data=true
```

## 5.登陆huggingface

设置国内镜像加速

```
export HF_ENDPOINT=https://hf-mirror.com
```

通过运行此命令将您的令牌添加到CLI:

```
hf auth login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
```

验证登录

```
HF_USER=$(hf auth whoami | head -n 1)
echo $HF_USER
```

上传数据集到huggingface

```
hf upload jokeru/pick_and_place ~/.cache/huggingface/lerobot/jokeru/pick_and_place \
  --repo-type dataset \
  --revision "v3.0"
```

## 6.采集数据集 (Data Collection)

Record teleoperation data from the leader arms to the follower robot.

```bash
# Record a new episode
# Ensure cameras are connected.
uv run lerobot-record \
  --robot.type=piper_dual \
  --robot.left_port=can_left \
  --robot.right_port=can_right \
  --robot.read_only=true \
  --robot.cameras='{
      "left": {
        "type": "opencv",
        "index_or_path": "/dev/video4",
        "width": 640,
        "height": 480,
        "fps": 30,
        "rotation": 0
      },
      "right": {
        "type": "opencv",
        "index_or_path": "/dev/video12",
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

_Note: Adjust `episode_time_s` to match your task length since you cannot use keyboard shortcuts in headless mode._

### 其他可选参数:

```
  --dataset.episode_time_s=60 每个数据记录episode的持续时间(默认60秒)，可提前结束。
  --dataset.reset_time_s=60 每episode之后重置环境的时长(默认60秒)。
  --dataset.num_episodes=50 记录的总episode数(默认50)。
```

数据会保存到~/.cache/huggingface/lerobot/jokeru

录制过程中使用键盘控制

### 使用键盘快捷键控制数据采集

按右箭头(→):提前停止当前事件,或重置时间,然后切换到下一个

按左箭头(→):取消当前事件并重新录制

按ESC:立即停止会话,编码视频并上传数据集

### 合并数据集

```
# 合并多个数据集（要求所有数据集特征完全一致）
lerobot-edit-dataset \
  --repo_id jokeru/pick_and_place \
  --operation.type merge \
  --operation.repo_ids "['jokeru/record_apple', 'jokeru/record_banana','jokeru/record_watermelon','jokeru/record_tape']" \
  --push_to_hub true
```

## 7.可视化数据集 (Dataset Visualization)

Verify the recorded data (cameras and joint positions) using Rerun.
This will open a Rerun window where you can inspect `observation/images` and `observation/state`.

```bash
# Visualize Episode 0
uv run lerobot-dataset-viz --repo-id local/lerobot_new_dataset --root ~/.cache/huggingface/lerobot/local/lerobot_new_dataset --episode-index 0
```

## Replaying Episode

```bash
lerobot-replay \
    --robot.type=piper_dual \
    --robot.left_port=can_left \
    --robot.right_port=can_right \
    --dataset.repo_id=local/lerobot_new_dataset \
    --robot.cameras='{"left":{"type":"opencv","index_or_path":"/dev/video4","width":640,"height":480,"fps":30,"rotation":0},"right":{"type":"opencv","index_or_path":"/dev/video12","width":640,"height":480,"fps":30,"rotation":0},"middle":{"type":"opencv","index_or_path":"/dev/video6","width":640,"height":480,"fps":30,"rotation":0}}' \
    --dataset.episode=0
```

## 8.全部失能

```
python utils/teleop_disable.py
```

## 9.ACT

### Training (训练模型)

Train an ACT policy on the newly collected dataset.

```bash
# Train command
uv run lerobot-train \
  --policy.type=act \
  --dataset.repo_id=local/lerobot_new_dataset \
  --output_dir=outputs/train/act_piper_new \
  --job_name=act_piper_new \
  --wandb.mode=offline \
  --policy.push_to_hub=false \
  --steps=100000 \
  --save_freq=10000
```

### 上传 model或checkpoints 到huggingface

上传model

```
hf upload jokeru/pick_and_place ~/.cache/huggingface/lerobot/jokeru/pick_and_place \
  --repo-type model \
  --revision "main"
```

### 测试ACT

### Testing on Real Robot (真机测试)

As requested, use `lerobot-record` to test the policy (Inference). This will run the policy and record the result.

```bash
# Test policy using lerobot-record
uv run lerobot-record \
  --robot.type=piper_dual \
  --robot.left_port=can_left \
  --robot.right_port=can_right \
  --robot.cameras='{"left":{"type":"opencv","index_or_path":"/dev/video4","width":640,"height":480,"fps":30,"rotation":0},"right":{"type":"opencv","index_or_path":"/dev/video12","width":640,"height":480,"fps":30,"rotation":0},"middle":{"type":"opencv","index_or_path":"/dev/video6","width":640,"height":480,"fps":30,"rotation":0}}' \
  --dataset.repo_id=local/eval_recording_test \
  --dataset.num_episodes=2 \
  --policy.type=act \
  --policy.temporal_ensemble_coeff=0.01 \
  --policy.pretrained_path=/home/droplab/workspace/piper_lerobot/outputs/train/act_piper_new/checkpoints/last/pretrained_model \
  --dataset.single_task="Dual arm evaluation task" \
--display_data=true \
--dataset.rename_map='{"left":"observation.image_0","right":"observation.image_1","middle":"observation.image_2"}' \
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
