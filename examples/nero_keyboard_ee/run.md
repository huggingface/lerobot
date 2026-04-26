# NERO Keyboard Joint Teleoperate

## 运行方式

'''
lerobot-find-cameras opencv # or realsense for Intel Realsense cameras
'''


**必须在本机终端里运行**（SSH / 桌面终端均可），不能通过脚本或管道执行，因为键盘输入需要直接读取终端 stdin。

```bash
# 1. 打开终端，激活环境
conda activate lerobot

# 2. 进入项目目录
cd /home/yuhang/projects/lerobot

# 3. 拉起 CAN0 接口（每次重启后需要执行）
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000 
sudo ip link set can0 up

# 4. 启动
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

## 按键映射

直接按键即可控制，无需按住任何使能键。

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

## 注意事项

- 机械臂断开时**不会下电**（`disable_torque_on_disconnect=False`），防止不自锁掉落
- 速度 10%，如需调整改 `--robot.speed_percent`
- 关节增量 0.05 rad/步，夹爪增量 2.0/步，可改 `--teleop.joint_step` 和 `--teleop.gripper_step`
- 如需重新开启死人开关（Shift 使能），加 `--teleop.require_deadman=True`


lerobot-teleoperate     --robot.type=nero_follower     --robot.channel=can0     --robot.cameras='{ front: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, side: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, back: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: "MJPG"}  }'      --robot.interface=socketcan     --robot.firmeware_version=default     --robot.speed_percent=10     --robot.disable_torque_on_disconnect=False     --teleop.type=keyboard_joint     --teleop.joint_step=0.05     --teleop.gripper_step=2.0     --fps=30     --display_data=true
---

## SmolVLA 零样本推理

### 1. 下载模型

网络不好时使用镜像站：

```bash
# 方式1：huggingface_hub
HF_ENDPOINT=https://hf-mirror.com python3 -c "from huggingface_hub import snapshot_download; snapshot_download('lerobot/smolvla_base', local_dir='$HOME/models/smolvla_base')"

# 方式2：git clone
git lfs install
git clone https://hf-mirror.com/lerobot/smolvla_base ~/models/smolvla_base
```

模型目录下必须包含：

```
config.json
model.safetensors
policy_postprocessor.json
policy_postprocessor_step_0_unnormalizer_processor.safetensors
policy_preprocessor.json
policy_preprocessor_step_5_normalizer_processor.safetensors
```

### 2. 运行推理

SmolVLA 模型期望 3 个相机（camera1/camera2/camera3），NERO 有 3 个（front/side/back），需要用 `--dataset.rename_map` 映射相机名。

```bash
conda activate lerobot

HF_HUB_OFFLINE=1 lerobot-record     --robot.type=nero_follower     --robot.channel=can0     --robot.interface=socketcan     --robot.firmeware_version=default     --robot.speed_percent=10     --robot.disable_torque_on_disconnect=False     --robot.cameras='{ front: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, side: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, back: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: "MJPG"} }'     --policy.path=$HOME/models/smolvla_base     --dataset.rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.side": "observation.images.camera2", "observation.images.back": "observation.images.camera3"}'     --dataset.repo_id=yuhang/eval_nero_smolvla_test14     --dataset.num_episodes=1     --dataset.single_task="pick up the 
green cube"     --dataset.push_to_hub=false     --display_data=True
```

> 注意：无微调的 `smolvla_base` 是通用基础模型，在 NERO Follower 上效果可能很差，需微调后才能有好的表现。