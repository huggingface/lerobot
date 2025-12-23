# 基于 SO101 的 HIL-SERL 复现指南

本指南将帮助您基于 LeRobot 的 SO101 机器人复现 HIL-SERL（Human-in-the-Loop Sample-Efficient Reinforcement Learning）训练流程。

## 前置要求

- 一个游戏手柄（推荐）或键盘用于控制机器人
- NVIDIA GPU
- SO101 follower 机械臂（用于执行任务）
- SO101 leader 机械臂（可选，用于遥操作，也可使用游戏手柄或键盘）
- 机器人的 URDF 文件（用于运动学计算）

## 安装步骤

### 1. 安装 LeRobot 和 HIL-SERL 依赖

```bash
# 安装 LeRobot 基础包和 HIL-SERL 扩展
pip install -e ".[hilserl]"

# 安装 Feetech SDK（SO101 使用 Feetech 电机）
pip install -e ".[feetech]"
```

### 2. 准备 URDF 文件

从 [SO-ARM100 仓库](https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf) 下载 SO101 的 URDF 文件：

```bash
# 创建目录并下载 URDF 文件
mkdir -p ./SO101
# 将 so101_new_calib.urdf 文件保存到 ./SO101/ 目录下
```

## 配置步骤

### 1. 查找机器人工作空间边界

在收集演示数据之前，需要确定机器人的操作边界。这有助于：
- 限制机器人的操作空间到任务相关区域
- 在末端执行器空间而非关节空间进行训练（通常更容易学习）

使用以下脚本查找边界：

```bash
lerobot-find-joint-limits \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodemXXXXX \  # 替换为您的 follower 端口
  --robot.id=black \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodemYYYYY \  # 替换为您的 leader 端口
  --teleop.id=blue
```

**工作流程：**
1. 运行脚本，通过 leader 移动 follower 到任务所需的空间范围
2. 脚本会记录最小和最大末端执行器位置，例如：
   ```
   Max ee position [0.2417 0.2012 0.1027]
   Min ee position [0.1663 -0.0823 0.0336]
   ```
3. 将这些值用于后续配置中的 `end_effector_bounds` 字段

### 2. 创建配置文件

创建环境配置文件（例如 `env_config_so101.json`）：

```json
{
  "env": {
    "type": "gym_manipulator",
    "name": "real_robot",
    "fps": 10,
    "processor": {
      "control_mode": "gamepad",
      "observation": {
        "display_cameras": false
      },
      "image_preprocessing": {
        "crop_params_dict": {},
        "resize_size": [128, 128]
      },
      "gripper": {
        "use_gripper": true,
        "gripper_penalty": 0.0
      },
      "reset": {
        "reset_time_s": 5.0,
        "control_time_s": 20.0,
        "terminate_on_success": true
      },
      "inverse_kinematics": {
        "urdf_path": "./SO101/so101_new_calib.urdf",
        "target_frame_name": "gripper_frame_link",
        "end_effector_bounds": {
          "min": [0.16, -0.08, 0.03],
          "max": [0.24, 0.2, 0.1]
        },
        "end_effector_step_sizes": {
          "x": 0.02,
          "y": 0.02,
          "z": 0.02
        }
      }
    },
    "robot": {
      "type": "so101_follower",
      "port": "/dev/tty.usbmodemXXXXX",
      "id": "my_so101_follower",
      "use_degrees": true,
      "cameras": {
        "front": {
          "type": "opencv",
          "index_or_path": 0,
          "width": 640,
          "height": 480,
          "fps": 10
        },
        "side": {
          "type": "opencv",
          "index_or_path": 1,
          "width": 640,
          "height": 480,
          "fps": 10
        }
      }
    },
    "teleop": {
      "type": "gamepad",
      "use_gripper": true
    }
  },
  "dataset": {
    "repo_id": "your_username/task_name",
    "root": null,
    "task": "pick_and_lift",
    "num_episodes_to_record": 15,
    "replay_episode": 0,
    "push_to_hub": false
  },
  "mode": "record",
  "device": "cpu"
}
```

**关键配置说明：**

- `processor.inverse_kinematics`: 配置末端执行器控制
  - `urdf_path`: URDF 文件路径
  - `target_frame_name`: 末端执行器框架名称（通常是 "gripper_frame_link"）
  - `end_effector_bounds`: 从 `lerobot-find-joint-limits` 获得的工作空间边界
  - `end_effector_step_sizes`: 每个轴的最大步长（米）

- `robot`: SO101 follower 配置
  - `port`: USB 端口（使用 `lerobot-find-port` 查找）
  - `use_degrees`: 设置为 `true`（SO101 使用度数）

- `teleop`: 遥操作设备配置
  - 使用 `gamepad` 或 `so101_leader`

## 数据收集流程

### 1. 收集演示数据

设置 `mode` 为 `"record"` 并运行：

```bash
python -m lerobot.rl.gym_manipulator --config_path env_config_so101.json
```

**录制过程：**
1. 机器人会重置到配置文件中 `env.processor.reset.fixed_reset_joint_positions` 定义的初始位置
2. 使用游戏手柄或 leader 完成任务
3. 按下"成功"按钮结束回合（奖励为 1）
4. 如果达到时间限制或按下"失败"按钮，回合以奖励 0 结束
5. 可以按"重新录制"按钮重新录制回合
6. 录制完所有回合后，数据集会自动保存

### 2. 处理数据集 - 确定图像裁剪区域

视觉强化学习对背景干扰很敏感，需要裁剪图像到相关的工作空间区域。

使用交互式裁剪工具：

```bash
python -m lerobot.rl.crop_dataset_roi --repo-id your_username/task_name
```

**工作流程：**
1. 脚本会显示每个相机视图的第一帧
2. 在相关的工作空间区域周围绘制矩形
3. 按 'c' 确认选择
4. 对所有相机视图重复此操作
5. 脚本会输出裁剪参数并创建新的裁剪数据集

示例输出：
```
Selected Rectangular Regions of Interest (top, left, height, width):
observation.images.side: [180, 207, 180, 200]
observation.images.front: [180, 250, 120, 150]
```

### 3. 更新配置中的裁剪参数

将裁剪参数添加到训练配置中：

```json
{
  "env": {
    "processor": {
      "image_preprocessing": {
        "crop_params_dict": {
          "observation.images.side": [180, 207, 180, 200],
          "observation.images.front": [180, 250, 120, 150]
        },
        "resize_size": [128, 128]
      }
    }
  }
}
```

**推荐图像分辨率：**
- 大多数基于视觉的策略在 **128×128**（默认）或 **64×64** 像素的方形输入上验证
- 建议设置 `resize_size` 为 `[128, 128]`，或如果需要节省 GPU 内存和带宽则使用 `[64, 64]`

## 训练奖励分类器（可选）

奖励分类器可以自动检测任务成功，无需手动标注每个时间步。

### 1. 收集奖励分类器数据集

修改配置以收集带标签的数据集：

```json
{
  "env": {
    "processor": {
      "reset": {
        "terminate_on_success": false
      }
    }
  },
  "dataset": {
    "num_episodes_to_record": 20
  },
  "mode": "record"
}
```

**重要：** 对于奖励分类器训练，设置 `terminate_on_success: false` 以收集足够的正样本。

### 2. 训练分类器

创建奖励分类器训练配置（`reward_classifier_train_config.json`）：

```json
{
  "policy": {
    "type": "reward_classifier",
    "model_name": "helper2424/resnet10",
    "model_type": "cnn",
    "num_cameras": 2,
    "num_classes": 2,
    "hidden_dim": 256,
    "dropout_rate": 0.1,
    "learning_rate": 1e-4,
    "device": "cuda",
    "use_amp": true,
    "input_features": {
      "observation.images.front": {
        "type": "VISUAL",
        "shape": [3, 128, 128]
      },
      "observation.images.side": {
        "type": "VISUAL",
        "shape": [3, 128, 128]
      }
    }
  },
  "dataset": {
    "repo_id": "your_username/task_name",
    "task": "pick_and_lift"
  }
}
```

训练分类器：

```bash
lerobot-train --config_path reward_classifier_train_config.json
```

### 3. 奖励分类器的输出说明

奖励分类器在处理图像观察时会输出以下内容：

**`predict()` 方法的输出（`ClassifierOutput` 对象）：**
- `logits`: 原始输出值（未归一化的分数）
  - 二分类：形状为 `[batch_size]` 的标量 logits
  - 多分类：形状为 `[batch_size, num_classes]` 的 logits
- `probabilities`: 概率值（归一化后的）
  - 二分类：使用 sigmoid 函数，形状为 `[batch_size]`，值域 [0, 1]
  - 多分类：使用 softmax 函数，形状为 `[batch_size, num_classes]`，每行和为 1
- `hidden_states`: 编码器的隐藏状态表示，形状为 `[batch_size, hidden_dim]`

**`predict_reward()` 方法的输出（用于环境中的实际奖励计算）：**
- **二分类模式**（`num_classes=2`，最常见）：
  - 输入：图像批次和阈值（默认 0.5）
  - 输出：`0` 或 `1`（torch.Tensor）
    - 如果 `probabilities > threshold`，返回 `1.0`（成功）
    - 否则返回 `0.0`（失败）
- **多分类模式**（`num_classes > 2`）：
  - 输出：类别索引（torch.Tensor），通过 `argmax(probabilities)` 获得

**在环境中的使用：**
当 `RewardClassifierProcessorStep` 处理每个时间步时：
1. 从观察中提取图像（所有包含 "image" 的键）
2. 调用 `predict_reward()` 得到成功预测（0 或 1）
3. 如果 `success == 1`：
   - 设置 `reward = success_reward`（默认 1.0）
   - 如果 `terminate_on_success=True`，设置 `done = True` 终止回合
4. 在 `info` 字典中记录 `reward_classifier_frequency`（分类器推理频率，Hz）

**示例：**
```python
# 二分类示例
# 输入：图像观察
images = {
    "observation.images.front": torch.Tensor([batch_size, 3, 128, 128]),
    "observation.images.side": torch.Tensor([batch_size, 3, 128, 128])
}

# predict_reward 输出
success = classifier.predict_reward(images, threshold=0.7)
# success: tensor([1.0])  # 成功
# 或
# success: tensor([0.0])  # 失败

# 在环境中，如果 success == 1：
# reward = 1.0
# done = True (如果 terminate_on_success=True)
```

### 4. 在训练中使用奖励分类器

在环境配置中添加奖励分类器：

```json
{
  "env": {
    "processor": {
      "reward_classifier": {
        "pretrained_path": "path_to_your_pretrained_model",
        "success_threshold": 0.7,
        "success_reward": 1.0
      },
      "reset": {
        "terminate_on_success": true
      }
    }
  }
}
```

**配置参数说明：**
- `pretrained_path`: 训练好的分类器模型路径
- `success_threshold`: 成功判断的概率阈值（0.0-1.0）
  - 概率超过此阈值时判定为成功
  - 建议从 0.5 开始，根据验证集表现调整
- `success_reward`: 成功时给予的奖励值（通常为 1.0）
- `terminate_on_success`: 是否在检测到成功时立即终止回合
  - `true`: 自动终止，适合大多数任务
  - `false`: 继续执行，适合需要收集更多成功状态数据的场景

## Actor-Learner 训练

HIL-SERL 使用分布式 actor-learner 架构进行训练。

### 1. 创建训练配置

创建训练配置文件（`train_config_hilserl_so101.json`）：

```json
{
  "policy": {
    "type": "sac",
    "device": "cuda",
    "storage_device": "cuda",
    "temperature_init": 1e-2,
    "actor_learner_config": {
      "policy_parameters_push_frequency": 2.0
    },
    "input_features": {
      "observation.images.front": {
        "type": "VISUAL",
        "shape": [3, 128, 128]
      },
      "observation.images.side": {
        "type": "VISUAL",
        "shape": [3, 128, 128]
      },
      "observation.state": {
        "type": "FLOATING_POINT",
        "shape": [6]
      }
    },
    "output_features": {
      "action": {
        "type": "FLOATING_POINT",
        "shape": [4]
      }
    }
  },
  "dataset": {
    "repo_id": "your_username/task_name",
    "task": "pick_and_lift"
  },
  "env": {
    "type": "gym_manipulator",
    "name": "real_robot",
    "fps": 10,
    "processor": {
      "control_mode": "gamepad",
      "image_preprocessing": {
        "crop_params_dict": {
          "observation.images.side": [180, 207, 180, 200],
          "observation.images.front": [180, 250, 120, 150]
        },
        "resize_size": [128, 128]
      },
      "inverse_kinematics": {
        "urdf_path": "./SO101/so101_new_calib.urdf",
        "target_frame_name": "gripper_frame_link",
        "end_effector_bounds": {
          "min": [0.16, -0.08, 0.03],
          "max": [0.24, 0.2, 0.1]
        },
        "end_effector_step_sizes": {
          "x": 0.02,
          "y": 0.02,
          "z": 0.02
        }
      },
      "gripper": {
        "use_gripper": true
      },
      "reset": {
        "reset_time_s": 5.0,
        "control_time_s": 20.0,
        "terminate_on_success": true
      }
    },
    "robot": {
      "type": "so101_follower",
      "port": "/dev/tty.usbmodemXXXXX",
      "id": "my_so101_follower",
      "use_degrees": true,
      "cameras": {
        "front": {
          "type": "opencv",
          "index_or_path": 0,
          "width": 640,
          "height": 480,
          "fps": 10
        },
        "side": {
          "type": "opencv",
          "index_or_path": 1,
          "width": 640,
          "height": 480,
          "fps": 10
        }
      }
    },
    "teleop": {
      "type": "gamepad",
      "use_gripper": true
    }
  },
  "wandb": {
    "enable": true,
    "project": "hilserl_so101"
  }
}
```

### 2. 启动 Learner 进程

在第一个终端中启动 learner：

```bash
python -m lerobot.rl.learner --config_path train_config_hilserl_so101.json
```

Learner 会：
- 初始化策略网络
- 准备重放缓冲区
- 打开 gRPC 服务器与 actors 通信
- 处理转换并更新策略

### 3. 启动 Actor 进程

在第二个终端中启动 actor：

```bash
python -m lerobot.rl.actor --config_path train_config_hilserl_so101.json
```

Actor 会：
- 通过 gRPC 连接到 learner
- 初始化环境
- 执行策略 rollout 收集经验
- 将转换发送给 learner
- 接收更新的策略参数

### 4. 人工干预

训练过程中的关键是人机交互：

- **干预方式：** 按下游戏手柄右上方的触发按钮（或键盘的 `space` 键）暂停策略动作并接管控制
- **干预策略：**
  - 在训练开始时允许策略探索几个回合
  - 避免长时间干预，只在机器人偏离轨道时快速纠正
  - 一旦策略开始完成任务（即使不完美），可以限制干预为简单的快速动作（如抓取命令）
- **理想行为：** 干预率应该随着训练逐渐下降（可在 WandB 仪表板中监控）

## 关键超参数调优

以下配置值对训练稳定性和速度有重要影响：

- **`temperature_init`** (`policy.temperature_init`): SAC 的初始熵温度
  - 较高值鼓励更多探索
  - 较低值使策略更早确定性
  - 建议起始值：`1e-2`
  - 设置过高可能使人工干预无效并减慢学习

- **`policy_parameters_push_frequency`** (`policy.actor_learner_config.policy_parameters_push_frequency`): learner 向 actor 推送权重的间隔（秒）
  - 默认：`4 s`
  - 建议：减少到 **1-2 s** 以提供更新的权重（代价是更多网络流量）
  - 仅在连接较慢时增加，因为这会降低样本效率

- **`storage_device`** (`policy.storage_device`): learner 保存策略参数的设备
  - 默认：`"cpu"`
  - 如果有空闲 GPU 内存，设置为 `"cuda"`
  - 将权重保留在 GPU 上可以移除 CPU→GPU 传输开销，显著增加每秒的 learner 更新次数

## 监控和调试

如果配置中设置了 `wandb.enable: true`，可以通过 [Weights & Biases](https://wandb.ai/site/) 仪表板实时监控训练进度。

## 故障排除

### 端口权限问题（Linux）

```bash
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1
```

### 找不到 URDF 文件

确保 URDF 文件路径正确，并且文件存在于指定位置。

### 机器人连接问题

- 检查 USB 连接
- 确认端口号正确（使用 `lerobot-find-port`）
- 检查电源连接

## 总结

完成以上步骤后，您应该能够：

1. ✅ 安装和配置 LeRobot HIL-SERL
2. ✅ 配置 SO101 机器人进行末端执行器控制
3. ✅ 收集和预处理演示数据
4. ✅ 训练奖励分类器（可选）
5. ✅ 使用 actor-learner 架构进行在线强化学习训练
6. ✅ 通过人工干预指导策略学习

祝您训练顺利！🎉

> [!TIP]
> 如有问题或需要帮助，请访问 [Discord](https://discord.com/invite/s3KuuzsPFb)。

