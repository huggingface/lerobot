# Nero Leader Teleoperator

AgileX Nero 7-DOF 机械臂 Leader 端遥操作，基于 pyAgxArm SDK + CAN 总线通信。

## 前置条件

```bash
pip install pyAgxArm python-can
```

激活 CAN 通道：

```bash
sudo ip link set can0 up type can bitrate 1000000
sudo ip link set can1 up type can bitrate 1000000
```

## 遥操作命令

```bash
lerobot-teleoperate \
    --robot.type=nero_follower \
    --robot.channel=can1 \
    --teleop.type=nero_leader \
    --teleop.port=can0 \
    --display_data=true
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--teleop.port` | `can0` | Leader 臂 CAN 通道 |
| `--robot.channel` | `can0` | Follower 臂 CAN 通道 |
| `--teleop.firmware_version` | `V111` | 固件版本，≤1.10 改为 `DEFAULT` |
| `--teleop.can_interface` | `socketcan` | CAN 接口类型 |
| `--teleop.enable_drag_teach` | `true` | 零力拖动（重力补偿），leader 连接时不使能电机 |
| `--teleop.reset_can_on_connect` | `true` | 连接时重置 CAN 通道 |
| `--teleop.mirror_sign` | `[-1,1,-1,1,-1,-1,1]` | 面对面镜像符号，-1=翻转，1=保持 |
| `--teleop.enable_retries` | `50` | 使能重试次数（leader 不使用） |
| `--robot.speed_percent` | `50` | Follower 运动速度 (0-100) |

## 常用场景

### 关闭零力拖动

不用重力补偿，手动拖动 Leader：

```bash
lerobot-teleoperate \
    --robot.type=nero_follower \
    --robot.channel=can1 \
    --teleop.type=nero_leader \
    --teleop.port=can0 \
    --teleop.enable_drag_teach=false
```

### 固件 ≤1.10

```bash
lerobot-teleoperate \
    --robot.type=nero_follower \
    --robot.channel=can1 \
    --robot.firmeware_version=default \
    --teleop.type=nero_leader \
    --teleop.port=can0 \
    --teleop.firmware_version=DEFAULT
```

### Python API

```python
from lerobot.teleoperators.nero_leader import NeroLeader, NeroLeaderConfig
from lerobot.robots.nero_follower import NEOFollower, NEOFollowerRobotConfig

leader = NeroLeader(NeroLeaderConfig(port="can0"))
follower = NEOFollower(NEOFollowerRobotConfig(channel="can1"))

leader.connect()
follower.connect()

while True:
    action = leader.get_action()
    follower.send_action(action)
```

## action_features

| Key | 类型 | 单位 |
|---|---|---|
| `joint1.pos` ~ `joint7.pos` | `float` | rad |
| `gripper.pos` | `float` | deg (0-100) |

与 `nero_follower` 的 action_features 完全对齐，无需额外映射。
