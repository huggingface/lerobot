# NERO 机械臂 LeRobot 集成实现计划

> **For Hermes:** 使用 OpenCode 实现此计划，逐步完成每个 Task。

**Goal:** 将 AgileX NERO 7轴机械臂 + AGX Gripper 夹爪集成到 LeRobot 框架中，作为新的机器人类型 `nero_follower`。

**Architecture:** 参照 so_follower 模式，创建 `nero_follower/` 子目录，包含配置类和机器人类。NERO 通过 pyAgxArm SDK 的 CAN 总线通信，不使用 LeRobot 的 FeetechMotorsBus，而是直接包装 pyAgxArm API 为 LeRobot Robot 接口。关节角度统一使用弧度制。

**Tech Stack:** Python 3.12+, pyAgxArm SDK (CAN), LeRobot Robot ABC, draccus 配置系统

---

## 关键设计决策

1. **不用 FeetechMotorsBus** — NERO 用 CAN 总线 + pyAgxArm SDK，与 so_follower 的串口总线舵机完全不同
2. **7轴 + 夹爪** — NERO 有 7 个旋转关节(joint1-7) + 1 个夹爪(gripper)，共 8 个自由度
3. **弧度制** — pyAgxArm 原生使用弧度，LeRobot 策略也用弧度，保持一致
4. **move_j 控制** — send_action 使用 `move_j()` 关节空间运动
5. **夹爪控制** — 使用 `init_effector(AGX_GRIPPER)` + `move_gripper_m()` / `move_gripper_deg()`
6. **无需校准** — NERO 有绝对编码器，不需要 so_follower 那样的手动校准流程
7. **firmware_version 配置** — 支持 NeroFW.DEFAULT (<=1.10) 和 NeroFW.V111 (>=1.11)

---

## 需要创建/修改的文件

### 新建文件 (4个)
1. `src/lerobot/robots/nero_follower/__init__.py`
2. `src/lerobot/robots/nero_follower/config_nero_follower.py`
3. `src/lerobot/robots/nero_follower/nero_follower.py`

### 修改文件 (2个)
4. `src/lerobot/robots/utils.py` — 添加 nero_follower 到 make_robot_from_config
5. `pyproject.toml` — 添加 nero optional dependency

---

## Task 1: 创建配置类

**Files:**
- Create: `src/lerobot/robots/nero_follower/config_nero_follower.py`

```python
#!/usr/bin/env python

from dataclasses import dataclass, field
from typing import Literal

from lerobot.cameras import CameraConfig
from ..config import RobotConfig


@dataclass
class NEROFollowerConfig:
    """Configuration for NERO follower robot."""

    # CAN channel (e.g. "can0" on Linux)
    channel: str = "can0"

    # Communication interface
    interface: str = "socketcan"

    # Firmware version: "default" (<=1.10) or "v111" (>=1.11)
    firmware_version: str = "default"

    # Auto set motion mode
    auto_set_motion_mode: bool = True

    # Enable joint limits
    enable_joint_limits: bool = True

    # Speed percentage (0-100)
    speed_percent: int = 50

    # Max relative target for safety clipping (radians)
    max_relative_target: float | None = None

    # Disable torque on disconnect
    disable_torque_on_disconnect: bool = True

    # Cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)


@RobotConfig.register_subclass("nero_follower")
@dataclass
class NEROFollowerRobotConfig(RobotConfig, NEROFollowerConfig):
    pass
```

---

## Task 2: 创建机器人类

**Files:**
- Create: `src/lerobot/robots/nero_follower/nero_follower.py`

关键实现要点：
- `__init__`: 创建 pyAgxArm 配置和实例，但暂不 connect
- `connect()`: robot.connect() + enable() + init_effector(AGX_GRIPPER) + set_speed_percent()
- `disconnect()`: disable() + robot.disconnect()
- `get_observation()`: get_joint_angles() 获取7关节弧度 + gripper 状态 + 相机
- `send_action()`: move_j() 发送关节目标 + move_gripper_deg() 控制夹爪
- `observation_features`: joint1.pos ~ joint7.pos (float) + gripper.pos (float) + cameras
- `action_features`: 同 observation_features
- `is_calibrated`: 始终 True（绝对编码器）
- `calibrate()`: no-op
- `configure()`: set_speed_percent()

---

## Task 3: 创建 __init__.py

**Files:**
- Create: `src/lerobot/robots/nero_follower/__init__.py`

---

## Task 4: 注册到 make_robot_from_config

**Files:**
- Modify: `src/lerobot/robots/utils.py`

在 elif 链中添加:
```python
elif config.type == "nero_follower":
    from .nero_follower import NEOFollower
    return NEOFollower(config)
```

---

## Task 5: 添加 optional dependency

**Files:**
- Modify: `pyproject.toml`

添加:
```toml
nero = ["lerobot[can-dep]"]
```

---

## Task 6: 测试验证

1. `python -c "from lerobot.robots.nero_follower import NEOFollower"` — 导入测试
2. `python -c "from lerobot.robots import make_robot_from_config; ..."` — 工厂测试
3. 连接真实硬件测试（如果有 CAN 设备）

---

## NERO 关节映射

| LeRobot 名称 | pyAgxArm 名称 | 索引 | 限位(rad) |
|---|---|---|---|
| joint1 | joint1 | 1 | [-2.705, 2.705] |
| joint2 | joint2 | 2 | [-1.745, 1.745] |
| joint3 | joint3 | 3 | [-2.758, 2.758] |
| joint4 | joint4 | 4 | [-1.012, 2.147] |
| joint5 | joint5 | 5 | [-2.758, 2.758] |
| joint6 | joint6 | 6 | [-0.733, 0.960] |
| joint7 | joint7 | 7 | [-1.571, 1.571] |
| gripper | - | - | [0, 100] (归一化) |

## 夹爪控制说明

- 初始化: `robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)`
- 校准: `effector.calibrate_gripper()` (首次使用)
- 设置最大行程: `effector.set_gripper_teaching_pendant_param(max_range_config=0.07)`
- 运动: `effector.move_gripper_m(0.07)` (米) 或 `effector.move_gripper_deg(0)` (度)
- 状态: `effector.get_gripper_ctrl_states()`
