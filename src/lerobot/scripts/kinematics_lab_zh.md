# kinematics_lab 中文使用文档

`kinematics_lab.py` 是一个面向 SO100/SO101 follower 机械臂的交互式正/逆运动学实验工具。默认情况下，它只基于 URDF 文件计算 FK/IK，不会连接或驱动真实机械臂；只有显式传入 `--connect --port ...` 后，才会读取真实关节位置并发送运动目标。

## 适用场景

- 根据当前关节角计算末端位姿，也就是正运动学 FK。
- 输入末端目标位置，求解对应关节角，也就是逆运动学 IK。
- 在不连接硬件的情况下验证 URDF、末端坐标系和关节顺序。
- 连接真实 SO100/SO101 follower 机械臂后，交互式测试小幅运动。
- 可选开启 Rerun 可视化，观察简化的 3D 连杆骨架和末端位置。

## 依赖安装

该脚本依赖可选的 `kinematics` 依赖组，其中核心求解器是 `placo`。请在 LeRobot 仓库根目录、并且在运行脚本的同一个 Python 环境中安装：

```powershell
python -m pip install -e ".[kinematics]"
```

如果在 Windows 上安装 `placo` 时遇到 `nmake` 或 C/C++ 编译器错误，可以优先从 conda-forge 安装：

```powershell
conda install -c conda-forge placo
python -m pip install -e .
```

Bash/zsh 环境中可以使用：

```shell
python -m pip install -e '.[kinematics]'
```

如需确认当前环境是否能导入 `placo`：

```powershell
python -c "import sys, placo; print(sys.executable); print(placo.__file__)"
```

## URDF 和标定文件

`kinematics_lab` 需要 URDF 描述机械臂的几何结构、连杆长度、关节轴和末端坐标系。LeRobot 的电机标定 JSON 只负责把电机原始位置映射为校准后的关节角，它不会生成 URDF，也不能替代 URDF。

对于 SO101 风格机械臂，可以使用 SO-ARM100 仓库中的校准 URDF：

```text
https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
```

注意不要只下载 `.urdf` 文件。该 URDF 会引用旁边的 STL 网格文件，例如 `assets/base_motor_holder_so101_v1.stl`，所以需要保持整个 `Simulation/SO101` 文件夹的相对目录结构：

```text
assets/urdf/so101/
|-- so101_new_calib.urdf
`-- assets/
    |-- base_motor_holder_so101_v1.stl
    |-- base_so101_v2.stl
    `-- ...
```

可以用 sparse clone 只拉取需要的目录：

```powershell
git clone --depth 1 --filter=blob:none --sparse https://github.com/TheRobotStudio/SO-ARM100.git external\SO-ARM100
cd external\SO-ARM100
git sparse-checkout set Simulation/SO101
cd ..\..
```

运行时传入：

```powershell
--urdf-path external\SO-ARM100\Simulation\SO101\so101_new_calib.urdf
```

如果连接真实机械臂，请使用和 LeRobot 标定命令相同的 `--id`，这样读取电机位置和发送动作时会使用正确的标定文件。

## 查看 URDF 中的关节和坐标系

如果不确定 `--target-frame-name` 或关节名称，可以直接在 URDF 中查看：

```powershell
Select-String -Path external\SO-ARM100\Simulation\SO101\so101_new_calib.urdf -Pattern '<joint name=','<link name='
```

脚本默认末端坐标系是：

```text
gripper_frame_link
```

如果 URDF 中包含额外关节，或者关节顺序不是期望顺序，可以用 `--joint-names` 显式指定参与运动学求解的 URDF 关节名。

## 干跑模式

干跑模式只计算 FK/IK，不连接、不移动真实硬件。建议第一次使用时先干跑，确认 URDF、末端坐标系、单位和关节方向是否合理。

PowerShell：

```powershell
python -m lerobot.scripts.kinematics_lab `
    --urdf-path path\to\so101_new_calib.urdf `
    --target-frame-name gripper_frame_link `
    --units mm
```

Bash/zsh：

```shell
python -m lerobot.scripts.kinematics_lab \
    --urdf-path path/to/so101_new_calib.urdf \
    --target-frame-name gripper_frame_link \
    --units mm
```

示例交互：

```text
kinematics> joints 0 -45 45 0 0 30
kinematics> fk
kinematics> ik 180 0 120
kinematics> quit
```

其中 `joints` 输入的是角度制，`ik` 输入的是由 `--units` 决定的笛卡尔位置单位。上面的例子中单位是毫米。

## 连接真实机械臂

连接硬件前先查找串口：

```shell
lerobot-find-port
```

Windows 端口通常类似 `COM5`，Linux/macOS 通常类似 `/dev/ttyACM0`、`/dev/ttyUSB0` 或 `/dev/tty.usbmodem*`。

第一次连接真实机械臂时建议：

- 保持工作空间清空。
- 一只手靠近电源或急停。
- 使用较小的 `--max-relative-target`，限制每条命令允许的相对关节变化。
- 先执行 `read` 同步真实机械臂当前姿态，再执行 `fk` 和小幅 `move`。

PowerShell 示例：

```powershell
python -m lerobot.scripts.kinematics_lab `
    --urdf-path path\to\so101_new_calib.urdf `
    --connect `
    --robot-type so101_follower `
    --port COM5 `
    --id classroom_so101 `
    --target-frame-name gripper_frame_link `
    --units mm `
    --max-relative-target 5
```

Bash/zsh 示例：

```shell
python -m lerobot.scripts.kinematics_lab \
    --urdf-path path/to/so101_new_calib.urdf \
    --connect \
    --robot-type so101_follower \
    --port /dev/ttyACM0 \
    --id classroom_so101 \
    --target-frame-name gripper_frame_link \
    --units mm \
    --max-relative-target 5
```

推荐第一次真机交互流程：

```text
kinematics> read
kinematics> fk
kinematics> ik 180 0 120
kinematics> move 180 0 120
kinematics> move 185 0 120
kinematics> move 185 5 120 30
kinematics> read
kinematics> quit
```

`ik` 只求解并打印目标关节角，不发送给硬件。`move` 会先求 IK，再发送给真实机械臂。`move x y z gripper` 的第四个值用于设置夹爪关节角，单位为度，前提是机器人暴露了夹爪关节。

## 交互命令

```text
read
    从真实机器人读取当前关节角并打印 FK。需要 --connect。

joints q1 q2 q3 q4 q5 [gripper]
    手动设置当前关节猜测值，单位为度，并打印 FK。

fk
    基于当前关节猜测值打印末端位姿。

ik x y z
    对指定的绝对末端位置求 IK。位置单位由 --units 决定，并保持当前末端姿态。

move x y z [gripper]
    对指定的绝对末端位置求 IK，然后发送关节目标到真实机器人。需要 --connect。

send
    将当前关节猜测值直接发送给真实机器人。需要 --connect。

help
    显示命令帮助。

quit
    断开连接并退出。
```

## 常用启动参数

| 参数 | 说明 |
| --- | --- |
| `--urdf-path` | 必填，机器人 URDF 文件路径。 |
| `--target-frame-name` | 作为末端执行器的 URDF frame，默认 `gripper_frame_link`。 |
| `--joint-names` | 可选，显式指定参与 FK/IK 的 URDF 关节名列表。 |
| `--units` | 交互式笛卡尔命令使用的单位，可选 `m`、`cm`、`mm`，默认 `m`。 |
| `--visualize` | 打开 Rerun 3D 可视化窗口。 |
| `--connect` | 连接真实 follower 机械臂。未设置时只做干跑计算。 |
| `--robot-type` | 真机类型，可选 `so100_follower` 或 `so101_follower`，默认 `so100_follower`。 |
| `--port` | 电机总线串口，例如 `COM5` 或 `/dev/ttyACM0`。使用 `--connect` 时必填。 |
| `--id` | 机器人标定 ID，应与 LeRobot 标定流程使用的 ID 一致。 |
| `--calibration-dir` | 可选，指定标定文件目录；不填时使用 LeRobot 默认查找路径。 |
| `--max-relative-target` | follower 机器人安全层的单次命令最大相对关节变化，默认 `10.0`。首次真机测试建议调小。 |
| `--no-calibrate` | 连接时跳过校准。仅在确认机械臂已正确校准且有意跳过启动校准流程时使用。 |

## FK/IK 计算细节

- `RobotKinematics` 使用 `placo.RobotWrapper` 加载 URDF。
- 求解时固定 base：`solver.mask_fbase(True)`。
- 关节输入和输出在脚本交互层都使用角度制。
- 内部求解时会把角度转换成弧度。
- `ik x y z` 和 `move x y z` 会保留当前末端姿态，只替换末端位置。
- IK 的位置约束权重默认为 `1.0`，姿态约束权重默认为 `0.01`。
- 如果当前关节数组中包含夹爪等额外关节，IK 会保留这些额外关节值；`move` 的第四个参数可以覆盖夹爪目标。

## 常见问题

### 缺少 placo

报错中如果出现 `Missing optional kinematics dependency placo`，说明当前 Python 环境没有安装运动学可选依赖。按本文“依赖安装”部分安装即可。

### 找不到 URDF 网格文件

如果报错提示 mesh asset 找不到，通常是只下载了 `.urdf`，没有下载旁边的 `assets/` STL 文件夹。请下载完整的 `Simulation/SO101` 目录，并保持 URDF 内引用的相对路径不变。

### move 没有移动机械臂

如果没有传入 `--connect`，`move` 只会计算 IK 并提示未发送硬件命令。需要连接真机时，请提供 `--connect --robot-type ... --port ... --id ...`。

### read 或 send 提示 requires --connect

`read` 和 `send` 都必须连接真实机器人后才能执行。干跑模式下只能使用 `joints`、`fk`、`ik` 等纯计算命令。

### IK 结果看起来不合理

优先检查以下内容：

- `--units` 是否和输入坐标一致。
- `--target-frame-name` 是否确实是期望末端 frame。
- URDF 的关节顺序是否符合机械臂实际顺序。
- 是否需要用 `--joint-names` 显式指定参与求解的关节。
- 当前关节猜测值是否接近真实姿态。真机模式下建议先执行 `read`。

## 安全建议

真实机械臂测试时，先用 `ik` 看目标关节角，再用小步 `move` 发送动作。第一次调试时建议使用 `--max-relative-target 5` 或更小值，并逐步扩大运动范围。

## 在自己的程序中调用正/逆运动学

如果以后要基于正逆运动学编写自己的控制程序，推荐直接调用 `lerobot.model.kinematics.RobotKinematics`，而不是从 `kinematics_lab.py` 里复制交互式命令循环。`kinematics_lab.py` 更像一个调试工具；真正可复用的 FK/IK 封装在 `RobotKinematics` 里。

### 基本导入

```python
import numpy as np

from lerobot.model.kinematics import RobotKinematics
```

初始化时需要传入 URDF 路径和末端坐标系名称：

```python
kinematics = RobotKinematics(
    urdf_path="assets/urdf/so101_new_calib.urdf",
    target_frame_name="gripper_frame_link",
)
```

如果 URDF 里有额外关节，或者你希望明确指定参与求解的关节顺序，可以传入 `joint_names`：

```python
kinematics = RobotKinematics(
    urdf_path="assets/urdf/so101_new_calib.urdf",
    target_frame_name="gripper_frame_link",
    joint_names=[
        "Rotation",
        "Pitch",
        "Elbow",
        "Wrist_Pitch",
        "Wrist_Roll",
    ],
)
```

实际关节名请以你的 URDF 文件为准，可以通过搜索 `<joint name=` 查看。

### 调用正运动学 FK

正运动学输入关节角，输出末端位姿矩阵。关节角单位是度。

```python
current_joints_deg = np.array([0, -45, 45, 0, 0, 30], dtype=float)

ee_pose = kinematics.forward_kinematics(current_joints_deg)

print("末端位置 xyz，单位 m:", ee_pose[:3, 3])
print("末端 4x4 变换矩阵:")
print(ee_pose)
```

返回的 `ee_pose` 是一个 `4 x 4` 齐次变换矩阵：

```text
[[R11 R12 R13 x]
 [R21 R22 R23 y]
 [R31 R32 R33 z]
 [0   0   0   1]]
```

其中左上角 `3 x 3` 是末端姿态旋转矩阵，最后一列前三个值是末端位置 `x, y, z`，单位为米。

### 调用逆运动学 IK

逆运动学输入当前关节角和目标末端位姿，输出新的关节角。关节输入和输出单位都是度。

常见做法是先用 FK 获取当前末端姿态，然后只修改目标位置，让 IK 保持当前姿态：

```python
current_joints_deg = np.array([0, -45, 45, 0, 0, 30], dtype=float)

current_pose = kinematics.forward_kinematics(current_joints_deg)
target_pose = current_pose.copy()

# 目标位置，单位 m。这里等价于 x=180 mm, y=0 mm, z=120 mm。
target_pose[:3, 3] = np.array([0.180, 0.000, 0.120])

target_joints_deg = kinematics.inverse_kinematics(
    current_joint_pos=current_joints_deg,
    desired_ee_pose=target_pose,
)

print("IK 求解得到的关节角，单位 deg:", target_joints_deg)
```

如果 `current_joints_deg` 中包含夹爪等额外关节，而 `joint_names` 只包含机械臂运动学关节，`inverse_kinematics` 会保留额外关节的原值。

### 只约束位置，弱化姿态

`inverse_kinematics` 支持调整位置和姿态权重：

```python
target_joints_deg = kinematics.inverse_kinematics(
    current_joint_pos=current_joints_deg,
    desired_ee_pose=target_pose,
    position_weight=1.0,
    orientation_weight=0.0,
)
```

当 `orientation_weight=0.0` 时，IK 基本只关心末端位置，不强制保持姿态。对于自由度较少、姿态约束容易导致无解或结果不稳定的机械臂，这种方式更容易得到可用的位置解。

### 毫米和米的换算

`RobotKinematics` 内部使用 URDF 的标准单位，也就是米。自己写程序时，如果用户输入是毫米，需要手动换算：

```python
target_xyz_mm = np.array([180, 0, 120], dtype=float)
target_xyz_m = target_xyz_mm / 1000.0

target_pose = kinematics.forward_kinematics(current_joints_deg).copy()
target_pose[:3, 3] = target_xyz_m
```

反过来，如果想把 FK 输出显示成毫米：

```python
ee_pose = kinematics.forward_kinematics(current_joints_deg)
ee_xyz_mm = ee_pose[:3, 3] * 1000.0
print(ee_xyz_mm)
```

### 连接真机并发送 IK 结果

如果要在程序中连接真实 SO100/SO101 follower 机械臂，可以参考 `kinematics_lab.py` 中的 `_make_follower_robot`、`_read_robot_joints` 和 `_joint_action`。核心流程是：

```python
from lerobot.robots import make_robot_from_config
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig

config = SO101FollowerConfig(
    port="COM5",
    id="classroom_so101",
    calibration_dir=None,
    max_relative_target=5,
    use_degrees=True,
)

robot = make_robot_from_config(config)
robot.connect(calibrate=True)

try:
    observation = robot.get_observation()
    current_joints_deg = np.array(
        [observation[f"{motor}.pos"] for motor in robot.bus.motors],
        dtype=float,
    )

    current_pose = kinematics.forward_kinematics(current_joints_deg)
    target_pose = current_pose.copy()
    target_pose[:3, 3] = np.array([0.180, 0.000, 0.120])

    target_joints_deg = kinematics.inverse_kinematics(
        current_joint_pos=current_joints_deg,
        desired_ee_pose=target_pose,
    )

    action = {
        f"{motor}.pos": float(target_joints_deg[i])
        for i, motor in enumerate(robot.bus.motors)
    }
    robot.send_action(action)
finally:
    if robot.is_connected:
        robot.disconnect()
```

SO100 follower 的写法类似，只需要把配置类换成：

```python
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
```

并创建 `SO100FollowerConfig(...)`。

### 推荐程序结构

实际写控制程序时，建议把逻辑分成三层：

```text
输入层
    读取键盘、手柄、轨迹文件、视觉目标或上层策略输出。

运动学层
    用 RobotKinematics 做 FK/IK，负责坐标单位、目标位姿和关节角转换。

执行层
    连接 LeRobot robot，读取 observation，发送 action，并处理安全限制。
```

这样做的好处是：干跑时可以只运行输入层和运动学层；接入真机时再打开执行层。调试会清楚很多，也更容易避免把错误目标直接发给硬件。

### 一个最小可复用函数

下面这个函数把“当前关节角 + 目标 xyz 毫米”转换成目标关节角：

```python
def solve_ik_xyz_mm(
    kinematics: RobotKinematics,
    current_joints_deg: np.ndarray,
    target_xyz_mm: np.ndarray,
) -> np.ndarray:
    current_pose = kinematics.forward_kinematics(current_joints_deg)
    target_pose = current_pose.copy()
    target_pose[:3, 3] = target_xyz_mm / 1000.0

    return kinematics.inverse_kinematics(
        current_joint_pos=current_joints_deg,
        desired_ee_pose=target_pose,
    )
```

使用示例：

```python
target_joints_deg = solve_ik_xyz_mm(
    kinematics=kinematics,
    current_joints_deg=np.array([0, -45, 45, 0, 0, 30], dtype=float),
    target_xyz_mm=np.array([180, 0, 120], dtype=float),
)
```

建议先打印 `target_joints_deg` 并用干跑验证，再连接真实机械臂发送。
