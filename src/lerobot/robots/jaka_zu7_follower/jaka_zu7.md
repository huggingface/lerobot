# JAKA Zu 7 Follower

把 JAKA Zu 7 六轴机械臂作为 LeRobot 从臂(follower)。关节 key 与 SO-ARM101 主臂的 6 个 arm 关节一致(`shoulder_pan` / `heavy_joint` / `shoulder_lift` / `elbow_flex` / `wrist_flex` / `wrist_roll`),直接 1:1 映射到 JAKA 的 J1..J6,用 `joint_signs` / `joint_offsets_deg` 修正方向与零位。主臂的 `gripper.pos` 被忽略(本 JAKA 未配夹爪)。

## 架构:两进程 socket 桥接

JAKA 官方 `jkrc` SDK 只能在 Python ≤3.10 import,而 LeRobot 要求 ≥3.12,两者**不能同进程**。因此:

- **桥接 server**(Python 3.10,conda env `jaka`):`jaka_bridge_server.py`,持有 `jkrc` SDK,通过 localhost TCP 提供「读关节位置 / 写关节目标」。
- **follower**(Python 3.12,conda env `lerobot`):`jaka_zu7_follower.py`,是 socket 客户端,**不 import jkrc**。

位置统一用「度」表示,deg ↔ rad 的转换在 server 的 SDK 边界完成。

## 交互式验证(主臂未就绪时)

主臂机械尚未完成、暂时无法摇操时,可以用 `jaka_interactive.py` 先验证桥接链路能像之前的 `jaka_zu7_control.py` 一样实时控制 JAKA 从臂。运行后给两个选项:**[1] 摇操** / **[2] 单独控制 JAKA 从臂**。

```powershell
conda activate jaka
cd E:\A_practice3\lerobot-main\lerobot-main\src\lerobot\robots\jaka_zu7_follower
python jaka_interactive.py --ip 10.5.5.100
```

- 选项 **2** 会自动拉起 `jaka_bridge_server.py` 子进程(日志在 `%TEMP%\jaka_bridge_server.log`),连接后进入交互式命令循环,命令集与 `jaka_zu7_control.py` 对齐(`status` / `move` / `servo` / `sinTest` / `servo_vr` / `abort` / `clear` 等)。`sinTest` 的 8ms 紧循环跑在 server 端(3.10,本地),保持与原测试代码一致的节拍精度。
- 选项 **1** 只打印摇操的启动命令(需主臂就绪)。
- 若已在别处手动起了 server,加 `--no-spawn` 直接连接。

## 启动

### 1) 先起桥接 server(Python 3.10)

```powershell
conda activate jaka
cd E:\A_practice3\lerobot-main\lerobot-main\src\lerobot\robots\jaka_zu7_follower
python jaka_bridge_server.py --ip 10.5.5.100 --sdk-dir "E:\A_practice3\jaka_sdk\Windows\python3\x64"
```

server 依次执行 login → power_on → enable_robot → 开伺服 + NLF 限速(默认 60°/s),然后监听 `127.0.0.1:10086` 等待连接。

### 2) 再起遥操(Python 3.12)

```powershell
conda activate lerobot
cd E:\A_practice3\lerobot-main\lerobot-main
lerobot-teleoperate `
  --robot.type=jaka_zu7_follower `
  --robot.bridge_host=127.0.0.1 `
  --robot.bridge_port=10086 `
  --robot.max_joint_velocity_deg_s=15 `
  --teleop.type=so100_leader `
  --teleop.port=COMx `
  --fps=60
```

## 关键配置

- `bridge_host` / `bridge_port`:桥接 server 地址(默认 `127.0.0.1:10086`)。
- `max_joint_velocity_deg_s` / `max_joint_acceleration_deg_s2`:lerobot 端软限速(度 / 度每秒平方)。首次联调先用低值(5~15),确认映射正确后再加大。
- `joint_signs` / `joint_offsets_deg`:SO 关节 → JAKA 关节的 1:1 映射修正,`controller_deg = sign * leader_deg + offset`,各 6 个值,按 JAKA J1..J6 顺序。默认全 `1` / `0`,需在真实设备上实测标定。

## 安全

- 首次联调用低 `max_joint_velocity_deg_s`,急停可及,逐个确认主臂关节驱动的是预期的 JAKA 轴与方向。
- server 端 NLF 限速(`--vr`,默认 60°/s)是硬件兜底;lerobot 端软限速是精细控制,两层叠加更安全。
- 退伺服时 `servo_move_enable(False)` 返回 -61 是 SDK 误报(实际已退出),server 已按此处理。
