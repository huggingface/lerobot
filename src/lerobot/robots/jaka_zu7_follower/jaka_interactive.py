#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""JAKA Zu 7 交互式控制台:摇操 / 单独控制从臂,二选一。

目的:主臂(SO 领导臂)机械尚未完成、暂时无法摇操时,先验证新工程的桥接链路
能像之前的 jaka_zu7_control.py 一样实时控制 JAKA 从臂。

运行环境:Python 3.10(jaka conda env)。「单独控制」需要在此环境拉起
jaka_bridge_server.py 子进程(server 依赖 jkrc SDK)。本脚本自身只用标准库 socket,
但默认用 sys.executable 拉起 server,所以请在 jaka env 下运行(或用 --server-python
指定 3.10 解释器)。

用法:
  conda activate jaka
  python jaka_interactive.py --ip 10.5.5.100
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time

JOINT_NAMES = ("J1", "J2", "J3", "J4", "J5", "J6")


class BridgeClient:
    """socket 客户端,连接 jaka_bridge_server.py,发 JSON 命令、读响应。"""

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self._sock = None
        self._recv = None

    def connect(self):
        self._sock = socket.create_connection((self.host, self.port), timeout=5.0)
        # 阻塞读:长命令(sine/servo_move/joint_move)可能持续数十秒,
        # 不能设短读超时;中断靠 Ctrl+C -> abort。
        self._sock.settimeout(None)
        self._recv = self._sock.makefile("rb")

    def close(self):
        if self._sock is not None:
            try:
                self._request({"cmd": "exit"})
            except Exception:
                pass
        for obj in (self._recv, self._sock):
            try:
                if obj is not None:
                    obj.close()
            except Exception:
                pass
        self._recv = None
        self._sock = None

    def _request(self, req, interruptible=False):
        if self._sock is None:
            raise RuntimeError("尚未连接 bridge server")
        self._sock.sendall((json.dumps(req) + "\n").encode("utf-8"))
        while True:
            try:
                line = self._recv.readline()
                break
            except KeyboardInterrupt:
                if not interruptible:
                    raise
                # 长命令运行中:向 server 发 abort,让它回位并返回结果。
                try:
                    self._sock.sendall(b'{"cmd":"abort"}\n')
                    print("\n[中止] 已请求 server 中止当前运动并回位,请稍候 ...")
                except Exception:
                    pass
        if not line:
            raise RuntimeError("bridge server 连接已关闭")
        resp = json.loads(line.decode("utf-8"))
        if resp.get("status") == "error":
            raise RuntimeError(f"bridge 错误: {resp.get('error')}")
        return resp

    def joints(self):
        return self._request({"cmd": "get"})["joints"]

    def status(self):
        return self._request({"cmd": "status"})


# ---- 交互式命令 -------------------------------------------------------------

HELP = """可用命令(与 jaka_zu7_control.py 对齐,命令通过桥接 server 下发):
  ---- 状态读取 ----
  status                 读取并打印机械臂状态(急停/限位/关节/末端/伺服/速度)
  joints                 打印关节位置(度)
  tcp                    打印末端位姿(mm + 度)

  ---- 点到点运动(慢速,阻塞)----
  move  <6个角度>        所有关节绝对运动(度),例: move 0 0 0 0 0 0
  rel   <6个增量>        所有关节增量运动(度),例: rel 0 0 0 0 5 0
  joint <n> <deg>        只动第 n 个关节到绝对角度(度),例: joint 6 30
  jrel  <n> <增量>       只动第 n 个关节一个增量(度),例: jrel 6 5

  ---- 实时伺服 ----
  servo_on / servo_off   开启/关闭伺服模式
  servo <6个角度>        所有关节伺服平滑到位,例: servo 0 0 0 0 0 30
  sj    <n> <deg>        只动第 n 个关节伺服到位(其余保持),例: sj 6 30
  sinTest [j] [amp] [freq] [s]  正弦波实时跟随测试(默认 joint5、±5°、0.5Hz、30s),
                              统计节拍+跟随误差;运行中 Ctrl+C 可安全回位

  ---- 速度/安全 ----
  speed [rad/s]          查看/设置 joint_move 速度(默认 0.2 rad/s)
  servo_vr [deg/s]       查看/设置伺服限速(默认 15°/s)
  abort                  中止当前运动(退出卡住的伺服/运动状态)
  clear                  清除控制器错误/报警(保护性停止后调用)

  help / quit
"""


def _parse_n(args, n, name):
    if len(args) != n:
        raise ValueError(f"命令 '{name}' 需要 {n} 个数值,收到 {len(args)} 个。")
    try:
        return [float(a) for a in args]
    except ValueError:
        raise ValueError(f"命令 '{name}' 的参数必须是数字。") from None


def _parse_jn(s):
    n = int(s)
    if n < 1 or n > 6:
        raise ValueError("关节编号必须在 1..6 之间。")
    return n


def _fmt_joints(joints):
    return "[" + ", ".join(f"{j:8.2f}" for j in joints) + "]"


def _cmd_status(client):
    st = client.status()
    print("---- 机械臂当前状态 ----")
    print(f"  急停 estop     : {st['estop']}")
    print(f"  限位 on_limit  : {st['on_limit']}")
    print(f"  到位 in_pos    : {st['in_pos']}")
    print(f"  运动倍率       : {st['rapidrate'] * 100:.0f}%")
    print(f"  伺服模式       : {'开' if st['servo_on'] else '关'}")
    print(f"  关节位置(度)   : {_fmt_joints(st['joints'])}")
    tcp = st["tcp"]
    print(f"  末端位姿       : x={tcp[0]:7.2f} mm, y={tcp[1]:7.2f} mm, z={tcp[2]:7.2f} mm, "
          f"rx={tcp[3]:7.2f}°, ry={tcp[4]:7.2f}°, rz={tcp[5]:7.2f}°")
    print(f"  速度配置       : joint_move={st['speed']} rad/s, 伺服限速={st['vr']:.1f}°/s")


def _cmd_sine(client, parts):
    if len(parts) > 5:
        raise ValueError("用法: sinTest [关节号1-6] [幅度deg] [频率Hz] [时长s]")
    joint = _parse_jn(parts[1]) if len(parts) > 1 else 5
    amp = float(parts[2]) if len(parts) > 2 else 5.0
    freq = float(parts[3]) if len(parts) > 3 else 0.5
    duration = float(parts[4]) if len(parts) > 4 else 30.0
    print(f"\n===== 正弦跟随测试 =====")
    print(f"joint{joint},幅度 ±{amp:.1f}°,频率 {freq}Hz,时长 {duration}s")
    print("测试中,请勿靠近机械臂。Ctrl+C 可中断(会安全回位)。")
    resp = client._request(
        {"cmd": "sine", "joint": joint, "amp": amp, "freq": freq, "duration": duration},
        interruptible=True,
    )
    if resp.get("warn"):
        print(f"[警告] {resp['warn']}")
    if resp.get("aborted"):
        print("[已中止] 测试被提前中断,已回位。")
    print("\n===== 测试结果 =====")
    iv = resp.get("intervals")
    if iv:
        print(f"节拍间隔(ms): 平均 {iv['mean']} | std {iv['std']} | min {iv['min']} | max {iv['max']} | 样本 {iv['n']}")
        print("  -> 目标 8.0ms;std<2 良好,>5 说明节拍抖动明显")
    fe = resp.get("follow_errors")
    if fe:
        print(f"跟随误差(度): 平均 {fe['mean']} | std {fe['std']} | max {fe['max']} | 样本 {fe['n']}")
        print("  -> 平均误差<1° 跟得紧;若随频率升高而变大,说明滞后增大")
    print("已回位。观察:摆动是否平滑、有无振动异响。")


def run_interactive(client):
    print(HELP)
    while True:
        try:
            raw = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue
        parts = raw.split()
        cmd = parts[0].lower()
        try:
            if cmd in ("quit", "q", "exit"):
                break
            elif cmd == "help":
                print(HELP)
            elif cmd == "status":
                _cmd_status(client)
            elif cmd == "joints":
                print("关节位置(度): " + _fmt_joints(client.joints()))
            elif cmd == "tcp":
                tcp = client.status()["tcp"]
                print("末端位姿: x=%.2f mm, y=%.2f mm, z=%.2f mm, rx=%.2f°, ry=%.2f°, rz=%.2f°"
                      % tuple(tcp))
            elif cmd == "move":
                client._request({"cmd": "joint_move", "joints": _parse_n(parts[1:], 6, "move"), "mode": 0})
                print("  joint_move 已完成")
            elif cmd == "rel":
                client._request({"cmd": "joint_move", "joints": _parse_n(parts[1:], 6, "rel"), "mode": 1})
                print("  joint_move(增量) 已完成")
            elif cmd == "joint":
                if len(parts) != 3:
                    raise ValueError("用法: joint <关节号1-6> <角度deg>")
                n = _parse_jn(parts[1])
                target = client.joints()
                target[n - 1] = float(parts[2])
                print(f"[单关节] joint_{n} -> {parts[2]}°(其余保持)")
                client._request({"cmd": "joint_move", "joints": target, "mode": 0})
                print("  joint_move 已完成")
            elif cmd == "jrel":
                if len(parts) != 3:
                    raise ValueError("用法: jrel <关节号1-6> <增量deg>")
                n = _parse_jn(parts[1])
                target = client.joints()
                target[n - 1] = target[n - 1] + float(parts[2])
                print(f"[单关节] joint_{n} 增量 {parts[2]}° -> {target[n - 1]:.2f}°")
                client._request({"cmd": "joint_move", "joints": target, "mode": 0})
                print("  joint_move 已完成")
            elif cmd == "servo_on":
                client._request({"cmd": "servo_on"})
                print("  伺服模式已开启")
            elif cmd == "servo_off":
                client._request({"cmd": "servo_off"})
                print("  伺服模式已关闭")
            elif cmd == "servo":
                client._request({"cmd": "servo_move", "joints": _parse_n(parts[1:], 6, "servo")},
                                interruptible=True)
                print("  servo 到位")
            elif cmd == "sj":
                if len(parts) != 3:
                    raise ValueError("用法: sj <关节号1-6> <角度deg>")
                n = _parse_jn(parts[1])
                target = client.joints()
                target[n - 1] = float(parts[2])
                client._request({"cmd": "servo_move", "joints": target}, interruptible=True)
                print("  servo 到位")
            elif cmd == "sintest":
                _cmd_sine(client, parts)
            elif cmd == "speed":
                if len(parts) > 1:
                    client._request({"cmd": "set_speed", "speed": float(parts[1])})
                    print(f"  joint_move 速度已设为 {float(parts[1])} rad/s")
                else:
                    print(f"当前 joint_move 速度: {client.status()['speed']} rad/s")
            elif cmd == "servo_vr":
                if len(parts) > 1:
                    resp = client._request({"cmd": "set_vr", "vr": float(parts[1])})
                    print(f"  伺服限速已设为 {resp['vr']:.1f}°/s")
                else:
                    print(f"当前伺服限速: {client.status()['vr']:.1f}°/s")
            elif cmd == "abort":
                client._request({"cmd": "abort"})
                print("  已发送中止运动指令")
            elif cmd == "clear":
                client._request({"cmd": "clear"})
                print("  已清除控制器错误")
            else:
                print(f"未知命令 '{cmd}',输入 help 查看帮助。")
        except Exception as e:
            print(f"[错误] {e}")


# ---- server 子进程管理 ------------------------------------------------------

def _server_log_tail(log_path, n=25):
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return "(无日志)"


def _spawn_server(args):
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jaka_bridge_server.py")
    log_path = os.path.join(tempfile.gettempdir(), "jaka_bridge_server.log")
    cmd = [
        args.server_python,
        script,
        "--ip", args.ip,
        "--sdk-dir", args.sdk_dir,
        "--host", args.host,
        "--port", str(args.port),
        "--vr", str(args.vr),
        "--speed", str(args.speed),
        "--no-servo-start",
    ]
    logf = open(log_path, "wb")
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"  # 让子进程 stdout 用 UTF-8,日志里的中文可读
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    return proc, logf, log_path


def _connect_with_retry(client, proc, log_path, timeout=30.0):
    """反复尝试建立真实连接,直到成功或超时。连接成功即就绪。

    注意:不能用一个「探测连接」去试探端口是否就绪 —— 那会被 server 当作一个
    空客户端 accept 后立刻断开,从而误触发 server 退出(导致后续连接 10061 被拒)。
    """
    deadline = time.time() + timeout
    while True:
        if proc.poll() is not None:
            raise RuntimeError(
                f"bridge server 启动失败(exit {proc.returncode})。日志({log_path}):\n{_server_log_tail(log_path)}"
            )
        try:
            client.connect()
            return
        except OSError:
            if time.time() >= deadline:
                raise RuntimeError(f"等待 bridge server 就绪超时。日志({log_path}):\n{_server_log_tail(log_path)}")
            time.sleep(0.2)


def _teleop_instructions(args):
    print("""
===== 摇操(SO 主臂 -> JAKA 从臂)=====
主臂机械尚未完成,暂时无法摇操。机械就绪后按下面两步启动:

  [终端 A] 起桥接 server(Python 3.10,jaka env):
      conda activate jaka
      python jaka_bridge_server.py --ip {ip} --sdk-dir "{sdk}"

  [终端 B] 起摇操(Python 3.12,lerobot env):
      conda activate lerobot
      cd E:\\A_practice3\\lerobot-main\\lerobot-main
      lerobot-teleoperate --robot.type=jaka_zu7_follower --teleop.type=so100_leader --teleop.port=COMx --fps=60
""".format(ip=args.ip, sdk=args.sdk_dir))


def run_standalone(args):
    """拉起 server 子进程 -> 连接 -> 进入交互式命令循环。"""
    proc = None
    logf = None
    log_path = None
    client = BridgeClient(args.host, args.port)
    if args.no_spawn:
        print(f"使用已运行的 bridge server({args.host}:{args.port}),不拉起子进程。")
        client.connect()
    else:
        print(f"正在拉起 bridge server 子进程({args.server_python}) ...")
        proc, logf, log_path = _spawn_server(args)
        print(f"  server 日志 → {log_path}")
        try:
            _connect_with_retry(client, proc, log_path)
        except Exception as e:
            print(f"[错误] {e}")
            if proc is not None and proc.poll() is None:
                proc.terminate()
            if logf is not None:
                logf.close()
            return
    print("已连接 bridge server,进入交互式控制。\n")
    try:
        run_interactive(client)
    finally:
        client.close()
        if proc is not None:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.terminate()
            if logf is not None:
                logf.close()


def main():
    ap = argparse.ArgumentParser(description="JAKA Zu 7 交互式控制台(摇操 / 单独控制)")
    ap.add_argument("--ip", default="10.5.5.100", help="JAKA 控制柜 IP")
    ap.add_argument("--sdk-dir", default=r"E:\A_practice3\jaka_sdk\Windows\python3\x64")
    ap.add_argument("--host", default="127.0.0.1", help="bridge server 地址")
    ap.add_argument("--port", type=int, default=10086, help="bridge server 端口")
    ap.add_argument("--vr", type=float, default=15.0, help="伺服限速(°/s,默认 15)")
    ap.add_argument("--speed", type=float, default=0.2, help="joint_move 速度(rad/s,默认 0.2)")
    ap.add_argument("--server-python", default=sys.executable,
                    help="拉起 server 用的解释器(默认 sys.executable,需为 3.10/jaka env)")
    ap.add_argument("--no-spawn", action="store_true", help="不拉起子进程,连接已运行的 server")
    args = ap.parse_args()

    print("=" * 60)
    print(" JAKA Zu 7 从臂控制台")
    print("=" * 60)
    while True:
        try:
            choice = input("请选择:\n  [1] 摇操(SO 主臂 -> JAKA,主臂就绪后)\n  [2] 单独控制 JAKA 从臂\n  [q] 退出\n>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if choice in ("1",):
            _teleop_instructions(args)
            input("\n按回车返回主菜单 ...")
        elif choice in ("2",):
            run_standalone(args)
            print("\n已退出单独控制。")
        elif choice in ("q", "quit", "exit", ""):
            break
        else:
            print("无效选项,请重新选择。\n")


if __name__ == "__main__":
    main()
