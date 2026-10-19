#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""JAKA Zu 7 摇操桥接 server(必须在 Python 3.10 环境运行,jaka conda env)。

独立进程,持有 jkrc SDK,通过 localhost TCP 为两类客户端服务:
  1) lerobot(Python 3.12)里的 follower 摇操客户端:只用 get / set / exit;
  2) jaka_interactive.py 交互式控制台(单独控制 JAKA):用下面更丰富的命令集。

解决 jkrc 只能在 3.10、lerobot 强制 3.12 的版本冲突。

复用 jaka_zu7_control.py 已实测验证的调用方式:
  - timeBeginPeriod(1):Windows 定时器精度,缺失会导致 servo 跟随振动
  - login -> power_on -> enable_robot(伺服按需开启)
  - servo_j(rad, MOVE_ABS=0):单 8ms 插补周期(servo_j_extend 本控制柜不支持,返回错误码 2)
  - servo_move_enable(False) 返回 -61 是误报,实际已退出
  - get_joint_position() 返回 (errno, joints_rad)
  - joint_move(rad, mode, is_block, speed) 4 参数

协议(换行分隔 JSON):
  客户端 -> server:
    {"cmd":"get"}                               读 6 关节位置(度)
    {"cmd":"set","joints":[deg*6]}              发一个伺服目标(单 8ms 拍,自动开伺服)
    {"cmd":"status"}                            读完整状态(关节/末端/急停/限位/伺服/速度)
    {"cmd":"joint_move","joints":[deg*6],"mode":0|1,"speed":rad_s}  点到点(阻塞)
    {"cmd":"servo_on"} / {"cmd":"servo_off"}    开/关伺服模式
    {"cmd":"servo_move","joints":[deg*6],"duration":s}  伺服平滑到位(duration 缺省自动)
    {"cmd":"sine","joint":n,"amp":deg,"freq":Hz,"duration":s}  正弦跟随,返回节拍/误差统计
    {"cmd":"set_vr","vr":deg_s}                 设置伺服限速
    {"cmd":"set_speed","speed":rad_s}           设置 joint_move 速度
    {"cmd":"abort"} / {"cmd":"clear"}           中止运动 / 清除错误
    {"cmd":"exit"}                              退出伺服并断开
  server -> 客户端:
    {"joints":[deg*6]}                                     (对应 get)
    {"status":"ok","servo_on":true,...}                     (对应 set/servo_*/joint_move/set_*)
    {"status":"ok","joints":[...],"tcp":[...],...}          (对应 status)
    {"status":"ok","aborted":false,"intervals":{...},"follow_errors":{...}}  (对应 sine)
    {"status":"bye"}                                        (对应 exit)
    {"status":"error","error":"..."}

用法:
  python jaka_bridge_server.py --ip 10.5.5.100 \
      --sdk-dir r"E:\A_practice3\jaka_sdk\Windows\python3\x64"
"""

import argparse
import json
import math
import os
import select
import socket
import sys
import time

MOVE_ABS = 0   # MoveMode: 绝对
MOVE_INCR = 1  # MoveMode: 增量

# 关节限位(弧度),来自官方 URDF。
JOINT_LIMITS_RAD = [
    (-6.28, 6.28),   # joint_1
    (-1.48, 4.62),   # joint_2
    (-3.05, 3.05),   # joint_3
    (-1.48, 4.62),   # joint_4
    (-6.28, 6.28),   # joint_5
    (-6.28, 6.28),   # joint_6
]

SERVO_PERIOD_S = 0.008  # servo_j 单插补周期


def _setup_jkrc(sdk_dir):
    """把 SDK 目录加入 sys.path / DLL 搜索路径,再导入 jkrc。"""
    sdk_dir = sdk_dir or os.environ.get("JAKA_SDK_DIR")
    if not sdk_dir:
        raise SystemExit("未指定 jkrc SDK 目录:用 --sdk-dir 或环境变量 JAKA_SDK_DIR。")
    sdk_dir = os.path.abspath(sdk_dir)
    if not os.path.isfile(os.path.join(sdk_dir, "jkrc.pyd")):
        raise SystemExit(f"在 '{sdk_dir}' 下找不到 jkrc.pyd,请检查 --sdk-dir。")
    if sdk_dir not in sys.path:
        sys.path.insert(0, sdk_dir)
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(sdk_dir)  # 让 jkrc.pyd 找到同目录 jakaAPI.dll
    import jkrc
    return jkrc


def _result(ret):
    """jkrc 返回统一成 (errno, payload)。"""
    if isinstance(ret, (tuple, list)):
        errno = int(ret[0]) if ret else 0
        return errno, (ret[1] if len(ret) >= 2 else None)
    return int(ret), None


def _check(ret, op):
    errno, payload = _result(ret)
    if errno != 0:
        raise RuntimeError(f"JAKA SDK '{op}' 失败,错误码 {errno}")
    return payload


def _stats(values):
    """返回 {n, mean, std, min, max},空序列返回 None。"""
    if not values:
        return None
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return {
        "n": n,
        "mean": round(mean, 2),
        "std": round(var ** 0.5, 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


def _clamp_deg(deg_list):
    """按关节限位裁剪目标(度),非法值抛异常。"""
    out = []
    for i, (d, (lo_r, hi_r)) in enumerate(zip(deg_list, JOINT_LIMITS_RAD), start=1):
        lo, hi = math.degrees(lo_r), math.degrees(hi_r)
        if not math.isfinite(d):
            raise ValueError(f"joint_{i} 目标含非法值 (NaN/inf)")
        if d < lo or d > hi:
            print(f"  [bridge 警告] joint_{i} 目标 {d:.2f}° 超出限位,已裁剪")
            d = max(lo, min(hi, d))
        out.append(d)
    return out


def _read_joints_deg(rc):
    errno, joints = _result(rc.get_joint_position())
    if errno != 0:
        raise RuntimeError(f"get_joint_position 错误码 {errno}")
    return [math.degrees(float(j)) for j in joints]


def _apply_nlf(rc, vr):
    _check(rc.servo_move_use_joint_NLF(vr, vr * 3.0, vr * 10.0), "servo_move_use_joint_NLF")


def _ensure_servo_on(rc, state):
    if state["servo_on"]:
        return
    _check(rc.servo_move_enable(True), "servo_move_enable(True)")
    _apply_nlf(rc, state["vr"])
    state["servo_on"] = True
    print("[bridge] 伺服模式已开启")


def _servo_off(rc, state):
    if not state["servo_on"]:
        return
    try:
        rc.motion_abort()
    except Exception:
        pass
    errno, _ = _result(rc.servo_move_enable(False))
    if errno in (0, -61):
        state["servo_on"] = False
        print("[bridge] 伺服模式已退出(-61 为误报,可忽略)" if errno == -61 else "[bridge] 伺服模式已关闭")
    else:
        raise RuntimeError(f"关闭伺服模式失败,错误码 {errno}")


def _abort_pending(conn, buf_holder):
    """非阻塞读 socket,把新字节并入 buf_holder;若收到 abort 命令或连接关闭则返回 True。"""
    try:
        ready, _, _ = select.select([conn], [], [], 0.0)
    except (OSError, ValueError):
        return True
    if not ready:
        return False
    data = conn.recv(4096)
    if not data:
        return True
    buf_holder[0] += data
    while b"\n" in buf_holder[0]:
        line, buf_holder[0] = buf_holder[0].split(b"\n", 1)
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line.decode("utf-8"))
            if req.get("cmd") == "abort":
                return True
        except Exception:
            continue
    return False


def _servo_move_loop(rc, state, conn, buf_holder, target_deg, duration):
    """伺服平滑走到 target(度),按 servo_vr 限速线性插值,支持运行中 abort。返回是否被中止。"""
    _ensure_servo_on(rc, state)
    cur = _read_joints_deg(rc)
    target = list(target_deg)
    for i, (d, (lo_r, hi_r)) in enumerate(zip(target, JOINT_LIMITS_RAD)):
        lo, hi = math.degrees(lo_r), math.degrees(hi_r)
        if not math.isfinite(d):
            raise ValueError(f"joint_{i + 1} 目标含非法值 (NaN/inf)")
        if d < lo or d > hi:
            target[i] = max(lo, min(hi, d))
    max_delta = max(abs(t - c) for t, c in zip(target, cur))
    if max_delta < 0.01:
        return False
    if duration is None or duration <= 0:
        duration = max_delta / max(state["vr"], 1e-6) + 0.8
    n = max(2, int(duration / SERVO_PERIOD_S))
    t0 = time.perf_counter()
    for i in range(1, n + 1):
        if _abort_pending(conn, buf_holder):
            return True
        alpha = i / n
        way = [c + (t - c) * alpha for c, t in zip(cur, target)]
        _check(rc.servo_j([math.radians(w) for w in way], MOVE_ABS), "servo_j")
        deadline = t0 + i * SERVO_PERIOD_S
        now = time.perf_counter()
        if now < deadline:
            time.sleep(deadline - now)
        while time.perf_counter() < deadline:
            pass
    return False


# ---- 命令处理 ---------------------------------------------------------------


def _h_get(rc):
    return {"joints": _read_joints_deg(rc)}


def _h_set(rc, state, req):
    joints = req.get("joints")
    if not isinstance(joints, (list, tuple)) or len(joints) != 6:
        n = len(joints) if isinstance(joints, (list, tuple)) else "非序列"
        return {"status": "error", "error": f"joints 必须是 6 个元素,收到 {n}"}
    _ensure_servo_on(rc, state)
    rad = [math.radians(float(j)) for j in joints]
    errno, _ = _result(rc.servo_j(rad, MOVE_ABS))
    if errno != 0:
        return {"status": "error", "error": f"servo_j 错误码 {errno}"}
    return {"status": "ok"}


def _h_status(rc, state):
    try:
        joints = _read_joints_deg(rc)
        tcp = _check(rc.get_tcp_position(), "get_tcp_position")
        tcp_deg = [tcp[0], tcp[1], tcp[2],
                   math.degrees(tcp[3]), math.degrees(tcp[4]), math.degrees(tcp[5])]
        estop = bool(_check(rc.is_in_estop(), "is_in_estop"))
        on_limit = bool(_check(rc.is_on_limit(), "is_on_limit"))
        in_pos = bool(_check(rc.is_in_pos(), "is_in_pos"))
        rapidrate = float(_check(rc.get_rapidrate(), "get_rapidrate"))
        return {
            "status": "ok",
            "joints": joints,
            "tcp": tcp_deg,
            "estop": estop,
            "on_limit": on_limit,
            "in_pos": in_pos,
            "rapidrate": rapidrate,
            "servo_on": state["servo_on"],
            "vr": state["vr"],
            "speed": state["speed"],
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _h_joint_move(rc, state, req):
    joints = req.get("joints")
    if not isinstance(joints, (list, tuple)) or len(joints) != 6:
        n = len(joints) if isinstance(joints, (list, tuple)) else "非序列"
        return {"status": "error", "error": f"joints 必须是 6 个元素,收到 {n}"}
    mode = int(req.get("mode", MOVE_ABS))
    speed = float(req.get("speed", state["speed"]))
    raw = [float(j) for j in joints]
    if not all(math.isfinite(v) for v in raw):
        return {"status": "error", "error": "joints 含非法值 (NaN/inf)"}
    # 绝对模式按限位裁剪;增量模式不做绝对限位裁剪。
    target = _clamp_deg(raw) if mode == MOVE_ABS else raw
    rad = [math.radians(d) for d in target]
    try:
        _check(rc.joint_move(rad, mode, True, speed), "joint_move")
    except Exception as e:
        return {"status": "error", "error": str(e)}
    return {"status": "ok"}


def _h_servo_move(rc, state, conn, buf_holder, req):
    joints = req.get("joints")
    if not isinstance(joints, (list, tuple)) or len(joints) != 6:
        n = len(joints) if isinstance(joints, (list, tuple)) else "非序列"
        return {"status": "error", "error": f"joints 必须是 6 个元素,收到 {n}"}
    target = _clamp_deg([float(j) for j in joints])
    duration = req.get("duration")
    duration = float(duration) if duration is not None else None
    aborted = _servo_move_loop(rc, state, conn, buf_holder, target, duration)
    return {"status": "ok", "aborted": aborted}


def _h_sine(rc, state, conn, buf_holder, req):
    joint = int(req.get("joint", 5))
    if not 1 <= joint <= 6:
        return {"status": "error", "error": "关节编号必须在 1..6"}
    jidx = joint - 1
    amp = float(req.get("amp", 5.0))
    freq = float(req.get("freq", 0.5))
    duration = float(req.get("duration", 30.0))

    peak_vel = amp * 2 * math.pi * freq
    warn = ""
    if peak_vel > state["vr"]:
        warn = f"峰值角速度 {peak_vel:.1f}°/s 超过伺服限速 {state['vr']:.1f}°/s,将跟不上正弦"

    _ensure_servo_on(rc, state)
    base = _read_joints_deg(rc)
    lo = math.degrees(JOINT_LIMITS_RAD[jidx][0])
    hi = math.degrees(JOINT_LIMITS_RAD[jidx][1])
    if base[jidx] + amp > hi or base[jidx] - amp < lo:
        new_amp = min(amp, hi - base[jidx], base[jidx] - lo)
        if new_amp <= 0:
            return {"status": "error",
                    "error": f"joint{joint} 当前 {base[jidx]:.2f}° 已贴近限位 [{lo:.1f},{hi:.1f}],无法摆动"}
        warn = (warn + "; " if warn else "") + f"幅度 {amp:.1f}° 超出限位余量,已裁剪为 {new_amp:.2f}°"
        amp = new_amp

    n_total = int(duration / SERVO_PERIOD_S)
    intervals = []       # 相邻 servo_j 实际间隔 (ms)
    follow_errors = []   # |目标 - 实际| (度)
    t0 = time.perf_counter()
    prev_ts = t0
    aborted = False
    try:
        for i in range(n_total):
            if _abort_pending(conn, buf_holder):
                aborted = True
                break
            t = i * SERVO_PERIOD_S
            target_deg = base[jidx] + amp * math.sin(2 * math.pi * freq * t)
            target = list(base)
            target[jidx] = target_deg
            _check(rc.servo_j([math.radians(x) for x in target], MOVE_ABS), "servo_j")

            ts = time.perf_counter()
            intervals.append((ts - prev_ts) * 1000.0)
            prev_ts = ts

            if i % 20 == 0:
                e, pos = _result(rc.get_joint_position())
                if e == 0:
                    follow_errors.append(abs(target_deg - math.degrees(pos[jidx])))

            deadline = t0 + (i + 1) * SERVO_PERIOD_S
            while time.perf_counter() < deadline:
                time.sleep(0.0002)
    finally:
        # 无论正常结束还是被中止,都平滑回位。
        try:
            _servo_move_loop(rc, state, conn, buf_holder, base, 2.0)
        except Exception as ex:
            print(f"[bridge 警告] 正弦测试回位失败:{ex}")

    return {
        "status": "ok",
        "aborted": aborted,
        "warn": warn,
        "intervals": _stats(intervals),
        "follow_errors": _stats(follow_errors),
    }


def _h_set_vr(rc, state, req):
    vr = float(req.get("vr", state["vr"]))
    state["vr"] = max(0.5, vr)
    if state["servo_on"]:
        _apply_nlf(rc, state["vr"])
    return {"status": "ok", "vr": state["vr"]}


def _h_set_speed(state, req):
    speed = float(req.get("speed", state["speed"]))
    state["speed"] = max(0.01, speed)
    return {"status": "ok", "speed": state["speed"]}


def _dispatch(rc, state, conn, buf_holder, req):
    cmd = req.get("cmd")
    if cmd == "get":
        return _h_get(rc)
    if cmd == "set":
        return _h_set(rc, state, req)
    if cmd == "status":
        return _h_status(rc, state)
    if cmd == "joint_move":
        return _h_joint_move(rc, state, req)
    if cmd == "servo_on":
        _ensure_servo_on(rc, state)
        return {"status": "ok", "servo_on": True}
    if cmd == "servo_off":
        _servo_off(rc, state)
        return {"status": "ok", "servo_on": False}
    if cmd == "servo_move":
        return _h_servo_move(rc, state, conn, buf_holder, req)
    if cmd == "sine":
        return _h_sine(rc, state, conn, buf_holder, req)
    if cmd == "set_vr":
        return _h_set_vr(rc, state, req)
    if cmd == "set_speed":
        return _h_set_speed(state, req)
    if cmd == "abort":
        _check(rc.motion_abort(), "motion_abort")
        return {"status": "ok"}
    if cmd == "clear":
        _check(rc.clear_error(), "clear_error")
        return {"status": "ok"}
    if cmd == "exit":
        return {"status": "bye"}
    return {"status": "error", "error": f"未知命令 {cmd!r}"}


def _shutdown(rc, state):
    _servo_off(rc, state)
    try:
        _check(rc.logout(), "logout")
        print("[bridge] logout 成功")
    except Exception as e:
        print(f"[bridge] logout 失败:{e}")


def main():
    # 提高 Windows 定时器精度到 1ms(否则 servo 跟随会振动)
    try:
        import ctypes
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="JAKA Zu 7 摇操桥接 server")
    ap.add_argument("--ip", default="10.5.5.100", help="JAKA 控制柜 IP")
    ap.add_argument("--sdk-dir", default=r"E:\A_practice3\jaka_sdk\Windows\python3\x64")
    ap.add_argument("--host", default="127.0.0.1", help="桥接监听地址")
    ap.add_argument("--port", type=int, default=10086, help="桥接监听端口")
    ap.add_argument("--vr", type=float, default=15.0, help="NLF 伺服限速(°/s,默认 15)")
    ap.add_argument("--speed", type=float, default=0.2, help="joint_move 速度(rad/s,默认 0.2)")
    ap.add_argument("--no-servo-start", action="store_true", help="启动时不自动开伺服(交互式单独控制用)")
    ap.add_argument("--power-on-delay", type=float, default=0.0, help="power_on 后等待秒数")
    args = ap.parse_args()

    jkrc = _setup_jkrc(args.sdk_dir)
    print(f"[bridge] 连接 JAKA 控制柜 {args.ip} ...")
    rc = jkrc.RC(args.ip)
    _check(rc.login(), "login")
    print("[bridge] login 成功")
    _check(rc.power_on(), "power_on")
    print("[bridge] power_on 成功")
    if args.power_on_delay > 0:
        time.sleep(args.power_on_delay)
    _check(rc.enable_robot(), "enable_robot")
    print("[bridge] enable_robot 成功(已上电并使能)")

    state = {"servo_on": False, "vr": args.vr, "speed": args.speed}
    if not args.no_servo_start:
        _ensure_servo_on(rc, state)
        print(f"[bridge] NLF 限速 {state['vr']:.1f}°/s")
    else:
        print("[bridge] 伺服未开启(按需开启)")

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    print(f"[bridge] 监听 {args.host}:{args.port},等待客户端连接 ...")

    conn = None
    buf_holder = [b""]
    exit_requested = False
    try:
        conn, addr = srv.accept()
        print(f"[bridge] 客户端已连接:{addr}")
        while not exit_requested:
            data = conn.recv(4096)
            if not data:
                print("[bridge] 客户端已断开")
                break
            buf_holder[0] += data
            while b"\n" in buf_holder[0]:
                line, buf_holder[0] = buf_holder[0].split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    req = json.loads(line.decode("utf-8"))
                    resp = _dispatch(rc, state, conn, buf_holder, req)
                except Exception as e:
                    resp = {"status": "error", "error": str(e)}
                try:
                    conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))
                except OSError:
                    exit_requested = True
                    break
                if resp.get("status") == "bye":
                    exit_requested = True
                    break
    except KeyboardInterrupt:
        print("\n[bridge] 收到 Ctrl+C")
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        try:
            srv.close()
        except Exception:
            pass
        _shutdown(rc, state)


if __name__ == "__main__":
    main()
