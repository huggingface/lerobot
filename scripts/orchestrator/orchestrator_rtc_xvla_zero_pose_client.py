import logging
import threading
import time
from contextlib import suppress
from dataclasses import asdict, dataclass
from pprint import pformat
from queue import Empty, Queue
from types import ModuleType

import draccus
import numpy as np
from orchestrator_rtc_xvla_client_only import (
    RTCXVLAClientOnlyConfig,
    _to_robot_client_config,
)

from lerobot.rtc_inference.robot_client import RobotClient
from lerobot.utils.import_utils import register_third_party_plugins

msvcrt: ModuleType | None
try:
    import msvcrt
except ImportError:
    msvcrt = None


@dataclass
class RTCXVLAZeroPoseClientConfig(RTCXVLAClientOnlyConfig):
    homing_duration_start: float = 8.0
    homing_duration_after_stop: float = 8.0
    gripper_home_value: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if self.homing_duration_start <= 0:
            raise ValueError("homing_duration_start must be > 0")
        if self.homing_duration_after_stop <= 0:
            raise ValueError("homing_duration_after_stop must be > 0")


@dataclass
class _RuntimeState:
    running: bool = False
    started_once: bool = False
    awaiting_disconnect: bool = False
    disconnected: bool = False
    action_receiver_thread: threading.Thread | None = None
    control_loop_thread: threading.Thread | None = None


def _keyboard_listener(command_queue: Queue[str], stop_event: threading.Event) -> None:
    """Capture keyboard commands in background to avoid prompt drift when logs are noisy."""
    if msvcrt is None:
        # Fallback for non-Windows environments.
        while not stop_event.is_set():
            try:
                line = input().strip().lower()
            except EOFError:
                return
            if not line:
                continue
            command_queue.put(line[0])
        return

    while not stop_event.is_set():
        if not msvcrt.kbhit():
            time.sleep(0.05)
            continue

        key = msvcrt.getwch()

        # Ignore extended keys (arrows, function keys, etc.).
        if key in ("\x00", "\xe0"):
            if msvcrt.kbhit():
                _ = msvcrt.getwch()
            continue

        if key in ("\r", "\n"):
            continue

        command_queue.put(key.lower())


def _to_float(value) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _extract_current_action_from_observation(
    current_obs: dict,
    joint_names: list[str],
) -> dict[str, float]:
    missing = [k for k in joint_names if k not in current_obs]
    if not missing:
        return {k: _to_float(current_obs[k]) for k in joint_names}

    state = current_obs.get("observation.state")
    if state is not None:
        if hasattr(state, "detach"):
            state = state.detach()
        if hasattr(state, "cpu"):
            state = state.cpu()
        if hasattr(state, "numpy"):
            state = state.numpy()
        state_array = np.asarray(state, dtype=np.float64).reshape(-1)
        if state_array.size == len(joint_names):
            return {k: float(state_array[i]) for i, k in enumerate(joint_names)}

    raise KeyError(f"Missing joint keys in observation and invalid observation.state fallback: {missing}")


def _home_to_zero(
    robot,
    homing_duration: float,
    start_action: dict[str, float],
    joint_names: list[str],
    gripper_home_value: float,
) -> dict[str, float]:
    print(f"[HOME] Homing to zero pose over {homing_duration:.1f}s")

    target_action = dict.fromkeys(joint_names, 0.0)
    gripper_name = next((k for k in joint_names if "gripper" in k.lower() or "jaw" in k.lower()), None)
    if gripper_name:
        target_action[gripper_name] = gripper_home_value

    hz = 50.0
    steps = max(1, int(homing_duration * hz))
    sleep_time = 1.0 / hz

    for i in range(1, steps + 1):
        alpha = i / steps
        smooth_alpha = (1.0 - np.cos(alpha * np.pi)) / 2.0

        interp_action = {
            k: start_action[k] + smooth_alpha * (target_action[k] - start_action[k]) for k in joint_names
        }
        robot.send_action(interp_action)
        time.sleep(sleep_time)

    robot.send_action(target_action)
    return target_action


def _home_robot_to_zero(robot, homing_duration: float, gripper_home_value: float) -> None:
    current_obs = robot.get_observation()
    joint_names = list(robot.action_features.keys())

    try:
        start_action = _extract_current_action_from_observation(current_obs, joint_names)
    except KeyError as exc:
        print(f"[WARN] Cannot infer current joint action from observation: {exc}")
        print("[WARN] Fallback to immediate zero target.")
        start_action = dict.fromkeys(joint_names, 0.0)

    _home_to_zero(robot, homing_duration, start_action, joint_names, gripper_home_value)


def _start_client_stream(client: RobotClient, task: str, state: _RuntimeState) -> bool:
    if state.running:
        print("[INFO] Client is already running.")
        return False
    if state.started_once:
        print("[INFO] Client stream already ran once and was stopped.")
        print("[INFO] Restart script if you want to run another stream session.")
        return False

    if not client.start():
        print("[ERROR] Failed to connect to policy server.")
        return False

    state.action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
    state.control_loop_thread = threading.Thread(
        target=client.control_loop,
        kwargs={"task": task},
        daemon=True,
    )

    state.action_receiver_thread.start()
    state.control_loop_thread.start()

    state.running = True
    state.started_once = True
    print("[RUN] Client stream started.")
    return True


def _stop_client_stream_preserve_robot(client: RobotClient, state: _RuntimeState) -> None:
    if not state.running:
        return

    print("[STOP] Stopping client stream (preserve robot connection)...")
    client.shutdown_event.set()

    if state.control_loop_thread is not None:
        state.control_loop_thread.join(timeout=2.0)
    if state.action_receiver_thread is not None:
        state.action_receiver_thread.join(timeout=2.0)

    # If receive thread is still blocked on gRPC, close channel to force unblock.
    if state.action_receiver_thread is not None and state.action_receiver_thread.is_alive():
        try:
            client.channel.close()
        except Exception as exc:
            print(f"[WARN] channel.close() failed: {exc}")
        state.action_receiver_thread.join(timeout=1.0)

    state.running = False
    print("[STOP] Client stream stopped.")


def _stop_and_disconnect(client: RobotClient, state: _RuntimeState) -> None:
    if state.disconnected:
        return

    print("[FINAL] Calling client.stop() to disconnect...")
    try:
        client.stop()
    except Exception as exc:
        print(f"[WARN] client.stop() failed: {exc}")
        # Fallbacks in case stop() fails midway.
        with suppress(Exception):
            client.channel.close()
        with suppress(Exception):
            client.robot.disconnect()

    state.running = False
    state.disconnected = True


def run_zero_pose_client_orchestrator(cfg: RTCXVLAZeroPoseClientConfig) -> None:
    client_cfg = _to_robot_client_config(cfg)
    client = RobotClient(client_cfg)
    state = _RuntimeState()
    command_queue: Queue[str] = Queue()
    keyboard_stop_event = threading.Event()
    keyboard_thread = threading.Thread(
        target=_keyboard_listener,
        args=(command_queue, keyboard_stop_event),
        daemon=True,
    )

    try:
        print("[INIT] Homing robot to zero pose before any inference...")
        _home_robot_to_zero(
            robot=client.robot,
            homing_duration=cfg.homing_duration_start,
            gripper_home_value=cfg.gripper_home_value,
        )
        print("Đã về Zero-pose")
        print("[KEY] Bấm phím c để chạy client-server.")
        print("[KEY] Bấm phím q để dừng stream và về zero-pose.")
        print("[KEY] Ctrl+C để thoát và disconnect ngay.")
        keyboard_thread.start()

        while True:
            try:
                cmd = command_queue.get(timeout=0.2)
            except Empty:
                continue

            if state.awaiting_disconnect:
                print("[INFO] Post-q input detected. Disconnecting now...")
                _stop_and_disconnect(client=client, state=state)
                return

            if cmd == "c":
                _start_client_stream(client=client, task=cfg.task, state=state)
                continue

            if cmd == "q":
                if not state.running:
                    print("[INFO] Client is not running.")
                    continue

                _stop_client_stream_preserve_robot(client=client, state=state)
                _home_robot_to_zero(
                    robot=client.robot,
                    homing_duration=cfg.homing_duration_after_stop,
                    gripper_home_value=cfg.gripper_home_value,
                )
                print("Đã về Zero-pose")
                state.awaiting_disconnect = True
                print("[INFO] Client stream is stopped.")
                print("[INFO] Nhấn Ctrl+C (ưu tiên) hoặc nhập phím bất kỳ để disconnect hoàn toàn.")
                continue

            print(f"[INFO] Ignored key: {cmd}. Use c or q.")

    except KeyboardInterrupt:
        print("\n[INFO] Shutdown signal received.")
        _stop_and_disconnect(client=client, state=state)
    finally:
        keyboard_stop_event.set()
        keyboard_thread.join(timeout=0.5)

        if not state.disconnected:
            _stop_client_stream_preserve_robot(client=client, state=state)
            _stop_and_disconnect(client=client, state=state)


@draccus.wrap()
def main(cfg: RTCXVLAZeroPoseClientConfig) -> None:
    logging.info(pformat(asdict(cfg)))
    run_zero_pose_client_orchestrator(cfg)


if __name__ == "__main__":
    register_third_party_plugins()
    main()
