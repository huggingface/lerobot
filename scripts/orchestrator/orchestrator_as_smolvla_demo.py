import threading
import time

import draccus
import numpy as np

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.robot_client import RobotClient

# =====================================================================
# ⚙️ CẤU HÌNH GRIPPER CỦA BẠN (CHỈNH TẠI ĐÂY)
# =====================================================================
# Giá trị để kẹp MỞ hoàn toàn (để thả cái lọ ra)
GRIPPER_OPEN_VALUE = 60.0  # Replace with your real hardware value (e.g. 1.5 or -1.0)

# Giá trị kẹp khi ở tư thế Zero Pose (kết thúc Demo)
GRIPPER_HOME_VALUE = 0.0  # Thường là 0.0 nếu bạn calib Zero là tư thế đóng/nghỉ
# =====================================================================


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


def _extract_current_action_from_observation(current_obs: dict, joint_names: list[str]) -> dict[str, float]:
    missing = [k for k in joint_names if k not in current_obs]
    if missing:
        raise KeyError(f"Missing joint keys in observation: {missing}")
    return {k: _to_float(current_obs[k]) for k in joint_names}


def _open_gripper_if_present(action_dict: dict[str, float], joint_names: list[str]) -> None:
    gripper_name = next((k for k in joint_names if "gripper" in k.lower() or "jaw" in k.lower()), None)
    if gripper_name:
        print(f"[PHASE 2] Opening gripper (target: {GRIPPER_OPEN_VALUE})...")
        action_dict[gripper_name] = GRIPPER_OPEN_VALUE


def _home_to_zero(
    robot, homing_duration: float, start_action: dict[str, float], joint_names: list[str]
) -> None:
    print(f"[PHASE 2] Homing to zero pose over {homing_duration:.1f}s")

    # Đích đến: Các khớp cánh tay về 0.0, Gripper về giá trị Home đã định nghĩa
    target_action = dict.fromkeys(joint_names, 0.0)
    gripper_name = next((k for k in joint_names if "gripper" in k.lower() or "jaw" in k.lower()), None)
    if gripper_name:
        target_action[gripper_name] = GRIPPER_HOME_VALUE

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


def _vla_timer_watcher(client: RobotClient, duration: float):
    print("[TIMER] Waiting for first action from AI...")
    while client.running:
        with client.latest_action_lock:
            if client.latest_action >= 0:
                break
        time.sleep(0.01)

    if not client.running:
        return

    print(f"[TIMER] First action received. Starting countdown: {duration:.1f}s...")
    time.sleep(duration)

    if client.running:
        print(f"\n[PHASE 1] Reached {duration:.1f}s. Stopping VLA stream...")
        client.shutdown_event.set()
        try:
            client.channel.close()
        except Exception as exc:
            print(f"[WARN] channel.close() failed: {exc}")


def run_full_demo(cfg: RobotClientConfig, vla_duration: float = 18.0, homing_duration: float = 10.0) -> None:
    client = RobotClient(cfg)
    try:
        print(f"\n[PHASE 1] Start VLA (Duration: {vla_duration:.1f}s after first movement)")
        if not client.start():
            print("[ERROR] Failed to connect to server.")
            return

        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
        action_receiver_thread.start()

        watcher_thread = threading.Thread(target=_vla_timer_watcher, args=(client, vla_duration), daemon=True)
        watcher_thread.start()

        try:
            client.control_loop(task=cfg.task)
        finally:
            action_receiver_thread.join(timeout=1.0)
            watcher_thread.join(timeout=1.0)

        print("[PHASE 1] Completed. Transitioning to Homing...")

        # PHASE 2
        robot = client.robot
        current_obs = robot.get_observation()
        joint_names = list(robot.action_features.keys())

        # 1. Thả kẹp tại vị trí hiện tại
        action_dict = _extract_current_action_from_observation(current_obs, joint_names)
        _open_gripper_if_present(action_dict, joint_names)
        robot.send_action(action_dict)
        time.sleep(1.5)  # Chờ thả rơi vật thể an toàn

        # 2. Thu tay về
        _home_to_zero(robot, homing_duration, action_dict, joint_names)
        print("[PHASE 2] Homing complete.")

        # PHASE 3: STANDBY
        print("\n[PHASE 3] STANDBY MODE")
        print("Robot is holding zero pose. Press Ctrl + C to fully stop.")
        while True:
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n[INFO] Shutdown signal received.")
    finally:
        try:
            print("[FINAL] Disconnecting safely...")
            client.robot.disconnect()
        except Exception as exc:
            print(f"[WARN] Disconnect failed: {exc}")


@draccus.wrap()
def main(cfg: RobotClientConfig) -> None:
    run_full_demo(cfg, vla_duration=1000.0, homing_duration=15.0)


if __name__ == "__main__":
    main()
