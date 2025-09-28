import cv2
import numpy as np
from openpi_client import websocket_client_policy, image_tools
import time
from lerobot.robots import make_robot_from_config
from lerobot.robots.so101_follower import SO101FollowerConfig

# Configuration constants
WS_HOST = "127.0.0.1"
WS_PORT = 9000
CAM_HIGH = 2
CAM_WRIST = 0
CTRL_HZ = 5.0          # 制御周期 10 Hz
INFER_EVERY = 6         # 推論は6制御周期に1回（= horizon と合わせる）
ALPHA = 0.9             # EMA係数（0<ALPHA<=1）
MAX_STEP_DEG = 45.0      # 1制御周期あたり関節の最大変化量[deg]
ROBOT_PORT = "/dev/ttyACM1"
ROBOT_ID = "my_follower_arm"
JOINT_ORDER = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]


def main():
    # Connect to policy server
    client = websocket_client_policy.WebsocketClientPolicy(host=WS_HOST, port=WS_PORT)
    # Initialize robot
    cfg = SO101FollowerConfig(id=ROBOT_ID, port=ROBOT_PORT, cameras={}, use_degrees=True)
    robot = make_robot_from_config(cfg)
    robot.connect(calibrate=False)

    # Open cameras
    cap_h = cv2.VideoCapture(CAM_HIGH)
    cap_w = cv2.VideoCapture(CAM_WRIST)

    print("Entering main control loop. Press Ctrl-C to exit.")
    try:
        t_next = time.monotonic()
        ema = None
        buf = []            # アクションバッファ（チャンク）
        step_idx = 0
        while True:
            # 周期整定
            t_next += 1.0/CTRL_HZ
            now = time.monotonic()

            # capture
            ok_h, frame_h = cap_h.read()
            ok_w, frame_w = cap_w.read()
            if not ok_h or not ok_w:
                print("Camera read failed, skipping iteration")
                time.sleep(max(0.0, t_next - now))
                continue

            # 画像前処理（Aloha: CHW, uint8）※サイズ変更はサーバ側に任せる
            img_h = cv2.cvtColor(frame_h, cv2.COLOR_BGR2RGB)
            img_w = cv2.cvtColor(frame_w, cv2.COLOR_BGR2RGB)
            img_h = np.transpose(image_tools.convert_to_uint8(img_h), (2,0,1)).copy(order="C")
            img_w = np.transpose(image_tools.convert_to_uint8(img_w), (2,0,1)).copy(order="C")

            # 状態取得
            obs_robot = robot.get_observation()
            state = np.array([obs_robot.get(f"{j}.pos", 0.0) for j in JOINT_ORDER], dtype=np.float32)
            print(f"[STATE] {' '.join(f'{s:7.1f}' for s in state)}")

            # 必要なときだけ infer（チャンク満了 or 指定周期）
            if (not buf) or (step_idx % INFER_EVERY == 0):
                obs = {"prompt": "Pick and Place the gray cube to circle on table",
                       "images": {"cam_high": img_h, "cam_left_wrist": img_w},
                       "state": state.tolist()}
                out = client.infer(obs)
                buf = list(np.asarray(out["actions"], dtype=np.float32))  # list of (6,)
                step_idx = 0

            # バッファ先頭を取り出し（6次元のみ使用）
            a = (buf.pop(0) if buf else np.zeros(14, np.float32))[:6]
            # # EMA平滑化（Temporal Ensemble）
            # ema = a if ema is None else (1-ALPHA)*ema + ALPHA*a
            # a_smooth = ema
            # # 速度制限（前回送付値prevからの変化をクリップ）
            # if 'prev' not in locals(): prev = state  # 初回は現在角度を基準
            # delta = np.clip(a_smooth - prev, -MAX_STEP_DEG, MAX_STEP_DEG)
            # a_limited = prev + delta
            # prev = a_limited.copy()
            a_limited = a
            # 送出
            cmd = {f"{j}.pos": float(a_limited[i]) for i, j in enumerate(JOINT_ORDER)}
            print(f"[ACTION]", cmd)
            robot.send_action(cmd)
            step_idx += 1
            # 次周期までスリープ
            time.sleep(max(0.0, t_next - time.monotonic()))
    except KeyboardInterrupt:
        print("Received interrupt, shutting down")
    finally:
        cap_h.release()
        cap_w.release()
        robot.disconnect()


if __name__ == "__main__":
    main()