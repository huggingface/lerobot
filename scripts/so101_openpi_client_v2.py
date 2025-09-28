import cv2
import numpy as np
from openpi_client import websocket_client_policy, image_tools
import time
from lerobot.robots import make_robot_from_config
from lerobot.robots.so101_follower import SO101FollowerConfig

# Configuration constants
WS_HOST = "127.0.0.1"
WS_PORT = 9000
CAM_HIGH = 0
CAM_WRIST = 2
FPS = 10
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
        while True:
            # capture images
            ok_h, frame_h = cap_h.read()
            ok_w, frame_w = cap_w.read()
            if not ok_h or not ok_w:
                print("Camera read failed, skipping iteration")
                time.sleep(1.0 / FPS)
                continue

            # Convert BGR->RGB (OpenCV gives BGR). Let server handle resize/padding.
            img_h = cv2.cvtColor(frame_h, cv2.COLOR_BGR2RGB)
            img_w = cv2.cvtColor(frame_w, cv2.COLOR_BGR2RGB)
            img_h = image_tools.convert_to_uint8(img_h)
            img_w = image_tools.convert_to_uint8(img_w)
            img_h = np.transpose(img_h, (2, 0, 1)).copy(order="C")  # (3,H,W)
            img_w = np.transpose(img_w, (2, 0, 1)).copy(order="C")  # (3,H,W)


            # get robot state
            obs_robot = robot.get_observation()
            state = np.array([obs_robot.get(f"{j}.pos", 0.0) for j in JOINT_ORDER], dtype=np.float32)

            # build observation
            # OpenPI (pi0.5 Aloha-style) server expects keys: 'images': {cam_high, cam_left_wrist}, 'state', 'prompt'
            # because training repack mapped:
            #   cam_high <- observation.images.above
            #   cam_left_wrist <- observation.images.front
            #   state <- observation.state
            obs = {
                "prompt": "Pick and Place the gray cube to circle on table",
                "images": {                   # Aloha入力
                    "cam_high": img_h,
                    "cam_left_wrist": img_w,
                },
                "state": state.tolist(),
            }
            # 念のため互換キーも同報（無害）
            obs["image"] = obs["images"]
            print(obs["state"])
            print("cam_high", img_h.shape, img_h.dtype, img_h.flags['C_CONTIGUOUS'])
            print("cam_left_wrist", img_w.shape, img_w.dtype, img_w.flags['C_CONTIGUOUS'])  
            # Basic sanity check
            for k, v in obs["images"].items():
                if v.ndim != 3 or v.shape[2] != 3:
                    print(f"Warning: camera {k} unexpected shape {v.shape}")

            # infer actions
            output = client.infer(obs)
            actions = np.asarray(output.get("actions", []), dtype=np.float32)
            print("Received", len(actions), "actions from policy")

            # send each action to robot
            for idx, a in enumerate(actions):
                cmd = {f"{j}.pos": float(a[i]) for i, j in enumerate(JOINT_ORDER)}
                print(f"[ACTION {idx}]", cmd)
                robot.send_action(cmd)
                time.sleep(1.0 / FPS)
    except KeyboardInterrupt:
        print("Received interrupt, shutting down")
    finally:
        cap_h.release()
        cap_w.release()
        robot.disconnect()


if __name__ == "__main__":
    main()