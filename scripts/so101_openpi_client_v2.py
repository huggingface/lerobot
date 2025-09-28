import cv2, numpy as np, time
import traceback
from openpi_client import websocket_client_policy, image_tools
from lerobot.robots import make_robot_from_config
from lerobot.robots.so101_follower import SO101FollowerConfig

WS_HOST = "127.0.0.1"
WS_PORT = 8000
CAM_FRONT = 0    # 机側（front）
CAM_ABOVE = 2    # 真上（above）
FPS = 10          # 推論頻度を下げて安定性向上
ROBOT_PORT = "/dev/ttyACM1"
ROBOT_ID = "my_follower_arm"
JOINT_ORDER = ["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]

DRY_RUN = False  # 実際にロボットにアクションを送信
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_DELAY = 2.0

def to_chw_uint8(img_bgr):
    """BGRイメージをCHW uint8形式に変換（OpenPI用）"""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = image_tools.convert_to_uint8(img)
    return np.transpose(img, (2,0,1)).copy(order="C")  # (3,H,W), uint8, C連続

def validate_observation(obs):
    """観測データの検証"""
    try:
        # イメージ形状チェック
        img_front = obs["observation/image"]
        img_above = obs["observation/wrist_image"]
        state = obs["observation/state"]
        
        if not isinstance(img_front, np.ndarray) or img_front.shape[0] != 3:
            raise ValueError(f"front image shape invalid: {img_front.shape}")
        if not isinstance(img_above, np.ndarray) or img_above.shape[0] != 3:
            raise ValueError(f"above image shape invalid: {img_above.shape}")
        if len(state) != 6:
            raise ValueError(f"state dimension invalid: {len(state)}")
            
        return True
    except Exception as e:
        print(f"観測データ検証エラー: {e}")
        return False

def create_client():
    """WebSocketクライアント作成（エラーハンドリング付き）"""
    try:
        client = websocket_client_policy.WebsocketClientPolicy(host=WS_HOST, port=WS_PORT)
        print(f"WebSocket接続成功: {WS_HOST}:{WS_PORT}")
        return client
    except Exception as e:
        print(f"WebSocket接続失敗: {e}")
        return None


def main():
    # 初期WebSocket接続
    client = create_client()
    if not client:
        print("初期WebSocket接続に失敗しました")
        return

    # ロボット接続
    cfg = SO101FollowerConfig(id=ROBOT_ID, port=ROBOT_PORT, cameras={}, use_degrees=True)
    robot = make_robot_from_config(cfg)
    robot.connect(calibrate=True)

    # カメラ初期化
    cap_front = cv2.VideoCapture(CAM_FRONT)
    cap_above = cv2.VideoCapture(CAM_ABOVE)
    
    # カメラ設定最適化
    for cap in [cap_front, cap_above]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("制御ループ開始。Ctrl-Cで停止。")
    try:
        step_count = 0
        
        while True:
            ok_f, frame_front = cap_front.read()
            ok_a, frame_above = cap_above.read()
            if not ok_f or not ok_a:
                print("カメラ読み込み失敗")
                time.sleep(1.0/FPS); continue

            img_front = to_chw_uint8(frame_front)
            img_above = to_chw_uint8(frame_above)

            obs_robot = robot.get_observation()
            state = np.array([obs_robot.get(f"{j}.pos", 0.0) for j in JOINT_ORDER], dtype=np.float32)

            # SO101Inputsトランスフォームが期待する形式に合わせる
            obs = {
                "observation/image": img_front, 
                "observation/wrist_image": img_above,
                "observation/state": state,
                "prompt": "Pick and Place the gray cube to circle on table"  # デフォルトプロンプト
            }
            
            # データ検証
            if not validate_observation(obs):
                print("観測データが無効です。スキップします。")
                time.sleep(1.0/FPS)
                continue
            
            # デバッグ用画像保存（50ステップごと）
            if step_count % 50 == 0:
                ts = int(time.time() * 1000)
                cv2.imwrite(f"debug_front_{ts}.png", frame_front)
                cv2.imwrite(f"debug_above_{ts}.png", frame_above)

            print(f"Step {step_count}, State: [{', '.join(f'{s:6.1f}' for s in state)}]")
            
            # 推論実行
            out = client.infer(obs)
            actions = np.asarray(out.get("actions", []), dtype=np.float32)
            print(f"受信アクション形状: {actions.shape}")
            
            if actions.size > 0 and actions.ndim > 1:
                for i in range(len(actions)):
                    action = actions[i]
                    print(f"受信アクション {i}: [{', '.join(f'{a:6.1f}' for a in action)}]")
                    print(f"受信アクション: [{', '.join(f'{a:6.1f}' for a in action)}]")
                    
                    if not DRY_RUN:
                        cmd = {f"{j}.pos": float(action[i]) for i, j in enumerate(JOINT_ORDER)}
                        robot.send_action(cmd)
                        time.sleep(1.0/FPS)
                        # print(f"[ACTION] {cmd}")
                                
            step_count += 1
            time.sleep(1.0/FPS)
    finally:
        print("クリーンアップ中...")
        cap_front.release()
        cap_above.release() 
        robot.disconnect()
        print("クリーンアップ完了")

if __name__ == "__main__":
    main()
