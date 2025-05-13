import argparse
import os
import time
import traceback

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from vx300s_env import VX300sEnv

FILEPATH = "./MJCF/so-arm101/scene.xml"
MODELNAME = "ppo_vx300s_reach_goal"
MODELFILE = f"{MODELNAME}.zip"
env = VX300sEnv(FILEPATH)
check_env(env)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--delete", action="store_true", help="既存モデルを削除して再学習する"
)
args = parser.parse_args()

if args.delete and os.path.exists(MODELFILE):
    print(f"[INFO] モデルファイル {MODELFILE} を削除します。")
    os.remove(MODELFILE)

if not os.path.exists(f"{MODELNAME}.zip"):
    # モデル定義＆学習
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=64,
        verbose=1,
        device="cpu",
        normalize_advantage=True,
        tensorboard_log="./ppo_tensorboard/",
    )
    # model.learn(total_timesteps=100_000)
    model.learn(total_timesteps=200_000)
    model.save(MODELNAME)
else:
    model = PPO.load(MODELNAME, env=env)

# テスト
obs, _ = env.reset()
try:
    for _ in range(10000):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"reward: {reward}")
        env.render()

        if terminated:
            print("terminated!")
            obs, _ = env.reset()
            time.sleep(3)

        if truncated:
            print("truncated!")
            obs, _ = env.reset()
            time.sleep(3)

        if not env.is_running:
            break
        time.sleep(env.model.opt.timestep)
except Exception as e:
    print(e, traceback.print_exc())
finally:
    # 終了時にスレッドが残ってしまいプロセスが終わらないので強制終了
    env.close()
    os._exit(0)
