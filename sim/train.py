import os
import time
import traceback

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from vx300s_env import VX300sEnv

FILEPATH = "./MJCF/so-arm101/scene.xml"
env = VX300sEnv(FILEPATH)
check_env(env)


# モデル定義＆学習
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# テスト
obs, _ = env.reset()
try:
    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"reward: {reward}")
        env.render()
        if not env.is_running:
            break
        time.sleep(env.model.opt.timestep)
except Exception as e:
    print(e, traceback.print_exc())
finally:
    # 終了時にスレッドが残ってしまいプロセスが終わらないので強制終了
    env.close()
    os._exit(0)
