import os
import time
import traceback

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from vx300s_env import VX300sEnv

FILEPATH = "./MJCF/so-arm101/scene.xml"
MODELNAME = "ppo_vx300s_reach_goal"
env = VX300sEnv(FILEPATH)
check_env(env)


if not os.path.exists(f"{MODELNAME}.zip"):
    # モデル定義＆学習
    model = PPO("MlpPolicy", env, verbose=1, device="cpu")
    # model.learn(total_timesteps=100_000)
    model.learn(total_timesteps=10_000)
    model.save(MODELNAME)
else:
    model = PPO.load(MODELNAME, env=env)

# テスト
obs, _ = env.reset()
try:
    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"reward: {reward}")
        env.render()

        if done:
            obs, _ = env.reset()

        if not env.is_running:
            break
        time.sleep(env.model.opt.timestep)
except Exception as e:
    print(e, traceback.print_exc())
finally:
    # 終了時にスレッドが残ってしまいプロセスが終わらないので強制終了
    env.close()
    os._exit(0)
