import os
import time
import traceback

import mujoco
import mujoco.viewer
from stable_baselines3.common.env_checker import check_env

from vx300s_env import VX300sEnv

FILEPATH = "./MJCF/so-arm101/scene.xml"
env = VX300sEnv(FILEPATH)
check_env(env)

obs, _ = env.reset()
try:
    while True:
        mujoco.mj_step(env.model, env.data)
        reward, distance = env.get_reward()
        print(f"{reward=:3.5f}, {distance=:3.5f}")

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
