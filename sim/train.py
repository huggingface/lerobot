from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from vx300s_env import VX300sEnv

FILEPATH = "./MJCF/so-arm101/scene.xml"
# 環境を作成
# env = VX300sEnv(FILEPATH, render_mode="human")
env = VX300sEnv(FILEPATH, render_mode="human")
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
        print(f"obs: {obs}, reward: {reward}")
        env.render()  # 非同期レンダリング
finally:
    env.close()
