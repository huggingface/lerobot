from stable_baselines3.common.env_checker import check_env
from vx300s_env import VX300sEnv

FILEPATH = "./MJCF/so-arm101/scene.xml"
# 環境を作成
# env = VX300sEnv(FILEPATH, render_mode="human")
env = VX300sEnv(FILEPATH, render_mode="human")
check_env(env)

# try:
#     while env.viewer.is_running():
#         goal_site_id = mujoco.mj_name2id(
#             env.model, mujoco.mjtObj.mjOBJ_SITE, "goal_site"
#         )
#         env.model.site_pos[goal_site_id] = np.array([0.1, 0.0, 0.1])
#         env.model.site_pos[goal_site_id] = env.goal

#         pinch_id = mujoco.mj_name2id(
#             env.model,
#             mujoco.mjtObj.mjOBJ_SITE,
#             "pinch",
#         )
#         pinch_site_id = mujoco.mj_name2id(
#             env.model, mujoco.mjtObj.mjOBJ_SITE, "pinch_site"
#         )
#         env.model.site_pos[pinch_site_id] = env.model.site_pos[pinch_id]

#         print(
#             f"{pinch_id=}, {pinch_site_id=}, {env.model.site_pos[pinch_site_id]=}, {env.model.site_pos[pinch_id]=}"
#         )

#         mujoco.mj_forward(env.model, env.data)
#         env.render()
# except KeyboardInterrupt:
#     env.close()

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
        # print(f"obs: {obs}, reward: {reward}")
        print(f"reward: {reward}")
        env.render()  # 非同期レンダリング
finally:
    env.close()
