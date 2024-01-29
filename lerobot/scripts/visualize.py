import pickle
from pathlib import Path

import imageio
import simxarm

if __name__ == "__main__":

    task = "lift"
    dataset_dir = Path(f"data/xarm_{task}_medium")
    dataset_path = dataset_dir / f"buffer.pkl"
    print(f"Using offline dataset '{dataset_path}'")
    with open(dataset_path, "rb") as f:
        dataset_dict = pickle.load(f)

    required_keys = [
        "observations",
        "next_observations",
        "actions",
        "rewards",
        "dones",
        "masks",
    ]
    for k in required_keys:
        if k not in dataset_dict and k[:-1] in dataset_dict:
            dataset_dict[k] = dataset_dict.pop(k[:-1])

    out_dir = Path("tmp/2023_01_26_xarm_lift_medium")
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = dataset_dict["observations"]["rgb"][:100]
    frames = frames.transpose(0, 2, 3, 1)
    imageio.mimsave(out_dir / "test.mp4", frames, fps=30)

    frames = []
    cfg = {}

    env = simxarm.make(
        task=task,
        obs_mode="all",
        image_size=84,
        action_repeat=cfg.get("action_repeat", 1),
        frame_stack=cfg.get("frame_stack", 1),
        seed=1,
    )

    obs = env.reset()
    frame = env.render(mode="rgb_array", width=384, height=384)
    frames.append(frame)

    # def is_first_obs(obs):
    #     nonlocal first_obs
    #     print(((dataset_dict["observations"]["state"][i]-obs["state"])**2).sum())
    #     print(((dataset_dict["observations"]["rgb"][i]-obs["rgb"])**2).sum())

    for i in range(25):
        action = dataset_dict["actions"][i]

        print(f"#{i}")
        # print(obs["state"])
        # print(dataset_dict["observations"]["state"][i])
        print(((dataset_dict["observations"]["state"][i] - obs["state"]) ** 2).sum())
        print(((dataset_dict["observations"]["rgb"][i] - obs["rgb"]) ** 2).sum())

        obs, reward, done, info = env.step(action)
        frame = env.render(mode="rgb_array", width=384, height=384)
        frames.append(frame)

        print(reward)
        print(dataset_dict["rewards"][i])

        print(done)
        print(dataset_dict["dones"][i])

        if dataset_dict["dones"][i]:
            obs = env.reset()
            frame = env.render(mode="rgb_array", width=384, height=384)
            frames.append(frame)

    # imageio.mimsave(out_dir / 'test_rollout.mp4', frames, fps=60)
