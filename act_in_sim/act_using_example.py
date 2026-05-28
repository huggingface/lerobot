import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
import numpy as np
import matplotlib.pyplot as plt
from lerobot.robots.assembling_sim import AssemblingSim, AssemblingSimCut, AssemblingSimConfig

FPS = 20

MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 10000


def main():
    device = torch.device("cuda")  # or "cuda" or "cpu"
    model_id = "outputs/assembling/act_s1/last"
    model = ACTPolicy.from_pretrained(model_id)

    dataset_id = "local/ACT_assembling_sim_s1"
    # This only downloads the metadata for the dataset, ~10s of MB even for large-scale datasets
    dataset_metadata = LeRobotDatasetMetadata(dataset_id)
    # preprocess, postprocess = make_pre_post_processors(model.config, dataset_stats=dataset_metadata.stats)
    preprocess, postprocess = make_pre_post_processors(model.config, pretrained_path=model_id, dataset_stats=dataset_metadata.stats)

    sim_config = AssemblingSimConfig(
        xml_path="scene.xml",
        sim_timestep=0.001,
        control_hz=FPS,
        mode="fast",   # "realtime" | "fast"
        max_episode_steps=1000,
        use_task_space=True,
        render_mode="all",   # None | "human" | "rgb_array" | "all"
        camera_names=["cam_front", "cam_side", "cam_gripper", "cam_state"],
        resolution=(224, 224),

        action_pos_scale=1000,
        action_angle_scale=100
    )

    robot = AssemblingSimCut(sim_config)

    robot.connect()

    # plt.ion()  # интерактивный режим

    fig, ax = plt.subplots()
    img = ax.imshow(np.zeros((224,224,3)))

    for _ in range(MAX_EPISODES):
        for _ in range(MAX_STEPS_PER_EPISODE):
            obs = robot.get_observation()

            print(obs["cam_state"])
            obs_frame = build_inference_frame(
                observation=obs, ds_features=dataset_metadata.features, device=device
            )

            obs = preprocess(obs_frame)

            # print(obs_frame)

            # img.set_data(obs['observation.images.front'].cpu().numpy()[0].transpose((1,2,0)))
            # img.set_data(obs_frame['observation.images.front'].cpu().numpy()[0].transpose((1,2,0)))
            # plt.pause(0.01)  # ~30 FPS


            action = model.select_action(obs)
            action = postprocess(action)

            action = make_robot_action(action, dataset_metadata.features)

            robot.send_action(action)

        print("Episode finished! Starting new episode...")


if __name__ == "__main__":
    main()
