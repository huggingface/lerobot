import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
import numpy as np
import matplotlib.pyplot as plt
from lerobot.robots.assembling_sim import AssemblingSim, AssemblingSimCut, AssemblingSimConfig
from state_models import StateClassifier, transform

from PIL import Image

FPS = 20

MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 10000

def get_model(model_id, dataset_id):
    model = ACTPolicy.from_pretrained(model_id)
    dataset_metadata = LeRobotDatasetMetadata(dataset_id)
    preprocess, postprocess = make_pre_post_processors(model.config, pretrained_path=model_id, dataset_stats=dataset_metadata.stats)

    return model, preprocess, postprocess, dataset_metadata

def main():
    device = torch.device("cuda")  # or "cuda" or "cpu"

    model = StateClassifier().to("cuda")
    model.load_state_dict(torch.load("act_in_sim/models/state_model.pt", map_location="cuda"))  # или "cuda"
    model.eval()

    model_1, preprocess_1, postprocess_1, dataset_metadata_1 = get_model("outputs/assembling/act_s1/last", "local/ACT_assembling_sim_s1")
    model_2, preprocess_2, postprocess_2, dataset_metadata_2 = get_model("outputs/assembling/act_s2/last", "local/ACT_assembling_sim_s2")

    sim_config = AssemblingSimConfig(
        xml_path="scene.xml",
        sim_timestep=0.001,
        control_hz=FPS,
        mode="realtime",   # "realtime" | "fast"
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

    fig, ax = plt.subplots()
    img = ax.imshow(np.zeros((224,224,3)))

    for _ in range(MAX_EPISODES):
        for _ in range(MAX_STEPS_PER_EPISODE):
            obs = robot.get_observation()

            image = obs["cam_state"]

            img = Image.fromarray(image, mode="RGB")
            input_tensor = transform(img).unsqueeze(0).to("cuda")  # добавляем batch dimension: (1, 3, 224, 224)

            with torch.no_grad():
                prediction, conf = model.predict(input_tensor)  # prediction shape: (1, 4)

                stage = prediction.cpu().numpy()[0]
                print("Stage: ", stage)
                print("Conf: ", conf.cpu().numpy()[0])
                print()

            if stage == 0:
                obs_frame = build_inference_frame(
                    observation=obs, ds_features=dataset_metadata_1.features, device=device
                )

                obs = preprocess_1(obs_frame)
                action = model_1.select_action(obs)
                action = postprocess_1(action)
                action = make_robot_action(action, dataset_metadata_1.features)
                robot.send_action(action)

            elif stage >= 1:
                obs_frame = build_inference_frame(
                    observation=obs, ds_features=dataset_metadata_2.features, device=device
                )

                obs = preprocess_2(obs_frame)
                action = model_2.select_action(obs)
                action = postprocess_2(action)
                action = make_robot_action(action, dataset_metadata_2.features)
                robot.send_action(action)

        print("Episode finished! Starting new episode...")


if __name__ == "__main__":
    main()
