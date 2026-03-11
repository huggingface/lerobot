import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 20


def main():
    device = torch.device("mps")  # or "cuda" or "cpu"
    model_id = "<user>/robot_learning_tutorial_diffusion"

    model = DiffusionPolicy.from_pretrained(model_id)

    dataset_id = "lerobot/svla_so101_pickplace"
    # This only downloads the metadata for the dataset, ~10s of MB even for large-scale datasets
    dataset_metadata = LeRobotDatasetMetadata(dataset_id)
    preprocess, postprocess = make_pre_post_processors(
        model.config, model_id, dataset_stats=dataset_metadata.stats
    )

    # # find ports using lerobot-find-port
    follower_port = ...  # something like "/dev/tty.usbmodem58760431631"

    # # the robot ids are used the load the right calibration files
    follower_id = ...  # something like "follower_so100"

    # Robot and environment configuration
    # Camera keys must match the name and resolutions of the ones used for training!
    # You can check the camera keys expected by a model in the info.json card on the model card on the Hub
    camera_config = {
        "side": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
        "up": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=30),
    }

    robot_cfg = SO100FollowerConfig(port=follower_port, id=follower_id, cameras=camera_config)
    robot = SO100Follower(robot_cfg)
    robot.connect()

    for _ in range(MAX_EPISODES):
        for _ in range(MAX_STEPS_PER_EPISODE):
            obs = robot.get_observation()
            obs_frame = build_inference_frame(
                observation=obs, ds_features=dataset_metadata.features, device=device
            )

            obs = preprocess(obs_frame)

            action = model.select_action(obs)
            action = postprocess(action)
            action = make_robot_action(action, dataset_metadata.features)
            robot.send_action(action)

        print("Episode finished! Starting new episode...")


if __name__ == "__main__":
    main()
