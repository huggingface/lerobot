import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 20


def main():
    device = torch.device("mps")  # or "cuda" or "cpu"
    model_id = "lerobot/pi0_base"

    model = PI0Policy.from_pretrained(model_id)

    preprocess, postprocess = make_pre_post_processors(
        model.config,
        model_id,
        # This overrides allows to run on MPS, otherwise defaults to CUDA (if available)
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # find ports using lerobot-find-port
    follower_port = ...  # something like "/dev/tty.usbmodem58760431631"

    # the robot ids are used the load the right calibration files
    follower_id = ...  # something like "follower_so100"

    # Robot and environment configuration
    # Camera keys must match the name and resolutions of the ones used for training!
    # You can check the camera keys expected by a model in the info.json card on the model card on the Hub
    camera_config = {
        "base_0_rgb": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
        "left_wrist_0_rgb": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=30),
        "right_wrist_0_rgb": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=30),
    }

    robot_cfg = SO100FollowerConfig(port=follower_port, id=follower_id, cameras=camera_config)
    robot = SO100Follower(robot_cfg)
    robot.connect()

    task = ""  # something like "pick the red block"
    robot_type = ""  # something like "so100_follower" for multi-embodiment datasets

    # This is used to match the raw observation keys to the keys expected by the policy
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    for _ in range(MAX_EPISODES):
        for _ in range(MAX_STEPS_PER_EPISODE):
            obs = robot.get_observation()
            obs_frame = build_inference_frame(
                observation=obs, ds_features=dataset_features, device=device, task=task, robot_type=robot_type
            )

            obs = preprocess(obs_frame)

            action = model.select_action(obs)
            action = postprocess(action)
            action = make_robot_action(action, dataset_features)
            robot.send_action(action)

        print("Episode finished! Starting new episode...")


if __name__ == "__main__":
    main()
