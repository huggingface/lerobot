import time

import torch

from lerobot.common.cameras.opencv import OpenCVCameraConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.robots import make_robot_from_config
from lerobot.common.robots.koch_screwdriver_follower import KochScrewdriverFollowerConfig
from lerobot.common.utils.robot_utils import busy_wait

inference_time_s = 20
fps = 30
device = "cuda"  # TODO: On Mac, use "mps" or "cpu"

# ckpt_path = "outputs/train/act_koch_screwdriver_with_validation_16/checkpoints/last/pretrained_model"
ckpt_path = (
    "outputs/screwdriver_attach_orange_panel_cleaned_t90_v10_clean_5/checkpoints/060000/pretrained_model"
)
policy = ACTPolicy.from_pretrained(ckpt_path)
policy.to(device)

# Configure robot using the new approach
robot_config = KochScrewdriverFollowerConfig(
    port="/dev/servo_5837053138",
    id="koch_screwdriver_follower_testing",
    cameras={
        "screwdriver": OpenCVCameraConfig(index_or_path="/dev/video0", width=800, height=600, fps=30),
        "side": OpenCVCameraConfig(index_or_path="/dev/video2", width=800, height=600, fps=30),
        "top": OpenCVCameraConfig(index_or_path="/dev/video6", width=800, height=600, fps=30),
    },
)
robot = make_robot_from_config(robot_config)
robot.connect()

for t in range(inference_time_s * fps):
    start_time = time.perf_counter()

    # Read the follower state and access the frames from the cameras
    observation = robot.get_observation()

    # Convert to pytorch format: channel first and float32 in [0,1]
    # with batch dimension
    processed_observation = {}

    # Collect motor states in the correct order
    # Based on koch_screwdriver_follower, we have 5 position states + 1 velocity state
    state_values = []
    state_values.append(observation["shoulder_pan.pos"])
    state_values.append(observation["shoulder_lift.pos"])
    state_values.append(observation["elbow_flex.pos"])
    state_values.append(observation["wrist_flex.pos"])
    state_values.append(observation["wrist_roll.pos"])
    state_values.append(observation["screwdriver.vel"])

    # Combine into a single state tensor
    state_tensor = torch.tensor(state_values, dtype=torch.float32).unsqueeze(0)
    processed_observation["observation.state"] = state_tensor.to(device)

    # Process images
    for cam_name in ["screwdriver", "side", "top"]:
        if cam_name in observation:
            # Convert numpy image to tensor: HWC -> CHW, normalize to [0,1]
            image = torch.from_numpy(observation[cam_name]).float() / 255.0
            image = image.permute(2, 0, 1).contiguous()
            image = image.unsqueeze(0)  # Add batch dimension
            processed_observation[f"observation.images.{cam_name}"] = image.to(device)

    # Compute the next action with the policy
    # based on the current observation
    action = policy.select_action(processed_observation)
    # Remove batch dimension
    action = action.squeeze(0)
    # Move to cpu, if not already the case
    action = action.to("cpu")

    # Convert action tensor to dictionary format expected by robot
    action_dict = {
        "shoulder_pan.pos": action[0].item(),
        "shoulder_lift.pos": action[1].item(),
        "elbow_flex.pos": action[2].item(),
        "wrist_flex.pos": action[3].item(),
        "wrist_roll.pos": action[4].item(),
        "screwdriver.vel": action[5].item(),
    }

    robot.send_action(action_dict)

    # Print t ever module fps
    if t % fps == 0:
        print(f"t: {t}")

    dt_s = time.perf_counter() - start_time
    busy_wait(1 / fps - dt_s)
