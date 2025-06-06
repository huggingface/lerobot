from lerobot.common.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.common.utils.control_utils import predict_action
from lerobot.common.utils.utils import get_safe_torch_device

NB_CYCLES_CLIENT_CONNECTION = 1000

robot_config = LeKiwiClientConfig(remote_ip="172.18.134.136", id="lekiwi")
robot = LeKiwiClient(robot_config)

robot.connect()

policy = ACTPolicy.from_pretrained("pepijn223/act_lekiwi_circle")
policy.reset()

obs_features = hw_to_dataset_features(robot.observation_features, "observation")

print("Running inference")
i = 0
while i < NB_CYCLES_CLIENT_CONNECTION:
    obs = robot.get_observation()

    observation_frame = build_dataset_frame(obs_features, obs, prefix="observation")
    action_values = predict_action(
        observation_frame, policy, get_safe_torch_device(policy.config.device), policy.config.use_amp
    )
    action = {key: action_values[i].item() for i, key in enumerate(robot.action_features)}
    robot.send_action(action)
    i += 1

robot.disconnect()
