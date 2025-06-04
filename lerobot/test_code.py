import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.policies.factory import make_policy
from lerobot.common.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.common.robots.lekiwi.lekiwi_client import LeKiwiClient
from lerobot.common.utils.control_utils import predict_action
from lerobot.common.utils.utils import get_safe_torch_device
from lerobot.configs.policies import PreTrainedConfig

NB_CYCLES_CLIENT_CONNECTION = 1000

robot_config = LeKiwiClientConfig(remote_ip="172.18.134.136", id="lekiwi")
robot = LeKiwiClient(robot_config)

robot.connect()

model_config = PreTrainedConfig.from_pretrained("pepijn223/act_lekiwi_circle")
metadata = LeRobotDatasetMetadata(repo_id="pepijn223/lekiwi1749025613")

policy = make_policy(cfg=model_config, ds_meta=metadata)
policy.reset()

print("Running inference")
i = 0
while i < NB_CYCLES_CLIENT_CONNECTION:
    obs = robot.get_observation()

    for key, value in obs.items():
        if isinstance(value, torch.Tensor):
            obs[key] = value.detach().cpu().numpy()

    action_values = predict_action(
        obs, policy, get_safe_torch_device(policy.config.device), policy.config.use_amp
    )
    action = {
        key: action_values[i].item() if isinstance(action_values[i], torch.Tensor) else action_values[i]
        for i, key in enumerate(robot.action_features)
    }
    robot.send_action(action)
    i += 1

robot.disconnect()
